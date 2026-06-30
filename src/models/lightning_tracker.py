"""
PyTorch Lightning Module for Siamese Object Tracker v2.

Full training pipeline including:
    - Focal Loss (classification)
    - GIoU Loss (regression)
    - Multi-scale Feature-level Knowledge Distillation with spatial attention masks
    - QAT hooks (torch.ao.quantization)
    - TensorBoard logging: scalars, images, and tracking visualizations
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torchvision.utils as vutils
import numpy as np

from .siam_tracker import SiamTracker
from .head import DepthwiseSeparableConv


# ============================================================================
# Teacher Model (ResNet-50 SiamRPN++)
# ============================================================================

class TeacherSiamRPN(nn.Module):
    """
    Teacher SiamRPN++ with ResNet-50 backbone for multi-scale knowledge distillation.

    Extracts intermediate feature maps from layer2 (512ch), layer3 (1024ch),
    and layer4 (2048ch) for multi-scale feature distillation.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        except ImportError:
            resnet = models.resnet50(pretrained=pretrained)

        # Split ResNet-50 into stages
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
        )
        self.layer2 = resnet.layer2     # 512 channels
        self.layer3 = resnet.layer3     # 1024 channels
        self.layer4 = resnet.layer4     # 2048 channels

        # Modify layer3 and layer4 strides for tracking (dilated)
        self._modify_stride_dilation(self.layer3, stride=1, dilation=2)
        self._modify_stride_dilation(self.layer4, stride=1, dilation=4)

        # Channel dimensions for multi-scale distillation
        self.layer_channels = [512, 1024, 2048]

    @staticmethod
    def _modify_stride_dilation(layer, stride=1, dilation=2):
        """Modify a ResNet layer to use dilated convolutions with stride 1."""
        for i, block in enumerate(layer.children()):
            if i == 0:
                if block.downsample is not None:
                    for sub in block.downsample.children():
                        if isinstance(sub, nn.Conv2d):
                            sub.stride = (stride, stride)
                block.conv2.stride = (stride, stride)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)
            else:
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)

    def extract_multiscale(self, x):
        """
        Extract multi-scale features from the search region.

        Args:
            x: [B, 3, 255, 255]
        Returns:
            dict with keys 'layer2', 'layer3', 'layer4' containing feature maps.
        """
        x = self.stem(x)                                               # [B, 256, H, W]
        f2 = self.layer2(x)                                            # [B, 512, H/8, W/8]
        f3 = self.layer3(f2)                                           # [B, 1024, H/8, W/8]
        f4 = self.layer4(f3)                                           # [B, 2048, H/8, W/8]
        return {"layer2": f2, "layer3": f3, "layer4": f4}

    def load_weights(self, weights_path):
        """Load pre-trained teacher weights."""
        if not os.path.exists(weights_path):
            print(f"Teacher weights not found at {weights_path}. Using ImageNet init.")
            return

        print(f"Loading teacher weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint) \
            if isinstance(checkpoint, dict) else checkpoint

        model_dict = self.state_dict()
        matched = {}
        for k, v in state_dict.items():
            clean = k.replace('backbone.', '').replace('neck.', 'cls_adjust_')
            if clean in model_dict and model_dict[clean].shape == v.shape:
                matched[clean] = v

        print(f"Matched {len(matched)}/{len(model_dict)} keys from teacher checkpoint.")
        model_dict.update(matched)
        self.load_state_dict(model_dict, strict=False)


# ============================================================================
# Loss Functions
# ============================================================================

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    α-balanced focal loss for classification.

    Args:
        logits:  [B, 1, H, W] — raw logits.
        targets: [B, 1, H, W] — binary labels (0 or 1).
        alpha:   weighting factor for positive class.
        gamma:   focusing parameter.
    Returns:
        scalar loss.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    return (focal_weight * ce).mean()


def giou_loss(pred, target, mask):
    """
    Generalized IoU loss for bounding box regression.

    Args:
        pred:   [B, 4, H, W] — predicted offsets (l, t, r, b), positive.
        target: [B, 4, H, W] — target offsets (l, t, r, b), positive.
        mask:   [B, 1, H, W] — binary mask (1.0 for positive locations).
    Returns:
        scalar loss.
    """
    mask_bool = mask.bool()
    if not mask_bool.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_pos = pred[mask_bool.expand_as(pred)].reshape(-1, 4)          # [N_pos, 4]
    tgt_pos = target[mask_bool.expand_as(target)].reshape(-1, 4)       # [N_pos, 4]

    l_p, t_p, r_p, b_p = pred_pos.unbind(dim=1)
    l_t, t_t, r_t, b_t = tgt_pos.unbind(dim=1)

    # Predicted / target box areas
    area_p = (l_p + r_p) * (t_p + b_p)
    area_t = (l_t + r_t) * (t_t + b_t)

    # Intersection
    w_i = torch.min(l_p, l_t) + torch.min(r_p, r_t)
    h_i = torch.min(t_p, t_t) + torch.min(b_p, b_t)
    inter = torch.clamp(w_i, min=0) * torch.clamp(h_i, min=0)

    union = area_p + area_t - inter
    iou = (inter + 1e-6) / (union + 1e-6)

    # Enclosing box
    w_c = torch.max(l_p, l_t) + torch.max(r_p, r_t)
    h_c = torch.max(t_p, t_t) + torch.max(b_p, b_t)
    area_c = w_c * h_c + 1e-6

    giou = iou - (area_c - union) / area_c
    return (1.0 - giou).mean()


# ============================================================================
# Spatial Attention Mask Generator
# ============================================================================

def generate_spatial_attention_mask(bbox, feat_h, feat_w, img_size=255, sigma_scale=0.3):
    """
    Generate a 2D Gaussian attention mask centered on the GT bounding box.

    Args:
        bbox:        [B, 4] — (cx, cy, w, h) in image coordinates.
        feat_h:      feature map height.
        feat_w:      feature map width.
        img_size:    input image size (for coordinate scaling).
        sigma_scale: Gaussian sigma as fraction of bbox size.
    Returns:
        mask: [B, 1, feat_h, feat_w] — soft attention mask, values in [0, 1].
    """
    B = bbox.shape[0]
    device = bbox.device

    # Scale bbox center to feature map coordinates
    cx = bbox[:, 0] / img_size * feat_w                                # [B]
    cy = bbox[:, 1] / img_size * feat_h                                # [B]
    w = bbox[:, 2] / img_size * feat_w                                 # [B]
    h = bbox[:, 3] / img_size * feat_h                                 # [B]

    # Gaussian sigmas
    sigma_x = torch.clamp(w * sigma_scale, min=1.0)                   # [B]
    sigma_y = torch.clamp(h * sigma_scale, min=1.0)                   # [B]

    # Grid
    grid_y = torch.arange(feat_h, device=device, dtype=torch.float32)
    grid_x = torch.arange(feat_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')            # [H, W]

    # Broadcast and compute Gaussian
    xx = xx.unsqueeze(0)                                               # [1, H, W]
    yy = yy.unsqueeze(0)                                               # [1, H, W]
    cx = cx.reshape(B, 1, 1)
    cy = cy.reshape(B, 1, 1)
    sigma_x = sigma_x.reshape(B, 1, 1)
    sigma_y = sigma_y.reshape(B, 1, 1)

    gauss = torch.exp(-0.5 * ((xx - cx) ** 2 / (sigma_x ** 2 + 1e-6) +
                               (yy - cy) ** 2 / (sigma_y ** 2 + 1e-6)))
    # Normalize to [0, 1]
    gauss = gauss / (gauss.amax(dim=(1, 2), keepdim=True) + 1e-6)

    return gauss.unsqueeze(1)                                          # [B, 1, H, W]


# ============================================================================
# Lightning Module
# ============================================================================

class SiamTrackerLightning(pl.LightningModule):
    """
    Complete training pipeline for Siamese Object Tracker v2.

    Features:
        - Focal Loss + GIoU Loss for student training
        - Multi-scale feature KD from ResNet-50 teacher
        - Spatial attention masks for foreground-focused distillation
        - QAT hooks for INT8 mobile export readiness
        - TensorBoard image & tracking visualizations at every step
    """

    def __init__(self,
                 # Model config
                 backbone_cfg: dict = None,
                 memory_cfg: dict = None,
                 fusion_cfg: dict = None,
                 head_cfg: dict = None,
                 # Teacher
                 teacher_weights_path: str = None,
                 teacher_pretrained: bool = True,
                 teacher_layer_channels: list = None,
                 # Loss weights
                 alpha_focal: float = 0.25,
                 gamma_focal: float = 2.0,
                 lambda_reg: float = 1.0,
                 beta_kd: float = 1.0,
                 pos_region_scale: float = 0.6,
                 # Optimizer
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 betas: tuple = (0.9, 0.999),
                 eta_min: float = 1e-6,
                 # QAT
                 enable_qat: bool = False,
                 qconfig_backend: str = "fbgemm",
                 # Logging
                 log_images_every_n_steps: int = 1,
                 num_vis_samples: int = 4,
                 # Search grid
                 grid_sz: int = 32,
                 stride: int = 8,
                 offset: int = 0):
        super().__init__()
        self.save_hyperparameters()

        # --- Student model ---
        self.student = SiamTracker(
            backbone_cfg=backbone_cfg or {},
            memory_cfg=memory_cfg or {},
            fusion_cfg=fusion_cfg or {},
            head_cfg=head_cfg or {},
        )

        # --- Teacher model ---
        self.teacher = None
        if teacher_weights_path:
            self.teacher = TeacherSiamRPN(pretrained=teacher_pretrained)
            self.teacher.load_weights(teacher_weights_path)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # --- Multi-scale projection layers (student→teacher dimensions) ---
        if self.teacher is not None:
            t_channels = teacher_layer_channels or [512, 1024, 2048]
            s_channels = [
                self.student.backbone.scale1_channels,                 # 24
                self.student.backbone.scale2_channels,                 # 32
                self.student.backbone.scale3_channels,                 # 64
            ]
            self.distill_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_ch),
                ) for s_ch, t_ch in zip(s_channels, t_channels)
            ])
        else:
            self.distill_projs = None

        # --- Search region grid for target computation ---
        self.grid_sz = grid_sz
        self.grid_stride = stride
        self.grid_offset = offset
        grid_range = torch.arange(self.grid_sz, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        self.register_buffer("grid_xc", self.grid_offset + grid_x * self.grid_stride)
        self.register_buffer("grid_yc", self.grid_offset + grid_y * self.grid_stride)

        # --- QAT setup ---
        if enable_qat:
            self._setup_qat(qconfig_backend)

        # --- Denormalization constants for visualization ---
        self.register_buffer("_vis_mean",
                             torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("_vis_std",
                             torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # QAT Setup
    # ------------------------------------------------------------------

    def _setup_qat(self, backend="fbgemm"):
        """Insert fake-quantize observers for quantization-aware training."""
        import torch.ao.quantization as quant

        if backend == "qnnpack":
            torch.backends.quantized.engine = "qnnpack"

        qconfig = quant.get_default_qat_qconfig(backend)
        self.student.qconfig = qconfig

        # Prepare model for QAT (fuses BN, inserts observers)
        quant.prepare_qat(self.student, inplace=True)
        print(f"[QAT] Model prepared with '{backend}' qconfig.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, templates, search):
        return self.student(templates, search)

    # ------------------------------------------------------------------
    # Target Computation
    # ------------------------------------------------------------------

    def _compute_targets(self, bbox_search):
        """
        Compute classification labels and regression targets.

        Args:
            bbox_search: [B, 4] — (cx, cy, w, h) in 255×255 search space.
        Returns:
            cls_targets: [B, 1, H, W] — binary labels.
            reg_targets: [B, 4, H, W] — (l, t, r, b) offsets.
        """
        B = bbox_search.shape[0]
        grid_xc = self.grid_xc.unsqueeze(0).expand(B, -1, -1)         # [B, H, W]
        grid_yc = self.grid_yc.unsqueeze(0).expand(B, -1, -1)

        cx = bbox_search[:, 0].reshape(B, 1, 1)
        cy = bbox_search[:, 1].reshape(B, 1, 1)
        w = bbox_search[:, 2].reshape(B, 1, 1)
        h = bbox_search[:, 3].reshape(B, 1, 1)

        # Full box bounds
        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - h / 2, cy + h / 2

        # Positive region (center sub-box)
        ps = self.hparams.pos_region_scale
        px_min, px_max = cx - ps * w / 2, cx + ps * w / 2
        py_min, py_max = cy - ps * h / 2, cy + ps * h / 2

        cls_targets = ((grid_xc >= px_min) & (grid_xc <= px_max) &
                       (grid_yc >= py_min) & (grid_yc <= py_max)).float()
        cls_targets = cls_targets.unsqueeze(1)                         # [B, 1, H, W]

        # Regression targets: l, t, r, b distances
        l = grid_xc - x_min
        t = grid_yc - y_min
        r = x_max - grid_xc
        b = y_max - grid_yc
        reg_targets = torch.stack([l, t, r, b], dim=1)                # [B, 4, H, W]

        return cls_targets, reg_targets

    # ------------------------------------------------------------------
    # Distillation Loss
    # ------------------------------------------------------------------

    def _distillation_loss(self, student_scales, search, bbox_search):
        """
        Multi-scale feature-level knowledge distillation with spatial attention.

        Args:
            student_scales: dict with 'scale1', 'scale2', 'scale3' feature maps.
            search:         [B, 3, 255, 255] — search image for teacher forward.
            bbox_search:    [B, 4] — GT bbox for attention mask.
        Returns:
            scalar distillation loss.
        """
        if self.teacher is None or self.distill_projs is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Teacher multi-scale features
        with torch.no_grad():
            self.teacher.eval()
            teacher_feats = self.teacher.extract_multiscale(search)

        scale_keys = ["scale1", "scale2", "scale3"]
        teacher_keys = ["layer2", "layer3", "layer4"]

        total_loss = torch.tensor(0.0, device=self.device)

        for i, (s_key, t_key) in enumerate(zip(scale_keys, teacher_keys)):
            s_feat = student_scales[s_key]                             # [B, C_s, H, W]
            t_feat = teacher_feats[t_key]                              # [B, C_t, H', W']

            # Project student to teacher dimension
            s_proj = self.distill_projs[i](s_feat)                     # [B, C_t, H, W]

            # Resize to match teacher spatial dims if needed
            if s_proj.shape[2:] != t_feat.shape[2:]:
                s_proj = F.interpolate(s_proj, size=t_feat.shape[2:],
                                       mode='bilinear', align_corners=False)

            # Spatial attention mask centered on GT bbox
            attn_mask = generate_spatial_attention_mask(
                bbox_search, t_feat.shape[2], t_feat.shape[3],
                img_size=255, sigma_scale=0.3,
            )                                                          # [B, 1, H', W']

            # Weighted MSE
            diff = (s_proj - t_feat) ** 2                              # [B, C_t, H', W']
            weighted = diff * attn_mask                                # broadcast over C
            total_loss = total_loss + weighted.mean()

        return total_loss / len(scale_keys)

    # ------------------------------------------------------------------
    # IoU Metric
    # ------------------------------------------------------------------

    def _compute_iou(self, pred_offsets, target_offsets, mask):
        """
        Compute mean IoU for positive locations.

        Args:
            pred_offsets:   [B, 4, H, W] — predicted (l, t, r, b).
            target_offsets: [B, 4, H, W] — target (l, t, r, b).
            mask:           [B, 1, H, W] — positive mask.
        Returns:
            mean IoU (scalar).
        """
        mask_bool = mask.bool()
        if not mask_bool.any():
            return torch.tensor(0.0, device=pred_offsets.device)

        p = pred_offsets[mask_bool.expand_as(pred_offsets)].reshape(-1, 4)
        t = target_offsets[mask_bool.expand_as(target_offsets)].reshape(-1, 4)

        l_p, t_p, r_p, b_p = p.unbind(1)
        l_t, t_t, r_t, b_t = t.unbind(1)

        area_p = (l_p + r_p) * (t_p + b_p)
        area_t = (l_t + r_t) * (t_t + b_t)
        w_i = torch.min(l_p, l_t) + torch.min(r_p, r_t)
        h_i = torch.min(t_p, t_t) + torch.min(b_p, b_t)
        inter = torch.clamp(w_i, min=0) * torch.clamp(h_i, min=0)
        union = area_p + area_t - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou.mean()

    # ------------------------------------------------------------------
    # Visualization Helpers
    # ------------------------------------------------------------------

    def _denormalize(self, img_tensor):
        """
        Denormalize a batch of images for visualization.

        Args:
            img_tensor: [B, 3, H, W] — normalized images.
        Returns:
            [B, 3, H, W] — pixel values in [0, 1].
        """
        return torch.clamp(img_tensor * self._vis_std + self._vis_mean, 0, 1)

    def _draw_bbox_on_image(self, img, bbox, color=(0, 255, 0), thickness=2):
        """
        Draw a bounding box on a numpy image.

        Args:
            img:    [H, W, 3] uint8 numpy array.
            bbox:   [cx, cy, w, h] in pixel coordinates.
            color:  BGR tuple.
        Returns:
            img with drawn bbox.
        """
        import cv2
        if np.isnan(bbox).any() or np.isinf(bbox).any():
            return img
        cx, cy, w, h = bbox
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _log_tracking_images(self, templates, search, cls_logits, reg_offsets,
                             bbox_search, prefix, batch_idx):
        """
        Log tracking visualizations to TensorBoard.

        Logs:
            - Template image grid (last frame from sequence)
            - Search image with GT bbox (green) and predicted bbox (red)
            - Score heatmap overlay

        Args:
            templates:   [B, T, 3, 127, 127]
            search:      [B, 3, 255, 255]
            cls_logits:  [B, 1, H, W]
            reg_offsets: [B, 4, H, W]
            bbox_search: [B, 4]
            prefix:      'train' or 'val'
            batch_idx:   current batch index
        """
        n = min(self.hparams.num_vis_samples, search.shape[0])

        # --- Template grid (last frame) ---
        last_templates = self._denormalize(templates[:n, -1])          # [n, 3, 127, 127]
        template_grid = vutils.make_grid(last_templates, nrow=n, padding=2)
        self.logger.experiment.add_image(
            f"{prefix}/templates", template_grid, self.global_step)

        # --- Search with bbox overlays ---
        search_vis = self._denormalize(search[:n])                     # [n, 3, 255, 255]
        score_maps = torch.sigmoid(cls_logits[:n, 0])                  # [n, H, W]

        # Find predicted bbox from max score location
        for i in range(n):
            img_np = (search_vis[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            gt_bbox = bbox_search[i].cpu().numpy()

            # Draw GT bbox (green)
            self._draw_bbox_on_image(img_np, gt_bbox, color=(0, 255, 0), thickness=2)

            # Predicted bbox from max response
            score = score_maps[i].detach().cpu().numpy()
            j_max, i_max = np.unravel_index(np.argmax(score), score.shape)
            reg = reg_offsets[i].detach().cpu().numpy()                 # [4, H, W]
            l, t, r, b = reg[:, j_max, i_max]
            pred_cx = self.grid_offset + i_max * self.grid_stride
            pred_cy = self.grid_offset + j_max * self.grid_stride
            pred_w, pred_h = l + r, t + b
            self._draw_bbox_on_image(img_np, [pred_cx, pred_cy, pred_w, pred_h],
                                     color=(255, 0, 0), thickness=2)

            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            self.logger.experiment.add_image(
                f"{prefix}/search_bbox_{i}", img_tensor, self.global_step)

        # --- Score heatmap ---
        heatmap = score_maps[:n].unsqueeze(1)                          # [n, 1, H, W]
        heatmap_up = F.interpolate(heatmap, size=(255, 255),
                                   mode='bilinear', align_corners=False)
        heatmap_grid = vutils.make_grid(heatmap_up, nrow=n, padding=2)
        self.logger.experiment.add_image(
            f"score_heatmap/{prefix}", heatmap_grid, self.global_step)

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        templates, search, bbox_search, bboxes_templates = batch
        # templates:        [B, T, 3, 127, 127]
        # search:           [B, 3, 255, 255]
        # bbox_search:      [B, 4]
        # bboxes_templates: [B, T, 4]

        # ========== NaN CHECK: Inputs ==========
        input_checks = {
            "templates": templates,
            "search": search,
            "bbox_search": bbox_search,
            "bboxes_templates": bboxes_templates,
        }
        for name, tensor in input_checks.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                print(f"\n{'='*60}")
                print(f"[NaN DEBUG] NaN/Inf detected in INPUT '{name}' at step {self.global_step}")
                print(f"  shape: {tensor.shape}, dtype: {tensor.dtype}")
                print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                print(f"  min: {tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
                print(f"  max: {tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
                print(f"  mean: {tensor[~torch.isnan(tensor)].mean().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
                print(f"{'='*60}\n")
                raise ValueError(f"NaN/Inf in input '{name}' at step {self.global_step}")

        # Student forward with multi-scale features
        cls_logits, reg_offsets, search_scales = self.student(
            templates, search, return_multiscale=True)
        # cls_logits:  [B, 1, H, W]
        # reg_offsets: [B, 4, H, W]

        # ========== NaN CHECK: Model Outputs ==========
        output_checks = {
            "cls_logits": cls_logits,
            "reg_offsets": reg_offsets,
        }
        if search_scales is not None:
            for k, v in search_scales.items():
                output_checks[f"search_scales.{k}"] = v
        for name, tensor in output_checks.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                print(f"\n{'='*60}")
                print(f"[NaN DEBUG] NaN/Inf detected in MODEL OUTPUT '{name}' at step {self.global_step}")
                print(f"  shape: {tensor.shape}, dtype: {tensor.dtype}")
                print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                finite = tensor[torch.isfinite(tensor)]
                if finite.numel() > 0:
                    print(f"  finite min: {finite.min().item():.6f}")
                    print(f"  finite max: {finite.max().item():.6f}")
                    print(f"  finite mean: {finite.mean().item():.6f}")
                    print(f"  finite std: {finite.std().item():.6f}")
                else:
                    print(f"  ALL values are NaN/Inf!")
                print(f"{'='*60}\n")
                raise ValueError(f"NaN/Inf in model output '{name}' at step {self.global_step}")

        # Compute targets
        cls_targets, reg_targets = self._compute_targets(bbox_search)

        # ========== NaN CHECK: Targets ==========
        target_checks = {
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
        }
        for name, tensor in target_checks.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"\n{'='*60}")
                print(f"[NaN DEBUG] NaN/Inf detected in TARGET '{name}' at step {self.global_step}")
                print(f"  shape: {tensor.shape}")
                print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
                print(f"  bbox_search values: {bbox_search}")
                print(f"{'='*60}\n")
                raise ValueError(f"NaN/Inf in target '{name}' at step {self.global_step}")

        # --- Losses ---
        loss_cls = focal_loss(cls_logits, cls_targets,
                              alpha=self.hparams.alpha_focal,
                              gamma=self.hparams.gamma_focal)

        loss_reg = giou_loss(reg_offsets, reg_targets, cls_targets)

        loss_distill = self._distillation_loss(
            search_scales, search, bbox_search)

        # ========== NaN CHECK: Individual Losses ==========
        loss_checks = {
            "loss_cls (focal)": loss_cls,
            "loss_reg (giou)": loss_reg,
            "loss_distill (kd)": loss_distill,
        }
        for name, loss_val in loss_checks.items():
            if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                print(f"\n{'='*60}")
                print(f"[NaN DEBUG] NaN/Inf detected in LOSS '{name}' at step {self.global_step}")
                print(f"  loss value: {loss_val.item()}")
                print(f"  cls_logits stats: min={cls_logits.min().item():.4f}, max={cls_logits.max().item():.4f}, "
                      f"mean={cls_logits.mean().item():.4f}")
                print(f"  reg_offsets stats: min={reg_offsets.min().item():.4f}, max={reg_offsets.max().item():.4f}, "
                      f"mean={reg_offsets.mean().item():.4f}")
                print(f"  cls_targets: pos_count={cls_targets.sum().item():.0f}, "
                      f"total={cls_targets.numel()}")
                print(f"  reg_targets stats: min={reg_targets.min().item():.4f}, max={reg_targets.max().item():.4f}")
                print(f"  bbox_search: {bbox_search}")
                print(f"{'='*60}\n")
                raise ValueError(f"NaN/Inf in loss '{name}' at step {self.global_step}")

        total_loss = (loss_cls +
                      self.hparams.lambda_reg * loss_reg +
                      self.hparams.beta_kd * loss_distill)

        # ========== NaN CHECK: Total Loss ==========
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n{'='*60}")
            print(f"[NaN DEBUG] NaN/Inf in TOTAL LOSS at step {self.global_step}")
            print(f"  loss_cls:     {loss_cls.item():.6f}")
            print(f"  loss_reg:     {loss_reg.item():.6f}")
            print(f"  loss_distill: {loss_distill.item():.6f}")
            print(f"  total_loss:   {total_loss.item()}")
            print(f"  lambda_reg:   {self.hparams.lambda_reg}")
            print(f"  beta_kd:      {self.hparams.beta_kd}")
            print(f"{'='*60}\n")
            raise ValueError(f"NaN/Inf in total loss at step {self.global_step}")

        # --- Logging ---
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_cls', loss_cls, on_step=True, on_epoch=True)
        self.log('train_loss_reg', loss_reg, on_step=True, on_epoch=True)
        self.log('train_loss_distill', loss_distill, on_step=True, on_epoch=True)

        # IoU metric
        with torch.no_grad():
            train_iou = self._compute_iou(reg_offsets, reg_targets, cls_targets)
        self.log('train_iou', train_iou, on_step=True, on_epoch=True)

        # Image logging at every step
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            if batch_idx % self.hparams.log_images_every_n_steps == 0:
                self._log_tracking_images(
                    templates, search, cls_logits, reg_offsets,
                    bbox_search, prefix="train", batch_idx=batch_idx)

        return total_loss

    # ------------------------------------------------------------------
    # Validation Step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        templates, search, bbox_search, bboxes_templates = batch

        cls_logits, reg_offsets = self.student(templates, search)
        cls_targets, reg_targets = self._compute_targets(bbox_search)

        loss_cls = focal_loss(cls_logits, cls_targets,
                              alpha=self.hparams.alpha_focal,
                              gamma=self.hparams.gamma_focal)
        loss_reg = giou_loss(reg_offsets, reg_targets, cls_targets)

        val_loss = loss_cls + self.hparams.lambda_reg * loss_reg

        # IoU and success rate
        iou = self._compute_iou(reg_offsets, reg_targets, cls_targets)
        success = (iou > 0.5).float() if iou.dim() > 0 else (iou > 0.5).float()

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_loss_cls', loss_cls, on_epoch=True)
        self.log('val_loss_reg', loss_reg, on_epoch=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)
        self.log('val_success_rate', success, on_epoch=True, prog_bar=True)

        # Image logging at every step
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            if batch_idx % self.hparams.log_images_every_n_steps == 0:
                self._log_tracking_images(
                    templates, search, cls_logits, reg_offsets,
                    bbox_search, prefix="val", batch_idx=batch_idx)

        return val_loss

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Only train student parameters (teacher is frozen)
        params = list(self.student.parameters())
        if self.distill_projs is not None:
            params += list(self.distill_projs.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.eta_min,
        )

        return [optimizer], [scheduler]
