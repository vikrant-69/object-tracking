import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from .siam_tracker import SiamTracker
from .head import DepthwiseXCorr

class TeacherSiamRPN(nn.Module):
    """
    Teacher SiamRPN++ model with a ResNet-50 backbone.
    Used during training for knowledge distillation.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load standard ResNet-50
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        except ImportError:
            resnet = models.resnet50(pretrained=pretrained)
            
        # Extract features (conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
        # In SiamRPN++, layer3 and layer4 are modified to have stride 1 and dilation 2 and 4.
        self.backbone_layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        self.backbone_layer3 = resnet.layer3
        self.backbone_layer4 = resnet.layer4
        
        # Modify layer3 (change stride from 2 to 1, add dilation=2)
        self._modify_layer_stride_dilation(self.backbone_layer3, stride=1, dilation=2)
        # Modify layer4 (change stride from 2 to 1, add dilation=4)
        self._modify_layer_stride_dilation(self.backbone_layer4, stride=1, dilation=4)
        
        # SiamRPN++ fuses features from layer2, layer3, and layer4.
        # For simplicity in distillation, we extract layer3 features and reduce channels to 256.
        # ResNet-50 layer3 has 1024 channels.
        self.cls_adjust_template = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.cls_adjust_search = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        
        self.xcorr = DepthwiseXCorr()
        self.cls_head = nn.Conv2d(256, 1, kernel_size=1) # Logits

    def _modify_layer_stride_dilation(self, layer, stride=1, dilation=2):
        # Modify the first bottleneck block of the layer which does the downsampling
        for name, module in layer.named_modules():
            if 'Conv2d' in module.__class__.__name__:
                pass
        
        # Iterate over Bottleneck blocks
        for i, block in enumerate(layer.children()):
            if i == 0:
                # First block has downsample and conv2 stride 2
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

    def extract_features(self, x):
        feat = self.backbone_layer0(x)
        feat = self.backbone_layer3(feat)
        return feat

    def forward(self, template, search):
        # Template is 127x127 -> outputs [B, 1024, 15, 15] or [B, 1024, 16, 16] depending on pooling/strides
        # Search is 255x255 -> outputs [B, 1024, 31, 31] or [B, 1024, 32, 32]
        feat_z = self.extract_features(template)
        feat_x = self.extract_features(search)
        
        cls_z = self.cls_adjust_template(feat_z)
        cls_x = self.cls_adjust_search(feat_x)
        
        response_map = self.xcorr(cls_x, cls_z)
        cls_logits = self.cls_head(response_map)
        return cls_logits

    def load_weights(self, weights_path):
        if not os.path.exists(weights_path):
            print(f"Teacher weights file not found at {weights_path}. Training with ImageNet initializations.")
            return
            
        print(f"Loading teacher weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Check if the checkpoint contains state_dict key or is state_dict itself
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        
        # Extract matching keys (many tracking checkpoints have prefixes like "backbone." or "neck.")
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Strip prefixes if necessary
            clean_k = k
            if k.startswith('backbone.'):
                clean_k = k.replace('backbone.', '')
            elif k.startswith('neck.'):
                clean_k = k.replace('neck.', 'cls_adjust_')
                
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                pretrained_dict[clean_k] = v
                
        print(f"Matched {len(pretrained_dict)} / {len(model_dict)} keys from teacher checkpoint.")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)


class SiamTrackerLightning(pl.LightningModule):
    def __init__(self, 
                 student_pretrained=True, 
                 teacher_weights_path=None, 
                 lr=1e-3, 
                 weight_decay=1e-4, 
                 lambda_reg=1.0, 
                 beta_kd=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate student model (MobileNet-V2 + Anchor-free Head)
        self.student = SiamTracker(pretrained_backbone=student_pretrained)
        
        # Instantiate teacher model (ResNet-50) only if weights path is provided
        if teacher_weights_path:
            self.teacher = TeacherSiamRPN(pretrained=True)
            self.teacher.load_weights(teacher_weights_path)
            # Freeze teacher weights
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None
        
        # Set up search region grid coordinates for target mapping
        # Student output size: 17x17. Spatial stride: 8. Center offset: 64.
        self.grid_sz = 17
        self.stride = 8
        self.offset = 64
        
        # Create grids
        grid_range = torch.arange(self.grid_sz, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        
        # Map indices to search patch coordinates (255x255 space)
        # Shape: [17, 17]
        self.register_buffer("grid_xc", self.offset + grid_x * self.stride)
        self.register_buffer("grid_yc", self.offset + grid_y * self.stride)

    def forward(self, template, search):
        return self.student(template, search)

    def _compute_targets(self, bbox_search):
        """
        Compute classification labels and regression targets for the 17x17 grid.
        Args:
            bbox_search: [B, 4] representing [cx, cy, w, h] in 255x255 space.
        Returns:
            cls_targets: [B, 1, 17, 17] with 1.0 for positive, 0.0 for negative.
            reg_targets: [B, 4, 17, 17] containing target [l, t, r, b] offsets.
        """
        batch_size = bbox_search.shape[0]
        
        # Expand grids to batch dimension
        grid_xc = self.grid_xc.unsqueeze(0).expand(batch_size, -1, -1) # [B, 17, 17]
        grid_yc = self.grid_yc.unsqueeze(0).expand(batch_size, -1, -1) # [B, 17, 17]
        
        # Extract ground truth boxes
        cx = bbox_search[:, 0].unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        cy = bbox_search[:, 1].unsqueeze(1).unsqueeze(2)
        w = bbox_search[:, 2].unsqueeze(1).unsqueeze(2)
        h = bbox_search[:, 3].unsqueeze(1).unsqueeze(2)
        
        # Calculate bounding box bounds
        x_min = cx - w / 2
        x_max = cx + w / 2
        y_min = cy - h / 2
        y_max = cy + h / 2
        
        # Positive region is a center sub-box (e.g. 0.6 scale of bounding box) to prevent drift
        pos_scale = 0.6
        px_min = cx - pos_scale * w / 2
        px_max = cx + pos_scale * w / 2
        py_min = cy - pos_scale * h / 2
        py_max = cy + pos_scale * h / 2
        
        # Classification target (1 if inside positive region, else 0)
        is_in_pos_x = (grid_xc >= px_min) & (grid_xc <= px_max)
        is_in_pos_y = (grid_yc >= py_min) & (grid_yc <= py_max)
        cls_targets = (is_in_pos_x & is_in_pos_y).float().unsqueeze(1) # [B, 1, 17, 17]
        
        # Bounding box regression targets: distances l, t, r, b
        l = grid_xc - x_min
        t = grid_yc - y_min
        r = x_max - grid_xc
        b = y_max - grid_yc
        reg_targets = torch.stack([l, t, r, b], dim=1) # [B, 4, 17, 17]
        
        return cls_targets, reg_targets

    def _iou_loss(self, pred, target, mask):
        """
        Compute Intersection over Union (IoU) Loss for positive locations.
        Args:
            pred: Predicted offsets [B, 4, 17, 17] containing [l, t, r, b]
            target: Target offsets [B, 4, 17, 17]
            mask: Binary mask [B, 1, 17, 17] (1.0 for positive locations)
        """
        # Apply mask
        mask = mask.bool()
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        pred_pos = pred[mask.expand_as(pred)].view(-1, 4)
        target_pos = target[mask.expand_as(target)].view(-1, 4)
        
        l_p, t_p, r_p, b_p = pred_pos.unbind(dim=1)
        l_t, t_t, r_t, b_t = target_pos.unbind(dim=1)
        
        # Areas
        area_pred = (l_p + r_p) * (t_p + b_p)
        area_target = (l_t + r_t) * (t_t + b_t)
        
        # Intersections
        w_inter = torch.min(l_p, l_t) + torch.min(r_p, r_t)
        h_inter = torch.min(t_p, t_t) + torch.min(b_p, b_t)
        
        # Compute IoU
        inter = torch.clamp(w_inter, min=0) * torch.clamp(h_inter, min=0)
        union = area_pred + area_target - inter
        
        iou = (inter + 1e-6) / (union + 1e-6)
        iou_loss = -torch.log(torch.clamp(iou, min=1e-6))
        return iou_loss.mean()

    def training_step(self, batch, batch_idx):
        template, search, bbox_search = batch
        
        # Student forward pass
        cls_logits_s, reg_offsets_s = self.student(template, search)
        
        # Compute student targets
        cls_targets, reg_targets = self._compute_targets(bbox_search)
        
        # Classification loss (Binary Cross Entropy)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits_s, cls_targets)
        
        # Regression loss (IoU Loss on positive locations)
        loss_reg = self._iou_loss(reg_offsets_s, reg_targets, cls_targets)
        
        # Knowledge Distillation (KD) Loss
        loss_kd = torch.tensor(0.0, device=self.device)
        if self.hparams.beta_kd > 0 and self.teacher is not None:
            with torch.no_grad():
                cls_logits_t = self.teacher(template, search)
            # Resize student logits to match teacher logits shape (usually 25x25)
            cls_logits_s_resized = F.interpolate(
                cls_logits_s, size=cls_logits_t.shape[2:], mode='bilinear', align_corners=False
            )
            loss_kd = F.mse_loss(cls_logits_s_resized, cls_logits_t)
            
        # Total loss
        total_loss = loss_cls + self.hparams.lambda_reg * loss_reg + self.hparams.beta_kd * loss_kd
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_cls', loss_cls, on_step=False, on_epoch=True)
        self.log('train_loss_reg', loss_reg, on_step=False, on_epoch=True)
        self.log('train_loss_kd', loss_kd, on_step=False, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        template, search, bbox_search = batch
        
        cls_logits_s, reg_offsets_s = self.student(template, search)
        cls_targets, reg_targets = self._compute_targets(bbox_search)
        
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits_s, cls_targets)
        loss_reg = self._iou_loss(reg_offsets_s, reg_targets, cls_targets)
        
        total_loss = loss_cls + self.hparams.lambda_reg * loss_reg
        
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_loss_cls', loss_cls, on_epoch=True)
        self.log('val_loss_reg', loss_reg, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]
