"""
Siamese Object Tracker — Model Assembly.

Assembles:
    MobileViTBackbone      — shared Siamese feature extractor.
    TemporalMemoryModule   — FIFO token cache + SSM aggregation.
    LinearCrossAttention   — target↔search fusion.
    RPNHead                — classification + regression output.

Training forward pass:
    1. Backbone(template_sequence) → T sets of target tokens.
    2. TemporalMemory(target_tokens) → aggregated Q.
    3. Backbone(search) → search tokens (K, V).
    4. LinearCrossAttention(Q, K/V) → fused search tokens.
    5. RPNHead(fused) → cls_logits, reg_offsets.
"""

import torch
import torch.nn as nn

from .backbone import MobileViTBackbone
from .temporal_memory import TemporalMemoryModule
from .head import LinearCrossAttention, RPNHead


class SiamTracker(nn.Module):
    """
    Complete Siamese Object Tracker with linear attention and temporal memory.

    Args:
        backbone_cfg:   dict of kwargs for MobileViTBackbone.
        memory_cfg:     dict of kwargs for TemporalMemoryModule.
        fusion_cfg:     dict of kwargs for LinearCrossAttention.
        head_cfg:       dict of kwargs for RPNHead.
    """

    def __init__(self,
                 backbone_cfg: dict = None,
                 memory_cfg: dict = None,
                 fusion_cfg: dict = None,
                 head_cfg: dict = None):
        super().__init__()

        backbone_cfg = backbone_cfg or {}
        memory_cfg = memory_cfg or {}
        fusion_cfg = fusion_cfg or {}
        head_cfg = head_cfg or {}

        # Shared Siamese backbone
        self.backbone = MobileViTBackbone(**backbone_cfg)
        dim = self.backbone.out_channels                               # 64

        # Temporal memory (token cache + SSM)
        memory_cfg.setdefault("dim", dim)
        self.temporal_memory = TemporalMemoryModule(**memory_cfg)

        # Linear cross-attention fusion
        fusion_cfg.setdefault("dim", dim)
        self.fusion = LinearCrossAttention(**fusion_cfg)

        # Lightweight RPN head
        head_cfg.setdefault("in_channels", dim)
        self.head = RPNHead(**head_cfg)

        # Cached template features for inference
        self._feat_z = None
        self._search_spatial = None

    def forward(self, templates: torch.Tensor, search: torch.Tensor,
                return_multiscale: bool = False):
        """
        Training forward pass with a sequence of template frames.

        Args:
            templates: [B, T, 3, 127, 127] — T sequential template frames.
            search:    [B, 3, 255, 255]     — single search frame.
            return_multiscale: if True, return multi-scale features for distillation.
        Returns:
            cls_logits:  [B, 1, H_s, W_s]  — target presence logits.
            reg_offsets: [B, 4, H_s, W_s]  — bounding box offsets.
            (optional) search_scales: dict of multi-scale search features.
        """
        B, T, C, Hz, Wz = templates.shape

        # --- Extract template tokens for each frame ---
        # Reshape to process all template frames in one batch
        templates_flat = templates.reshape(B * T, C, Hz, Wz)           # [B*T, 3, 127, 127]
        feat_z_flat = self.backbone(templates_flat, return_multiscale=False)
        # feat_z_flat: [B*T, D, H_z, W_z]  e.g. [B*T, 64, 16, 16]

        _, D, H_z, W_z = feat_z_flat.shape
        N_z = H_z * W_z                                               # 256 tokens per template

        # Reshape to [B, T, N_z, D] for temporal memory
        feat_z_tokens = feat_z_flat.flatten(2).transpose(1, 2)         # [B*T, N_z, D]
        feat_z_tokens = feat_z_tokens.reshape(B, T, N_z, D)           # [B, T, N_z, D]

        # --- Temporal memory aggregation → Query tokens ---
        q_tokens = self.temporal_memory(feat_z_tokens)                 # [B, N_z, D]

        # --- Extract search features ---
        if return_multiscale:
            feat_x, search_scales = self.backbone(search, return_multiscale=True)
        else:
            feat_x = self.backbone(search, return_multiscale=False)
            search_scales = None
        # feat_x: [B, D, H_x, W_x]  e.g. [B, 64, 32, 32]

        _, _, H_x, W_x = feat_x.shape
        kv_tokens = feat_x.flatten(2).transpose(1, 2)                 # [B, H_x*W_x, D]

        # --- Linear cross-attention fusion ---
        fused_tokens = self.fusion(q_tokens, kv_tokens)                # [B, H_x*W_x, D]

        # --- RPN Head ---
        cls_logits, reg_offsets = self.head(fused_tokens,
                                           spatial_h=H_x, spatial_w=W_x)
        # cls_logits:  [B, 1, H_x, W_x]
        # reg_offsets: [B, 4, H_x, W_x]
        
        if return_multiscale:
            return cls_logits, reg_offsets, search_scales
        return cls_logits, reg_offsets

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def init_template(self, template: torch.Tensor):
        """
        Initialize tracking with the first template frame.

        Args:
            template: [1, 3, 127, 127] — initial target crop.
        """
        self.eval()
        with torch.no_grad():
            feat_z = self.backbone(template)                           # [1, D, H_z, W_z]
            tokens = feat_z.flatten(2).transpose(1, 2).squeeze(0)      # [N_z, D]
            N_z = tokens.shape[0]
            self.temporal_memory.init_cache(
                num_tokens=N_z,
                device=template.device,
                dtype=template.dtype,
            )
            self.temporal_memory.push_and_aggregate(tokens)
            self._feat_z = tokens                                      # cache for reference

    def track(self, search: torch.Tensor):
        """
        Track target in the search region using cached temporal memory.

        Args:
            search: [1, 3, 255, 255] — search region crop.
        Returns:
            cls_logits:  [1, 1, H_x, W_x]
            reg_offsets: [1, 4, H_x, W_x]
        """
        if self._feat_z is None:
            raise ValueError("Call init_template() before track().")

        with torch.no_grad():
            feat_x = self.backbone(search)                             # [1, D, H_x, W_x]
            _, _, H_x, W_x = feat_x.shape
            kv_tokens = feat_x.flatten(2).transpose(1, 2)             # [1, H_x*W_x, D]

            # Get aggregated Q from temporal memory
            q_tokens = self.temporal_memory._cache.get_sequence()       # [n, N_z, D]
            q_tokens = self.temporal_memory.ssm(q_tokens)               # [N_z, D]
            q_tokens = q_tokens.unsqueeze(0)                           # [1, N_z, D]

            # Fusion + head
            fused = self.fusion(q_tokens, kv_tokens)                   # [1, H_x*W_x, D]
            cls_logits, reg_offsets = self.head(fused, H_x, W_x)
            return cls_logits, reg_offsets

    def update_template(self, template: torch.Tensor):
        """
        Push a new template frame into the temporal memory during tracking.

        Args:
            template: [1, 3, 127, 127]
        """
        with torch.no_grad():
            feat_z = self.backbone(template)                           # [1, D, H_z, W_z]
            tokens = feat_z.flatten(2).transpose(1, 2).squeeze(0)      # [N_z, D]
            self.temporal_memory._cache.push(tokens)


# ============================================================================
# Quick shape test
# ============================================================================

if __name__ == "__main__":
    model = SiamTracker()

    # Training forward
    templates = torch.randn(2, 4, 3, 127, 127)    # B=2, T=4 template frames
    search = torch.randn(2, 3, 255, 255)

    cls, reg = model(templates, search)
    print(f"Cls logits: {cls.shape}")               # [2, 1, 32, 32]
    print(f"Reg offsets: {reg.shape}")               # [2, 4, 32, 32]

    # With multi-scale
    cls, reg, scales = model(templates, search, return_multiscale=True)
    for k, v in scales.items():
        print(f"  {k}: {v.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")

    # Inference
    model.init_template(torch.randn(1, 3, 127, 127))
    cls_i, reg_i = model.track(torch.randn(1, 3, 255, 255))
    print(f"\nInference cls: {cls_i.shape}")         # [1, 1, 32, 32]
    print(f"Inference reg: {reg_i.shape}")           # [1, 4, 32, 32]
