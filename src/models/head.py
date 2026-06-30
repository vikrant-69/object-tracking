"""
Linear Attention Fusion + Lightweight RPN Head for Siamese Object Tracking.

Replaces traditional depthwise cross-correlation with linear cross-attention.

Components:
    DepthwiseSeparableConv  — Depthwise 3×3 + Pointwise 1×1 conv block.
    LinearCrossAttention    — O(N) cross-attention: Q=target, K/V=search.
    RPNHead                 — Classification (target presence) + Regression (bbox offsets).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Depthwise Separable Convolution
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3×3 + Pointwise 1×1 convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, C_out, H, W]
        """
        return self.conv(x)


# ============================================================================
# Linear Cross-Attention (replaces cross-correlation)
# ============================================================================

def _elu_feature_map(x):
    """φ(x) = ELU(x) + 1  — ensures positivity for linear attention kernel."""
    return F.elu(x) + 1.0


class LinearCrossAttention(nn.Module):
    """
    O(N) linear cross-attention between target (Q) and search (K, V) tokens.

    Computation:
        φ(Q) · (φ(K)^T · V) / (φ(Q) · sum(φ(K)))

    This completely replaces depthwise cross-correlation layers.

    Args:
        dim:   token dimension for Q, K, V.
        heads: number of attention heads.
    """

    def __init__(self, dim: int = 64, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.25

        # Separate projections for Q (target) and K, V (search)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: target tokens attend to search tokens.

        Args:
            q_tokens:  [B, T_q, D] — aggregated target tokens (from temporal memory).
            kv_tokens: [B, T_s, D] — search region tokens (from backbone).
        Returns:
            [B, T_s, D] — search tokens enriched with target information.
        """
        B, T_q, D = q_tokens.shape
        _, T_s, _ = kv_tokens.shape
        h = self.heads
        d = self.head_dim

        # Project Q, K, V
        Q_proj = self.q_proj(q_tokens).reshape(B, T_q, h, d).permute(0, 2, 1, 3)   # [B, h, T_q, d]
        K_proj = self.k_proj(kv_tokens).reshape(B, T_s, h, d).permute(0, 2, 1, 3)  # [B, h, T_s, d]
        V_s = self.v_proj(kv_tokens).reshape(B, T_s, h, d).permute(0, 2, 1, 3)     # [B, h, T_s, d]

        # Force float32 for attention math to prevent fp16 overflow in einsum accumulations
        with torch.amp.autocast('cuda', enabled=False):
            Q_proj = Q_proj.float()
            K_proj = K_proj.float()
            V_s = V_s.float()

            # L2 normalize projections before scaling and feature map
            Q_proj = F.normalize(Q_proj, p=2, dim=-1)
            K_proj = F.normalize(K_proj, p=2, dim=-1)

            # Apply feature map φ with scaling
            Q = _elu_feature_map(Q_proj * self.scale)                                 # [B, h, T_q, d]
            K = _elu_feature_map(K_proj * self.scale)                                 # [B, h, T_s, d]

            # 1. Target attending to search positions
            KV = torch.einsum("bhsd,bhse->bhde", K, V_s)                              # [B, h, d, d]
            Z = 1.0 / (torch.einsum("bhqd,bhd->bhq", Q,
                                     K.sum(dim=2)) + 1e-6)                            # [B, h, T_q]
            QKV = torch.einsum("bhqd,bhde->bhqe", Q, KV)                              # [B, h, T_q, d]
            QKV = QKV * Z.unsqueeze(-1)                                               # [B, h, T_q, d]

            # 2. Search attending to target (actual tracking output)
            Q_s = _elu_feature_map(K_proj * self.scale)                               # [B, h, T_s, d]
            K_t = _elu_feature_map(Q_proj * self.scale)                               # [B, h, T_q, d]
            V_t = self.v_proj(q_tokens.float()).reshape(B, T_q, h, d).permute(0, 2, 1, 3)  # [B, h, T_q, d]

            KtVt = torch.einsum("bhqd,bhqe->bhde", K_t, V_t)                          # [B, h, d, d]
            Z_s = 1.0 / (torch.einsum("bhsd,bhd->bhs", Q_s,
                                        K_t.sum(dim=2)) + 1e-6)                       # [B, h, T_s]

            out = torch.einsum("bhsd,bhde->bhse", Q_s, KtVt)                          # [B, h, T_s, d]
            out = out * Z_s.unsqueeze(-1)                                             # [B, h, T_s, d]

        # Merge heads
        out = out.permute(0, 2, 1, 3).reshape(B, T_s, D)                          # [B, T_s, D]
        out = self.out_proj(out)                                                  # [B, T_s, D]

        # Residual with search tokens + normalize
        out = self.norm(out + kv_tokens)                                          # [B, T_s, D]
        return out


# ============================================================================
# Lightweight RPN Head
# ============================================================================

class RPNHead(nn.Module):
    """
    Lightweight Region Proposal Network head for object tracking.

    Takes fused search tokens (from LinearCrossAttention), reshapes them
    to a spatial feature map, and predicts:
        - Classification: target presence score per spatial location (0→1).
        - Regression: bounding box offsets (dx, dy, dw, dh) per location.

    Uses small depthwise separable 3×3 convolutions.

    Args:
        in_channels:     token / feature dimension.
        hidden_channels: intermediate channels in the conv heads.
        num_cls_convs:   number of DWSepConv layers in classification branch.
        num_reg_convs:   number of DWSepConv layers in regression branch.
    """

    def __init__(self, in_channels=64, hidden_channels=64,
                 num_cls_convs=1, num_reg_convs=1):
        super().__init__()

        # Classification branch
        cls_layers = []
        ch = in_channels
        for _ in range(num_cls_convs):
            cls_layers.append(DepthwiseSeparableConv(ch, hidden_channels, kernel_size=3, padding=1))
            ch = hidden_channels
        cls_layers.append(nn.Conv2d(ch, 1, kernel_size=1))             # logit output
        self.cls_head = nn.Sequential(*cls_layers)

        # Regression branch
        reg_layers = []
        ch = in_channels
        for _ in range(num_reg_convs):
            reg_layers.append(DepthwiseSeparableConv(ch, hidden_channels, kernel_size=3, padding=1))
            ch = hidden_channels
        reg_layers.append(nn.Conv2d(ch, 4, kernel_size=1))             # 4 offsets
        self.reg_head = nn.Sequential(*reg_layers)

    def forward(self, fused_tokens: torch.Tensor, spatial_h: int, spatial_w: int):
        """
        Args:
            fused_tokens: [B, H*W, D] — search tokens after cross-attention fusion.
            spatial_h:    height of the spatial feature map.
            spatial_w:    width of the spatial feature map.
        Returns:
            cls_logits:   [B, 1, H, W] — target presence logits.
            reg_offsets:  [B, 4, H, W] — bounding box offsets (l, t, r, b).
        """
        B, N, D = fused_tokens.shape

        # Reshape tokens to spatial feature map
        feat = fused_tokens.transpose(1, 2).reshape(B, D, spatial_h, spatial_w)  # [B, D, H, W]

        # Classification
        cls_logits = self.cls_head(feat)                                          # [B, 1, H, W]

        # Regression (positive offsets via exp, clamped for stability)
        reg_offsets = self.reg_head(feat)                                         # [B, 4, H, W]
        reg_offsets = torch.exp(torch.clamp(reg_offsets, min=-10.0, max=10.0))

        return cls_logits, reg_offsets


# ============================================================================
# Quick shape test
# ============================================================================

if __name__ == "__main__":
    B, D = 2, 64
    T_q, T_s = 256, 1024  # 16×16 and 32×32 tokens
    H_s, W_s = 32, 32

    # Test LinearCrossAttention
    fusion = LinearCrossAttention(dim=D, heads=4)
    q = torch.randn(B, T_q, D)
    kv = torch.randn(B, T_s, D)
    fused = fusion(q, kv)
    print(f"Fused tokens shape: {fused.shape}")    # [2, 1024, 64]

    # Test RPNHead
    head = RPNHead(in_channels=D, hidden_channels=64)
    cls_map, reg_map = head(fused, spatial_h=H_s, spatial_w=W_s)
    print(f"Cls map shape: {cls_map.shape}")         # [2, 1, 32, 32]
    print(f"Reg map shape: {reg_map.shape}")         # [2, 4, 32, 32]

    total = sum(p.numel() for p in fusion.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Total parameters (fusion+head): {total:,}")
