"""
MobileViT-XS Variant Backbone for Siamese Object Tracking.

Architecture:
    stem → stage1 (InvRes) → stage2 (InvRes) → stage3 (InvRes)
         → stage4 (InvRes + LinearAttentionTransformer, dilated)
         → stage5 (InvRes + LinearAttentionTransformer, dilated)

Multi-scale feature taps:
    scale1 = end of stage2  (24 ch)
    scale2 = end of stage3  (32 ch)
    scale3 = end of stage5  (64 ch)  ← main output

Input sizes (Siamese twin, shared weights):
    Template: [B, 3, 127, 127] → [B, 64, 16, 16]
    Search:   [B, 3, 255, 255] → [B, 64, 32, 32]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building Blocks
# ============================================================================

class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + Activation (default ReLU6)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 groups=1, dilation=1, bias=False, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, groups=groups, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style inverted residual block (MBConv).

    expand → depthwise 3×3 → project (linear bottleneck).
    Residual connection when stride == 1 and in_ch == out_ch.
    """

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4, dilation=1):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        padding = dilation  # for 3×3 kernel with dilation

        layers = []
        # Expand (pointwise)
        if expand_ratio != 1:
            layers.append(ConvBnAct(in_ch, mid_ch, kernel_size=1, padding=0))
        # Depthwise 3×3
        layers.append(ConvBnAct(mid_ch, mid_ch, kernel_size=3, stride=stride,
                                padding=padding, groups=mid_ch, dilation=dilation))
        # Project (pointwise, no activation)
        layers.append(ConvBnAct(mid_ch, out_ch, kernel_size=1, padding=0, act=False))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x                                              # [B, C, H, W]
        return out


# ============================================================================
# Linear Attention (O(N) complexity)
# ============================================================================

def _elu_feature_map(x):
    """φ(x) = ELU(x) + 1  — always positive, differentiable."""
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    """
    O(N) linear attention with ELU+1 feature map.

    Computes:  Attn(Q, K, V) = φ(Q) · (φ(K)^T · V) / (φ(Q) · Σ_φ(K))

    Args:
        dim:   token embedding dimension.
        heads: number of attention heads.
    """

    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.25

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, D]  — N tokens of dimension D.
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        h = self.heads

        qkv = self.to_qkv(x)                                          # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, h, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                             # [3, B, h, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]                             # each [B, h, N, d]

        # Force float32 for attention math to prevent fp16 overflow in einsum accumulations
        with torch.amp.autocast('cuda', enabled=False):
            q = q.float()
            k = k.float()
            v = v.float()

            # L2 normalize and scale inputs before feature map
            q = F.normalize(q, p=2, dim=-1) * self.scale
            k = F.normalize(k, p=2, dim=-1) * self.scale

            # Apply feature map
            q = _elu_feature_map(q)                                       # [B, h, N, d]
            k = _elu_feature_map(k)                                       # [B, h, N, d]

            # Linear attention: O(N*d^2) instead of O(N^2*d)
            kv = torch.einsum("bhnd,bhne->bhde", k, v)                    # [B, h, d, d]
            z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q,
                                     k.sum(dim=2)) + 1e-6)               # [B, h, N]
            out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)           # [B, h, N, d]

        out = out.transpose(1, 2).reshape(B, N, D)                    # [B, N, D]
        return self.to_out(out)


class FeedForward(nn.Module):
    """Simple MLP: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim, mult=2.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LinearAttentionTransformerBlock(nn.Module):
    """
    Spatial transformer block using Linear Attention.

    Pipeline:
        Conv feature map [B, C, H, W]
        → Unfold to tokens [B, N, C]
        → L × (LayerNorm → LinearAttention → residual → LayerNorm → FFN → residual)
        → Fold back to [B, C, H, W]
        → 1×1 Conv projection (fuse local + global features)
    """

    def __init__(self, dim, depth=2, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                LinearAttention(dim, heads=heads, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mult=mlp_ratio, dropout=dropout),
            ]))
        # 1×1 projection to fuse local (pre-unfold) + global (post-fold) features
        self.proj = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        local_feat = x                                                 # [B, C, H, W]

        # Unfold: spatial → token sequence
        tokens = x.flatten(2).transpose(1, 2)                         # [B, H*W, C]

        for ln1, attn, ln2, ffn in self.layers:
            tokens = tokens + attn(ln1(tokens))                        # [B, N, C]
            tokens = tokens + ffn(ln2(tokens))                         # [B, N, C]

        # Fold back
        global_feat = tokens.transpose(1, 2).reshape(B, C, H, W)      # [B, C, H, W]

        # Fuse local + global via concat and 1×1 conv
        fused = torch.cat([local_feat, global_feat], dim=1)            # [B, 2C, H, W]
        out = self.proj(fused)                                         # [B, C, H, W]
        return out


# ============================================================================
# Full Backbone
# ============================================================================

class MobileViTBackbone(nn.Module):
    """
    MobileViT-XS variant backbone for Siamese tracking.

    Hierarchy:
        stem  → 16ch, /2
        stage1 → 16ch, /2
        stage2 → 24ch, /4   ← scale1 tap
        stage3 → 32ch, /8   ← scale2 tap
        stage4 → 48ch, /8 (dilated) + transformer
        stage5 → 64ch, /8 (dilated) + transformer  ← scale3 tap (output)

    Template [B,3,127,127] → [B,64,16,16]   (stride 8)
    Search   [B,3,255,255] → [B,64,32,32]   (stride 8)
    """

    def __init__(self,
                 stem_channels=16,
                 stage_configs=None,
                 transformer_depth=2,
                 heads_stage4=2,
                 heads_stage5=4,
                 mlp_ratio=2.0,
                 dropout=0.0):
        super().__init__()

        # Default stage configs: [out_ch, stride, expand_ratio, num_blocks]
        if stage_configs is None:
            stage_configs = [
                [16, 1, 1, 1],      # stage1
                [24, 2, 4, 2],      # stage2  (scale1 tap)
                [32, 2, 4, 3],      # stage3  (scale2 tap)
                [48, 1, 4, 1],      # stage4  (dilated, transformer)
                [64, 1, 4, 1],      # stage5  (dilated, transformer, scale3 tap)
            ]

        # --- Stem ---
        self.stem = ConvBnAct(3, stem_channels, kernel_size=3, stride=2, padding=1)

        # --- Stages 1-3: pure inverted residuals ---
        in_ch = stem_channels
        # Stage 1
        s1_cfg = stage_configs[0]
        self.stage1 = self._make_stage(in_ch, s1_cfg[0], s1_cfg[1], s1_cfg[2], s1_cfg[3])
        in_ch = s1_cfg[0]

        # Stage 2 (scale1 tap)
        s2_cfg = stage_configs[1]
        self.stage2 = self._make_stage(in_ch, s2_cfg[0], s2_cfg[1], s2_cfg[2], s2_cfg[3])
        in_ch = s2_cfg[0]

        # Stage 3 (scale2 tap)
        s3_cfg = stage_configs[2]
        self.stage3 = self._make_stage(in_ch, s3_cfg[0], s3_cfg[1], s3_cfg[2], s3_cfg[3])
        in_ch = s3_cfg[0]

        # --- Stages 4-5: inverted residuals (dilated) + transformer ---
        s4_cfg = stage_configs[3]
        self.stage4_conv = self._make_stage(in_ch, s4_cfg[0], s4_cfg[1], s4_cfg[2],
                                            s4_cfg[3], dilation=2)
        self.stage4_transformer = LinearAttentionTransformerBlock(
            dim=s4_cfg[0], depth=transformer_depth, heads=heads_stage4,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        in_ch = s4_cfg[0]

        s5_cfg = stage_configs[4]
        self.stage5_conv = self._make_stage(in_ch, s5_cfg[0], s5_cfg[1], s5_cfg[2],
                                            s5_cfg[3], dilation=2)
        self.stage5_transformer = LinearAttentionTransformerBlock(
            dim=s5_cfg[0], depth=transformer_depth, heads=heads_stage5,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        self.out_channels = s5_cfg[0]  # 64

        # Scale tap channel counts (for distillation projection layers)
        self.scale1_channels = s2_cfg[0]  # 24
        self.scale2_channels = s3_cfg[0]  # 32
        self.scale3_channels = s5_cfg[0]  # 64

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, stride, expand_ratio, num_blocks, dilation=1):
        """Build a sequence of InvertedResidual blocks for one stage."""
        blocks = []
        # First block handles stride / channel change
        blocks.append(InvertedResidual(in_ch, out_ch, stride=stride,
                                       expand_ratio=expand_ratio, dilation=dilation))
        # Remaining blocks: stride=1, same channels
        for _ in range(1, num_blocks):
            blocks.append(InvertedResidual(out_ch, out_ch, stride=1,
                                           expand_ratio=expand_ratio, dilation=dilation))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_multiscale=False):
        """
        Args:
            x: [B, 3, H, W] — input image (127×127 or 255×255).
            return_multiscale: if True, also return intermediate feature maps.
        Returns:
            features: [B, 64, H/8, W/8]
            (optional) dict of multi-scale features
        """
        x = self.stem(x)                                               # [B, 16, H/2, W/2]
        x = self.stage1(x)                                             # [B, 16, H/2, W/2]
        x = self.stage2(x)                                             # [B, 24, H/4, W/4]
        scale1 = x                                                     # multi-scale tap #1

        x = self.stage3(x)                                             # [B, 32, H/8, W/8]
        scale2 = x                                                     # multi-scale tap #2

        x = self.stage4_conv(x)                                        # [B, 48, H/8, W/8]
        x = self.stage4_transformer(x)                                 # [B, 48, H/8, W/8]

        x = self.stage5_conv(x)                                        # [B, 64, H/8, W/8]
        x = self.stage5_transformer(x)                                 # [B, 64, H/8, W/8]
        scale3 = x                                                     # multi-scale tap #3

        if return_multiscale:
            return x, {"scale1": scale1, "scale2": scale2, "scale3": scale3}
        return x


# ============================================================================
# Quick shape test
# ============================================================================

if __name__ == "__main__":
    model = MobileViTBackbone()
    z = torch.randn(2, 3, 127, 127)
    x = torch.randn(2, 3, 255, 255)

    feat_z, scales_z = model(z, return_multiscale=True)
    feat_x, scales_x = model(x, return_multiscale=True)

    print("--- Template (127×127) ---")
    print(f"  Output:  {feat_z.shape}")        # [2, 64, 16, 16]
    print(f"  scale1:  {scales_z['scale1'].shape}")  # [2, 24, 32, 32]
    print(f"  scale2:  {scales_z['scale2'].shape}")  # [2, 32, 16, 16]
    print(f"  scale3:  {scales_z['scale3'].shape}")  # [2, 64, 16, 16]

    print("--- Search (255×255) ---")
    print(f"  Output:  {feat_x.shape}")        # [2, 64, 32, 32]
    print(f"  scale1:  {scales_x['scale1'].shape}")  # [2, 24, 64, 64]
    print(f"  scale2:  {scales_x['scale2'].shape}")  # [2, 32, 32, 32]
    print(f"  scale3:  {scales_x['scale3'].shape}")  # [2, 64, 32, 32]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
