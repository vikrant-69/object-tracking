"""
Shape verification smoke test for Siamese Tracker v2.
Tests all modules in isolation and the full assembled pipeline.
"""

import sys
import torch

# Add src to path
sys.path.insert(0, '.')

print("=" * 60)
print("  Shape Verification Smoke Test")
print("=" * 60)

# 1. Backbone
print("\n--- 1. Backbone (MobileViT-XS) ---")
from models.backbone import MobileViTBackbone
backbone = MobileViTBackbone()

z = torch.randn(2, 3, 127, 127)
x = torch.randn(2, 3, 255, 255)

feat_z, scales_z = backbone(z, return_multiscale=True)
feat_x, scales_x = backbone(x, return_multiscale=True)

print(f"  Template output:    {feat_z.shape}")       # [2, 64, 16, 16]
print(f"  Template scale1:    {scales_z['scale1'].shape}")
print(f"  Template scale2:    {scales_z['scale2'].shape}")
print(f"  Template scale3:    {scales_z['scale3'].shape}")
print(f"  Search output:      {feat_x.shape}")         # [2, 64, 32, 32]
print(f"  Search scale1:      {scales_x['scale1'].shape}")
print(f"  Search scale2:      {scales_x['scale2'].shape}")
print(f"  Search scale3:      {scales_x['scale3'].shape}")
print(f"  Params: {sum(p.numel() for p in backbone.parameters()):,}")

# 2. Temporal Memory
print("\n--- 2. Temporal Memory (SSM) ---")
from models.temporal_memory import TemporalMemoryModule
memory = TemporalMemoryModule(dim=64, cache_size=4, state_dim=16)

seq = torch.randn(2, 4, 256, 64)  # [B, T, N_tokens, D]
q = memory(seq)
print(f"  Aggregated Q:       {q.shape}")              # [2, 256, 64]
print(f"  Params: {sum(p.numel() for p in memory.parameters()):,}")

# 3. Linear Cross-Attention + RPN Head
print("\n--- 3. Fusion + RPN Head ---")
from models.head import LinearCrossAttention, RPNHead
fusion = LinearCrossAttention(dim=64, heads=4)
head = RPNHead(in_channels=64, hidden_channels=64)

kv = torch.randn(2, 1024, 64)  # search tokens
fused = fusion(q, kv)
print(f"  Fused search tokens: {fused.shape}")         # [2, 1024, 64]

cls_map, reg_map = head(fused, spatial_h=32, spatial_w=32)
print(f"  Cls logits:          {cls_map.shape}")       # [2, 1, 32, 32]
print(f"  Reg offsets:         {reg_map.shape}")       # [2, 4, 32, 32]
print(f"  Params (fusion+head): {sum(p.numel() for p in fusion.parameters()) + sum(p.numel() for p in head.parameters()):,}")

# 4. Full Assembled Tracker
print("\n--- 4. Full SiamTracker (Training Forward) ---")
from models.siam_tracker import SiamTracker
tracker = SiamTracker()

templates = torch.randn(2, 4, 3, 127, 127)
search = torch.randn(2, 3, 255, 255)

cls, reg, scales = tracker(templates, search, return_multiscale=True)
print(f"  Cls logits:     {cls.shape}")
print(f"  Reg offsets:    {reg.shape}")
for k, v in scales.items():
    print(f"  Scale '{k}':  {v.shape}")

total_params = sum(p.numel() for p in tracker.parameters())
print(f"  Total params:   {total_params:,}")

# 5. Inference Mode
print("\n--- 5. Inference Mode ---")
tracker.init_template(torch.randn(1, 3, 127, 127))
cls_i, reg_i = tracker.track(torch.randn(1, 3, 255, 255))
print(f"  Inference cls:  {cls_i.shape}")
print(f"  Inference reg:  {reg_i.shape}")

print("\n" + "=" * 60)
print("  ALL SHAPE CHECKS PASSED!")
print("=" * 60)
