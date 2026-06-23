import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseXCorr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, search, template):
        """
        Perform batch-wise depthwise cross-correlation.
        Args:
            search: Tensor of shape [B, C, H_x, W_x] (e.g., [B, 64, 32, 32])
            template: Tensor of shape [B, C, H_z, W_z] (e.g., [B, 64, 16, 16])
        Returns:
            corr_map: Tensor of shape [B, C, H_out, W_out]
        """
        B, C, H_z, W_z = template.shape
        _, _, H_x, W_x = search.shape
        
        # Unfold search tensor
        search_unfold = F.unfold(search, kernel_size=(H_z, W_z))
        
        # Reshape search_unfold
        H_out = H_x - H_z + 1
        W_out = W_x - W_z + 1
        search_unfold = search_unfold.view(B, C, H_z * W_z, H_out, W_out)
        
        # Reshape template
        template_flat = template.view(B, C, H_z * W_z, 1, 1)
        
        # Multiply and sum
        out = (search_unfold * template_flat).sum(dim=2)
        return out


class DepthwiseSeparableConv(nn.Module):
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
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiamTrackerHead(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=64):
        super().__init__()
        
        # Feature adjustment layers for template and search branches
        self.cls_adjust_template = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )
        self.cls_adjust_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )
        
        self.reg_adjust_template = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )
        self.reg_adjust_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )
        
        # Cross correlation
        self.xcorr = DepthwiseXCorr()
        
        # Classification head (outputs 1 channel: target vs background logits)
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )
        
        # Regression head (outputs 4 channels: l, t, r, b offsets)
        self.reg_head = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            # Output values must be positive, so we'll apply exp or relu during forward/loss.
            nn.Conv2d(hidden_channels, 4, kernel_size=1)
        )

    def forward(self, feat_z, feat_x):
        """
        Args:
            feat_z: Template features [B, C, H_z, W_z]
            feat_x: Search features [B, C, H_x, W_x]
        Returns:
            cls_map: Logits for target presence [B, 1, H_out, W_out]
            reg_map: Bounding box offsets [B, 4, H_out, W_out]
        """
        # Classification branch correlation
        cls_z = self.cls_adjust_template(feat_z)
        cls_x = self.cls_adjust_search(feat_x)
        cls_corr = self.xcorr(cls_x, cls_z)
        
        # Regression branch correlation
        reg_z = self.reg_adjust_template(feat_z)
        reg_x = self.reg_adjust_search(feat_x)
        reg_corr = self.xcorr(reg_x, reg_z)
        
        # Bounding box head outputs
        cls_logits = self.cls_head(cls_corr)
        reg_offsets = self.reg_head(reg_corr)
        
        # Ensure regression offsets are positive (distances to boundaries)
        # Using softplus or exp is common in anchor-free trackers (FCOS uses exp).
        # We will use exp to match standard FCOS, scaled by a factor or clipped for stability.
        reg_offsets = torch.exp(torch.clamp(reg_offsets, min=-10.0, max=10.0))
        
        return cls_logits, reg_offsets

if __name__ == "__main__":
    # Check shapes
    head = SiamTrackerHead(in_channels=64, hidden_channels=64)
    feat_z = torch.randn(2, 64, 16, 16)
    feat_x = torch.randn(2, 64, 32, 32)
    
    cls_map, reg_map = head(feat_z, feat_x)
    print("Class map shape:", cls_map.shape) # Expected: [2, 1, 17, 17]
    print("Reg map shape:", reg_map.shape)   # Expected: [2, 4, 17, 17]
