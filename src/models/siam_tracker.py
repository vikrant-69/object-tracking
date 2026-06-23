import torch
import torch.nn as nn
from .backbone import MobileNetV2Backbone
from .head import SiamTrackerHead

class SiamTracker(nn.Module):
    def __init__(self, pretrained_backbone=True, hidden_channels=64):
        super().__init__()
        self.backbone = MobileNetV2Backbone(pretrained=pretrained_backbone)
        self.head = SiamTrackerHead(in_channels=self.backbone.out_channels, hidden_channels=hidden_channels)
        
        # Cache for template features during tracking inference
        self.feat_z = None

    def forward(self, template, search):
        """
        Standard forward pass for training.
        Args:
            template: [B, 3, 127, 127]
            search: [B, 3, 255, 255]
        """
        feat_z = self.backbone(template)
        feat_x = self.backbone(search)
        
        cls_logits, reg_offsets = self.head(feat_z, feat_x)
        return cls_logits, reg_offsets

    def init_template(self, template):
        """
        Extract and cache template features (for tracking inference).
        Args:
            template: [1, 3, 127, 127]
        """
        self.eval()
        with torch.no_grad():
            self.feat_z = self.backbone(template)

    def track(self, search):
        """
        Track target in the search region using cached template features.
        Args:
            search: [1, 3, 255, 255]
        """
        if self.feat_z is None:
            raise ValueError("Template features are not initialized. Call init_template first.")
            
        feat_x = self.backbone(search)
        cls_logits, reg_offsets = self.head(self.feat_z, feat_x)
        return cls_logits, reg_offsets
