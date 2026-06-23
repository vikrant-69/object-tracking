import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load standard MobileNet-V2
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
        except ImportError:
            base_model = models.mobilenet_v2(pretrained=pretrained)
            
        features = base_model.features
        
        # Slice features up to block 8 (inclusive)
        # Block 0: ConvBNActivation (stride 2) - output: 32 ch, size /2
        # Block 1: InvertedResidual (stride 1) - output: 16 ch, size /2
        # Block 2: InvertedResidual (stride 2) - output: 24 ch, size /4
        # Block 3: InvertedResidual (stride 1) - output: 24 ch, size /4
        # Block 4: InvertedResidual (stride 2) - output: 32 ch, size /8
        # Block 5: InvertedResidual (stride 1) - output: 32 ch, size /8
        # Block 6: InvertedResidual (stride 1) - output: 32 ch, size /8
        # Block 7: InvertedResidual (stride 2 -> change to 1) - output: 64 ch, size /8
        # Block 8: InvertedResidual (stride 1) - output: 64 ch, size /8
        
        self.layers = nn.Sequential(*[features[i] for i in range(9)])
        
        # Modify Block 7 stride and dilation
        # features[7] is an InvertedResidual block. Let's modify it.
        block7 = self.layers[7]
        self._modify_block_stride_dilation(block7, stride=1, dilation=2)
        
        # Modify Block 8 dilation to match Block 7
        block8 = self.layers[8]
        self._modify_block_stride_dilation(block8, stride=1, dilation=2)
        
        # Output channels for Block 8 is 64
        self.out_channels = 64

    def _modify_block_stride_dilation(self, block, stride=1, dilation=2):
        # Scan block modules for Conv2d layers
        convs = []
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                convs.append(m)
                
        # Modify stride and dilation in the depthwise conv (the one with groups > 1)
        for sub_layer in convs:
            if sub_layer.groups > 1:
                # Modify stride
                sub_layer.stride = (stride, stride)
                # Modify dilation
                sub_layer.dilation = (dilation, dilation)
                # Adjust padding to maintain spatial size: pad = dilation * (kernel_size - 1) // 2
                padding = dilation * (sub_layer.kernel_size[0] - 1) // 2
                sub_layer.padding = (padding, padding)
                
        # Update shortcut connection check
        # InvertedResidual uses: block.use_res_connect = block.stride == 1 and in_channels == out_channels
        # Since we modified stride to 1, we update use_res_connect based on the input and output channels of the block
        in_channels = convs[0].in_channels
        out_channels = convs[-1].out_channels
        block.use_res_connect = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Quick shape verification
    model = MobileNetV2Backbone(pretrained=False)
    z = torch.randn(2, 3, 127, 127)
    x = torch.randn(2, 3, 255, 255)
    
    feat_z = model(z)
    feat_x = model(x)
    
    print("Template features shape:", feat_z.shape) # Expected: [2, 64, 16, 16]
    print("Search features shape:", feat_x.shape)   # Expected: [2, 64, 32, 32]
