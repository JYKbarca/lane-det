import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # We assume input features are [c2, c3, c4, c5]
        # But we only use [c3, c4, c5] for FPN usually, or all.
        # Let's use all [c2, c3, c4, c5] to get high res features.
        # c2: stride 4, c3: stride 8, c4: stride 16, c5: stride 32
        
        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        # inputs: [c2, c3, c4, c5]
        # Build laterals
        laterals = [conv(x) for x, conv in zip(inputs, self.lateral_convs)]
        
        # Top-down path
        # Start from the last level (c5)
        num_levels = len(laterals)
        for i in range(num_levels - 1, 0, -1):
            # Upsample top level and add to current level
            top = laterals[i]
            bottom = laterals[i-1]
            
            # Upsample top to match bottom size
            h, w = bottom.shape[-2:]
            top_up = F.interpolate(top, size=(h, w), mode='nearest')
            laterals[i-1] = bottom + top_up
            
        # Apply 3x3 convs
        outs = [conv(x) for x, conv in zip(laterals, self.fpn_convs)]
        
        # Return all levels: [p2, p3, p4, p5]
        # p2 is highest resolution (stride 4)
        return outs

class LaneFPN(nn.Module):
    """
    FPN specialized for Lane Detection.
    Fuses features into a single high-resolution map.
    """
    def __init__(self, in_channels_list, out_channels, use_c2=False):
        super(LaneFPN, self).__init__()
        self.use_c2 = use_c2
        
        # If use_c2 is False, we only use [c3, c4, c5]
        # in_channels_list should match the input features
        # We assume inputs are always [c2, c3, c4, c5]
        
        # Indices of features to use
        # c2(0), c3(1), c4(2), c5(3)
        self.start_idx = 0 if use_c2 else 1
        self.active_in_channels = in_channels_list[self.start_idx:]
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_c in self.active_in_channels:
            l_conv = nn.Conv2d(in_c, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
    def forward(self, inputs):
        # inputs: [c2, c3, c4, c5]
        # Select active inputs
        active_inputs = inputs[self.start_idx:]
        
        # Build laterals
        laterals = [conv(x) for x, conv in zip(active_inputs, self.lateral_convs)]
        
        # Top-down path
        num_levels = len(laterals)
        for i in range(num_levels - 1, 0, -1):
            top = laterals[i]
            bottom = laterals[i-1]
            h, w = bottom.shape[-2:]
            top_up = F.interpolate(top, size=(h, w), mode='bilinear', align_corners=False)
            laterals[i-1] = bottom + top_up
            
        # Apply 3x3 convs
        outs = [conv(x) for x, conv in zip(laterals, self.fpn_convs)]
        
        # Return all levels for multi-scale feature pooling
        # If use_c2=False, this is [p3, p4, p5]
        # If use_c2=True, this is [p2, p3, p4, p5]
        return outs

