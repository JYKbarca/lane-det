import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorFeaturePooler(nn.Module):
    """
    Pool per-anchor feature sequences from feature map(s) via grid_sample.
    Now supports Multi-Scale inputs:
    If a list of features [P2, P3, P4, P5] is provided, it will pool from all 
    levels and fuse them together using a 1x1 convolution back to the original channel count.

    Input:
      - features_list: a single tensor [B, C, Hf, Wf] OR a list of tensors
      - anchors.anchor_xs: [NumAnchors, NumY]
      - anchors.y_samples: [NumY]
      - img_h, img_w: original image size used to normalize coordinates
    Output:
      - pooled_features: [B, C, NumAnchors, NumY]
    """

    def __init__(self, in_channels=128, num_levels=4, align_corners: bool = True, padding_mode: str = "zeros"):
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        
        # 1x1 convolution to fuse multi-scale concatenated features back to in_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_levels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list, anchors, img_h, img_w):
        # Convert single feature tensor to list for unified processing
        if not isinstance(features_list, (list, tuple)):
            features_list = [features_list]
            
        device = features_list[0].device
        dtype = features_list[0].dtype

        if isinstance(anchors.anchor_xs, np.ndarray):
            anchor_xs = torch.from_numpy(anchors.anchor_xs).to(device, dtype)
            y_samples = torch.from_numpy(anchors.y_samples).to(device, dtype)
        else:
            anchor_xs = anchors.anchor_xs.to(device, dtype)
            y_samples = anchors.y_samples.to(device, dtype)

        batch_size = features_list[0].shape[0]

        # Normalize from image coords to [-1, 1] for grid_sample.
        x_norm = (anchor_xs / (img_w - 1)) * 2 - 1  # [NumAnchors, NumY]
        y_norm = (y_samples / (img_h - 1)) * 2 - 1  # [NumY]
        y_norm = y_norm.unsqueeze(0).expand_as(x_norm)  # [NumAnchors, NumY]

        grid = torch.stack([x_norm, y_norm], dim=-1)  # [NumAnchors, NumY, 2]
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, NumAnchors, NumY, 2]

        pooled_list = []
        for feat in features_list:
            pooled = F.grid_sample(
                feat,
                grid,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            )  # [B, C, NumAnchors, NumY]
            pooled_list.append(pooled)
            
        # If there's only 1 level (or legacy usage), just return it directly
        if len(pooled_list) == 1:
            return pooled_list[0]
            
        # Concatenate along channel dimension: [B, num_levels * C, NumAnchors, NumY]
        pooled_cat = torch.cat(pooled_list, dim=1)
        
        # 1x1 conv dimension reduction and fusion: [B, C, NumAnchors, NumY]
        pooled_fused = self.fuse_conv(pooled_cat)
        
        return pooled_fused
