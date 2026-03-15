import torch
import torch.nn as nn
from .anchor_feature_pooler import AnchorFeaturePooler

class AnchorHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_anchors=None,
        use_masked_pooling=True,
        debug_shapes=False,
        use_refinement=False,
    ):
        super(AnchorHead, self).__init__()
        # Default num_levels is 4 assuming use_c2=True inside LaneFPN configuration.
        self.pooler = AnchorFeaturePooler(
            in_channels=in_channels,
            num_levels=4,
            align_corners=True, 
            padding_mode="zeros"
        )

        self.use_masked_pooling = bool(use_masked_pooling)
        self.debug_shapes = bool(debug_shapes)
        self.use_refinement = bool(use_refinement)
        self._debug_printed = False
        
        # Shared feature extraction
        self.conv_shared = nn.Conv1d(in_channels, in_channels, 1)
        self.bn_shared = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        
        # Classification head
        # Takes pooled features (mean over Y) -> [B, C, Num_Anchors]
        self.cls_head = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, 1, 1) # Output: [B, 1, Num_Anchors]
        )
        
        # Stage1 regression head (coarse offsets).
        self.reg_head_stage1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, 1, 1)  # Output: [B, 1, Num_Anchors * Num_Y]
        )

        # Stage2 regression head (delta offsets).
        self.reg_head_stage2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, 1, 1),  # Output: [B, 1, Num_Anchors * Num_Y]
        )

    def pool_anchors(self, features, anchors, img_h, img_w):
        """
        Pool features for each anchor using grid_sample.
        features: [B, C, H_feat, W_feat]
        anchors: AnchorSet object
        img_h, img_w: Original image height and width
        """
        return self.pooler(features, anchors, img_h, img_w)

    def _get_anchor_valid_mask(self, anchors, device, dtype):
        """
        Returns:
          mask: [1, 1, NumAnchors, NumY] float tensor in {0, 1}
        """
        anchor_valid_mask = anchors.valid_mask
        if isinstance(anchor_valid_mask, torch.Tensor):
            mask = anchor_valid_mask.to(device=device, dtype=dtype)
        else:
            mask = torch.as_tensor(anchor_valid_mask, device=device, dtype=dtype)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, features, anchors, img_h, img_w, return_aux=False):
        # features: [B, C, H, W]
        # anchors: AnchorSet
        
        # 1. Pool features
        # [B, C, Num_Anchors, Num_Y]
        pooled = self.pool_anchors(features, anchors, img_h, img_w)
        
        b, c, num_anchors, num_y = pooled.shape
        
        # 2. Classification
        # Mean pool over Y dimension -> [B, C, Num_Anchors]
        if self.use_masked_pooling:
            valid_mask = self._get_anchor_valid_mask(anchors, pooled.device, pooled.dtype)
            pooled_sum = (pooled * valid_mask).sum(dim=3)
            valid_count = valid_mask.sum(dim=3).clamp(min=1.0)
            pooled_mean = pooled_sum / valid_count
        else:
            pooled_mean = pooled.mean(dim=3)
        
        cls_logits = self.cls_head(pooled_mean) # [B, 1, Num_Anchors]
        cls_logits = cls_logits.squeeze(1) # [B, Num_Anchors]
        
        # 3. Regression stage1
        # Reshape to [B, C, Num_Anchors * Num_Y] to apply Conv1d
        pooled_flat = pooled.view(b, c, -1)
        reg_stage1 = self.reg_head_stage1(pooled_flat) # [B, 1, Num_Anchors * Num_Y]
        reg_stage1 = reg_stage1.view(b, num_anchors, num_y) # [B, Num_Anchors, Num_Y]

        # 4. Regression stage2 refinement
        if self.use_refinement:
            reg_delta_stage2 = self.reg_head_stage2(pooled_flat) # [B, 1, Num_Anchors * Num_Y]
            reg_delta_stage2 = reg_delta_stage2.view(b, num_anchors, num_y)
            reg_final = reg_stage1 + reg_delta_stage2
        else:
            reg_delta_stage2 = torch.zeros_like(reg_stage1)
            reg_final = reg_stage1

        if self.debug_shapes and (not self._debug_printed):
            print(
                "[AnchorHead Debug] "
                f"pooled={tuple(pooled.shape)}, "
                f"pooled_mean={tuple(pooled_mean.shape)}, "
                f"cls_logits={tuple(cls_logits.shape)}, "
                f"reg_stage1={tuple(reg_stage1.shape)}, "
                f"reg_delta_stage2={tuple(reg_delta_stage2.shape)}, "
                f"reg_final={tuple(reg_final.shape)}"
            )
            self._debug_printed = True

        if return_aux:
            aux = {
                "reg_stage1": reg_stage1,
                "reg_delta_stage2": reg_delta_stage2,
                "reg_final": reg_final,
            }
            return cls_logits, reg_final, aux

        return cls_logits, reg_final
