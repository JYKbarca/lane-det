import torch
import torch.nn as nn
from .anchor_feature_pooler import AnchorFeaturePooler


class LaneSequenceHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, num_layers=3, kernel_size=3):
        super().__init__()
        hidden_channels = int(in_channels if hidden_channels is None else hidden_channels)
        num_layers = max(1, int(num_layers))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length sequence modeling")
        padding = kernel_size // 2

        layers = []
        cur_in = int(in_channels)
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(cur_in, hidden_channels, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            cur_in = hidden_channels
        self.feature_extractor = nn.Sequential(*layers)
        self.pred = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.pred(x)


class AnchorHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_anchors=None,
        use_masked_pooling=True,
        debug_shapes=False,
        use_refinement=False,
        reg_seq_layers=3,
        reg_hidden_channels=None,
        reg_kernel_size=3,
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

        # Classification head
        # Takes pooled features (mean over Y) -> [B, C, Num_Anchors]
        self.cls_head = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, 1, 1) # Output: [B, 1, Num_Anchors]
        )
        
        # Sequence heads model lane geometry along the Y dimension instead of
        # treating all sampled points as independent flattened tokens.
        self.reg_head_stage1 = LaneSequenceHead(
            in_channels=in_channels,
            hidden_channels=reg_hidden_channels,
            num_layers=reg_seq_layers,
            kernel_size=reg_kernel_size,
        )

        # Stage2 regression head (delta offsets).
        self.reg_head_stage2 = LaneSequenceHead(
            in_channels=in_channels,
            hidden_channels=reg_hidden_channels,
            num_layers=reg_seq_layers,
            kernel_size=reg_kernel_size,
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
        # Model each anchor as a feature sequence along Y: [B * NumAnchors, C, NumY]
        seq_features = pooled.permute(0, 2, 1, 3).contiguous().view(b * num_anchors, c, num_y)
        reg_stage1 = self.reg_head_stage1(seq_features).squeeze(1)
        reg_stage1 = reg_stage1.view(b, num_anchors, num_y)

        # 4. Regression stage2 refinement
        if self.use_refinement:
            reg_delta_stage2 = self.reg_head_stage2(seq_features).squeeze(1)
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
