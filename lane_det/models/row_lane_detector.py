import torch.nn as nn

from .backbone import ResNet18
from .fpn import LaneFPN
from .row_lane_head import RowLaneHead


class RowLaneDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg.get("model", {})
        row_cfg = cfg.get("row", {})
        fpn_out = int(model_cfg.get("fpn_out", 128))

        self.backbone = ResNet18(pretrained=True)
        self.fpn = LaneFPN(
            in_channels_list=[64, 128, 256, 512],
            out_channels=fpn_out,
            use_c2=True,
        )
        self.head = RowLaneHead(
            in_channels=fpn_out,
            num_lanes=row_cfg.get("max_lanes", 5),
            num_y=row_cfg.get("num_y", cfg["dataset"]["y_samples"]),
            hidden_dim=model_cfg.get("row_hidden_dim", 256),
            dropout=model_cfg.get("row_dropout", 0.1),
            num_grids=row_cfg.get("num_grids", 100),
        )

    def forward(self, images):
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        if isinstance(fpn_features, (list, tuple)):
            feature_map = fpn_features[0]
        else:
            feature_map = fpn_features
        return self.head(feature_map)
