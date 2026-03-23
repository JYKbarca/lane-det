import torch
import torch.nn as nn


class RowLaneHead(nn.Module):
    def __init__(self, in_channels, num_lanes=5, num_y=56, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_lanes = int(num_lanes)
        self.num_y = int(num_y)

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.row_pool = nn.AdaptiveAvgPool2d((self.num_y, 1))
        self.row_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.lane_queries = nn.Parameter(torch.randn(self.num_lanes, hidden_dim))
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.coord_head = nn.Linear(hidden_dim, 1)
        self.valid_head = nn.Linear(hidden_dim, 1)

    def forward(self, feature_map):
        feat = self.proj(feature_map)
        row_feat = self.row_pool(feat).squeeze(-1)
        row_feat = self.row_encoder(row_feat).transpose(1, 2)

        lane_queries = self.lane_queries.unsqueeze(0).unsqueeze(2)
        lane_feat = row_feat.unsqueeze(1) + lane_queries
        lane_feat = self.fuse(lane_feat)

        exist_context = lane_feat.mean(dim=2)
        exist_logits = self.exist_head(exist_context).squeeze(-1)
        valid_logits = self.valid_head(lane_feat).squeeze(-1)
        x_coords = torch.sigmoid(self.coord_head(lane_feat).squeeze(-1))
        return exist_logits, valid_logits, x_coords
