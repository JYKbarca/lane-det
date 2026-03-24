import math

import torch
import torch.nn as nn


class RowLaneHead(nn.Module):
    def __init__(self, in_channels, num_lanes=5, num_y=56, hidden_dim=256, dropout=0.1, num_grids=100):
        super().__init__()
        self.num_lanes = int(num_lanes)
        self.num_y = int(num_y)
        self.num_grids = int(num_grids)
        self.score_scale = 1.0 / math.sqrt(max(float(hidden_dim), 1.0))

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.row_grid_pool = nn.AdaptiveAvgPool2d((self.num_y, self.num_grids))
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self.row_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.lane_queries = nn.Parameter(torch.randn(self.num_lanes, hidden_dim))
        fusion_dim = hidden_dim * 3
        self.row_fuse = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.no_lane_head = nn.Linear(hidden_dim, 1)

    def forward(self, feature_map):
        feat = self.proj(feature_map)
        grid_feat = self.row_grid_pool(feat)
        grid_feat = self.grid_encoder(grid_feat)

        row_feat = grid_feat.mean(dim=-1)
        row_feat = self.row_encoder(row_feat).transpose(1, 2)

        B, num_y, hidden_dim = row_feat.shape
        lane_queries = self.lane_queries.unsqueeze(0).unsqueeze(2).expand(B, self.num_lanes, num_y, hidden_dim)
        row_feat_expanded = row_feat.unsqueeze(1).expand(B, self.num_lanes, num_y, hidden_dim)
        interaction = row_feat_expanded * lane_queries
        lane_feat = torch.cat([row_feat_expanded, lane_queries, interaction], dim=-1)
        lane_feat = self.row_fuse(lane_feat)

        exist_context = lane_feat.mean(dim=2)
        exist_logits = self.exist_head(exist_context).squeeze(-1)

        position_logits = torch.einsum("bkyh,bhyg->bkyg", lane_feat, grid_feat) * self.score_scale
        no_lane_logits = self.no_lane_head(lane_feat).squeeze(-1)
        grid_logits = torch.cat([position_logits, no_lane_logits.unsqueeze(-1)], dim=-1)
        return exist_logits, grid_logits
