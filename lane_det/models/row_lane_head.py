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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.exist_head = nn.Linear(hidden_dim, self.num_lanes)
        self.coord_head = nn.Linear(hidden_dim, self.num_lanes * self.num_y)

    def forward(self, feature_map):
        feat = self.proj(feature_map)
        pooled = self.pool(feat)
        hidden = self.shared(pooled)
        exist_logits = self.exist_head(hidden)
        x_coords = self.coord_head(hidden).view(-1, self.num_lanes, self.num_y)
        return exist_logits, x_coords
