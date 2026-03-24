import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lane_det.models.row_lane_head import RowLaneHead


def _set_norm_identity(module):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, 1.0)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)
    if hasattr(module, "running_mean"):
        module.running_mean.zero_()
    if hasattr(module, "running_var"):
        module.running_var.fill_(1.0)


class TestRowLaneHead(unittest.TestCase):
    def test_output_shapes(self):
        head = RowLaneHead(in_channels=8, num_lanes=5, num_y=56, hidden_dim=16, dropout=0.0, num_grids=100)
        feat = torch.randn(2, 8, 20, 40)

        exist_logits, grid_logits = head(feat)

        self.assertEqual(exist_logits.shape, (2, 5))
        self.assertEqual(grid_logits.shape, (2, 5, 56, 101))

    def test_horizontal_layout_affects_grid_logits(self):
        head = RowLaneHead(in_channels=1, num_lanes=1, num_y=2, hidden_dim=1, dropout=0.0, num_grids=4)
        with torch.no_grad():
            for module in head.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    nn.init.constant_(module.weight, 1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    _set_norm_identity(module)
            head.lane_queries.fill_(1.0)

        head.eval()
        left_heavy = torch.tensor(
            [[[[1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0]]]]
        )
        right_heavy = torch.tensor(
            [[[[0.0, 0.0, 1.0, 1.0],
               [0.0, 0.0, 1.0, 1.0]]]]
        )

        _, left_logits = head(left_heavy)
        _, right_logits = head(right_heavy)

        self.assertFalse(torch.allclose(left_logits[..., :4], right_logits[..., :4]))


if __name__ == "__main__":
    unittest.main()
