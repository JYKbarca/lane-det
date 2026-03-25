import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from lane_det.utils.row_supervision import compute_coord_loss, compute_expected_x, compute_soft_grid_loss


class TestRowSupervision(unittest.TestCase):
    def test_expected_x_matches_weighted_average(self):
        logits = torch.tensor([[[[-20.0, -20.0, 20.0, 20.0, -20.0]]]], dtype=torch.float32)
        expected_x = compute_expected_x(logits, num_grids=5)
        self.assertAlmostEqual(float(expected_x.item()), 0.625, places=3)

    def test_soft_grid_loss_prefers_neighboring_bins(self):
        target_x = torch.tensor([[[0.6250]]], dtype=torch.float32)
        coord_mask = torch.tensor([[[1.0]]], dtype=torch.float32)

        aligned_logits = torch.tensor([[[[-4.0, -4.0, 4.0, 4.0, -4.0]]]], dtype=torch.float32)
        misaligned_logits = torch.tensor([[[[4.0, -4.0, -4.0, -4.0, -4.0]]]], dtype=torch.float32)

        aligned_loss = compute_soft_grid_loss(aligned_logits, target_x, coord_mask, num_grids=5)
        misaligned_loss = compute_soft_grid_loss(misaligned_logits, target_x, coord_mask, num_grids=5)

        self.assertLess(float(aligned_loss.item()), float(misaligned_loss.item()))

    def test_coord_loss_ignores_invalid_rows(self):
        pred_x = torch.tensor([[[0.50, 0.10]]], dtype=torch.float32)
        target_x = torch.tensor([[[0.50, 0.90]]], dtype=torch.float32)
        coord_mask = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)

        loss = compute_coord_loss(pred_x, target_x, coord_mask, num_grids=100)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
