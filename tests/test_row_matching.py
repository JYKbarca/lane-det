import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from lane_det.utils.row_matching import build_matched_targets, solve_lane_assignment


class TestRowMatching(unittest.TestCase):
    def test_solve_lane_assignment_finds_min_cost_permutation(self):
        cost = torch.tensor(
            [
                [4.0, 1.0],
                [1.0, 4.0],
            ],
            dtype=torch.float32,
        )
        assignment = solve_lane_assignment(cost)
        self.assertEqual(sorted(assignment), [(0, 1), (1, 0)])

    def test_build_matched_targets_aligns_swapped_queries(self):
        exist_logits = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
        row_valid_logits = torch.tensor(
            [[[5.0, -5.0], [-5.0, 5.0]]],
            dtype=torch.float32,
        )
        expected_x_norm = torch.tensor(
            [[[0.8, 0.0], [0.0, 0.2]]],
            dtype=torch.float32,
        )
        exist_targets = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        x_targets_norm = torch.tensor(
            [[[0.0, 0.2], [0.8, 0.0]]],
            dtype=torch.float32,
        )
        coord_masks = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]]],
            dtype=torch.float32,
        )

        matched_exist, matched_x, matched_mask, assignments = build_matched_targets(
            exist_logits,
            row_valid_logits,
            expected_x_norm,
            exist_targets,
            x_targets_norm,
            coord_masks,
        )

        self.assertTrue(torch.allclose(matched_exist, torch.tensor([[1.0, 1.0]])))
        self.assertTrue(torch.allclose(matched_x[0, 0], x_targets_norm[0, 1]))
        self.assertTrue(torch.allclose(matched_x[0, 1], x_targets_norm[0, 0]))
        self.assertTrue(torch.allclose(matched_mask[0, 0], coord_masks[0, 1]))
        self.assertTrue(torch.allclose(matched_mask[0, 1], coord_masks[0, 0]))
        self.assertEqual(sorted(assignments[0]), [(0, 1), (1, 0)])


if __name__ == "__main__":
    unittest.main()
