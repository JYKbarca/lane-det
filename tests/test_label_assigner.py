import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from lane_det.anchors.label_assigner import LabelAssigner


class TestLabelAssigner(unittest.TestCase):
    def test_threshold_positive_exists_when_topk_disabled(self):
        assigner = LabelAssigner(
            neg_thr=0.10,
            pos_thr=0.35,
            min_common_points=2,
            min_common_ratio=0.5,
            line_iou_width=20.0,
            topk_per_gt=0,
        )

        anchor_xs = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)
        anchor_valid = np.ones_like(anchor_xs, dtype=np.uint8)
        gt_lanes = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)
        gt_valid = np.ones_like(gt_lanes, dtype=np.uint8)

        assigned = assigner.assign(anchor_xs, anchor_valid, gt_lanes, gt_valid)

        self.assertGreater(float(assigned.cls_target[0]), 0.0)
        self.assertEqual(int(assigned.matched_gt_idx[0]), 0)

    def test_fallback_cannot_bypass_angle_gate(self):
        assigner = LabelAssigner(
            neg_thr=0.10,
            pos_thr=0.35,
            min_common_points=2,
            min_common_ratio=0.5,
            line_iou_width=20.0,
            geo_angle_thr=5.0,
            geo_x_bottom_thr=20.0,
            topk_per_gt=1,
            min_force_pos_iou=0.10,
        )

        y_samples = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float32)
        anchor_xs = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)
        anchor_valid = np.ones_like(anchor_xs, dtype=np.uint8)
        gt_lanes = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)
        gt_valid = np.ones_like(gt_lanes, dtype=np.uint8)
        anchor_angles = np.array([80.0], dtype=np.float32)
        anchor_x_bottom = np.array([50.0], dtype=np.float32)

        assigned = assigner.assign(
            anchor_xs,
            anchor_valid,
            gt_lanes,
            gt_valid,
            anchor_angles=anchor_angles,
            anchor_x_bottom=anchor_x_bottom,
            y_samples=y_samples,
        )

        self.assertEqual(float(assigned.cls_target[0]), 0.0)
        self.assertEqual(int(assigned.matched_gt_idx[0]), -1)
        self.assertEqual(float(assigned.best_iou[0]), -1.0)

    def test_match_stats_count_uses_matched_gt_idx(self):
        assigner = LabelAssigner(neg_thr=0.10, pos_thr=0.35)

        iou_mat = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
            ],
            dtype=np.float32,
        )
        best_iou = np.array([0.90, 0.80], dtype=np.float32)
        cls_target = np.array([0.95, 0.80], dtype=np.float32)
        matched_gt_idx = np.array([0, 1], dtype=np.int64)
        reason_masks = {
            "all_common_fail": np.zeros((2,), dtype=bool),
            "top_fail_any": np.zeros((2,), dtype=bool),
            "angle_fail_any": np.zeros((2,), dtype=bool),
            "xbottom_fail_any": np.zeros((2,), dtype=bool),
        }

        stats = assigner._build_match_stats(
            num_gt=2,
            iou_mat=iou_mat,
            best_iou=best_iou,
            cls_target=cls_target,
            matched_gt_idx=matched_gt_idx,
            reason_masks=reason_masks,
        )

        self.assertEqual(stats["per_gt_pos_count"], [1, 1])


if __name__ == "__main__":
    unittest.main()
