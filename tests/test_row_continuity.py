import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from lane_det.metrics.tusimple_converter import TuSimpleConverter
from lane_det.utils.row_continuity import build_continuous_valid_mask, extract_continuous_segments


class TestRowContinuity(unittest.TestCase):
    def test_short_segments_are_removed(self):
        valid_mask = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        x_coords = np.array([10, 12, 0, 40, 0, 50, 52, 54], dtype=np.float32)
        cleaned = build_continuous_valid_mask(
            valid_mask,
            x_coords=x_coords,
            min_segment_points=2,
            max_row_gap=0,
            max_dx=10.0,
        )
        np.testing.assert_array_equal(cleaned, np.array([1, 1, 0, 0, 0, 1, 1, 1], dtype=np.uint8))

    def test_large_x_jump_splits_segment(self):
        valid_mask = np.array([1, 1, 1, 1], dtype=np.uint8)
        x_coords = np.array([10, 12, 100, 102], dtype=np.float32)
        segments = extract_continuous_segments(
            valid_mask,
            x_coords=x_coords,
            min_segment_points=2,
            max_row_gap=0,
            max_dx=20.0,
        )
        self.assertEqual(len(segments), 2)
        np.testing.assert_array_equal(segments[0], np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(segments[1], np.array([2, 3], dtype=np.int64))

    def test_converter_preserves_gap_between_segments(self):
        converter = TuSimpleConverter(
            target_h_samples=[100, 110, 120, 130, 140],
            min_segment_points=2,
            max_row_gap=0,
            max_dx_ratio=1.0,
        )
        lane = {
            "x_list": np.array([10, 20, 0, 30, 40], dtype=np.float32),
            "y_samples": np.array([100, 110, 120, 130, 140], dtype=np.float32),
            "valid_mask": np.array([1, 1, 0, 1, 1], dtype=np.uint8),
        }
        result = converter.convert(
            [lane],
            raw_file="clips/test.jpg",
            img_w=100,
            img_h=200,
            ori_w=100,
            ori_h=200,
            target_h_samples=[100, 110, 120, 130, 140],
        )
        self.assertEqual(len(result["lanes"]), 1)
        self.assertEqual(result["lanes"][0][2], -2)
        self.assertNotEqual(result["lanes"][0][1], -2)
        self.assertNotEqual(result["lanes"][0][3], -2)


if __name__ == "__main__":
    unittest.main()
