import os
import sys
import unittest
import importlib.util

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_PATH = os.path.join(ROOT, "lane_det", "datasets", "row_target_builder.py")
SPEC = importlib.util.spec_from_file_location("row_target_builder_test_module", MODULE_PATH)
ROW_TARGET_BUILDER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROW_TARGET_BUILDER_MODULE)
RowTargetBuilder = ROW_TARGET_BUILDER_MODULE.RowTargetBuilder


class TestRowTargetBuilder(unittest.TestCase):
    def test_invalid_rows_use_ignore_index(self):
        builder = RowTargetBuilder(num_lanes=2, num_y=4, num_grids=10)
        lanes = np.array([[10.0, 20.0, 0.0, 40.0]], dtype=np.float32)
        valid_mask = np.array([[1, 1, 0, 1]], dtype=np.uint8)
        h_samples = np.array([100, 110, 120, 130], dtype=np.float32)

        target = builder.build(lanes, valid_mask, h_samples, img_width=100.0)

        self.assertEqual(target["grid_targets"][0, 2], -1)
        self.assertGreaterEqual(target["grid_targets"][0, 0], 0)
        self.assertLess(target["grid_targets"][0, 0], 10)
        self.assertEqual(target["coord_mask"][0, 2], 0.0)

    def test_ordering_uses_mean_of_all_valid_points(self):
        builder = RowTargetBuilder(num_lanes=2, num_y=6, num_grids=10)
        lanes = np.array(
            [
                [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
                [50.0, 50.0, 50.0, 60.0, 60.0, 60.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.ones_like(lanes, dtype=np.uint8)
        h_samples = np.array([100, 110, 120, 130, 140, 150], dtype=np.float32)

        target = builder.build(lanes, valid_mask, h_samples, img_width=128.0)

        np.testing.assert_array_equal(target["x_coords"][0], lanes[0])
        np.testing.assert_array_equal(target["x_coords"][1], lanes[1])


if __name__ == "__main__":
    unittest.main()
