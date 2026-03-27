import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from lane_det.utils.row_diagnostics import (
    finalize_row_diagnostic_stats,
    init_row_diagnostic_stats,
    update_row_diagnostic_stats,
)


class TestRowDiagnostics(unittest.TestCase):
    def test_row_recall_and_cleanup_keep_rate(self):
        stats = init_row_diagnostic_stats()
        update_row_diagnostic_stats(
            stats,
            gt_exist=[1, 0],
            gt_valid_mask=[
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ],
            raw_valid_mask=[
                [1, 1, 0, 1],
                [1, 1, 1, 1],
            ],
            clean_valid_mask=[
                [1, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        )
        metrics = finalize_row_diagnostic_stats(stats)
        self.assertAlmostEqual(metrics["RawRowRecall"], 0.75)
        self.assertAlmostEqual(metrics["FinalRowRecall"], 0.5)
        self.assertAlmostEqual(metrics["CleanupKeepRate"], 2.0 / 3.0)

    def test_upper_and_lower_row_recall_are_split_by_row_index(self):
        stats = init_row_diagnostic_stats()
        update_row_diagnostic_stats(
            stats,
            gt_exist=[1],
            gt_valid_mask=[[1, 1, 1, 1, 1, 1]],
            raw_valid_mask=[[1, 1, 1, 1, 1, 1]],
            clean_valid_mask=[[0, 1, 1, 1, 1, 0]],
        )
        metrics = finalize_row_diagnostic_stats(stats)
        self.assertAlmostEqual(metrics["UpperRowRecall"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["LowerRowRecall"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
