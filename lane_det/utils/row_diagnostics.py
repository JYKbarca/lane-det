import numpy as np


def init_row_diagnostic_stats():
    return {
        "gt_rows": 0,
        "gt_rows_upper": 0,
        "gt_rows_lower": 0,
        "raw_hits": 0,
        "clean_hits": 0,
        "clean_hits_upper": 0,
        "clean_hits_lower": 0,
    }


def update_row_diagnostic_stats(stats, gt_exist, gt_valid_mask, raw_valid_mask, clean_valid_mask):
    gt_exist = np.asarray(gt_exist, dtype=np.float32).reshape(-1) > 0
    gt_valid_mask = np.asarray(gt_valid_mask, dtype=np.float32)
    raw_valid_mask = np.asarray(raw_valid_mask, dtype=np.uint8)
    clean_valid_mask = np.asarray(clean_valid_mask, dtype=np.uint8)

    if gt_valid_mask.ndim != 2:
        raise ValueError("gt_valid_mask must have shape [num_lanes, num_y]")
    if raw_valid_mask.shape != gt_valid_mask.shape or clean_valid_mask.shape != gt_valid_mask.shape:
        raise ValueError("All row-valid masks must share the same shape")

    num_y = int(gt_valid_mask.shape[1])
    split_idx = max(num_y // 2, 1)
    upper_region = np.zeros((num_y,), dtype=bool)
    upper_region[:split_idx] = True
    lower_region = ~upper_region

    for lane_idx in np.flatnonzero(gt_exist):
        gt_mask = gt_valid_mask[lane_idx] > 0
        if not np.any(gt_mask):
            continue

        raw_mask = raw_valid_mask[lane_idx] > 0
        clean_mask = clean_valid_mask[lane_idx] > 0

        stats["gt_rows"] += int(gt_mask.sum())
        stats["raw_hits"] += int(np.logical_and(raw_mask, gt_mask).sum())
        stats["clean_hits"] += int(np.logical_and(clean_mask, gt_mask).sum())

        gt_upper = np.logical_and(gt_mask, upper_region)
        gt_lower = np.logical_and(gt_mask, lower_region)
        stats["gt_rows_upper"] += int(gt_upper.sum())
        stats["gt_rows_lower"] += int(gt_lower.sum())
        stats["clean_hits_upper"] += int(np.logical_and(clean_mask, gt_upper).sum())
        stats["clean_hits_lower"] += int(np.logical_and(clean_mask, gt_lower).sum())


def finalize_row_diagnostic_stats(stats):
    gt_rows = max(int(stats["gt_rows"]), 1)
    gt_rows_upper = max(int(stats["gt_rows_upper"]), 1)
    gt_rows_lower = max(int(stats["gt_rows_lower"]), 1)
    raw_hits = int(stats["raw_hits"])

    return {
        "RawRowRecall": float(raw_hits) / float(gt_rows),
        "FinalRowRecall": float(stats["clean_hits"]) / float(gt_rows),
        "UpperRowRecall": float(stats["clean_hits_upper"]) / float(gt_rows_upper),
        "LowerRowRecall": float(stats["clean_hits_lower"]) / float(gt_rows_lower),
        "CleanupKeepRate": float(stats["clean_hits"]) / float(max(raw_hits, 1)),
    }
