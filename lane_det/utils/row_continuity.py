import numpy as np


def extract_continuous_segments(valid_mask, x_coords=None, min_segment_points=2, max_row_gap=0, max_dx=None):
    valid_mask = np.asarray(valid_mask, dtype=np.uint8).reshape(-1)
    valid_indices = np.flatnonzero(valid_mask > 0)
    if valid_indices.size == 0:
        return []

    if x_coords is not None:
        x_coords = np.asarray(x_coords, dtype=np.float32).reshape(-1)
        if x_coords.shape[0] != valid_mask.shape[0]:
            raise ValueError("x_coords and valid_mask must have the same length")

    segments = []
    current = [int(valid_indices[0])]
    for idx in valid_indices[1:]:
        idx = int(idx)
        prev_idx = current[-1]
        row_gap = idx - prev_idx - 1
        jump_break = False
        if x_coords is not None and max_dx is not None:
            max_allowed_dx = float(max_dx) * max(idx - prev_idx, 1)
            jump_break = abs(float(x_coords[idx]) - float(x_coords[prev_idx])) > max_allowed_dx
        if row_gap > int(max_row_gap) or jump_break:
            if len(current) >= int(min_segment_points):
                segments.append(np.asarray(current, dtype=np.int64))
            current = [idx]
        else:
            current.append(idx)

    if len(current) >= int(min_segment_points):
        segments.append(np.asarray(current, dtype=np.int64))
    return segments


def build_continuous_valid_mask(valid_mask, x_coords=None, min_segment_points=2, max_row_gap=0, max_dx=None):
    cleaned_mask = np.zeros_like(np.asarray(valid_mask, dtype=np.uint8).reshape(-1))
    for segment in extract_continuous_segments(
        valid_mask,
        x_coords=x_coords,
        min_segment_points=min_segment_points,
        max_row_gap=max_row_gap,
        max_dx=max_dx,
    ):
        cleaned_mask[segment] = 1
    return cleaned_mask
