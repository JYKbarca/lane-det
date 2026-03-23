import numpy as np


class RowTargetBuilder:
    def __init__(self, num_lanes=5, num_y=56):
        self.num_lanes = int(num_lanes)
        self.num_y = int(num_y)

    def _lane_order_key(self, lane_xs, lane_mask):
        valid_idx = np.flatnonzero(lane_mask > 0)
        if valid_idx.size == 0:
            return float("inf")

        tail_idx = valid_idx[-min(3, valid_idx.size):]
        return float(np.mean(lane_xs[tail_idx]))

    def build(self, lanes, valid_mask, h_samples):
        lanes = np.asarray(lanes, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        h_samples = np.asarray(h_samples, dtype=np.float32)

        exist = np.zeros((self.num_lanes,), dtype=np.float32)
        x_coords = np.zeros((self.num_lanes, self.num_y), dtype=np.float32)
        coord_mask = np.zeros((self.num_lanes, self.num_y), dtype=np.float32)

        if lanes.ndim == 1:
            lanes = lanes[None, :]
        if valid_mask.ndim == 1:
            valid_mask = valid_mask[None, :]

        if lanes.shape[0] == 0 or h_samples.size == 0:
            return {
                "exist": exist,
                "x_coords": x_coords,
                "coord_mask": coord_mask,
                "row_h_samples": h_samples.astype(np.float32),
            }

        if lanes.shape[1] != self.num_y:
            raise ValueError(
                f"RowTargetBuilder expects num_y={self.num_y}, got lanes.shape[1]={lanes.shape[1]}"
            )

        lane_indices = list(range(lanes.shape[0]))
        lane_indices.sort(key=lambda idx: self._lane_order_key(lanes[idx], valid_mask[idx]))

        for slot_idx, lane_idx in enumerate(lane_indices[: self.num_lanes]):
            lane_mask = valid_mask[lane_idx] > 0
            if lane_mask.sum() == 0:
                continue
            exist[slot_idx] = 1.0
            x_coords[slot_idx] = lanes[lane_idx]
            coord_mask[slot_idx] = lane_mask.astype(np.float32)

        return {
            "exist": exist,
            "x_coords": x_coords,
            "coord_mask": coord_mask,
            "row_h_samples": h_samples.astype(np.float32),
        }
