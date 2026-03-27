import numpy as np


class RowTargetBuilder:
    def __init__(
        self,
        num_lanes=5,
        num_y=56,
        num_grids=100,
        order_tail_points=6,
        order_min_tail_points=3,
        order_bottom_weight=2.5,
    ):
        self.num_lanes = int(num_lanes)
        self.num_y = int(num_y)
        self.num_grids = int(num_grids)
        self.order_tail_points = int(order_tail_points)
        self.order_min_tail_points = int(order_min_tail_points)
        self.order_bottom_weight = float(order_bottom_weight)

    def _lane_order_key(self, lane_xs, lane_mask):
        valid_idx = np.flatnonzero(lane_mask > 0)
        if valid_idx.size == 0:
            return float("inf")

        if valid_idx.size < self.order_min_tail_points:
            return float(np.mean(lane_xs[valid_idx]))

        tail_count = min(self.order_tail_points, valid_idx.size)
        tail_idx = valid_idx[-tail_count:]
        tail_xs = lane_xs[tail_idx].astype(np.float32)
        weights = np.linspace(1.0, self.order_bottom_weight, tail_count, dtype=np.float32)
        return float(np.average(tail_xs, weights=weights))

    def build(self, lanes, valid_mask, h_samples, img_width):
        lanes = np.asarray(lanes, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        h_samples = np.asarray(h_samples, dtype=np.float32)

        exist = np.zeros((self.num_lanes,), dtype=np.float32)
        x_coords = np.zeros((self.num_lanes, self.num_y), dtype=np.float32)
        coord_mask = np.zeros((self.num_lanes, self.num_y), dtype=np.float32)
        grid_targets = np.full((self.num_lanes, self.num_y), -1, dtype=np.int64)

        if lanes.ndim == 1:
            lanes = lanes[None, :]
        if valid_mask.ndim == 1:
            valid_mask = valid_mask[None, :]

        if lanes.shape[0] == 0 or h_samples.size == 0:
            return {
                "exist": exist,
                "x_coords": x_coords,
                "coord_mask": coord_mask,
                "grid_targets": grid_targets,
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
            
            # Convert continuous x to discrete grid index
            # x_coords are in [0, img_width-1]
            for y_idx in range(self.num_y):
                if lane_mask[y_idx]:
                    x_val = lanes[lane_idx, y_idx]
                    grid_idx = int(round((x_val / max(img_width - 1.0, 1.0)) * (self.num_grids - 1)))
                    grid_idx = max(0, min(grid_idx, self.num_grids - 1))
                    grid_targets[slot_idx, y_idx] = grid_idx

        return {
            "exist": exist,
            "x_coords": x_coords,
            "coord_mask": coord_mask,
            "grid_targets": grid_targets,
            "row_h_samples": h_samples.astype(np.float32),
        }
