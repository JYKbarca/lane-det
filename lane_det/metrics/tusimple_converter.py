import numpy as np
import json
import os
from typing import List, Dict, Any

from lane_det.utils.row_continuity import extract_continuous_segments

class TuSimpleConverter:
    """
    Convert decoded lanes to TuSimple evaluation format.
    """
    def __init__(
        self,
        target_h_samples: List[int] = None,
        min_segment_points: int = 2,
        max_row_gap: int = 0,
        max_dx_ratio: float = 1.0,
    ):
        if target_h_samples is None:
            # Default TuSimple h_samples: 160 to 710, step 10
            self.target_h_samples = list(range(160, 720, 10))
        else:
            self.target_h_samples = target_h_samples
        self.min_segment_points = int(min_segment_points)
        self.max_row_gap = int(max_row_gap)
        self.max_dx_ratio = float(max_dx_ratio)

    def convert(
        self,
        lanes_list: List[Dict[str, Any]],
        raw_file: str,
        img_w: int,
        img_h: int,
        ori_w: int = 1280,
        ori_h: int = 720,
        target_h_samples: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            lanes_list: List of lane dicts (from LaneDecoder) for ONE image.
                        Each dict has 'x_list', 'y_samples', 'valid_mask'.
            raw_file: Path to image file (relative to root).
            img_w, img_h: Model input size (used for prediction).
            ori_w, ori_h: Original image size (for rescaling).
            
        Returns:
            Dict matching TuSimple format line.
        """
        output_lanes = []
        cur_h_samples = self.target_h_samples if target_h_samples is None else [int(y) for y in target_h_samples]
        
        scale_x = ori_w / img_w
        scale_y = ori_h / img_h
        
        for lane in lanes_list:
            full_valid_mask = np.asarray(lane["valid_mask"], dtype=np.uint8) > 0
            full_xs = np.asarray(lane["x_list"], dtype=np.float32)
            full_ys = np.asarray(lane["y_samples"], dtype=np.float32)
            if full_valid_mask.sum() < self.min_segment_points:
                continue

            segments = extract_continuous_segments(
                full_valid_mask,
                x_coords=full_xs,
                min_segment_points=self.min_segment_points,
                max_row_gap=self.max_row_gap,
                max_dx=self.max_dx_ratio * max(float(img_w) - 1.0, 1.0),
            )
            if not segments:
                continue

            lane_out = np.full(len(cur_h_samples), -2.0, dtype=np.float32)
            cur_h_samples_np = np.asarray(cur_h_samples, dtype=np.float32)
            for segment in segments:
                xs_ori = full_xs[segment] * scale_x
                ys_ori = full_ys[segment] * scale_y
                sort_idx = np.argsort(ys_ori)
                ys_sorted = ys_ori[sort_idx]
                xs_sorted = xs_ori[sort_idx]
                if ys_sorted.shape[0] < 2:
                    continue

                y_min = ys_sorted.min()
                y_max = ys_sorted.max()
                interp_mask = (cur_h_samples_np >= y_min) & (cur_h_samples_np <= y_max)
                if not np.any(interp_mask):
                    continue
                lane_out[interp_mask] = np.interp(cur_h_samples_np[interp_mask], ys_sorted, xs_sorted)

            lane_out_list = []
            for x in lane_out.tolist():
                if x == -2:
                    lane_out_list.append(-2)
                elif x < 0 or x >= ori_w:
                    lane_out_list.append(-2)
                else:
                    lane_out_list.append(float(x))

            valid_cnt = sum(1 for x in lane_out_list if x != -2)
            if valid_cnt > 2:
                output_lanes.append(lane_out_list)
                
        return {
            "raw_file": raw_file,
            "lanes": output_lanes,
            "h_samples": cur_h_samples,
            "run_time": 0
        }

    @staticmethod
    def save_json(results: List[Dict[str, Any]], path: str):
        with open(path, 'w') as f:
            for res in results:
                json.dump(res, f)
                f.write('\n')
