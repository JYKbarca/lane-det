import numpy as np
import json
import os
from typing import List, Dict, Any

class TuSimpleConverter:
    """
    Convert decoded lanes to TuSimple evaluation format.
    """
    def __init__(self, target_h_samples: List[int] = None):
        if target_h_samples is None:
            # Default TuSimple h_samples: 160 to 710, step 10
            self.target_h_samples = list(range(160, 720, 10))
        else:
            self.target_h_samples = target_h_samples

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
        cur_h_samples = self.target_h_samples if target_h_samples is None else list(target_h_samples)
        
        scale_x = ori_w / img_w
        scale_y = ori_h / img_h
        
        for lane in lanes_list:
            # Get valid points
            valid_mask = lane['valid_mask'] > 0
            if valid_mask.sum() < 2:
                continue
                
            # Get x, y in model coordinates
            xs = lane['x_list'][valid_mask]
            ys = lane['y_samples'][valid_mask]
            
            # Rescale to original coordinates
            xs_ori = xs * scale_x
            ys_ori = ys * scale_y
            
            # Sort by y (increasing) for interpolation
            sort_idx = np.argsort(ys_ori)
            ys_sorted = ys_ori[sort_idx]
            xs_sorted = xs_ori[sort_idx]

            # --- PolyFit Refinement (Improved) ---
            # Use weighted polynomial fitting to balance smoothness and adherence.
            if len(ys_sorted) > 6:
                try:
                    # Weights: Trust bottom points (large y) more than top points (small y).
                    # Top points often drift, so we give them low weight.
                    weights = (ys_sorted / 720.0) ** 2
                    
                    # Fit 3rd degree polynomial: x = ay^3 + by^2 + cy + d
                    # Degree 3 allows for changing curvature (e.g. entering a curve) better than degree 2.
                    # The weights ensure the curve is anchored by the reliable bottom points.
                    z = np.polyfit(ys_sorted, xs_sorted, 3, w=weights)
                    p = np.poly1d(z)
                    
                    # Replace jagged xs with smooth xs
                    xs_sorted = p(ys_sorted)
                except:
                    pass
            # --------------------------------
            
            # Interpolate to target h_samples
            # We use -2 to indicate invalid points (TuSimple convention)
            # np.interp returns extrapolated values if not handled, 
            # so we use left/right parameters.
            
            # However, we also want to mark points as -2 if they are too far 
            # from the predicted range? 
            # Usually, we only output points within the y-range covered by the lane.
            # But TuSimple requires values for ALL h_samples. 
            # If a lane doesn't cover a h_sample, it should be -2.
            
            # Check range
            y_min = ys_sorted.min()
            y_max = ys_sorted.max()
            
            interp_xs = np.interp(cur_h_samples, ys_sorted, xs_sorted, left=-2, right=-2)
            
            # Refine: if target_h is strictly outside [y_min, y_max], ensure it is -2.
            # (np.interp left/right handles strictly outside, but let's be explicit if needed)
            
            # Convert to list
            lane_out = []
            for i, x in enumerate(interp_xs):
                # Double check range (optional, np.interp handles it with left/right)
                if x == -2:
                    lane_out.append(-2)
                else:
                    # Also check if x is within image width
                    if x < 0 or x >= ori_w:
                        lane_out.append(-2)
                    else:
                        lane_out.append(float(x))
            
            # Only add lane if it has enough valid points?
            # TuSimple doesn't strictly require this, but empty lanes are useless.
            valid_cnt = sum(1 for x in lane_out if x != -2)
            if valid_cnt > 2:
                output_lanes.append(lane_out)
                
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
