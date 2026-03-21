import torch
import numpy as np
import warnings
from typing import List, Dict, Any

class LaneDecoder:
    """
    Decode model outputs (cls_logits, reg_preds) into lane coordinates.
    """
    def __init__(
        self,
        score_thr: float = 0.5,
        nms_thr: float = 45.0,
        use_polyfit: bool = True,
        nms_min_common_points: int = 8,
        nms_overlap_ratio_thr: float = 0.6,
        nms_top_dist_ratio: float = 1.25,
        nms_bottom_dist_ratio: float = 1.0,
    ):
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.use_polyfit = bool(use_polyfit)
        self.nms_min_common_points = int(nms_min_common_points)
        self.nms_overlap_ratio_thr = float(nms_overlap_ratio_thr)
        self.nms_top_dist_ratio = float(nms_top_dist_ratio)
        self.nms_bottom_dist_ratio = float(nms_bottom_dist_ratio)

    def _is_duplicate_lane(self, lane_a: Dict[str, Any], lane_b: Dict[str, Any]) -> bool:
        common_mask = (lane_a['valid_mask'] > 0) & (lane_b['valid_mask'] > 0)
        common_idx = np.where(common_mask)[0]
        common_len = int(common_idx.size)
        if common_len < self.nms_min_common_points:
            return False

        len_a = int((lane_a['valid_mask'] > 0).sum())
        len_b = int((lane_b['valid_mask'] > 0).sum())
        min_len = max(min(len_a, len_b), 1)
        overlap_ratio = common_len / float(min_len)
        if overlap_ratio < self.nms_overlap_ratio_thr:
            return False

        diffs = np.abs(lane_a['x_list'][common_idx] - lane_b['x_list'][common_idx])
        mean_dist = float(diffs.mean())
        
        if mean_dist >= self.nms_thr:
            return False

        seg_len = max(1, common_len // 3)
        top_dist = float(diffs[:seg_len].mean())
        bottom_dist = float(diffs[-seg_len:].mean())

        top_dist_thr = self.nms_thr * self.nms_top_dist_ratio
        bottom_dist_thr = self.nms_thr * self.nms_bottom_dist_ratio

        return top_dist < top_dist_thr and bottom_dist < bottom_dist_thr

    def decode(
        self, 
        cls_logits: torch.Tensor, 
        reg_preds: torch.Tensor, 
        anchors, 
        img_w: int, 
        img_h: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Args:
            cls_logits: [B, Num_Anchors]
            reg_preds: [B, Num_Anchors, Num_Y]
            anchors: AnchorSet object (containing anchor_xs, y_samples, etc.)
            img_w: Image width (for clipping)
            img_h: Image height
        
        Returns:
            List of list of lanes. One list of lanes per image in batch.
        """
        batch_size = cls_logits.shape[0]
        device = cls_logits.device
        
        # Get anchor data
        if isinstance(anchors.anchor_xs, np.ndarray):
            anchor_xs = torch.from_numpy(anchors.anchor_xs).to(device)
            anchor_valid_mask = torch.from_numpy(anchors.valid_mask).to(device)
            y_samples = anchors.y_samples # Keep as numpy for output
        else:
            anchor_xs = anchors.anchor_xs.to(device)
            anchor_valid_mask = anchors.valid_mask.to(device)
            y_samples = anchors.y_samples.cpu().numpy()

        # Sigmoid for scores
        scores = torch.sigmoid(cls_logits)
        
        decoded_batch = []
        
        for b in range(batch_size):
            # 1. Filter by score
            cur_scores = scores[b] # [Num_Anchors]
            mask = cur_scores > self.score_thr
            
            if not mask.any():
                decoded_batch.append([])
                continue
                
            keep_indices = torch.nonzero(mask, as_tuple=True)[0]
            keep_scores = cur_scores[keep_indices]
            keep_reg = reg_preds[b, keep_indices] # [K, Num_Y]
            keep_anchor_xs = anchor_xs[keep_indices] # [K, Num_Y]
            keep_anchor_mask = anchor_valid_mask[keep_indices] # [K, Num_Y]
            
            # 2. Recover coordinates
            # pred_x = anchor_x + offset
            pred_xs = keep_anchor_xs + keep_reg
            
            # 3. Handle validity
            pred_valid_mask = (pred_xs >= 0) & (pred_xs <= (img_w - 1))
            final_mask = keep_anchor_mask & pred_valid_mask.int() # [K, Num_Y]
            
            # Convert to numpy for output & NMS
            keep_scores_np = keep_scores.detach().cpu().numpy()
            pred_xs_np = pred_xs.detach().cpu().numpy()
            final_mask_np = final_mask.detach().cpu().numpy()
            
            # Prepare candidates for NMS
            candidates = []
            for k in range(len(keep_indices)):
                valid_len = final_mask_np[k].sum()
                if valid_len < 2:
                    continue
                candidates.append({
                    'score': float(keep_scores_np[k]),
                    'x_list': pred_xs_np[k],
                    'valid_mask': final_mask_np[k].astype(np.uint8),
                    'y_samples': y_samples,
                    'length': int(valid_len)
                })
            
            # Sort only by classification score so train-time score semantics
            # remain aligned with decode-time keep priority.
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 4. Apply Lane NMS when enabled
            if self.nms_thr is not None and self.nms_thr > 0:
                keep_lanes = []
                while len(candidates) > 0:
                    best_lane = candidates.pop(0)
                    keep_lanes.append(best_lane)

                    # Filter out overlapping lanes
                    remaining = []
                    for lane in candidates:
                        if not self._is_duplicate_lane(best_lane, lane):
                            remaining.append(lane)

                    candidates = remaining
            else:
                keep_lanes = candidates

            # 5. Polynomial Fit Smoothing (Fix for wavy lines and starting drift)
            final_lanes = []
            for lane in keep_lanes:
                valid_mask = lane['valid_mask'] > 0
                n_points = valid_mask.sum()
                if n_points < 2:
                    continue

                if self.use_polyfit:
                    y_valid = lane['y_samples'][valid_mask]
                    x_valid = lane['x_list'][valid_mask]

                    # Dynamic degree based on points to avoid RankWarning and overfitting
                    if n_points < 3:
                        deg = 1
                    elif n_points < 5:
                        deg = 2
                    else:
                        deg = 3

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', np.RankWarning)
                            # Fit x = f(y)
                            poly_coeffs = np.polyfit(y_valid, x_valid, deg=deg)
                            poly_func = np.poly1d(poly_coeffs)

                            # Re-calculate x for all valid y samples
                            x_smooth = poly_func(y_valid)

                            # Update the lane x_list
                            lane['x_list'][valid_mask] = x_smooth
                    except Exception:
                        pass # Keep original if fit fails
                
                final_lanes.append(lane)

            decoded_batch.append(final_lanes)
            
        return decoded_batch
