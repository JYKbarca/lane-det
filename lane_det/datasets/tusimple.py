import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from lane_det.anchors import AnchorGenerator, LabelAssigner

from .transforms import build_transforms


class TuSimpleDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.root = cfg["dataset"]["root"]
        self.y_samples = int(cfg["dataset"]["y_samples"])
        self.list_file = cfg["dataset"].get("list_file", None)
        self.transforms = build_transforms(cfg, is_train=(split == "train"))

        if self.list_file is None:
            # Default TuSimple list names
            if split == "train":
                self.list_file = os.path.join(self.root, "train.json")
            elif split == "val":
                self.list_file = os.path.join(self.root, "val.json")
            else:
                self.list_file = os.path.join(self.root, "test.json")

        if not os.path.exists(self.list_file):
            raise FileNotFoundError(
                f"TuSimple list file not found: {self.list_file}. "
                "Please set dataset.root or dataset.list_file."
            )

        self.samples = self._load_list(self.list_file)

        self.anchor_assigner = None
        self.anchor_cfg = cfg.get("anchor", None)
        self.anchor_cache = {}
        if self.anchor_cfg is not None:
            self.anchor_assigner = LabelAssigner.from_config(cfg)

    def _load_list(self, list_file):
        samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def _extend_gt_lanes(self, lanes, valid_mask, h_samples):
        # Extend GT lanes to the bottom of the image using linear extrapolation
        # This helps anchors match short lanes that don't reach the bottom
        num_gt, num_y = lanes.shape
        extended_lanes = lanes.copy()
        extended_mask = valid_mask.copy()
        
        h_samples = np.asarray(h_samples, dtype=np.float32)
        
        for i in range(num_gt):
            valid_idx = np.where(valid_mask[i] > 0)[0]
            if len(valid_idx) < 2:
                continue
                
            ys_valid = h_samples[valid_idx]
            xs_valid = lanes[i, valid_idx]
            
            # Use only the bottom-most points for better extrapolation accuracy
            # If we have enough points, use the last 6 points (closest to bottom)
            if len(valid_idx) > 6:
                # Sort by y (descending, i.e., larger y is lower in image)
                # h_samples usually are sorted. Let's assume they are.
                # If h_samples are 0..710, then larger index is lower.
                # Let's just take the last 6 valid points.
                ys_fit = ys_valid[-6:]
                xs_fit = xs_valid[-6:]
            else:
                ys_fit = ys_valid
                xs_fit = xs_valid
            
            # Fit line: x = k * y + b
            k, b = np.polyfit(ys_fit, xs_fit, 1)
            
            # Extrapolate to all y samples
            # We fill in where mask is 0
            invalid_idx = np.where(valid_mask[i] == 0)[0]
            if len(invalid_idx) > 0:
                ys_invalid = h_samples[invalid_idx]
                xs_pred = k * ys_invalid + b
                
                # Update extended lanes
                extended_lanes[i, invalid_idx] = xs_pred
                # Mark as valid for matching assignment
                extended_mask[i, invalid_idx] = 1
                
        return extended_lanes, extended_mask

    def _parse_lanes(self, item):
        h_samples = item.get("h_samples", [])
        lanes_raw = item.get("lanes", [])

        if not h_samples or not lanes_raw:
            return None, None

        num_y = len(h_samples)
        lanes = []
        valid_mask = []

        for lane in lanes_raw:
            lane = np.array(lane, dtype=np.float32)
            mask = (lane >= 0).astype(np.uint8)
            lane[lane < 0] = 0.0
            lanes.append(lane)
            valid_mask.append(mask)

        lanes = np.stack(lanes, axis=0)  # [N, num_y]
        valid_mask = np.stack(valid_mask, axis=0)  # [N, num_y]

        if num_y != self.y_samples:
            pass

        return lanes, valid_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.root, item["raw_file"])
        
        # Read image
        try:
            with open(img_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
            
        if image is None:
            raise FileNotFoundError(f"Image not found or invalid: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image size
        h_orig, w_orig = image.shape[:2]

        lanes, valid_mask = self._parse_lanes(item)
        if lanes is None:
             # Handle case with no lanes
             lanes = np.zeros((0, self.y_samples), dtype=np.float32)
             valid_mask = np.zeros((0, self.y_samples), dtype=np.uint8)
             h_samples = np.linspace(0, h_orig-1, self.y_samples)
        else:
             h_samples = np.array(item.get("h_samples", []), dtype=np.float32)

        # Apply transforms
        # Note: transforms might change image size and scale lanes/h_samples
        image, lanes, valid_mask, h_samples = self.transforms(image, lanes, valid_mask, h_samples)
        
        # Keep TuSimple's original y-sample semantics after resize.
        # We only resample if the sample count itself is inconsistent.
        h_img, w_img = image.shape[1], image.shape[2] # image is [C, H, W] after transforms

        # Only repair malformed inputs with inconsistent sample count.
        if len(h_samples) != self.y_samples:
            target_h_samples = np.linspace(
                float(np.min(h_samples)) if len(h_samples) > 0 else 0.0,
                float(np.max(h_samples)) if len(h_samples) > 0 else float(h_img - 1),
                self.y_samples,
                dtype=np.float32,
            )
            new_lanes = []
            new_valid_mask = []
            
            for i in range(len(lanes)):
                # Filter valid points for interpolation
                valid_idx = valid_mask[i] > 0
                if valid_idx.sum() < 2:
                    new_lanes.append(np.zeros_like(target_h_samples))
                    new_valid_mask.append(np.zeros_like(target_h_samples, dtype=np.uint8))
                    continue
                
                ys_valid = h_samples[valid_idx]
                xs_valid = lanes[i][valid_idx]
                
                # Sort by y just in case
                sort_idx = np.argsort(ys_valid)
                ys_valid = ys_valid[sort_idx]
                xs_valid = xs_valid[sort_idx]
                
                # Interpolate
                # We only interpolate within the range of original valid y
                # Points outside original range are invalid
                # Use np.interp for linear interpolation
                # But np.interp expects increasing x (here ys_valid must be increasing)
                # ys_valid is sorted above.
                
                xs_interp = np.interp(target_h_samples, ys_valid, xs_valid, left=-1000, right=-1000)
                
                # Create mask: valid if within original y range
                # Be careful with y direction. y increases downwards.
                # So min_y is top, max_y is bottom.
                y_min = ys_valid.min()
                y_max = ys_valid.max()
                
                # Mask points outside original valid range
                mask_interp = (target_h_samples >= y_min) & (target_h_samples <= y_max)
                
                # Also check if interpolation result is valid (not -1000)
                mask_interp = mask_interp & (xs_interp != -1000)
                
                # Set invalid points to 0
                xs_interp[~mask_interp] = 0
                
                new_lanes.append(xs_interp.astype(np.float32))
                new_valid_mask.append(mask_interp.astype(np.uint8))
            
            if len(new_lanes) > 0:
                lanes = np.stack(new_lanes)
                valid_mask = np.stack(new_valid_mask)
            else:
                lanes = np.zeros((0, self.y_samples), dtype=np.float32)
                valid_mask = np.zeros((0, self.y_samples), dtype=np.uint8)
                
            h_samples = target_h_samples

        sample = {
            "image": image,
            "lanes": lanes,
            "valid_mask": valid_mask,
            "meta": {
                "raw_file": item.get("raw_file", ""),
                "img_path": img_path,
                "h_samples": h_samples if h_samples is not None else [],
                "original_h_samples": item.get("h_samples", []), # Keep original for evaluation
            },
        }

        if self.anchor_assigner is not None:
             # Get anchors based on current image size and y_samples
             h, w = h_img, w_img
             anchors = self._get_anchor_set(len(h_samples), h_samples, h, w)
             
             # Create extended GT for assignment ONLY
             # DISABLED: Now that we have side anchors, we don't need to artificially extend lanes to the bottom.
             # This prevents matching bottom anchors to incorrect linear extrapolations of curved/short lanes.
             # if len(lanes) > 0:
             #    ext_lanes, ext_mask = self._extend_gt_lanes(lanes, valid_mask, h_samples)
             # else:
             #    ext_lanes, ext_mask = lanes, valid_mask
             
             # Use original lanes for matching
             ext_lanes, ext_mask = lanes, valid_mask
             
             assigned = self.anchor_assigner.assign(
                 anchors.anchor_xs,
                 anchors.valid_mask,
                 ext_lanes,      # Use extended lanes for matching
                 ext_mask,       # Use extended mask for matching
                 anchor_angles=anchors.angles,
                 anchor_x_bottom=anchors.x_bottom,
                 y_samples=anchors.y_samples,
             )
             
             # Fix the valid_mask in the assigned result to respect original GT
             # We want to regress to the extended lane ONLY if the original point was valid?
             # Or do we want to regress to the extended lane even if original point was invalid?
             # Usually, we only supervise on valid GT points.
             # But if we extended the GT, maybe we want to supervise on extended parts too?
             # Let's stick to original valid points for regression supervision to be safe.
             
             final_offset_mask = assigned.valid_mask.copy()
             
             # But wait, assigned.valid_mask is [Num_Anchors, Num_Y]
             # It is 1 where anchor is valid AND matched GT is valid (based on ext_mask passed to assigner)
             # If we used ext_mask for assignment, then assigned.valid_mask includes extended points.
             # We should mask out points that were NOT valid in original GT.
             
             # Get matched GT index for each anchor
             matched_gt_indices = assigned.matched_gt_idx # [Num_Anchors]
             
             # Create a mask of valid GT points for each matched anchor
             # [Num_Anchors, Num_Y]
             gt_valid_mask_for_anchors = np.zeros_like(final_offset_mask)
             
             for i, gt_idx in enumerate(matched_gt_indices):
                 if gt_idx >= 0 and gt_idx < len(valid_mask):
                     gt_valid_mask_for_anchors[i] = valid_mask[gt_idx]
             
             # Intersect
             final_offset_mask = final_offset_mask & gt_valid_mask_for_anchors

             sample.update(
                 {
                     "anchor_xs": anchors.anchor_xs,
                     "anchor_valid_mask": anchors.valid_mask,
                     "anchor_y_samples": anchors.y_samples,
                     "cls_target": assigned.cls_target,
                     "offset_label": assigned.offset_label,
                     "offset_valid_mask": final_offset_mask, # Use the corrected mask
                     "matched_gt_idx": assigned.matched_gt_idx,
                     "best_gt_idx": assigned.best_gt_idx,
                     "best_iou": assigned.best_iou,
                     "anchor_min_dist": assigned.min_dist,
                     "match_stats": assigned.match_stats,
                 }
             )

        return sample

    def _get_anchor_set(self, num_y, h_samples, h, w):
        # h, w are current image size (after transform)
        key = (int(w), int(h), int(num_y), tuple(np.asarray(h_samples, dtype=np.int32).tolist()))
        if key in self.anchor_cache:
            return self.anchor_cache[key]

        cache_dir = self.cfg.get("paths", {}).get("anchor_cache", "outputs/cache/anchors")

        generator = AnchorGenerator(
            img_size=[w, h],
            num_y=num_y,
            x_positions=self.anchor_cfg["x_positions"],
            angles=self.anchor_cfg["angles"],
            side_y_step=self.anchor_cfg.get("side_y_step", 20),
            side_y_start_ratio=self.anchor_cfg.get("side_y_start_ratio", 0.5),
            side_angle_min=self.anchor_cfg.get("side_angle_min", 10.0),
            cache_dir=cache_dir,
            use_cache=True,
            y_samples=h_samples,
        )
        anchors = generator.generate()
        self.anchor_cache[key] = anchors
        return anchors
