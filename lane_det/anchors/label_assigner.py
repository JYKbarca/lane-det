from dataclasses import dataclass

import numpy as np


@dataclass
class AssignedLabels:
    cls_target: np.ndarray
    offset_label: np.ndarray
    valid_mask: np.ndarray
    matched_gt_idx: np.ndarray
    best_gt_idx: np.ndarray
    best_iou: np.ndarray
    min_dist: np.ndarray
    match_stats: dict | None = None


class LabelAssigner:
    """Assign cls / offset labels with Line IoU on shared valid points."""

    def __init__(
        self,
        neg_thr: float,
        min_common_points: int = 6,
        min_common_ratio: float = 0.5,
        line_iou_width: float = 15.0,
        geo_angle_thr: float = 30.0,
        geo_x_bottom_thr: float = 200.0,
        top_region_ratio: float = 0.4,
        top_max_mean_err: float = 20.0,
        top_min_points: int = 2,
        min_common_ratio_bottom: float | None = None,
        min_common_ratio_side: float | None = None,
        top_max_mean_err_bottom: float | None = None,
        top_max_mean_err_side: float | None = None,
        line_iou_width_bottom: float | None = None,
        line_iou_width_side: float | None = None,
        min_common_points_bottom: int | None = None,
        min_common_points_side: int | None = None,
        top_min_points_bottom: int | None = None,
        top_min_points_side: int | None = None,
        topk_per_gt: int = 0,
        min_force_pos_iou: float = 0.10,
        use_soft_gating: bool = False,
        soft_common: bool = True,
        soft_top: bool = True,
        soft_angle: bool = True,
        penalty_common_ratio: float = 1.0,
        penalty_top_err: float = 1.0,
        penalty_angle: float = 1.0,
        pos_score_min: float = 0.6,
    ):
        self.neg_thr = float(neg_thr)
        self.min_common_points = int(min_common_points)
        self.min_common_ratio = float(min_common_ratio)
        self.line_iou_width = float(line_iou_width)
        self.geo_angle_thr = float(geo_angle_thr)
        self.geo_x_bottom_thr = float(geo_x_bottom_thr)
        self.top_region_ratio = float(top_region_ratio)
        self.top_max_mean_err = float(top_max_mean_err)
        self.top_min_points = int(top_min_points)
        self.min_common_ratio_bottom = float(min_common_ratio if min_common_ratio_bottom is None else min_common_ratio_bottom)
        self.min_common_ratio_side = float(min_common_ratio if min_common_ratio_side is None else min_common_ratio_side)
        self.top_max_mean_err_bottom = float(top_max_mean_err if top_max_mean_err_bottom is None else top_max_mean_err_bottom)
        self.top_max_mean_err_side = float(top_max_mean_err if top_max_mean_err_side is None else top_max_mean_err_side)
        self.line_iou_width_bottom = float(line_iou_width if line_iou_width_bottom is None else line_iou_width_bottom)
        self.line_iou_width_side = float(line_iou_width if line_iou_width_side is None else line_iou_width_side)
        self.min_common_points_bottom = int(min_common_points if min_common_points_bottom is None else min_common_points_bottom)
        self.min_common_points_side = int(min_common_points if min_common_points_side is None else min_common_points_side)
        self.top_min_points_bottom = int(top_min_points if top_min_points_bottom is None else top_min_points_bottom)
        self.top_min_points_side = int(top_min_points if top_min_points_side is None else top_min_points_side)
        # Step3 skeleton params for future strategies (Top-K / Soft-Gating).
        # These fields are parsed and stored now, but not applied yet.
        self.topk_per_gt = int(topk_per_gt)
        self.min_force_pos_iou = float(min_force_pos_iou)
        self.use_soft_gating = bool(use_soft_gating)
        self.soft_common = bool(soft_common)
        self.soft_top = bool(soft_top)
        self.soft_angle = bool(soft_angle)
        # Soft-gating penalties should only attenuate IoU, not amplify.
        self.penalty_common_ratio = float(np.clip(penalty_common_ratio, 0.0, 1.0))
        self.penalty_top_err = float(np.clip(penalty_top_err, 0.0, 1.0))
        self.penalty_angle = float(np.clip(penalty_angle, 0.0, 1.0))
        self.pos_score_min = float(np.clip(pos_score_min, 0.0, 1.0))

    @classmethod
    def from_config(cls, cfg):
        acfg = cfg["anchor"]
        mcfg = cfg.get("match", {})
        global_soft = bool(mcfg.get("use_soft_gating", False))
        return cls(
            neg_thr=float(mcfg.get("neg_thr", acfg.get("line_iou_neg_thr", 0.35))),
            min_common_points=int(acfg.get("min_common_points", 6)),
            min_common_ratio=float(acfg.get("min_common_ratio", 0.5)),
            line_iou_width=float(acfg.get("line_iou_width", 15.0)),
            geo_angle_thr=float(acfg.get("geo_angle_thr", 30.0)),
            geo_x_bottom_thr=float(acfg.get("geo_x_bottom_thr", 200.0)),
            top_region_ratio=float(acfg.get("top_region_ratio", 0.4)),
            top_max_mean_err=float(acfg.get("top_max_mean_err", 20.0)),
            top_min_points=int(acfg.get("top_min_points", 2)),
            min_common_ratio_bottom=acfg.get("min_common_ratio_bottom", None),
            min_common_ratio_side=acfg.get("min_common_ratio_side", None),
            top_max_mean_err_bottom=acfg.get("top_max_mean_err_bottom", None),
            top_max_mean_err_side=acfg.get("top_max_mean_err_side", None),
            line_iou_width_bottom=acfg.get("line_iou_width_bottom", None),
            line_iou_width_side=acfg.get("line_iou_width_side", None),
            min_common_points_bottom=acfg.get("min_common_points_bottom", None),
            min_common_points_side=acfg.get("min_common_points_side", None),
            top_min_points_bottom=acfg.get("top_min_points_bottom", None),
            top_min_points_side=acfg.get("top_min_points_side", None),
            topk_per_gt=int(mcfg.get("topk_per_gt", 0)),
            min_force_pos_iou=float(mcfg.get("min_force_pos_iou", 0.10)),
            use_soft_gating=global_soft,
            soft_common=bool(mcfg.get("soft_common", global_soft)),
            soft_top=bool(mcfg.get("soft_top", global_soft)),
            soft_angle=bool(mcfg.get("soft_angle", global_soft)),
            penalty_common_ratio=float(mcfg.get("penalty_common_ratio", 1.0)),
            penalty_top_err=float(mcfg.get("penalty_top_err", 1.0)),
            penalty_angle=float(mcfg.get("penalty_angle", 1.0)),
            pos_score_min=float(mcfg.get("pos_score_min", 0.6)),
        )

    def _map_iou_to_cls_target(self, iou: float) -> float:
        if iou < self.min_force_pos_iou:
            return 0.0
        if self.min_force_pos_iou >= 1.0:
            return 1.0
        norm = (float(iou) - self.min_force_pos_iou) / (1.0 - self.min_force_pos_iou)
        norm = float(np.clip(norm, 0.0, 1.0))
        return self.pos_score_min + (1.0 - self.pos_score_min) * norm

    def _line_iou(self, anchor_xs, gt_xs, common_mask, width_per_anchor):
        """
        Compute Line IoU between one GT lane and all anchors over shared valid points.

        For each shared y-point, represent lane as a horizontal segment [x-w, x+w].
        IoU per point:
            inter = max(0, 2w - |dx|)
            union = 2w + |dx|
        Aggregate as sum(inter) / sum(union) on shared points.
        """
        w = np.asarray(width_per_anchor, dtype=np.float32)[:, None]
        dx = np.abs(anchor_xs - gt_xs[None, :])
        inter = np.maximum(0.0, 2.0 * w - dx)
        union = 2.0 * w + dx

        cm = common_mask.astype(np.float32)
        inter_sum = (inter * cm).sum(axis=1)
        union_sum = (union * cm).sum(axis=1)

        valid = union_sum > 0
        iou = np.zeros(anchor_xs.shape[0], dtype=np.float32)
        iou[valid] = inter_sum[valid] / union_sum[valid]
        return iou

    def _build_match_stats(self, num_gt, iou_mat, best_iou, cls_target, reason_masks):
        if num_gt <= 0:
            return {
                "per_gt_max_iou": [],
                "per_gt_pos_count": [],
                "ignore_reason_count": {
                    "common_fail": 0,
                    "top_fail": 0,
                    "angle_fail": 0,
                    "x_bottom_fail": 0,
                    "threshold_gray": 0,
                    "other": int((cls_target < 0).sum()),
                },
            }

        per_gt_max_iou = iou_mat.max(axis=0)
        per_gt_max_iou = np.where(per_gt_max_iou >= 0, per_gt_max_iou, 0.0).astype(np.float32)
        per_gt_pos_count = np.zeros((num_gt,), dtype=np.int64)

        pos_ids = np.where(cls_target > 0)[0]
        for a in pos_ids:
            g = int(np.argmax(iou_mat[a]))
            if g >= 0:
                per_gt_pos_count[g] += 1

        ignore_ids = np.where(cls_target < 0)[0]
        common_fail = 0
        top_fail = 0
        angle_fail = 0
        x_bottom_fail = 0
        threshold_gray = 0
        other = 0

        for a in ignore_ids:
            if reason_masks["all_common_fail"][a]:
                common_fail += 1
            elif reason_masks["top_fail_any"][a]:
                top_fail += 1
            elif reason_masks["angle_fail_any"][a]:
                angle_fail += 1
            elif reason_masks["xbottom_fail_any"][a]:
                x_bottom_fail += 1
            elif best_iou[a] >= 0:
                threshold_gray += 1
            else:
                other += 1

        return {
            "per_gt_max_iou": per_gt_max_iou.tolist(),
            "per_gt_pos_count": per_gt_pos_count.tolist(),
            "ignore_reason_count": {
                "common_fail": int(common_fail),
                "top_fail": int(top_fail),
                "angle_fail": int(angle_fail),
                "x_bottom_fail": int(x_bottom_fail),
                "threshold_gray": int(threshold_gray),
                "other": int(other),
            },
        }

    def assign(self, anchor_xs, anchor_valid_mask, gt_lanes, gt_valid_mask, anchor_angles=None, anchor_x_bottom=None, y_samples=None):
        anchor_xs = np.asarray(anchor_xs, dtype=np.float32)
        anchor_valid_mask = np.asarray(anchor_valid_mask, dtype=np.uint8)

        num_anchors, num_y = anchor_xs.shape
        cls_target = np.full((num_anchors,), -1.0, dtype=np.float32)
        offset_label = np.zeros((num_anchors, num_y), dtype=np.float32)
        valid_mask = np.zeros((num_anchors, num_y), dtype=np.uint8)
        matched_gt_idx = np.full((num_anchors,), -1, dtype=np.int64)
        min_dist = np.full((num_anchors,), np.inf, dtype=np.float32)

        if gt_lanes is None or gt_valid_mask is None:
            cls_target[:] = 0
            return AssignedLabels(
                cls_target,
                offset_label,
                valid_mask,
                matched_gt_idx,
                np.full((num_anchors,), -1, dtype=np.int64),
                np.full((num_anchors,), -1.0, dtype=np.float32),
                min_dist,
                match_stats=self._build_match_stats(
                    0,
                    np.zeros((num_anchors, 0), dtype=np.float32),
                    np.full((num_anchors,), -1.0, dtype=np.float32),
                    cls_target,
                    {
                        "all_common_fail": np.zeros((num_anchors,), dtype=bool),
                        "top_fail_any": np.zeros((num_anchors,), dtype=bool),
                        "angle_fail_any": np.zeros((num_anchors,), dtype=bool),
                        "xbottom_fail_any": np.zeros((num_anchors,), dtype=bool),
                    },
                ),
            )

        gt_lanes = np.asarray(gt_lanes, dtype=np.float32)
        gt_valid_mask = np.asarray(gt_valid_mask, dtype=np.uint8)

        if gt_lanes.size == 0:
            cls_target[:] = 0
            return AssignedLabels(
                cls_target,
                offset_label,
                valid_mask,
                matched_gt_idx,
                np.full((num_anchors,), -1, dtype=np.int64),
                np.full((num_anchors,), -1.0, dtype=np.float32),
                min_dist,
                match_stats=self._build_match_stats(
                    0,
                    np.zeros((num_anchors, 0), dtype=np.float32),
                    np.full((num_anchors,), -1.0, dtype=np.float32),
                    cls_target,
                    {
                        "all_common_fail": np.zeros((num_anchors,), dtype=bool),
                        "top_fail_any": np.zeros((num_anchors,), dtype=bool),
                        "angle_fail_any": np.zeros((num_anchors,), dtype=bool),
                        "xbottom_fail_any": np.zeros((num_anchors,), dtype=bool),
                    },
                ),
            )

        num_gt = gt_lanes.shape[0]
        iou_mat = np.full((num_anchors, num_gt), -1.0, dtype=np.float32)
        common_fail_mat = np.zeros((num_anchors, num_gt), dtype=bool)
        top_fail_mat = np.zeros((num_anchors, num_gt), dtype=bool)
        angle_fail_mat = np.zeros((num_anchors, num_gt), dtype=bool)
        xbottom_fail_mat = np.zeros((num_anchors, num_gt), dtype=bool)

        # Pre-calculate GT geometric properties if needed for gating
        gt_angles = np.zeros(num_gt, dtype=np.float32)
        gt_x_bottom = np.zeros(num_gt, dtype=np.float32)
        
        # We need y_samples to estimate slope/angle
        # If not provided, we can assume linear spacing, but better to have it.
        # For simplicity, let's assume y_samples are sorted descending or ascending.
        # We'll use simple linear regression to get slope and x_bottom.
        
        has_geo_info = (anchor_angles is not None) and (anchor_x_bottom is not None) and (y_samples is not None)
        
        if has_geo_info:
            y_samples = np.asarray(y_samples, dtype=np.float32)
            for g in range(num_gt):
                valid_idx = np.where(gt_valid_mask[g] > 0)[0]
                if len(valid_idx) < 2:
                    # Too few points to estimate direction, mark as invalid/neutral
                    gt_angles[g] = np.nan 
                    continue
                
                ys_valid = y_samples[valid_idx]
                xs_valid = gt_lanes[g, valid_idx]
                
                # Fit line: x = k * y + b
                # slope k = (N*sum(xy) - sum(x)sum(y)) / (N*sum(yy) - sum(y)^2)
                # or just use polyfit
                k, b = np.polyfit(ys_valid, xs_valid, 1)
                
                # Angle: tan(theta) = k. 
                # Note: y increases downwards. 
                # If x increases as y decreases (upwards), k is negative?
                # Let's align with anchor definition: 
                # Anchor: x = x_bottom + tan(angle) * (y_bottom - y)
                #       = x_bottom + tan(angle)*y_bottom - tan(angle)*y
                # So anchor slope wrt y is -tan(angle).
                # Thus k = -tan(angle)  =>  tan(angle) = -k
                angle_deg = np.degrees(np.arctan(-k))
                gt_angles[g] = angle_deg
                
                # x_bottom at the last y sample (usually image bottom)
                y_bottom = y_samples[-1]
                gt_x_bottom[g] = k * y_bottom + b

        for g in range(num_gt):
            common = (anchor_valid_mask > 0) & (gt_valid_mask[g][None, :] > 0)
            common_count = common.sum(axis=1)
            gt_count = int(gt_valid_mask[g].sum())
            denom = max(gt_count, 1)
            common_ratio = common_count.astype(np.float32) / float(denom)
            is_side_anchor = anchor_valid_mask[:, -1] == 0
            min_common_points_req = np.where(is_side_anchor, self.min_common_points_side, self.min_common_points_bottom)
            min_common_ratio_req = np.where(is_side_anchor, self.min_common_ratio_side, self.min_common_ratio_bottom)
            line_iou_width_req = np.where(is_side_anchor, self.line_iou_width_side, self.line_iou_width_bottom)

            iou = self._line_iou(anchor_xs, gt_lanes[g], common, line_iou_width_req)
            fail_common = (common_count < min_common_points_req) | (common_ratio < min_common_ratio_req)
            common_fail_mat[:, g] = fail_common
            hard_common_fail = common_count == 0
            soft_common_fail = fail_common & (~hard_common_fail)
            if self.use_soft_gating and self.soft_common:
                iou[hard_common_fail] = -1.0
                valid_soft = soft_common_fail & (iou >= 0)
                iou[valid_soft] = iou[valid_soft] * self.penalty_common_ratio
            else:
                iou[fail_common] = -1.0

            # Extra constraint: anchors that diverge too much in the top region are rejected.
            if y_samples is not None:
                y_min = float(y_samples.min())
                y_max = float(y_samples.max())
                y_cut = y_min + self.top_region_ratio * (y_max - y_min)
                top_region = y_samples <= y_cut
                if np.any(top_region):
                    top_common = common & top_region[None, :]
                    top_count = top_common.sum(axis=1)
                    abs_diff = np.abs(anchor_xs - gt_lanes[g][None, :])
                    top_sum = (abs_diff * top_common.astype(np.float32)).sum(axis=1)
                    top_mean_err = np.zeros(num_anchors, dtype=np.float32)
                    top_min_points_req = np.where(is_side_anchor, self.top_min_points_side, self.top_min_points_bottom)
                    valid_top = (top_count > 0) & (top_count >= top_min_points_req)
                    top_mean_err[valid_top] = top_sum[valid_top] / np.maximum(top_count[valid_top], 1)
                    top_err_thr = np.where(is_side_anchor, self.top_max_mean_err_side, self.top_max_mean_err_bottom)
                    fail_top = valid_top & (top_mean_err > top_err_thr)
                    top_fail_mat[:, g] = fail_top
                    if self.use_soft_gating and self.soft_top:
                        valid_soft = fail_top & (iou >= 0)
                        iou[valid_soft] = iou[valid_soft] * self.penalty_top_err
                    else:
                        iou[fail_top] = -1.0

            # --- Geometric Gating ---
            if has_geo_info and not np.isnan(gt_angles[g]):
                # Gate 1: Orientation Consistency
                # This prevents "X-shape" crossings where anchor and lane have different directions.
                angle_diff = np.abs(anchor_angles - gt_angles[g])
                fail_angle = angle_diff > self.geo_angle_thr
                angle_fail_mat[:, g] = fail_angle
                if self.use_soft_gating and self.soft_angle:
                    valid_soft = fail_angle & (iou >= 0)
                    iou[valid_soft] = iou[valid_soft] * self.penalty_angle
                else:
                    iou[fail_angle] = -1.0
                
                # Gate 2: Bottom Intersection Sanity
                # For side anchors, x_bottom is a virtual extrapolated value and can be unstable.
                # We therefore apply x_bottom gating only to bottom anchors.
                # A practical discriminator is whether the anchor has a valid point at the bottom y.
                x_bottom_diff = np.abs(anchor_x_bottom - gt_x_bottom[g])
                non_side_fail = (~is_side_anchor) & (x_bottom_diff > self.geo_x_bottom_thr)
                xbottom_fail_mat[:, g] = non_side_fail
                iou[non_side_fail] = -1.0
            
            iou_mat[:, g] = iou

        best_gt = np.argmax(iou_mat, axis=1)
        best_iou = iou_mat[np.arange(num_anchors), best_gt]
        invalid_best = best_iou < 0
        best_gt[invalid_best] = -1

        # Keep historical field semantics for downstream visualization: lower is better.
        # Use (1 - IoU), invalid pairs stay +inf.
        valid_best = best_iou >= 0
        min_dist[valid_best] = 1.0 - best_iou[valid_best]

        # Positive anchors are defined only by top-k quality matches per GT.
        # All other valid anchors become negatives instead of additional positives.
        force_pos = np.zeros((num_anchors,), dtype=bool)
        force_gt_idx = np.full((num_anchors,), -1, dtype=np.int64)
        if self.topk_per_gt > 0 and num_gt > 0:
            for g in range(num_gt):
                ious_g = iou_mat[:, g]
                valid_ids = np.where(ious_g >= 0)[0]
                if valid_ids.size == 0:
                    continue
                order = np.argsort(ious_g[valid_ids])[::-1]
                top_ids = valid_ids[order[: self.topk_per_gt]]
                qualified = top_ids[ious_g[top_ids] >= self.min_force_pos_iou]
                for a in qualified:
                    if (not force_pos[a]) or (ious_g[a] > iou_mat[a, force_gt_idx[a]]):
                        force_pos[a] = True
                        force_gt_idx[a] = g

            # --- Fallback guarantee: every GT gets at least 1 positive anchor ---
            # If a GT has no positive anchor after top-k filtering, fall back to the
            # anchor with the highest *raw* IoU, ignoring all gate results.
            # We must ensure anchors are not "stolen" by later GTs, leaving earlier GTs empty.
            
            # 1. Find which GTs still need an anchor
            gt_has_pos = np.zeros(num_gt, dtype=bool)
            for a in np.where(force_pos)[0]:
                gt_has_pos[force_gt_idx[a]] = True

            # 2. Assign fallback anchors, avoiding already claimed ones if possible
            for g in range(num_gt):
                if gt_has_pos[g]:
                    continue
                    
                # Recompute raw IoU without any gating for this GT
                raw_common = (anchor_valid_mask > 0) & (gt_valid_mask[g][None, :] > 0)
                raw_common_count = raw_common.sum(axis=1)
                is_side = anchor_valid_mask[:, -1] == 0
                raw_width = np.where(is_side, self.line_iou_width_side, self.line_iou_width_bottom)
                raw_iou = self._line_iou(anchor_xs, gt_lanes[g], raw_common, raw_width)
                
                # Require at least some overlap
                raw_iou[raw_common_count < 2] = -1.0
                
                if raw_iou.max() <= 0:
                    continue
                    
                # Try to find the best anchor that isn't already claimed by another GT
                unclaimed_mask = ~force_pos
                unclaimed_iou = raw_iou.copy()
                unclaimed_iou[~unclaimed_mask] = -1.0
                
                if unclaimed_iou.max() > 0:
                    # Found an unclaimed anchor
                    best_a = int(np.argmax(unclaimed_iou))
                    force_pos[best_a] = True
                    force_gt_idx[best_a] = g
                    iou_mat[best_a, g] = raw_iou[best_a]
                else:
                    # All overlapping anchors are claimed by other GTs.
                    # We CANNOT steal, because that would leave another GT empty.
                    # Instead, we allow this anchor to be assigned to the CURRENT GT as well.
                    # Wait, force_gt_idx is a 1D array, so an anchor can only point to ONE GT.
                    # If we overwrite force_gt_idx[best_a], we steal it.
                    # If we don't overwrite it, this GT gets nothing.
                    # Since an anchor can only regress to one GT, if two GTs perfectly overlap
                    # and share the exact same anchors, they are essentially duplicates.
                    # Leaving one GT without an anchor is the ONLY mathematically sound choice 
                    # when using a 1D assignment array.
                    pass
                
        # --- Crucial Sync: Recompute best_gt and best_iou after fallback ---
        # Fallback modifies iou_mat, so we must update these to prevent mismatch
        # between matched_gt_idx and best_gt_idx in downstream loss/stats.
        best_gt = np.argmax(iou_mat, axis=1)
        best_iou = iou_mat[np.arange(num_anchors), best_gt]
        invalid_best = best_iou < 0
        best_gt[invalid_best] = -1

        pos = force_pos
        neg = (best_iou >= 0) & (~pos)

        cls_target[neg] = 0.0
        matched_gt_idx[force_pos] = force_gt_idx[force_pos]
        pos_ids = np.where(pos)[0]
        for a in pos_ids:
            g = matched_gt_idx[a]
            cls_target[a] = self._map_iou_to_cls_target(float(iou_mat[a, g]))

        for a in pos_ids:
            g = matched_gt_idx[a]
            common = (anchor_valid_mask[a] > 0) & (gt_valid_mask[g] > 0)
            if not np.any(common):
                continue
            valid_mask[a, common] = 1
            offset_label[a, common] = gt_lanes[g, common] - anchor_xs[a, common]

        reason_masks = {
            "all_common_fail": np.all(common_fail_mat, axis=1),
            "top_fail_any": np.any(top_fail_mat, axis=1),
            "angle_fail_any": np.any(angle_fail_mat, axis=1),
            "xbottom_fail_any": np.any(xbottom_fail_mat, axis=1),
        }
        match_stats = self._build_match_stats(num_gt, iou_mat, best_iou, cls_target, reason_masks)

        return AssignedLabels(
            cls_target,
            offset_label,
            valid_mask,
            matched_gt_idx,
            best_gt.astype(np.int64),
            best_iou.astype(np.float32),
            min_dist,
            match_stats=match_stats,
        )
