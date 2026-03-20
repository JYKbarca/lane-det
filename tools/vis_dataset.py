import argparse
import os
import sys

import cv2
import numpy as np

# Ensure project root is importable when running `python tools/xxx.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lane_det.datasets import TuSimpleDataset
from lane_det.utils.config import load_config


def denormalize(image_chw, mean, std):
    img = image_chw.transpose(1, 2, 0)
    img = img * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def draw_lane_points(image, lanes, valid_mask, ys, color=(0, 255, 0), radius=2):
    if lanes is None or valid_mask is None:
        return image
    img = image.copy()
    h, w = img.shape[:2]

    for lane, mask in zip(lanes, valid_mask):
        for x, y, m in zip(lane, ys, mask):
            if m == 0:
                continue
            xi = int(np.clip(x, 0, w - 1))
            yi = int(np.clip(y, 0, h - 1))
            cv2.circle(img, (xi, yi), radius, color, -1)
    return img


def draw_anchor_curve(image, x_values, mask, ys, color=(255, 64, 64), thickness=1):
    h, w = image.shape[:2]
    pts = []
    # Only draw points that are ACTUALLY inside the image
    for x, y in zip(x_values, ys):
        # Strict check: if point is outside image bounds, skip it
        # Do NOT clip, because clipping creates fake vertical lines at edges
        if x < 0 or x >= w or y < 0 or y >= h:
            if len(pts) > 0:
                # If we have accumulated points and now go out of bounds,
                # draw the current segment and start a new one (handle fragmentation)
                if len(pts) >= 2:
                    cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, color, thickness)
                elif len(pts) == 1:
                    cv2.circle(image, pts[0], 2, color, -1)
                pts = []
            continue
        
        xi = int(x)
        yi = int(y)
        pts.append((xi, yi))
        
    # Draw remaining points
    if len(pts) >= 2:
        cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, color, thickness)
    elif len(pts) == 1:
        cv2.circle(image, pts[0], 2, color, -1)


def draw_supervised_points(image, x_values, ys, supervised_mask, color=(0, 255, 255), radius=2):
    """Draw points that actually participate in regression loss."""
    h, w = image.shape[:2]
    for x, y, m in zip(x_values, ys, supervised_mask):
        if m == 0:
            continue
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        cv2.circle(image, (int(x), int(y)), radius, color, -1)


def summarize_distribution(values):
    if len(values) == 0:
        return "n=0"
    arr = np.asarray(values, dtype=np.float32)
    return (
        f"n={arr.size}, mean={arr.mean():.4f}, "
        f"p50={np.percentile(arr, 50):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to config yaml")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--max", type=int, default=20, help="Max samples to save")
    parser.add_argument("--show_match", action="store_true", help="Overlay matched positive anchors")
    parser.add_argument(
        "--show_reg_mask",
        action="store_true",
        help="Overlay regression-supervised points (offset_valid_mask) for selected anchors",
    )
    parser.add_argument("--topk", type=int, default=8, help="Max positive anchors to draw per image")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    os.makedirs(args.out, exist_ok=True)

    dataset = TuSimpleDataset(cfg, split="train")
    max_n = min(args.max, len(dataset))

    mean = cfg["dataset"].get("mean", [0.485, 0.456, 0.406])
    std = cfg["dataset"].get("std", [0.229, 0.224, 0.225])

    total_pos = 0
    total_neg = 0
    total_ignore = 0
    per_gt_max_iou_all = []
    per_gt_pos_count_all = []
    ignore_reason_total = {
        "common_fail": 0,
        "top_fail": 0,
        "angle_fail": 0,
        "x_bottom_fail": 0,
        "threshold_gray": 0,
        "other": 0,
    }

    for i in range(max_n):
        sample = dataset[i]
        img = denormalize(sample["image"], mean, std)
        h, w = img.shape[:2]

        if sample["lanes"] is not None:
            num_y = sample["lanes"].shape[1]
            h_samples = sample["meta"].get("h_samples", None)
            if h_samples is not None and len(h_samples) == num_y:
                ys = np.asarray(h_samples, dtype=np.float32)
            else:
                ys = np.linspace(0, h - 1, num_y, dtype=np.float32)
        else:
            ys = np.linspace(0, h - 1, cfg["dataset"]["y_samples"], dtype=np.float32)

        vis = draw_lane_points(img, sample["lanes"], sample["valid_mask"], ys, color=(0, 255, 0), radius=2)

        cls_target = sample.get("cls_target", None)
        min_dist = sample.get("anchor_min_dist", None)

        if cls_target is not None:
            pos_idx = np.where(cls_target > 0)[0]
            neg_idx = np.where(cls_target == 0)[0]
            ign_idx = np.where(cls_target < 0)[0]

            total_pos += int(pos_idx.size)
            total_neg += int(neg_idx.size)
            total_ignore += int(ign_idx.size)
            match_stats = sample.get("match_stats", None)
            if match_stats is not None:
                per_gt_max_iou_all.extend(match_stats.get("per_gt_max_iou", []))
                per_gt_pos_count_all.extend(match_stats.get("per_gt_pos_count", []))
                rs = match_stats.get("ignore_reason_count", {})
                for k in ignore_reason_total.keys():
                    ignore_reason_total[k] += int(rs.get(k, 0))

            if args.show_match and pos_idx.size > 0:
                # Group by matched GT index to ensure we see anchors for ALL lanes
                matched_gt = sample["matched_gt_idx"]
                unique_gts = np.unique(matched_gt[pos_idx])
                
                sel_indices = []
                for gt_idx in unique_gts:
                    if gt_idx < 0: continue
                    # Find anchors matching this specific GT
                    this_gt_mask = (matched_gt[pos_idx] == gt_idx)
                    this_gt_pos_idx = pos_idx[this_gt_mask]
                    
                    # Sort by distance for this GT
                    dists = min_dist[this_gt_pos_idx]
                    order = np.argsort(dists)
                    
                    # Take top K for THIS lane
                    k = max(1, args.topk)
                    sel_indices.extend(this_gt_pos_idx[order[:k]])
                
                sel = np.array(sel_indices)
                
                anchor_xs = sample["anchor_xs"]
                anchor_mask = sample["anchor_valid_mask"]
                reg_mask_all = sample.get("offset_valid_mask", None)
                anchor_ys = sample.get("anchor_y_samples", ys)
                if len(anchor_ys) != anchor_xs.shape[1]:
                    anchor_ys = ys

                for a in sel:
                    draw_anchor_curve(
                        vis,
                        anchor_xs[a],
                        anchor_mask[a],
                        anchor_ys,
                        color=(255, 64, 64),
                        thickness=1,
                    )
                    if args.show_reg_mask and reg_mask_all is not None:
                        draw_supervised_points(
                            vis,
                            anchor_xs[a],
                            anchor_ys,
                            reg_mask_all[a],
                            color=(0, 255, 255),
                            radius=2,
                        )

            text = f"pos={pos_idx.size} neg={neg_idx.size} ign={ign_idx.size}"
            cv2.putText(vis, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out_path = os.path.join(args.out, f"{i:04d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Saved {max_n} samples to {args.out}")
    if (total_pos + total_neg + total_ignore) > 0:
        print(
            "Anchor label summary: "
            f"pos={total_pos}, neg={total_neg}, ignore={total_ignore}"
        )
    if len(per_gt_max_iou_all) > 0:
        print("[Step2] per-GT max_iou distribution:")
        print(f"  {summarize_distribution(per_gt_max_iou_all)}")
    if len(per_gt_pos_count_all) > 0:
        print("[Step2] per-GT positive-count distribution:")
        print(f"  {summarize_distribution(per_gt_pos_count_all)}")
    if total_ignore > 0:
        print("[Step2] ignore source counts:")
        print(f"  common_fail={ignore_reason_total['common_fail']}")
        print(f"  top_fail={ignore_reason_total['top_fail']}")
        print(f"  angle_fail={ignore_reason_total['angle_fail']}")
        print(f"  x_bottom_fail={ignore_reason_total['x_bottom_fail']}")
        print(f"  threshold_gray={ignore_reason_total['threshold_gray']}")
        print(f"  other={ignore_reason_total['other']}")


if __name__ == "__main__":
    main()
