import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lane_det.anchors import AnchorSet
from lane_det.datasets.tusimple import TuSimpleDataset
from lane_det.models import LaneDetector
from lane_det.postprocess import LaneDecoder


def resolve_list_file(cfg, split):
    dataset_cfg = cfg.get("dataset", {})
    list_file = dataset_cfg.get("list_file", "")
    root = dataset_cfg.get("root", "")

    if split == "train":
        if list_file:
            return os.path.join(os.path.dirname(list_file), "train.json")
        return os.path.join(root, "train.json")

    if split == "val":
        if list_file:
            return os.path.join(os.path.dirname(list_file), "val.json")
        return os.path.join(root, "val.json")

    if list_file and os.path.basename(list_file) == "test_label.json":
        return list_file
    if root and os.path.basename(root.rstrip("/\\")) == "test_set":
        return os.path.join(os.path.dirname(root), "test_label.json")
    return os.path.join(root, "test.json")


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images = torch.stack([torch.from_numpy(b["image"]).float() for b in batch])
    first = batch[0]
    anchors = AnchorSet(
        anchor_xs=first["anchor_xs"],
        valid_mask=first["anchor_valid_mask"],
        x_bottom=first.get("anchor_x_bottom", None),
        angles=first.get("anchor_angles", None),
        y_samples=first["anchor_y_samples"],
    )

    cls_targets = None
    if "cls_target" in first:
        cls_targets = torch.stack([torch.from_numpy(b["cls_target"]).float() for b in batch])

    return {
        "images": images,
        "anchors": anchors,
        "metas": [b["meta"] for b in batch],
        "cls_targets": cls_targets,
        "match_stats": [b.get("match_stats", None) for b in batch],
    }


def summarize_array(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def build_score_hist(scores: List[float], bins: int = 10) -> Dict[str, object]:
    if len(scores) == 0:
        return {"bins": [], "counts": []}
    hist, edges = np.histogram(np.asarray(scores, dtype=np.float32), bins=bins, range=(0.0, 1.0))
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(len(hist))]
    return {"bins": labels, "counts": hist.astype(int).tolist()}


def decode_candidates(decoder, cls_logits, reg_preds, anchors, img_w, img_h):
    device = cls_logits.device
    if isinstance(anchors.anchor_xs, np.ndarray):
        anchor_xs = torch.from_numpy(anchors.anchor_xs).to(device)
        anchor_valid_mask = torch.from_numpy(anchors.valid_mask).to(device)
        y_samples = anchors.y_samples
    else:
        anchor_xs = anchors.anchor_xs.to(device)
        anchor_valid_mask = anchors.valid_mask.to(device)
        y_samples = anchors.y_samples.cpu().numpy()

    scores = torch.sigmoid(cls_logits)
    batch_records = []

    for b in range(cls_logits.shape[0]):
        cur_scores = scores[b]
        passed_thr = cur_scores > decoder.score_thr
        thr_indices = torch.nonzero(passed_thr, as_tuple=True)[0]
        thr_scores = cur_scores[thr_indices]
        thr_reg = reg_preds[b, thr_indices]
        thr_anchor_xs = anchor_xs[thr_indices]
        thr_anchor_mask = anchor_valid_mask[thr_indices]

        pred_xs = thr_anchor_xs + thr_reg
        pred_valid_mask = (pred_xs >= 0) & (pred_xs <= (img_w - 1))
        final_mask = thr_anchor_mask & pred_valid_mask.int()

        scores_np = thr_scores.detach().cpu().numpy()
        pred_xs_np = pred_xs.detach().cpu().numpy()
        final_mask_np = final_mask.detach().cpu().numpy()

        candidates = []
        valid_lengths = []
        for k in range(len(thr_indices)):
            valid_len = int(final_mask_np[k].sum())
            valid_lengths.append(valid_len)
            if valid_len < 2:
                continue
            candidates.append(
                {
                    "score": float(scores_np[k]),
                    "x_list": pred_xs_np[k],
                    "valid_mask": final_mask_np[k].astype(np.uint8),
                    "y_samples": y_samples,
                    "length": valid_len,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        pre_nms_count = len(candidates)

        if decoder.nms_thr is not None and decoder.nms_thr > 0:
            kept = []
            remaining = list(candidates)
            while remaining:
                best = remaining.pop(0)
                kept.append(best)
                remaining = [lane for lane in remaining if not decoder._is_duplicate_lane(best, lane)]
        else:
            kept = candidates

        batch_records.append(
            {
                "all_scores": cur_scores.detach().cpu().numpy().tolist(),
                "thr_scores": scores_np.tolist(),
                "thr_count": int(len(thr_indices)),
                "valid_candidate_count": int(pre_nms_count),
                "post_nms_count": int(len(kept)),
                "valid_lengths": valid_lengths,
            }
        )

    return batch_records


def main():
    parser = argparse.ArgumentParser(description="Diagnose lane-det checkpoints without retraining")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--score_thr", type=float, default=None, help="Override decode score threshold")
    parser.add_argument("--nms_thr", type=float, default=None, help="Override decode nms threshold")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples, 0 means all")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save diagnosis outputs")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    diag_cfg = copy.deepcopy(cfg)
    diag_cfg["dataset"]["list_file"] = resolve_list_file(cfg, args.split)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = TuSimpleDataset(diag_cfg, split=args.split)
    if args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaneDetector(diag_cfg).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    eval_cfg = cfg.get("eval", {})
    decoder = LaneDecoder(
        score_thr=float(eval_cfg.get("score_thr", 0.5) if args.score_thr is None else args.score_thr),
        nms_thr=float(eval_cfg.get("nms_thr", 30.0) if args.nms_thr is None else args.nms_thr),
        use_polyfit=not bool(eval_cfg.get("disable_polyfit", False)),
        nms_min_common_points=int(eval_cfg.get("nms_min_common_points", 8)),
        nms_overlap_ratio_thr=float(eval_cfg.get("nms_overlap_ratio_thr", 0.6)),
        nms_top_dist_ratio=float(eval_cfg.get("nms_top_dist_ratio", 1.25)),
        nms_bottom_dist_ratio=float(eval_cfg.get("nms_bottom_dist_ratio", 1.0)),
    )

    gt_lane_total = 0
    gt_zero_pos_total = 0
    gt_pos_counts = []
    per_image_records = []
    thr_counts = []
    valid_candidate_counts = []
    post_nms_counts = []
    all_score_values = []
    thr_score_values = []
    valid_lengths_all = []
    delta_abs_all = []
    delta_abs_scored = []
    stage1_abs_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Diagnosing"):
            if batch is None:
                continue

            images = batch["images"].to(device)
            anchors = batch["anchors"]
            metas = batch["metas"]
            cls_targets = batch["cls_targets"]
            match_stats_list = batch["match_stats"]

            cls_logits, reg_preds, aux = model(images, anchors, return_aux=True)
            img_h, img_w = images.shape[2], images.shape[3]

            batch_decode = decode_candidates(decoder, cls_logits, reg_preds, anchors, img_w, img_h)

            reg_stage1 = aux["reg_stage1"]
            reg_delta_stage2 = aux["reg_delta_stage2"]
            score_mask = torch.sigmoid(cls_logits) > decoder.score_thr

            delta_abs = reg_delta_stage2.abs()
            stage1_abs = reg_stage1.abs()
            delta_abs_all.extend(delta_abs.detach().cpu().reshape(-1).tolist())
            stage1_abs_all.extend(stage1_abs.detach().cpu().reshape(-1).tolist())

            if score_mask.any():
                selected = delta_abs[score_mask.unsqueeze(-1).expand_as(delta_abs)]
                delta_abs_scored.extend(selected.detach().cpu().reshape(-1).tolist())

            for i, rec in enumerate(batch_decode):
                meta = metas[i]
                thr_counts.append(rec["thr_count"])
                valid_candidate_counts.append(rec["valid_candidate_count"])
                post_nms_counts.append(rec["post_nms_count"])
                valid_lengths_all.extend(rec["valid_lengths"])
                all_score_values.extend(rec["all_scores"])
                thr_score_values.extend(rec["thr_scores"])

                image_record = {
                    "raw_file": meta.get("raw_file", ""),
                    "score_thr_pass_count": rec["thr_count"],
                    "valid_candidate_count": rec["valid_candidate_count"],
                    "post_nms_count": rec["post_nms_count"],
                }

                match_stats = match_stats_list[i]
                if match_stats is not None:
                    pos_counts = [int(v) for v in match_stats.get("per_gt_pos_count", [])]
                    gt_lane_total += len(pos_counts)
                    gt_zero_pos_total += sum(1 for v in pos_counts if v == 0)
                    gt_pos_counts.extend(pos_counts)
                    image_record["gt_lane_count"] = len(pos_counts)
                    image_record["gt_zero_pos_count"] = sum(1 for v in pos_counts if v == 0)

                per_image_records.append(image_record)

    summary = {
        "config": {
            "cfg": args.cfg,
            "ckpt": args.ckpt,
            "split": args.split,
            "dataset_size": len(dataset),
            "score_thr": decoder.score_thr,
            "nms_thr": decoder.nms_thr,
        },
        "gt_positive_coverage": {
            "gt_lane_total": int(gt_lane_total),
            "gt_zero_positive_total": int(gt_zero_pos_total),
            "gt_zero_positive_rate": float(gt_zero_pos_total / gt_lane_total) if gt_lane_total > 0 else None,
            "per_gt_positive_count": summarize_array(gt_pos_counts),
        },
        "decode_counts": {
            "score_thr_pass_count": summarize_array(thr_counts),
            "valid_candidate_count_before_nms": summarize_array(valid_candidate_counts),
            "post_nms_count": summarize_array(post_nms_counts),
            "valid_length_before_nms": summarize_array(valid_lengths_all),
        },
        "score_distribution": {
            "all_scores": summarize_array(all_score_values),
            "all_scores_hist": build_score_hist(all_score_values),
            "thr_scores": summarize_array(thr_score_values),
            "thr_scores_hist": build_score_hist(thr_score_values),
        },
        "stage2_delta": {
            "abs_delta_all_points": summarize_array(delta_abs_all),
            "abs_delta_points_on_scored_anchors": summarize_array(delta_abs_scored),
            "abs_stage1_all_points": summarize_array(stage1_abs_all),
        },
    }

    summary_path = os.path.join(args.out_dir, "diagnosis_summary.json")
    per_image_path = os.path.join(args.out_dir, "diagnosis_per_image.jsonl")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(per_image_path, "w", encoding="utf-8") as f:
        for item in per_image_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved per-image records to: {per_image_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
