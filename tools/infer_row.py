import argparse
import copy
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lane_det.datasets import RowTargetBuilder, TuSimpleDataset
from lane_det.metrics import TuSimpleConverter
from lane_det.models import RowLaneDetector
from tools.train_row import decode_row_predictions


def resolve_list_file(cfg, split):
    dataset_cfg = cfg.get("dataset", {})
    list_file = dataset_cfg.get("list_file", "")
    root = dataset_cfg.get("root", "")

    if split == "train":
        return os.path.join(os.path.dirname(list_file), "train.json") if list_file else os.path.join(root, "train.json")
    if split == "val":
        return os.path.join(os.path.dirname(list_file), "val.json") if list_file else os.path.join(root, "val.json")
    if list_file and os.path.basename(list_file) == "test_label.json":
        return list_file
    if root and os.path.basename(root.rstrip("/\\")) == "test_set":
        return os.path.join(os.path.dirname(root), "test_label.json")
    return os.path.join(root, "test.json")


def build_collate_fn(target_builder):
    def collate_fn(batch):
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return None

        images = torch.stack([torch.from_numpy(sample["image"]).float() for sample in batch])
        img_width = float(images.shape[-1])
        row_targets = [
            target_builder.build(
                sample["lanes"],
                sample["valid_mask"],
                sample["meta"].get("h_samples", []),
                img_width
            )
            for sample in batch
        ]
        row_h_samples = torch.stack(
            [torch.from_numpy(target["row_h_samples"]).float() for target in row_targets]
        )

        return {
            "images": images,
            "row_h_samples": row_h_samples,
            "metas": [sample["meta"] for sample in batch],
        }

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Infer Row-Based Lane Detector")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--out", type=str, default=None, help="Output json path")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=None, help="Dataset split")
    parser.add_argument("--exist_score_thr", type=float, default=None, help="Existence threshold override")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    row_cfg = cfg.setdefault("row", {})
    row_cfg.setdefault("max_lanes", 5)
    row_cfg.setdefault("num_y", int(cfg["dataset"]["y_samples"]))
    row_cfg.setdefault("num_grids", 100)
    score_thr = (
        float(args.exist_score_thr)
        if args.exist_score_thr is not None
        else float(cfg.get("eval", {}).get("exist_score_thr", 0.5))
    )

    split = args.split
    if split is None:
        split = "test" if os.path.basename(cfg["dataset"].get("list_file", "")) == "test_label.json" else "val"

    infer_cfg = copy.deepcopy(cfg)
    infer_cfg["dataset"]["list_file"] = resolve_list_file(cfg, split)
    out_path = args.out or os.path.join(
        cfg.get("paths", {}).get("output_root", "outputs"),
        "pred",
        f"row_{split}.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_builder = RowTargetBuilder(
        num_lanes=row_cfg["max_lanes"],
        num_y=row_cfg["num_y"],
        num_grids=row_cfg["num_grids"],
    )
    dataset = TuSimpleDataset(infer_cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(infer_cfg.get("test", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=build_collate_fn(target_builder),
    )
    print(f"Running split: {split}")
    print(f"List file: {infer_cfg['dataset']['list_file']}")
    print(f"Dataset size: {len(dataset)}")

    model = RowLaneDetector(infer_cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    converter = TuSimpleConverter()
    results = []

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            images = batch["images"].to(device)
            row_h_samples = batch["row_h_samples"].to(device)
            metas = batch["metas"]

            exist_logits, grid_logits = model(images)
            decoded_batch = decode_row_predictions(
                exist_logits,
                grid_logits,
                row_h_samples,
                score_thr,
                images.shape[-1],
                model.head.num_grids,
            )

            img_h, img_w = images.shape[-2:]
            for lanes, meta in zip(decoded_batch, metas):
                results.append(
                    converter.convert(
                        lanes,
                        meta["raw_file"],
                        img_w,
                        img_h,
                        ori_w=1280,
                        ori_h=720,
                        target_h_samples=meta.get("original_h_samples"),
                    )
                )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    TuSimpleConverter.save_json(results, out_path)
    print(f"Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
