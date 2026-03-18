import argparse
import copy
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lane_det.datasets.tusimple import TuSimpleDataset
from lane_det.models import LaneDetector
from lane_det.anchors import AnchorSet
from lane_det.postprocess import LaneDecoder
from lane_det.metrics import TuSimpleConverter


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
    if len(batch) == 0:
        return None
    
    images = torch.stack([torch.from_numpy(b["image"]).float() for b in batch])
    
    # Reconstruct AnchorSet (assuming same for all)
    first = batch[0]
    anchors = AnchorSet(
        anchor_xs=first["anchor_xs"],
        valid_mask=first["anchor_valid_mask"],
        x_bottom=None,
        angles=None,
        y_samples=first["anchor_y_samples"]
    )
    
    # Meta info
    metas = [b["meta"] for b in batch]
    
    return {
        "images": images,
        "anchors": anchors,
        "metas": metas
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--out", type=str, default=None, help="Path to save output json")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=None, help="Dataset split to run")
    parser.add_argument("--score_thr", type=float, default=0.5, help="Score threshold for decoding")
    parser.add_argument("--nms_thr", type=float, default=30.0, help="Lane NMS distance threshold in pixels; <= 0 disables NMS")
    parser.add_argument("--disable-polyfit", action="store_true", help="Disable polynomial smoothing in decoder")
    args = parser.parse_args()
    
    # Load config
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    default_out = os.path.join(cfg.get("paths", {}).get("output_root", "outputs"), "pred", "pred.json")
    out_path = args.out or default_out

    infer_cfg = copy.deepcopy(cfg)
    split = args.split
    if split is None:
        split = "test" if os.path.basename(infer_cfg["dataset"].get("list_file", "")) == "test_label.json" else "val"
    infer_cfg["dataset"]["list_file"] = resolve_list_file(cfg, split)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = TuSimpleDataset(infer_cfg, split=split)
    
    batch_size = infer_cfg.get("test", {}).get("batch_size", 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    print(f"Running split: {split}")
    print(f"List file: {infer_cfg['dataset']['list_file']}")
    print(f"Dataset size: {len(dataset)}")
    
    # Model
    model = LaneDetector(infer_cfg)
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Decoder & Converter
    decoder = LaneDecoder(
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        use_polyfit=not args.disable_polyfit,
    )
    converter = TuSimpleConverter() # Default h_samples
    
    results = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue
                
            images = batch["images"].to(device)
            # anchors need to be moved to device? 
            # AnchorSet usually holds numpy arrays, but LaneDecoder converts them.
            # However, model forward might expect tensors if it uses them.
            # In LaneDetector forward:
            # cls, reg = self.head(features, anchors, img_h, img_w)
            # AnchorHead.pool_anchors converts anchors to tensor.
            # So passing AnchorSet object is fine.
            
            anchors = batch["anchors"]
            metas = batch["metas"]
            
            # Forward
            cls_logits, reg_preds = model(images, anchors)
            
            # Decode
            img_h, img_w = images.shape[2], images.shape[3]
            decoded_batch = decoder.decode(cls_logits, reg_preds, anchors, img_w, img_h)
            
            # Convert
            for i, lanes in enumerate(decoded_batch):
                meta = metas[i]
                raw_file = meta["raw_file"]
                
                # TuSimple format
                res = converter.convert(
                    lanes,
                    raw_file,
                    img_w,
                    img_h,
                    ori_w=1280,
                    ori_h=720,
                    target_h_samples=meta["h_samples"],
                )
                results.append(res)
                
    # Save
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Saving results to {out_path}...")
    TuSimpleConverter.save_json(results, out_path)
    print("Done.")

if __name__ == "__main__":
    main()
