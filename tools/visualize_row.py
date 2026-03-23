import argparse
import json
import os

import cv2
import yaml
from tqdm import tqdm


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def draw_lanes(image, lanes, h_samples, color, thickness=2):
    for lane in lanes:
        pts = []
        for x, y in zip(lane, h_samples):
            if x < 0:
                continue
            pts.append((int(round(x)), int(round(y))))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(image, p0, p1, color, thickness)
    return image


def main():
    parser = argparse.ArgumentParser(description="Visualize row-based predictions against GT")
    parser.add_argument("--pred", required=True, help="Prediction jsonl path")
    parser.add_argument("--cfg", type=str, help="Config file for dataset root")
    parser.add_argument("--root", type=str, help="Dataset root override")
    parser.add_argument("--gt", type=str, default=None, help="Ground-truth jsonl path")
    parser.add_argument("--out", default="outputs/visualizations/row", help="Output directory")
    parser.add_argument("--num", type=int, default=20, help="Max images")
    args = parser.parse_args()

    root = args.root
    if not root and args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = cfg["dataset"]["root"]
    if not root:
        raise ValueError("Dataset root not specified. Use --root or --cfg.")

    gt_map = {}
    if args.gt:
        gt_items = load_jsonl(args.gt)
        gt_map = {item["raw_file"]: item for item in gt_items}

    pred_items = load_jsonl(args.pred)
    os.makedirs(args.out, exist_ok=True)

    saved = 0
    for item in tqdm(pred_items):
        if saved >= args.num:
            break

        raw_file = item["raw_file"]
        img_path = os.path.join(root, raw_file)
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        vis = image.copy()
        if raw_file in gt_map:
            gt = gt_map[raw_file]
            draw_lanes(vis, gt["lanes"], gt["h_samples"], color=(0, 255, 0), thickness=2)

        draw_lanes(vis, item["lanes"], item["h_samples"], color=(0, 0, 255), thickness=2)
        cv2.putText(vis, "GT: green  Pred: red", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        save_name = raw_file.replace("/", "_").replace("\\", "_")
        cv2.imwrite(os.path.join(args.out, save_name), vis)
        saved += 1

    print(f"Saved {saved} visualizations to {args.out}")


if __name__ == "__main__":
    main()
