import argparse
import json
import cv2
import os
import yaml
import numpy as np
from tqdm import tqdm

def draw_lanes(image, lanes, h_samples, color=(0, 255, 0), thickness=2):
    for lane in lanes:
        points = []
        for i, x in enumerate(lane):
            if x > 0: # TuSimple uses -2 for invalid
                y = h_samples[i]
                points.append((int(x), int(y)))
        
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i+1], color, thickness)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="pred.json", help="Path to prediction json")
    parser.add_argument("--cfg", type=str, help="Path to config file (to get dataset root)")
    parser.add_argument("--root", type=str, help="Dataset root directory (overrides config)")
    parser.add_argument("--out", type=str, default="outputs/visualizations/pred", help="Output directory")
    parser.add_argument("--num", type=int, default=20, help="Number of images to visualize")
    args = parser.parse_args()
    
    root = args.root
    if not root and args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            root = cfg["dataset"]["root"]
            
    if not root:
        print("Error: Dataset root not specified. Use --root or --cfg.")
        return

    if not os.path.exists(args.pred):
        print(f"Error: Prediction file not found: {args.pred}")
        return

    os.makedirs(args.out, exist_ok=True)
    
    print(f"Loading predictions from {args.pred}...")
    with open(args.pred, 'r') as f:
        lines = f.readlines()
        
    print(f"Visualizing {args.num} images...")
    
    count = 0
    for line in tqdm(lines):
        if count >= args.num:
            break
            
        data = json.loads(line)
        raw_file = data['raw_file']
        lanes = data['lanes']
        h_samples = data['h_samples']
        
        img_path = os.path.join(root, raw_file)
        if not os.path.exists(img_path):
            # Try removing leading slash if present
            if raw_file.startswith('/'):
                 img_path = os.path.join(root, raw_file[1:])
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Draw
        img = draw_lanes(img, lanes, h_samples)
                
        # Save
        # Flatten path for filename
        save_name = raw_file.replace('/', '_').replace('\\', '_')
        if save_name.startswith('_'):
            save_name = save_name[1:]
            
        cv2.imwrite(os.path.join(args.out, save_name), img)
        count += 1
        
    print(f"Visualized {count} images to {args.out}")

if __name__ == "__main__":
    main()
