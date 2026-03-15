import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge TuSimple label_data_*.json and split train/val.")
    parser.add_argument("--root", default="data/tusimple", help="TuSimple root that contains label_data_*.json")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("label_data_*.json"))
    if not files:
        raise FileNotFoundError(f"No label_data_*.json found under: {root}")

    all_records = []
    for fp in files:
        all_records.extend(read_jsonl(fp))

    if len(all_records) < 2:
        raise RuntimeError("Not enough records to split train/val.")

    rnd = random.Random(args.seed)
    rnd.shuffle(all_records)

    val_count = max(1, int(len(all_records) * args.val_ratio))
    val_records = all_records[:val_count]
    train_records = all_records[val_count:]

    train_path = root / "train.json"
    val_path = root / "val.json"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print(f"Source files: {len(files)}")
    print(f"Total records: {len(all_records)}")
    print(f"Train records: {len(train_records)} -> {train_path}")
    print(f"Val records:   {len(val_records)} -> {val_path}")


if __name__ == "__main__":
    main()
