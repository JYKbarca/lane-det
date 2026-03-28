import argparse
import os
import re

import matplotlib.pyplot as plt


TRAIN_AVG_PATTERN = re.compile(
    r"Epoch \[(\d+)/\d+\] Finished\..*Avg Loss: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
VAL_LOSS_PATTERN = re.compile(
    r"Epoch \[(\d+)/\d+\] Val Loss: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_log(log_file):
    train_loss_by_epoch = {}
    val_loss_by_epoch = {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            train_match = TRAIN_AVG_PATTERN.search(line)
            if train_match:
                train_loss_by_epoch[int(train_match.group(1))] = float(train_match.group(2))

            val_match = VAL_LOSS_PATTERN.search(line)
            if val_match:
                val_loss_by_epoch[int(val_match.group(1))] = float(val_match.group(2))

    return train_loss_by_epoch, val_loss_by_epoch


def plot_loss(train_loss_by_epoch, val_loss_by_epoch, output_path):
    train_epochs = sorted(train_loss_by_epoch)
    val_epochs = sorted(val_loss_by_epoch)

    plt.figure(figsize=(10, 6))
    plt.plot(
        train_epochs,
        [train_loss_by_epoch[epoch] for epoch in train_epochs],
        marker="o",
        linewidth=2,
        label="Train Avg Loss",
        color="tab:blue",
    )
    if val_epochs:
        plt.plot(
            val_epochs,
            [val_loss_by_epoch[epoch] for epoch in val_epochs],
            marker="s",
            linewidth=2,
            label="Val Loss",
            color="tab:orange",
        )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Loss plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot epoch-level loss curves from train.log")
    parser.add_argument("--log", type=str, required=True, help="Path to train.log file")
    parser.add_argument("--out", type=str, default="outputs/visualizations/loss_curve.png", help="Path to save plot image")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: Log file not found at {args.log}")
        return

    train_loss_by_epoch, val_loss_by_epoch = parse_log(args.log)
    if not train_loss_by_epoch:
        print("No epoch-level train loss data found in log file.")
        return
    if not val_loss_by_epoch:
        print("Warning: No val loss data found in log file. Plotting train loss only.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plot_loss(train_loss_by_epoch, val_loss_by_epoch, args.out)


if __name__ == "__main__":
    main()
