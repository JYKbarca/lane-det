import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file):
    steps = []
    total_loss = []
    cls_loss = []
    reg_loss = []
    
    # Regex to match log line
    # Example: Epoch [1/50], Step [10/816], Loss: 21.5384 (Cls: 0.0550, Reg: 21.4833)
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Step \[(\d+)/\d+\], Loss: ([-\d.]+) \(Cls: ([-\d.]+), Reg: ([-\d.]+)\)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                cls = float(match.group(4))
                reg = float(match.group(5))
                
                # Global step calculation depends on steps per epoch
                # We can just use simple index or calculate global step if we know steps per epoch
                # Here we just store the raw values
                steps.append(len(steps)) 
                total_loss.append(loss)
                cls_loss.append(cls)
                reg_loss.append(reg)
                
    return steps, total_loss, cls_loss, reg_loss

def plot_loss(steps, total, cls, reg, output_path):
    plt.figure(figsize=(12, 6))
    
    # Plot Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, total, label='Total Loss', color='blue', alpha=0.7)
    plt.title('Total Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Cls and Reg Loss
    plt.subplot(1, 2, 2)
    plt.plot(steps, cls, label='Cls Loss', color='red', alpha=0.7)
    plt.plot(steps, reg, label='Reg Loss', color='green', alpha=0.7)
    plt.title('Cls & Reg Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Loss plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Loss Curve from Log File")
    parser.add_argument("--log", type=str, required=True, help="Path to train.log file")
    parser.add_argument("--out", type=str, default="/root/autodl-tmp/outputs/lane-det/visualizations/loss_curve.png", help="Path to save plot image")
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Log file not found at {args.log}")
        return
        
    steps, total, cls, reg = parse_log(args.log)
    
    if not steps:
        print("No loss data found in log file.")
        return
        
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_loss(steps, total, cls, reg, args.out)

if __name__ == "__main__":
    main()
