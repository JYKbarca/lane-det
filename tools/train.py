import argparse
import copy
import os
import time
import datetime
import logging
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lane_det.datasets.tusimple import TuSimpleDataset
from lane_det.models import LaneDetector
from lane_det.losses import FocalLoss, RegLoss, SoftLineOverlapLoss
from lane_det.anchors import AnchorSet
from lane_det.postprocess import LaneDecoder
from lane_det.metrics import TuSimpleConverter
from evaluate import LaneEval

def setup_logger(output_dir):
    logger = logging.getLogger("LaneDet")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def collate_fn(batch):
    """
    Custom collate function to handle variable size tensors if any.
    """
    # Filter failed samples (None)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    # Stack images: [B, 3, H, W]
    images = []
    for b in batch:
        img = b["image"]
        # img is already [C, H, W] numpy float32 array from transforms
        img = torch.from_numpy(img).float()
        images.append(img)
        
    images = torch.stack(images)
    
    # Stack labels
    # cls_label: [B, Num_Anchors]
    cls_labels = torch.stack([torch.from_numpy(b["cls_label"]).float() for b in batch])
    
    # offset_label: [B, Num_Anchors, Num_Y]
    offset_labels = torch.stack([torch.from_numpy(b["offset_label"]).float() for b in batch])
    
    # offset_valid_mask: [B, Num_Anchors, Num_Y]
    offset_masks = torch.stack([torch.from_numpy(b["offset_valid_mask"]).float() for b in batch])
    
    # We need to reconstruct AnchorSet for the batch
    first_sample = batch[0]
    anchors = AnchorSet(
        anchor_xs=first_sample["anchor_xs"],
        valid_mask=first_sample["anchor_valid_mask"],
        x_bottom=None, # Not needed for forward
        angles=None,   # Not needed for forward
        y_samples=first_sample["anchor_y_samples"]
    )
    
    return {
        "images": images,
        "cls_labels": cls_labels,
        "offset_labels": offset_labels,
        "offset_masks": offset_masks,
        "anchors": anchors,
        "metas": [b.get("meta", {}) for b in batch],
    }


def resolve_val_list_file(cfg):
    list_file = cfg.get("dataset", {}).get("list_file", "")
    root = cfg.get("dataset", {}).get("root", "")
    if list_file:
        return os.path.join(os.path.dirname(list_file), "val.json")
    return os.path.join(root, "val.json")


def validate(model, loader, samples, cfg, device):
    model.eval()
    eval_cfg = cfg.get("eval", {})
    decoder = LaneDecoder(
        score_thr=float(eval_cfg.get("score_thr", 0.5)),
        nms_thr=float(eval_cfg.get("nms_thr", 30.0)),
        use_polyfit=not bool(eval_cfg.get("disable_polyfit", False)),
    )
    converter = TuSimpleConverter()
    gt_map = {item["raw_file"]: item for item in samples}

    total_acc = 0.0
    total_fp = 0.0
    total_fn = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            images = batch["images"].to(device)
            anchors = batch["anchors"]
            metas = batch["metas"]

            cls_logits, reg_preds = model(images, anchors)
            img_h, img_w = images.shape[2], images.shape[3]
            decoded_batch = decoder.decode(cls_logits, reg_preds, anchors, img_w, img_h)

            for i, lanes in enumerate(decoded_batch):
                raw_file = metas[i]["raw_file"]
                gt = gt_map.get(raw_file)
                if gt is None:
                    continue

                pred = converter.convert(
                    lanes,
                    raw_file,
                    img_w,
                    img_h,
                    ori_w=1280,
                    ori_h=720,
                    target_h_samples=gt["h_samples"],
                )
                acc, fp, fn = LaneEval.bench(
                    pred["lanes"],
                    gt["lanes"],
                    gt["h_samples"],
                    pred["run_time"],
                )
                total_acc += acc
                total_fp += fp
                total_fn += fn
                count += 1

    if count == 0:
        raise RuntimeError("No validation samples were evaluated.")

    return {
        "Accuracy": total_acc / count,
        "FP": total_fp / count,
        "FN": total_fn / count,
    }


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


def log_pretrain_match_stats(dataset, logger, max_samples=200):
    max_n = min(int(max_samples), len(dataset))
    if max_n <= 0:
        return

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
    total_pos = 0
    total_neg = 0
    total_ignore = 0

    for i in range(max_n):
        sample = dataset[i]
        cls_label = sample.get("cls_label", None)
        if cls_label is None:
            continue
        total_pos += int((cls_label == 1).sum())
        total_neg += int((cls_label == 0).sum())
        total_ignore += int((cls_label < 0).sum())

        match_stats = sample.get("match_stats", None)
        if match_stats is None:
            continue
        per_gt_max_iou_all.extend(match_stats.get("per_gt_max_iou", []))
        per_gt_pos_count_all.extend(match_stats.get("per_gt_pos_count", []))
        rs = match_stats.get("ignore_reason_count", {})
        for k in ignore_reason_total.keys():
            ignore_reason_total[k] += int(rs.get(k, 0))

    logger.info(f"[Step2] Pretrain match stats on first {max_n} samples")
    logger.info(f"[Step2] Anchor label summary: pos={total_pos}, neg={total_neg}, ignore={total_ignore}")
    if len(per_gt_max_iou_all) > 0:
        logger.info(f"[Step2] per-GT max_iou distribution: {summarize_distribution(per_gt_max_iou_all)}")
    if len(per_gt_pos_count_all) > 0:
        logger.info(f"[Step2] per-GT positive-count distribution: {summarize_distribution(per_gt_pos_count_all)}")
    if total_ignore > 0:
        logger.info(
            "[Step2] ignore source counts: "
            f"common_fail={ignore_reason_total['common_fail']}, "
            f"top_fail={ignore_reason_total['top_fail']}, "
            f"angle_fail={ignore_reason_total['angle_fail']}, "
            f"x_bottom_fail={ignore_reason_total['x_bottom_fail']}, "
            f"threshold_gray={ignore_reason_total['threshold_gray']}, "
            f"other={ignore_reason_total['other']}"
        )

def main():
    parser = argparse.ArgumentParser(description="Train Lane Detector")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--work-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--pre-stat-max", type=int, default=200, help="Max samples for pretrain match statistics")
    args = parser.parse_args()
    
    # Load config
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    default_work_dir = cfg.get("paths", {}).get("checkpoint_root", "outputs/checkpoints")
    work_root = args.work_dir or default_work_dir
        
    # Setup logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(work_root, timestamp)
    os.makedirs(work_dir, exist_ok=True)
    logger = setup_logger(work_dir)
    logger.info(f"Config: {cfg}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Dataset
    train_dataset = TuSimpleDataset(cfg, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0, # Windows usually needs 0 workers to avoid multiprocessing issues
        collate_fn=collate_fn,
        drop_last=True
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    log_pretrain_match_stats(train_dataset, logger, max_samples=args.pre_stat_max)

    val_cfg = copy.deepcopy(cfg)
    val_cfg["dataset"]["list_file"] = resolve_val_list_file(cfg)
    val_dataset = TuSimpleDataset(val_cfg, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("test", {}).get("batch_size", 1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False
    )
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Val list file: {val_cfg['dataset']['list_file']}")
    
    # Model
    model = LaneDetector(cfg)
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    
    # Learning Rate Scheduler
    lr_milestones = cfg["train"].get("lr_milestones", [12, 18])
    lr_gamma = cfg["train"].get("lr_gamma", 0.3)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # Gradient Accumulation Steps (Simulate larger batch size)
    accumulation_steps = 2
    
    # Losses
    cls_criterion = FocalLoss(
        alpha=cfg["loss"]["focal_alpha"],
        gamma=cfg["loss"]["focal_gamma"]
    )
    reg_criterion = RegLoss()
    reg_stage1_w = float(cfg.get("loss", {}).get("reg_loss_stage1_weight", 0.5))
    reg_stage2_w = float(cfg.get("loss", {}).get("reg_loss_stage2_weight", 1.0))
    use_refinement = bool(cfg.get("model", {}).get("use_refinement", False))
    
    use_line_loss = bool(cfg.get("loss", {}).get("use_line_overlap_loss", False))
    line_loss_weight = float(cfg.get("loss", {}).get("line_loss_weight", 0.1))
    line_sigma = float(cfg.get("loss", {}).get("line_sigma", 12.0))
    line_sigma_refined = float(cfg.get("loss", {}).get("line_sigma_refined", line_sigma))
    line_sigma_step = int(cfg.get("loss", {}).get("line_sigma_step", 12))
    line_min_points = int(cfg.get("loss", {}).get("line_min_valid_points", 3))
    if use_line_loss:
        line_criterion = SoftLineOverlapLoss(sigma=line_sigma, min_valid_points=line_min_points)
    
    # Resume
    start_epoch = 0
    best_acc = float("-inf")
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = float(checkpoint.get("best_acc", best_acc))
        # Synchronize scheduler if resuming
        for _ in range(start_epoch):
            scheduler.step()
        
    # Train loop
    num_epochs = cfg["train"]["epochs"]
    logger.info("Start training...")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        # Dynamic Sigma Adjustment for Line Loss
        if use_line_loss and epoch >= line_sigma_step:
            if line_criterion.sigma != line_sigma_refined:
                line_criterion.sigma = line_sigma_refined
                logger.info(f"Epoch {epoch+1}: Adjusted line_sigma to {line_sigma_refined} for refinement")
        
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_reg1_loss = 0.0
        epoch_reg2_loss = 0.0
        epoch_line_loss = 0.0
        
        optimizer.zero_grad()
        
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            images = batch["images"].to(device)
            cls_labels = batch["cls_labels"].to(device)
            offset_labels = batch["offset_labels"].to(device)
            offset_masks = batch["offset_masks"].to(device)
            anchors = batch["anchors"] 
            
            # Forward
            if use_refinement:
                cls_logits, reg_preds, aux = model(images, anchors, return_aux=True)
                reg_stage1 = aux["reg_stage1"]
                reg_stage2 = aux["reg_final"]
            else:
                cls_logits, reg_preds = model(images, anchors)
                reg_stage1 = reg_preds
                reg_stage2 = reg_preds
            
            # Loss
            cls_loss = cls_criterion(cls_logits, cls_labels)
            reg1_loss = reg_criterion(reg_stage1, offset_labels, offset_masks)
            reg2_loss = reg_criterion(reg_stage2, offset_labels, offset_masks)
            reg_loss = reg_stage1_w * reg1_loss + reg_stage2_w * reg2_loss
            
            loss = cfg["loss"]["cls_weight"] * cls_loss + cfg["loss"]["reg_weight"] * reg_loss
            
            line_loss_val = 0.0
            if use_line_loss:
                line1_loss = line_criterion(reg_stage1, offset_labels, offset_masks)
                line2_loss = line_criterion(reg_stage2, offset_labels, offset_masks)
                line_loss = reg_stage1_w * line1_loss + reg_stage2_w * line2_loss
                loss += line_loss_weight * line_loss
                line_loss_val = line_loss.item()
            
            (loss / accumulation_steps).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_reg1_loss += reg1_loss.item()
            epoch_reg2_loss += reg2_loss.item()
            if use_line_loss:
                epoch_line_loss += line_loss_val
            
            if i % 10 == 0:
                msg = (
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f} "
                    f"(Cls: {cls_loss.item():.4f}, Reg: {reg_loss.item():.4f}, "
                    f"Reg1: {reg1_loss.item():.4f}, Reg2: {reg2_loss.item():.4f})"
                )
                if use_line_loss:
                    msg += f", Line: {line_loss_val:.4f}"
                logger.info(msg)
                
        end_time = time.time()
        epoch_time = end_time - start_time
        
        avg_loss = epoch_loss / len(train_loader)
        avg_cls = epoch_cls_loss / len(train_loader)
        avg_reg = epoch_reg_loss / len(train_loader)
        avg_reg1 = epoch_reg1_loss / len(train_loader)
        avg_reg2 = epoch_reg2_loss / len(train_loader)
        avg_line = epoch_line_loss / len(train_loader) if use_line_loss else 0.0
        
        msg = (
            f"Epoch [{epoch+1}/{num_epochs}] Finished. Time: {epoch_time:.2f}s. "
            f"Avg Loss: {avg_loss:.4f} "
            f"(Cls: {avg_cls:.4f}, Reg: {avg_reg:.4f}, Reg1: {avg_reg1:.4f}, Reg2: {avg_reg2:.4f})"
        )
        if use_line_loss:
            msg += f", Line: {avg_line:.4f}"
        logger.info(msg)

        val_metrics = validate(model, val_loader, val_dataset.samples, cfg, device)
        logger.info(
            f"Validation: Accuracy={val_metrics['Accuracy']:.6f}, "
            f"FP={val_metrics['FP']:.6f}, "
            f"FN={val_metrics['FN']:.6f}"
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Current LR: {current_lr:.6f}")
        
        # Save checkpoint
        save_path = os.path.join(work_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }, save_path)
        
        # Save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }, os.path.join(work_dir, "last.pth"))

        if val_metrics["Accuracy"] > best_acc:
            best_acc = val_metrics["Accuracy"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "best_acc": best_acc,
            }, os.path.join(work_dir, "best.pth"))
            logger.info(f"Saved best checkpoint with Accuracy={best_acc:.6f}")

if __name__ == "__main__":
    main()
