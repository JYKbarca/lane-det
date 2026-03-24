import argparse
import copy
import datetime
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate import LaneEval
from lane_det.datasets import RowTargetBuilder, TuSimpleDataset
from lane_det.metrics import TuSimpleConverter
from lane_det.models import RowLaneDetector


def setup_logger(output_dir):
    logger = logging.getLogger("RowLaneDet")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def build_optimizer(model, cfg):
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))

    decay_params = []
    no_decay_params = []
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name.endswith("bias") or isinstance(module, norm_types):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return optim.AdamW(param_groups, lr=lr)


def resolve_val_list_file(cfg):
    list_file = cfg.get("dataset", {}).get("list_file", "")
    root = cfg.get("dataset", {}).get("root", "")
    if list_file:
        return os.path.join(os.path.dirname(list_file), "val.json")
    return os.path.join(root, "val.json")


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
            )
            for sample in batch
        ]

        exist_targets = torch.stack(
            [torch.from_numpy(target["exist"]).float() for target in row_targets]
        )
        x_targets = torch.stack(
            [torch.from_numpy(target["x_coords"]).float() for target in row_targets]
        )
        coord_masks = torch.stack(
            [torch.from_numpy(target["coord_mask"]).float() for target in row_targets]
        )
        row_h_samples = torch.stack(
            [torch.from_numpy(target["row_h_samples"]).float() for target in row_targets]
        )

        x_targets_norm = x_targets / max(img_width - 1.0, 1.0)

        return {
            "images": images,
            "exist_targets": exist_targets,
            "x_targets": x_targets,
            "x_targets_norm": x_targets_norm,
            "coord_masks": coord_masks,
            "row_h_samples": row_h_samples,
            "metas": [sample.get("meta", {}) for sample in batch],
        }

    return collate_fn


def masked_smooth_l1_loss(pred, target, mask, beta=1.0):
    if mask.numel() == 0:
        return pred.sum() * 0.0

    diff = torch.abs(pred - target)
    beta = float(beta)
    if beta < 1e-6:
        loss = diff
    else:
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    weighted = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom


def decode_row_predictions(exist_logits, valid_logits, x_coords_norm, row_h_samples, score_thr, img_w, valid_thr):
    exist_scores = torch.sigmoid(exist_logits)
    valid_scores = torch.sigmoid(valid_logits)
    lanes_batch = []
    x_coords = x_coords_norm * max(float(img_w) - 1.0, 1.0)

    for scores_b, valid_b, coords_b, ys_b in zip(exist_scores, valid_scores, x_coords, row_h_samples):
        lane_dicts = []
        for lane_score, lane_valid, lane_xs in zip(scores_b, valid_b, coords_b):
            if float(lane_score) < score_thr:
                continue
            valid_mask = (lane_valid >= valid_thr).to(torch.uint8)
            if int(valid_mask.sum()) < 2:
                continue
            lane_dicts.append(
                {
                    "x_list": lane_xs.detach().cpu().numpy(),
                    "y_samples": ys_b.detach().cpu().numpy(),
                    "valid_mask": valid_mask.detach().cpu().numpy(),
                }
            )
        lanes_batch.append(lane_dicts)

    return lanes_batch


def validate(model, loader, samples, cfg, device):
    model.eval()
    score_thr = float(cfg.get("eval", {}).get("exist_score_thr", 0.5))
    valid_thr = float(cfg.get("eval", {}).get("valid_thr", 0.5))
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
            row_h_samples = batch["row_h_samples"].to(device)
            metas = batch["metas"]

            exist_logits, valid_logits, x_coords_norm = model(images)
            decoded_batch = decode_row_predictions(
                exist_logits,
                valid_logits,
                x_coords_norm,
                row_h_samples,
                score_thr,
                images.shape[-1],
                valid_thr,
            )
            img_h, img_w = images.shape[-2:]

            for lane_dicts, meta in zip(decoded_batch, metas):
                raw_file = meta["raw_file"]
                gt = gt_map.get(raw_file)
                if gt is None:
                    continue
                pred = converter.convert(
                    lane_dicts,
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


def main():
    parser = argparse.ArgumentParser(description="Train Row-Based Lane Detector")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--work-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    row_cfg = cfg.setdefault("row", {})
    row_cfg.setdefault("max_lanes", 5)
    row_cfg.setdefault("num_y", int(cfg["dataset"]["y_samples"]))

    default_work_dir = cfg.get("paths", {}).get("checkpoint_root", "outputs/checkpoints")
    work_root = args.work_dir or default_work_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(work_root, timestamp)
    os.makedirs(work_dir, exist_ok=True)

    logger = setup_logger(work_dir)
    logger.info(f"Config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    target_builder = RowTargetBuilder(
        num_lanes=row_cfg["max_lanes"],
        num_y=row_cfg["num_y"],
    )
    collate_fn = build_collate_fn(target_builder)

    train_dataset = TuSimpleDataset(cfg, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")

    val_cfg = copy.deepcopy(cfg)
    val_cfg["dataset"]["list_file"] = resolve_val_list_file(cfg)
    val_dataset = TuSimpleDataset(val_cfg, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("test", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Val list file: {val_cfg['dataset']['list_file']}")

    model = RowLaneDetector(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = MultiStepLR(
        optimizer,
        milestones=cfg["train"].get("lr_milestones", [12, 18]),
        gamma=cfg["train"].get("lr_gamma", 0.3),
    )

    exist_criterion = nn.BCEWithLogitsLoss()
    coord_beta = float(cfg.get("loss", {}).get("smooth_l1_beta", 1.0))
    exist_weight = float(cfg.get("loss", {}).get("exist_weight", 1.0))
    coord_weight = float(cfg.get("loss", {}).get("coord_weight", 1.0))
    valid_weight = float(cfg.get("loss", {}).get("valid_weight", 1.0))

    start_epoch = 0
    best_acc = float("-inf")
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", best_acc))
        for _ in range(start_epoch):
            scheduler.step()

    num_epochs = int(cfg["train"]["epochs"])
    logger.info("Start training...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_exist_loss = 0.0
        epoch_coord_loss = 0.0
        epoch_valid_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            images = batch["images"].to(device)
            exist_targets = batch["exist_targets"].to(device)
            x_targets_norm = batch["x_targets_norm"].to(device)
            coord_masks = batch["coord_masks"].to(device)

            optimizer.zero_grad()
            exist_logits, valid_logits, x_coords_norm = model(images)

            lane_masks = coord_masks * exist_targets.unsqueeze(-1)
            exist_loss = exist_criterion(exist_logits, exist_targets)
            
            valid_loss_unreduced = nn.functional.binary_cross_entropy_with_logits(valid_logits, coord_masks, reduction='none')
            valid_loss = (valid_loss_unreduced * exist_targets.unsqueeze(-1)).sum() / (exist_targets.sum().clamp(min=1.0) * valid_logits.shape[-1])
            
            coord_loss = masked_smooth_l1_loss(x_coords_norm, x_targets_norm, lane_masks, beta=coord_beta)
            
            # Add L1 regularization for lane_queries to encourage differentiation
            query_reg_loss = 0.01 * torch.mean(torch.abs(model.head.lane_queries))
            
            loss = exist_weight * exist_loss + coord_weight * coord_loss + valid_weight * valid_loss + query_reg_loss

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_exist_loss += float(exist_loss.item())
            epoch_coord_loss += float(coord_loss.item())
            epoch_valid_loss += float(valid_loss.item())

            if step % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f} "
                    f"(Exist: {exist_loss.item():.4f}, Coord: {coord_loss.item():.4f}, Valid: {valid_loss.item():.4f}, QueryReg: {query_reg_loss.item():.4f})"
                )

        elapsed = time.time() - start_time
        denom = max(len(train_loader), 1)
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] Finished. Time: {elapsed:.2f}s. "
            f"Avg Loss: {epoch_loss / denom:.4f} "
            f"(Exist: {epoch_exist_loss / denom:.4f}, Coord: {epoch_coord_loss / denom:.4f}, Valid: {epoch_valid_loss / denom:.4f})"
        )

        val_metrics = validate(model, val_loader, val_dataset.samples, cfg, device)
        logger.info(
            f"Validation: Accuracy={val_metrics['Accuracy']:.6f}, "
            f"FP={val_metrics['FP']:.6f}, "
            f"FN={val_metrics['FN']:.6f}"
        )

        scheduler.step()
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "best_acc": best_acc,
        }
        torch.save(checkpoint, os.path.join(work_dir, f"epoch_{epoch+1}.pth"))
        torch.save(checkpoint, os.path.join(work_dir, "last.pth"))

        if val_metrics["Accuracy"] > best_acc:
            best_acc = val_metrics["Accuracy"]
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, os.path.join(work_dir, "best.pth"))
            logger.info(f"Saved best checkpoint with Accuracy={best_acc:.6f}")


if __name__ == "__main__":
    main()
