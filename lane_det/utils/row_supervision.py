import torch
import torch.nn.functional as F


def compute_expected_x(grid_logits, num_grids):
    coord_prob = torch.softmax(grid_logits, dim=-1)
    grid_indices = torch.arange(num_grids, device=grid_logits.device, dtype=grid_logits.dtype)
    expected_grid = (coord_prob * grid_indices).sum(dim=-1)
    return expected_grid / max(float(num_grids - 1), 1.0)


def compute_soft_grid_loss(grid_logits, x_targets_norm, coord_masks, num_grids):
    valid_mask = coord_masks > 0
    if not torch.any(valid_mask):
        return grid_logits.sum() * 0.0

    target_grid = x_targets_norm.clamp(0.0, 1.0) * max(float(num_grids - 1), 1.0)
    left_idx = torch.floor(target_grid).long().clamp_(0, num_grids - 1)
    right_idx = (left_idx + 1).clamp_max(num_grids - 1)
    right_weight = target_grid - left_idx.to(target_grid.dtype)
    left_weight = 1.0 - right_weight

    log_prob = F.log_softmax(grid_logits, dim=-1)
    left_log_prob = torch.gather(log_prob, dim=-1, index=left_idx.unsqueeze(-1)).squeeze(-1)
    right_log_prob = torch.gather(log_prob, dim=-1, index=right_idx.unsqueeze(-1)).squeeze(-1)

    loss = -(left_weight * left_log_prob + right_weight * right_log_prob)
    return loss[valid_mask].mean()


def compute_coord_loss(pred_x, target_x, coord_masks, num_grids):
    valid_mask = coord_masks > 0
    if not torch.any(valid_mask):
        return pred_x.sum() * 0.0

    beta = 1.0 / max(float(num_grids - 1), 1.0)
    diff = torch.abs(pred_x - target_x)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss[valid_mask].mean()
