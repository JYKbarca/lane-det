import itertools

import torch


def compute_match_cost_matrix(
    exist_logits,
    row_valid_logits,
    expected_x_norm,
    gt_exist,
    gt_x,
    gt_mask,
    exist_weight=0.2,
    row_valid_weight=1.0,
    coord_weight=2.0,
):
    gt_indices = torch.nonzero(gt_exist > 0.5, as_tuple=False).flatten()
    num_queries = int(exist_logits.shape[0])
    if gt_indices.numel() == 0:
        return gt_indices, expected_x_norm.new_zeros((num_queries, 0))

    exist_prob = torch.sigmoid(exist_logits.detach())
    row_valid_prob = torch.sigmoid(row_valid_logits.detach())
    pred_x = expected_x_norm.detach()

    cost_cols = []
    for gt_idx in gt_indices.tolist():
        gt_mask_lane = gt_mask[gt_idx]
        gt_x_lane = gt_x[gt_idx]
        valid_cost = torch.abs(row_valid_prob - gt_mask_lane.unsqueeze(0)).mean(dim=-1)

        valid_rows = gt_mask_lane > 0.5
        if torch.any(valid_rows):
            coord_cost = torch.abs(pred_x[:, valid_rows] - gt_x_lane[valid_rows].unsqueeze(0)).mean(dim=-1)
        else:
            coord_cost = pred_x.new_zeros((num_queries,))

        exist_cost = 1.0 - exist_prob
        total_cost = (
            exist_weight * exist_cost
            + row_valid_weight * valid_cost
            + coord_weight * coord_cost
        )
        cost_cols.append(total_cost)

    return gt_indices, torch.stack(cost_cols, dim=1)


def solve_lane_assignment(cost_matrix):
    num_queries, num_gt = cost_matrix.shape
    if num_gt == 0:
        return []

    if num_gt > num_queries:
        raise ValueError("Number of GT lanes cannot exceed the number of queries.")

    cost_cpu = cost_matrix.detach().cpu()
    best_cost = float("inf")
    best_assignment = None
    for pred_indices in itertools.permutations(range(num_queries), num_gt):
        total = 0.0
        for gt_col, pred_idx in enumerate(pred_indices):
            total += float(cost_cpu[pred_idx, gt_col])
        if total < best_cost:
            best_cost = total
            best_assignment = [(pred_idx, gt_col) for gt_col, pred_idx in enumerate(pred_indices)]

    return best_assignment or []


def build_matched_targets(
    exist_logits,
    row_valid_logits,
    expected_x_norm,
    exist_targets,
    x_targets_norm,
    coord_masks,
    exist_cost_weight=0.2,
    row_valid_cost_weight=1.0,
    coord_cost_weight=2.0,
):
    matched_exist = torch.zeros_like(exist_targets)
    matched_x = torch.zeros_like(x_targets_norm)
    matched_mask = torch.zeros_like(coord_masks)

    batch_size = int(exist_targets.shape[0])
    assignments = []
    for batch_idx in range(batch_size):
        gt_indices, cost_matrix = compute_match_cost_matrix(
            exist_logits[batch_idx],
            row_valid_logits[batch_idx],
            expected_x_norm[batch_idx],
            exist_targets[batch_idx],
            x_targets_norm[batch_idx],
            coord_masks[batch_idx],
            exist_weight=exist_cost_weight,
            row_valid_weight=row_valid_cost_weight,
            coord_weight=coord_cost_weight,
        )
        assignment = solve_lane_assignment(cost_matrix)
        assignment_pairs = []
        for pred_idx, gt_col in assignment:
            gt_idx = int(gt_indices[gt_col])
            matched_exist[batch_idx, pred_idx] = 1.0
            matched_x[batch_idx, pred_idx] = x_targets_norm[batch_idx, gt_idx]
            matched_mask[batch_idx, pred_idx] = coord_masks[batch_idx, gt_idx]
            assignment_pairs.append((int(pred_idx), gt_idx))
        assignments.append(assignment_pairs)

    return matched_exist, matched_x, matched_mask, assignments
