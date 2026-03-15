import torch
import torch.nn as nn

class SoftLineOverlapLoss(nn.Module):
    def __init__(self, sigma=12.0, min_valid_points=3):
        """
        Soft Line Overlap Loss
        :param sigma: Standard deviation for the Gaussian kernel to convert distance to similarity.
        :param min_valid_points: The minimum number of valid points for a lane to be considered in the loss.
        """
        super(SoftLineOverlapLoss, self).__init__()
        self.sigma = sigma
        self.min_valid_points = min_valid_points

    def forward(self, pred_offsets, target_offsets, valid_mask):
        """
        :param pred_offsets: [B, Num_Anchors, Num_Y] or [N, Num_Y]
        :param target_offsets: [B, Num_Anchors, Num_Y] or [N, Num_Y]
        :param valid_mask: [B, Num_Anchors, Num_Y] or [N, Num_Y]
        :return: soft line overlap loss scalar
        """
        # Flatten to [N, Num_Y]
        if pred_offsets.dim() == 3:
            pred_offsets = pred_offsets.view(-1, pred_offsets.shape[-1])
            target_offsets = target_offsets.view(-1, target_offsets.shape[-1])
            valid_mask = valid_mask.view(-1, valid_mask.shape[-1])

        mask = valid_mask.float()
        
        # Calculate squared distance
        d_sq = torch.pow(pred_offsets - target_offsets, 2)
        
        # Calculate soft similarity S [N, Num_Y]
        s = torch.exp(-d_sq / (2.0 * self.sigma ** 2))
        
        # Sum similarities per lane and count valid points
        s_sum_per_lane = torch.sum(s * mask, dim=1)
        valid_count_per_lane = torch.sum(mask, dim=1)
        
        # Filter valid lanes
        valid_lane_mask = valid_count_per_lane >= self.min_valid_points
        
        if valid_lane_mask.sum() == 0:
            # If no valid lanes match the criteria, return 0 with gradients
            return pred_offsets.sum() * 0.0
            
        # Calculate mean valid S per valid lane
        s_per_lane = s_sum_per_lane[valid_lane_mask] / valid_count_per_lane[valid_lane_mask]
        
        # Line loss is defined as 1 - S
        line_loss = 1.0 - s_per_lane
        
        # Return the mean of line losses
        return line_loss.mean()
