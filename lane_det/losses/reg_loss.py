import torch
import torch.nn as nn
import torch.nn.functional as F

class RegLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RegLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        """
        inputs: [N, num_y]
        targets: [N, num_y]
        mask: [N, num_y], 0 or 1
        """
        # Smooth L1 Loss
        loss = F.smooth_l1_loss(inputs, targets, reduction='none', beta=1.0)
        
        if mask is not None:
            loss = loss * mask
            
            if self.reduction == 'mean':
                # Only average over valid elements
                num_valid = mask.sum()
                if num_valid > 0:
                    return loss.sum() / num_valid
                else:
                    return loss.sum() * 0 # Avoid division by zero
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss
