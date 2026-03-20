import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C] or [N], logits (before sigmoid)
        targets: [N, C] or [N], values in {0, 1, -1}
        """
        # Filter ignore samples (targets < 0)
        valid_mask = (targets >= 0)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0).to(inputs.device)
            
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        # Calculate BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Calculate alpha weight
        # alpha_t = alpha if target=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            # Normalize positive and negative samples separately to balance gradients
            pos_mask = (targets == 1)
            neg_mask = (targets == 0)
            
            num_pos = pos_mask.sum().clamp(min=1.0)
            num_neg = neg_mask.sum().clamp(min=1.0)
            
            loss_pos = focal_loss[pos_mask].sum() / num_pos
            loss_neg = focal_loss[neg_mask].sum() / num_neg
            
            return loss_pos + loss_neg
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
