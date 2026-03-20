import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(QualityBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C] or [N], logits (before sigmoid)
        targets: [N, C] or [N], values in [-1, 1]
          -1: ignore
           0: negative
        (0, 1]: positive quality target
        """
        valid_mask = (targets >= 0)
        if valid_mask.sum() == 0:
            return inputs.sum() * 0.0

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        if self.reduction == 'mean':
            pos_mask = (targets > 0)
            neg_mask = (targets == 0)

            num_pos = pos_mask.sum().clamp(min=1.0)
            num_neg = neg_mask.sum().clamp(min=1.0)

            loss_pos = loss[pos_mask].sum() / num_pos
            loss_neg = loss[neg_mask].sum() / num_neg
            return loss_pos + loss_neg
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
