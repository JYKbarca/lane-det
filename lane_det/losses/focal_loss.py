import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(QualityFocalLoss, self).__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
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
        pred_sigmoid = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pos_mask = (targets > 0).to(inputs.dtype)
        neg_mask = (targets == 0).to(inputs.dtype)

        pos_weight = targets * pos_mask
        neg_weight = self.alpha * pred_sigmoid.pow(self.gamma) * neg_mask
        loss = bce * (pos_weight + neg_weight)

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
