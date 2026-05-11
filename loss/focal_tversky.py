"""Focal Tversky Loss for 3D binary segmentation.

Tversky index: TP / (TP + alpha*FP + beta*FN)
Focal exponent gamma focuses learning on hard-to-segment regions.
Default alpha=0.2, beta=0.8 penalises false negatives more (important for thin
neuron structures). Default gamma=2 mimics focal loss behaviour.
"""

import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.8,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        loss = 0.0
        for b in range(pred.shape[0]):
            p = pred[b].contiguous().view(-1)
            t = target[b].contiguous().view(-1)
            tp = (p * t).sum()
            fp = ((1.0 - t) * p).sum()
            fn = (t * (1.0 - p)).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss = loss + (1.0 - tversky) ** self.gamma
        return loss
