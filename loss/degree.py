import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .cldice import soft_skel


class DegreeLoss(nn.Module):

    def __init__(
        self,
        iter_: int = 3,
        sigmoid=False,
        mode="l1",
        hops: List[int] = [1],
        hop_mode="normal",
        reduction="mean",
    ) -> None:
        super().__init__()
        assert mode in ["l1", "l2"]
        assert hop_mode in ["normal"]
        assert reduction in ["mean", "sum", "max"]
        self.hops = hops
        self.hop_mode = hop_mode
        self.iter = iter_
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.mode = mode

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if not self.sigmoid:
            y_pred = y_pred.sigmoid()
        y_pred = soft_skel(y_pred, self.iter)
        y_true = soft_skel(y_true, self.iter)
        total_loss = 0.0
        max_hop = max(self.hops)
        if self.hop_mode == "normal":
            kernel = torch.zeros(
                (1, 1, max_hop * 2 + 1, max_hop * 2 + 1, max_hop * 2 + 1), dtype=torch.float32
            )
            for hop in self.hops:
                kernel[:, :, [max_hop - hop, max_hop + hop], max_hop - hop:max_hop + hop + 1, max_hop - hop:max_hop + hop + 1] = 1
                kernel[:, :, max_hop - hop:max_hop+1 + hop, [max_hop - hop, max_hop + hop], max_hop - hop:max_hop + hop + 1] = 1
                kernel[:, :, max_hop - hop:max_hop+1 + hop, max_hop - hop:max_hop + hop + 1, [max_hop - hop, max_hop + hop]] = 1
        kernel = kernel.to(y_pred.device)
        deg_diff = F.conv3d(y_pred - y_true, kernel, padding=max_hop)
        if self.mode == "l1":
            loss = deg_diff.abs()
        elif self.mode == "l2":
            loss = deg_diff.pow(2)
            total_loss += loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "max":
            loss = loss.max()
        return loss
