"""Focal loss for multi-class classification with per-class alpha weights.

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017).

    FL(p_t) = - alpha_t * (1 - p_t) ** gamma * log(p_t)

where ``p_t`` is the predicted probability of the true class and
``alpha_t`` is a per-class weight (typically inverse-frequency). When
``gamma == 0`` this reduces to weighted cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None,
                 gamma: float = 2.0,
                 reduction: str = "mean") -> None:
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = float(gamma)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be none|mean|sum, got {reduction}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        focal_factor = (1.0 - target_probs).clamp(min=1e-8).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device, logits.dtype)
            loss = loss * alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
