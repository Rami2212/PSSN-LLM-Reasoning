"""Optional class-weighted loss for the Black Hole detector."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def positive_class_weight(labels: torch.Tensor) -> float:
    """Return negatives/positives for ``pos_weight``; 1.0 if either class is absent."""
    labels = torch.as_tensor(labels).reshape(-1)
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


class BlackHoleLoss(nn.Module):
    """Binary cross-entropy on logits, optionally weighting low-value positives."""

    def __init__(self, positive_weight: float | None = None) -> None:
        super().__init__()
        if positive_weight is not None and positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        weight = None if positive_weight is None else torch.tensor(float(positive_weight))
        self.register_buffer("positive_weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.positive_weight,
        )
