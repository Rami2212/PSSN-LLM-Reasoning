"""Small neural classifier for contextual reasoning-sentence labels."""

from __future__ import annotations

import torch
from torch import nn

from .config import BlackHoleDetectorConfig


class BlackHoleDetector(nn.Module):
    """Predict low-value probability from a pooled sentence activation.

    Input: ``[batch_size, input_size]`` sentence-span activation vectors.
    ``forward_logits`` returns one unnormalized low-value score per row for a
    numerically stable BCE-with-logits training loss. ``forward`` returns the
    corresponding probability in ``[0, 1]``.
    """

    def __init__(self, config: BlackHoleDetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or BlackHoleDetectorConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, 1),
        )

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch_size]`` (or scalar for one vector)."""
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim not in (1, 2):
            raise ValueError("features must have shape [input_size] or [batch_size, input_size]")
        if features.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} features, got {features.shape[-1]}"
            )
        return self.network(features).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``P(low_value_in_context)`` for each input sentence."""
        return torch.sigmoid(self.forward_logits(features))

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Explicit probability API, equivalent to :meth:`forward`."""
        return self(features)
