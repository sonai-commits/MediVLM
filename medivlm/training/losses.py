"""Loss utilities - thin wrappers around the terms defined in the model."""
from __future__ import annotations

from typing import Optional

import torch


def combined_loss(
    loss_ce: torch.Tensor,
    loss_contrast: Optional[torch.Tensor],
    lambda_ce: float = 1.0,
    lambda_contrast: float = 0.7,
) -> torch.Tensor:
    """ L = lambda_1 * L_CE + lambda_2 * L_contrast."""
    if loss_contrast is None:
        return lambda_ce * loss_ce
    return lambda_ce * loss_ce + lambda_contrast * loss_contrast
