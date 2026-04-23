"""Projection heads + InfoNCE contrastive loss.

After the image encoder (CLIP) and the text encoder (ClinicalBERT) produce features in different spaces, a pair
of trainable MLPs project them into a shared d-dim space, where the
``contrastive_loss`` aligns positive image-text pairs.

This module is trainable end-to-end.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


def contrastive_loss(
    v: torch.Tensor,          # (B, D) projected visual features
    t: torch.Tensor,          # (B, D) projected text features
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE as used in CLIP / Eq. (1) of the paper.

    For each visual anchor v_i, the positive is t_i; all other t_j in the
    batch are negatives. We return the symmetric i2t + t2i mean loss.
    """
    assert v.shape == t.shape, "contrastive requires matched shapes"
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = (v @ t.t()) / temperature          # (B, B)
    target = torch.arange(v.size(0), device=v.device)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i2t + loss_t2i)
