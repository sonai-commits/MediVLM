"""Trainable cross-attention fusion of aligned visual and text features

F = CrossAttention(V_aligned, T_aligned)

We use visual tokens as the query (they will drive generation since
text may be unavailable at inference) and text tokens as key/value.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_expansion * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_expansion * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,              # (B, P, D)
        text: Optional[torch.Tensor],      # (B, L, D)
        text_mask: Optional[torch.Tensor] = None,  # (B, L), 1 = keep
    ) -> torch.Tensor:
        """Fuse visual (query) with text (key/value). If text is None,
        self-attend over visual tokens (inference without reports).
        Returns (B, P, D)."""
        q = self.norm1(visual)
        if text is None:
            kv = q
            key_padding_mask = None
        else:
            kv = self.norm1(text)
            if text_mask is not None:
                key_padding_mask = text_mask == 0  # True = ignore
            else:
                key_padding_mask = None
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        x = visual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
