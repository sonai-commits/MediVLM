"""CLIP-ViT-L/14 image encoder with positional priors.

The p=8 most confident anatomical patches
returned by Faster R-CNN are resized and pushed through a frozen
CLIP-ViT encoder. Each patch's (x, y, w, h) bounding-box coordinates
are then normalised and passed through an MLP to produce a positional
embedding with the same dim as the CLIP feature; the two are summed
(we also provide a concat variant) before being fed to the projection
layer / decoder.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPImageProcessor, CLIPVisionModel
except ImportError:  
    CLIPImageProcessor = None 
    CLIPVisionModel = None    


class PositionMLP(nn.Module):
    """Encodes the normalised bbox (x, y, w, h) into the CLIP feature dim."""

    def __init__(self, hidden_dim: int = 256, out_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # (B, P, 4)
        return self.net(coords)


class CLIPImageEncoder(nn.Module):
    """Frozen CLIP-ViT-L/14 vision tower; returns (B, P, D) patch embeddings.

    We do NOT use the hugging-face image processor here; the patches are
    already cropped, resized and [0,1]-normalised by the detector. We
    perform the CLIP mean/std normalisation inside the module for speed.
    """

    # CLIP-ViT-L/14 preprocessing constants (OpenAI)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        freeze: bool = True,
        pos_mlp_hidden: int = 256,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if CLIPVisionModel is None:
            raise ImportError(
                "transformers is required. Install via `pip install transformers`."
            )
        self.model_name = model_name
        self.image_size = image_size
        self.vision = CLIPVisionModel.from_pretrained(model_name)
        self.vision.eval()
        self.hidden_size = self.vision.config.hidden_size  # 1024 for CLIP-L/14
        self._frozen = bool(freeze)

        if freeze:
            for p in self.vision.parameters():
                p.requires_grad_(False)

        self.pos_mlp = PositionMLP(hidden_dim=pos_mlp_hidden, out_dim=self.hidden_size)

        self.register_buffer(
            "clip_mean",
            torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(self.CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):  
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.vision.eval()  # keep dropout off 
        return self

    # ------------------------------------------------------------------
    def _preprocess(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B*P, 3, h, w) in [0, 1]."""
        if patches.shape[-1] != self.image_size:
            patches = F.interpolate(
                patches, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return (patches - self.clip_mean) / self.clip_std

    def forward(
        self,
        patches: torch.Tensor,     # (B, P, 3, h, w)
        coords: torch.Tensor,      # (B, P, 4)
    ) -> torch.Tensor:
        """Return visual tokens of shape (B, P, D) ready for alignment/fusion."""
        assert patches.dim() == 5, "patches must be (B, P, 3, h, w)"
        B, P, C, H, W = patches.shape
        flat = patches.view(B * P, C, H, W)
        flat = self._preprocess(flat)
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.vision.parameters())):
            out = self.vision(pixel_values=flat)
        # pooler_output: (B*P, D); last_hidden_state: (B*P, 1+tokens, D)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            cls_feat = out.pooler_output
        else:
            cls_feat = out.last_hidden_state[:, 0]
        cls_feat = cls_feat.view(B, P, self.hidden_size)

        pos = self.pos_mlp(coords)  # (B, P, D)
        return cls_feat + pos
