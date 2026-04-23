"""MediVLM model"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.config import MediVLMConfig
from .detector import DetectorOutput, FasterRCNNDetector
from .image_encoder import CLIPImageEncoder
from .text_encoder import ClinicalBertTextEncoder
from .projection import ProjectionHead, contrastive_loss
from .fusion import CrossAttentionFusion
from .decoder import GPT2ReportDecoder


@dataclass
class MediVLMForwardOutput:
    loss: Optional[torch.Tensor]
    loss_ce: Optional[torch.Tensor]
    loss_contrast: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    fusion: torch.Tensor
    visual_tokens: torch.Tensor
    text_tokens: Optional[torch.Tensor]
    detections: List[DetectorOutput]


class MediVLM(nn.Module):
    def __init__(self, config: MediVLMConfig) -> None:
        super().__init__()
        self.config = config

        # ---- Module 1: Faster R-CNN detector (frozen) ----
        self.detector = FasterRCNNDetector(
            num_anatomical_classes=config.detector.num_anatomical_classes,
            weights_path=config.detector.weights_path,
            min_size=config.detector.min_size,
            max_size=config.detector.max_size,
            box_score_thresh=config.detector.box_score_thresh,
            box_nms_thresh=config.detector.box_nms_thresh,
            top_p_patches=config.detector.top_p_patches,
            patch_size=config.detector.patch_size,
            freeze=config.detector.freeze,
        )

        # ---- Module 2: CLIP-ViT image encoder (frozen) ----
        self.image_encoder = CLIPImageEncoder(
            model_name=config.image_encoder.model_name,
            freeze=config.image_encoder.freeze,
            pos_mlp_hidden=config.image_encoder.pos_mlp_hidden,
            image_size=config.image_encoder.image_size,
        )
        visual_dim = self.image_encoder.hidden_size

        # ---- Module 3: ClinicalBERT text encoder (frozen) ----
        self.text_encoder = ClinicalBertTextEncoder(
            model_name=config.text_encoder.model_name,
            freeze=config.text_encoder.freeze,
            max_length=config.text_encoder.max_length,
        )
        text_dim = self.text_encoder.hidden_size

        # ---- Module 4a: projection heads + contrastive alignment ----
        proj_dim = config.projection.proj_dim
        self.proj_visual = ProjectionHead(visual_dim, proj_dim, dropout=config.projection.dropout)
        self.proj_text = ProjectionHead(text_dim, proj_dim, dropout=config.projection.dropout)
        self.temperature = config.projection.temperature

        # ---- Module 4b: cross-attention fusion ----
        self.fusion = CrossAttentionFusion(
            dim=proj_dim,
            num_heads=config.fusion.num_heads,
            dropout=config.fusion.dropout,
        )

        # ---- Module 5: GPT-2 decoder ----
        self.decoder = GPT2ReportDecoder(
            model_name=config.decoder.model_name,
            trainable_blocks=config.decoder.trainable_blocks,
            fusion_dim=proj_dim,
            max_length=config.decoder.max_length,
            beam_size=config.decoder.beam_size,
            repetition_penalty=config.decoder.repetition_penalty,
            no_repeat_ngram_size=config.decoder.no_repeat_ngram_size,
            length_penalty=config.decoder.length_penalty,
        )

        # weights in the combined loss
        self.lambda_ce = config.training.lambda_ce
        self.lambda_contrast = config.training.lambda_contrast

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    def _stack_detector_outputs(
        self, detections: List[DetectorOutput]
    ) -> Dict[str, torch.Tensor]:
        """Stack per-image detector outputs into batched tensors (pads to top_p)."""
        patches = torch.stack([d.patches for d in detections], dim=0)        # (B, P, 3, h, w)
        coords = torch.stack([d.norm_coords for d in detections], dim=0)      # (B, P, 4)
        return {"patches": patches, "coords": coords}

    # ------------------------------------------------------------------
    def encode_image(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect -> crop -> CLIP -> (B, P, D_visual)."""
        detections = self.detector(images)
        stacked = self._stack_detector_outputs(detections)
        visual = self.image_encoder(stacked["patches"].to(images.device),
                                    stacked["coords"].to(images.device))
        return {"visual": visual, "detections": detections}

    def encode_text(self, texts: Optional[List[str]]) -> Optional[Dict[str, torch.Tensor]]:
        if texts is None:
            return None
        return self.text_encoder(texts)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        reports: Optional[List[str]] = None,
        return_detections: bool = False,
    ) -> MediVLMForwardOutput:
        """Training forward. ``reports`` must be provided to compute losses.

        If reports are None, the model simply produces the fused visual
        representation and tokenised generation-ready prefix.
        """
        enc_img = self.encode_image(images)
        visual = enc_img["visual"]            # (B, P, D_visual)
        detections = enc_img["detections"]

        enc_txt = self.encode_text(reports)
        text_tokens: Optional[torch.Tensor] = None
        text_pooled: Optional[torch.Tensor] = None
        text_mask: Optional[torch.Tensor] = None
        if enc_txt is not None:
            text_tokens = enc_txt["token_embeds"]            # (B, L, D_text)
            text_pooled = enc_txt["pooled"]                   # (B, D_text)
            text_mask = enc_txt["attention_mask"]

        # Project to shared space
        v_proj_tokens = self.proj_visual(visual)              # (B, P, D_proj)
        v_proj_pool = v_proj_tokens.mean(dim=1)                # (B, D_proj)

        if text_tokens is not None:
            t_proj_tokens = self.proj_text(text_tokens)        # (B, L, D_proj)
            t_proj_pool = self.proj_text(text_pooled)           # (B, D_proj)
        else:
            t_proj_tokens = None
            t_proj_pool = None

        # Cross-attention fusion
        fused = self.fusion(v_proj_tokens, t_proj_tokens, text_mask)   # (B, P, D_proj)

        loss_ce = None
        loss_contrast = None
        loss = None
        logits = None
        if reports is not None:
            # Tokenise reports for the GPT-2 side
            enc = self.decoder.tokenizer(
                reports,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.decoder.max_length,
            ).to(images.device)
            labels_input_ids = enc["input_ids"]
            dec_out = self.decoder(fused, labels_input_ids)
            loss_ce = dec_out["loss"]
            logits = dec_out["logits"]

            # Contrastive on pooled projected features
            loss_contrast = contrastive_loss(
                v_proj_pool, t_proj_pool, temperature=self.temperature
            )
            loss = self.lambda_ce * loss_ce + self.lambda_contrast * loss_contrast

        return MediVLMForwardOutput(
            loss=loss,
            loss_ce=loss_ce,
            loss_contrast=loss_contrast,
            logits=logits,
            fusion=fused,
            visual_tokens=v_proj_tokens,
            text_tokens=t_proj_tokens,
            detections=detections if return_detections else [],
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> List[str]:
        """Image-only inference: no ground-truth report needed."""
        enc_img = self.encode_image(images)
        v_tokens = self.proj_visual(enc_img["visual"])
        fused = self.fusion(v_tokens, text=None)
        return self.decoder.generate(
            fused, prompt=prompt, max_length=max_length, num_beams=num_beams
        )
