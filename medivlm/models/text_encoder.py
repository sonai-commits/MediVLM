"""ClinicalBERT text encoder (frozen) -> latent token embeddings.

ClinicalBERT (Huang et al. 2020) was
specifically designed for clinical text and the authors found that
swapping it in for the CLIP text tower gave better embeddings.

Weights default to the Huang et al. checkpoint on the HF Hub
(``medicalai/ClinicalBERT``) but any BERT-style model works. We
fall back to ``emilyalsentzer/Bio_ClinicalBERT`` if the primary
name is unavailable.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None     # type: ignore
    AutoTokenizer = None  # type: ignore


class ClinicalBertTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "medicalai/ClinicalBERT",
        freeze: bool = True,
        max_length: int = 128,
        fallback: str = "emilyalsentzer/Bio_ClinicalBERT",
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required.")
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception:  # network or model issues -> fallback
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.bert = AutoModel.from_pretrained(fallback)
        self.hidden_size = self.bert.config.hidden_size  # usually 768
        self._frozen = bool(freeze)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)
            self.bert.eval()

    def train(self, mode: bool = True):  
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.bert.eval()
        return self

    # ------------------------------------------------------------------
    def tokenize(self, texts: List[str]) -> dict:
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return tok

    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Return token embeddings, pooled embedding, and attention mask.

        If ``texts`` is provided, tokenisation happens on CPU inside the
        encoder, then tensors are moved to the encoder's device.
        """
        if texts is not None:
            tok = self.tokenize(texts)
            device = next(self.bert.parameters()).device
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
        assert input_ids is not None and attention_mask is not None

        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.bert.parameters())
        ):
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        token_embeds = out.last_hidden_state              # (B, L, D)
        if getattr(out, "pooler_output", None) is not None:
            pooled = out.pooler_output                     # (B, D)
        else:
            # mean pool over attention mask
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        return {
            "token_embeds": token_embeds,
            "pooled": pooled,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }
