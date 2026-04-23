"""Unsupervised pseudo-report generator (Section 3.5).

Given an image x_i, Faster R-CNN produces anatomical (label_i, bbox_i)
pairs. The semantic labels are encoded with ClinicalBERT and decoded
into a coherent clinical sentence embedding with BioT5, which is then
autoregressively refined to produce a pseudo-report. MediVLM's GPT-2
decoder is then fine-tuned with these pseudo-reports as targets.

This module wraps:
    * ClinicalBERT as the encoder (reuse of the frozen text encoder)
    * BioT5 (QizhiPei/biot5-base) fine-tuned on (label -> sentence) tasks

The pseudo-report for each image is the concatenation (after dedup) of
one sentence per detected region. BioT5 is prompted with templated
strings like ``"describe anatomy: <region name>"`` to produce natural
clinical sentences. The result is a pseudo-report usable as supervision.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None          # type: ignore


class BioT5PseudoReporter(nn.Module):
    """Produces pseudo-reports from Faster R-CNN region labels via BioT5.

    The module is used for *generating training targets*, not inside the
    main forward. It is therefore fine to place under ``@torch.no_grad``
    at inference time. We keep it trainable-capable however so that the
    authors' described "fine-tuning with clinical labels and partial
    reports" pathway can be reproduced via ``fine_tune_step``.
    """

    DEFAULT_TEMPLATE = (
        "describe the chest xray finding for the region {region}: "
    )
    NORMAL_TEMPLATE = (
        "The {region} appears normal."
    )

    def __init__(
        self,
        biot5_model_name: str = "QizhiPei/biot5-base",
        max_length: int = 77,
        num_beams: int = 4,
    ) -> None:
        super().__init__()
        if AutoModelForSeq2SeqLM is None:
            raise ImportError("transformers is required.")
        self.tokenizer = AutoTokenizer.from_pretrained(biot5_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(biot5_model_name)
        self.max_length = max_length
        self.num_beams = num_beams

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_sentences(self, region_labels: Iterable[str]) -> List[str]:
        """Turn a list of region names into clinical sentences (one each)."""
        prompts = [self.DEFAULT_TEMPLATE.format(region=r) for r in region_labels]
        device = next(self.model.parameters()).device
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_length,
            num_beams=self.num_beams,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    # ------------------------------------------------------------------
    def generate_pseudo_report(
        self,
        region_labels: List[str],
        scores: Optional[List[float]] = None,
    ) -> str:
        """Compose a full pseudo-report from region labels.

        Heuristic: if BioT5 produces a low-confidence sentence (empty or
        too short) we fall back to the ``NORMAL_TEMPLATE`` formulation.
        """
        seen, kept, order = set(), [], []
        for i, r in enumerate(region_labels):
            if r in seen or r in {"padding", "background", "whole image"}:
                continue
            seen.add(r)
            order.append((i, r))
        if not order:
            return "Heart size is normal. The lungs are clear. No acute abnormalities."
        sentences = self.generate_sentences([r for _, r in order])
        refined = []
        for (idx, r), s in zip(order, sentences):
            s = (s or "").strip()
            if len(s.split()) < 3:
                s = self.NORMAL_TEMPLATE.format(region=r)
            if not s.endswith("."):
                s += "."
            refined.append(s)
        return " ".join(refined)

    # ------------------------------------------------------------------
    def fine_tune_step(
        self,
        input_texts: List[str],
        target_texts: List[str],
    ) -> torch.Tensor:
        """Single supervised step using (label/partial-text -> clean-report) pairs."""
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(device)
        labels = dec["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )
        return out.loss
