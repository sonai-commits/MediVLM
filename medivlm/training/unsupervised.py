"""Unsupervised MediVLM training (Section 3.5).

Pipeline:
    1. Run the frozen Faster R-CNN on every training image -> region
       labels + bboxes.
    2. Feed the region labels to ClinicalBERT (inside MediVLM's text
       encoder) and decode them with BioT5 to produce a concrete
       clinical pseudo-report per image.
    3. Train MediVLM as usual, but replace the ground-truth report with
       the BioT5 pseudo-report as supervision for the GPT-2 decoder.
       The image-text contrastive term uses the same pseudo-report.

This file provides:

    * ``generate_pseudo_reports`` -- offline corpus builder
    * ``UnsupervisedTrainer``     -- wraps ``Trainer`` but replaces
                                     ground-truth reports with the
                                     pseudo-reports on the fly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.medivlm import MediVLM
from ..models.pseudo_labeler import BioT5PseudoReporter
from ..utils.config import MediVLMConfig
from ..utils.logger import get_logger

from .trainer import Trainer


logger = get_logger("unsup")


@torch.no_grad()
def generate_pseudo_reports(
    model: MediVLM,
    pseudo_labeler: BioT5PseudoReporter,
    loader: DataLoader,
    output_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, str]:
    """Run the detector over every image in ``loader`` and write
    {id: pseudo_report} to ``output_path``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    pseudo_labeler.to(device).eval()

    reports: Dict[str, str] = {}
    for batch in tqdm(loader, desc="pseudo-labeling"):
        images = batch["images"].to(device, non_blocking=True)
        detections = model.detector(images)
        for i, det in enumerate(detections):
            labels = det.label_names
            pseudo = pseudo_labeler.generate_pseudo_report(labels)
            reports[batch["ids"][i]] = pseudo

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    logger.info(f"wrote {len(reports)} pseudo reports to {out}")
    return reports


class UnsupervisedTrainer(Trainer):
    """Drop-in replacement for ``Trainer`` that substitutes pseudo-reports."""

    def __init__(
        self,
        model: MediVLM,
        config: MediVLMConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        pseudo_report_map: Optional[Dict[str, str]] = None,
        pseudo_labeler: Optional[BioT5PseudoReporter] = None,
    ) -> None:
        super().__init__(model, config, train_loader, val_loader, device)
        self.pseudo_report_map = pseudo_report_map or {}
        self.pseudo_labeler = pseudo_labeler
        if self.pseudo_labeler is not None:
            self.pseudo_labeler.to(self.device).eval()

    # ------------------------------------------------------------------
    def _maybe_build_pseudo_reports(self, batch) -> List[str]:
        """Prefer the pre-computed map; otherwise generate on-the-fly."""
        ids = batch["ids"]
        pseudo: List[Optional[str]] = [self.pseudo_report_map.get(i) for i in ids]

        missing = [i for i, p in enumerate(pseudo) if p is None]
        if missing and self.pseudo_labeler is not None:
            with torch.no_grad():
                images = batch["images"][missing].to(self.device, non_blocking=True)
                dets = self.model.detector(images)
                new = [self.pseudo_labeler.generate_pseudo_report(d.label_names) for d in dets]
            for m, n in zip(missing, new):
                pseudo[m] = n
                self.pseudo_report_map[ids[m]] = n
        # final fallback
        return [p or "Heart size is normal. The lungs are clear. No acute abnormalities."
                for p in pseudo]

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        pbar = tqdm(self.train_loader, desc=f"unsup epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["images"].to(self.device, non_blocking=True)
            reports = self._maybe_build_pseudo_reports(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out = self.model(images, reports=reports)
                loss = out.loss
            if loss is None:
                continue
            self.scaler.scale(loss).backward()
            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.training.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total += float(loss.item())
            count += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        return total / max(count, 1)
