"""Supervised training loop for MediVLM.

hyperparameters:
    * AdamW optimiser, lr = 2e-5
    * batch size = 32
    * 30-50 epochs
    * loss = 1.0 * CE + 0.7 * contrastive (tau = 0.07)
    * mixed precision (optional)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..evaluation.metrics import compute_nlg_metrics
from ..models.medivlm import MediVLM
from ..utils.checkpoints import save_checkpoint
from ..utils.config import MediVLMConfig
from ..utils.logger import get_logger


logger = get_logger("trainer")


class Trainer:
    def __init__(
        self,
        model: MediVLM,
        config: MediVLMConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.training.mixed_precision and self.device.type == "cuda")

        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float("inf")

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in trainable):,} "
            f"(total: {sum(p.numel() for p in self.model.parameters()):,})"
        )
        return torch.optim.AdamW(
            trainable,
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

    def _build_scheduler(self):
        total_steps = max(1, self.config.training.epochs * max(1, len(self.train_loader)))
        warmup = self.config.training.warmup_steps
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    def fit(self, epochs: Optional[int] = None) -> Dict[str, float]:
        epochs = epochs or self.config.training.epochs
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            logger.info(f"epoch {epoch:03d} | train_loss={train_loss:.4f}")
            if self.val_loader is not None and epoch % self.config.training.eval_every == 0:
                metrics = self.evaluate(self.val_loader, tag=f"val_epoch{epoch}")
                score = metrics.get("BLEU-4", metrics.get("BLEU-1", 0.0))
                logger.info(f"epoch {epoch:03d} | val {metrics}")
                if score >= self.best_metric:
                    self.best_metric = score
                    self.save(f"best.ckpt", metrics=metrics, epoch=epoch)
            if epoch % self.config.training.save_every == 0:
                self.save(f"epoch_{epoch}.ckpt", epoch=epoch)
        return {"best_metric": self.best_metric}

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            images = batch["images"].to(self.device, non_blocking=True)
            reports = batch["reports"]

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
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "ce": f"{float(out.loss_ce.item()):.3f}" if out.loss_ce is not None else "-",
                "contrast": f"{float(out.loss_contrast.item()):.3f}" if out.loss_contrast is not None else "-",
            })
        return total / max(count, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, tag: str = "val") -> Dict[str, float]:
        self.model.eval()
        refs: List[str] = []
        hyps: List[str] = []
        for batch in tqdm(loader, desc=tag, leave=False):
            images = batch["images"].to(self.device, non_blocking=True)
            gen = self.model.generate(images)
            hyps.extend(gen)
            refs.extend(batch["reports"])
        # save generations for later analysis
        out_dir = self.output_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "hyp.txt", "w") as fh, open(out_dir / "ref.txt", "w") as fr:
            for h, r in zip(hyps, refs):
                fh.write(h.strip() + "\n")
                fr.write(r.strip() + "\n")
        return compute_nlg_metrics(refs, hyps)

    # ------------------------------------------------------------------
    def save(self, name: str, **extra) -> None:
        path = self.output_dir / name
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config.to_dict(),
            **extra,
        )
        logger.info(f"saved checkpoint to {path}")
