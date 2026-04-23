#!/usr/bin/env python
"""Train MediVLM with full supervision (ground-truth reports)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medivlm.data import build_dataloader
from medivlm.models import MediVLM
from medivlm.training import Trainer
from medivlm.utils import load_config
from medivlm.utils.logger import get_logger


logger = get_logger("train_supervised")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    torch.manual_seed(cfg.training.seed)
    train_loader = build_dataloader(cfg.data, "train")
    val_loader = build_dataloader(cfg.data, "val", shuffle=False)

    model = MediVLM(cfg)
    if args.resume:
        from medivlm.utils import load_checkpoint
        load_checkpoint(args.resume, model=model)

    trainer = Trainer(model, cfg, train_loader, val_loader)
    trainer.fit(epochs=args.epochs)

    # final test evaluation
    test_loader = build_dataloader(cfg.data, "test", shuffle=False)
    metrics = trainer.evaluate(test_loader, tag="test")
    logger.info(f"test metrics: {metrics}")

    import json
    out = Path(cfg.training.output_dir) / "test_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
