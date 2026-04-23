#!/usr/bin/env python
"""Train MediVLM *without* ground-truth reports.

Pipeline:
    1. Load the dataset.
    2. Build MediVLM + BioT5 pseudo-labeler.
    3. Either load a pre-computed pseudo_reports.json (preferred for
       large corpora) or generate on-the-fly.
    4. Fine-tune MediVLM using BioT5 outputs as supervision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from medivlm.data import build_dataloader
from medivlm.models import MediVLM
from medivlm.models.pseudo_labeler import BioT5PseudoReporter
from medivlm.training import UnsupervisedTrainer, generate_pseudo_reports
from medivlm.utils import load_config, load_checkpoint
from medivlm.utils.logger import get_logger


logger = get_logger("train_unsup")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--pseudo-reports", default=None,
                   help="Optional pre-computed JSON map {id: pseudo_report}")
    p.add_argument("--dump-pseudo", default=None,
                   help="If given and pseudo-reports is absent, generate "
                        "pseudo reports offline and dump them here.")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def _load_pseudo(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.training.seed)

    train_loader = build_dataloader(cfg.data, "train")
    val_loader = build_dataloader(cfg.data, "val", shuffle=False)

    model = MediVLM(cfg)
    if args.resume:
        load_checkpoint(args.resume, model=model)
    pseudo_labeler = BioT5PseudoReporter(
        biot5_model_name=cfg.pseudo_labeler.biot5_model_name,
        max_length=cfg.pseudo_labeler.max_length,
        num_beams=cfg.pseudo_labeler.num_beams,
    )

    pseudo_map = _load_pseudo(args.pseudo_reports)
    if not pseudo_map and args.dump_pseudo:
        pseudo_map = generate_pseudo_reports(
            model=model,
            pseudo_labeler=pseudo_labeler,
            loader=train_loader,
            output_path=args.dump_pseudo,
        )

    trainer = UnsupervisedTrainer(
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        pseudo_report_map=pseudo_map,
        pseudo_labeler=pseudo_labeler if not pseudo_map else None,
    )
    trainer.fit(epochs=args.epochs)

    test_loader = build_dataloader(cfg.data, "test", shuffle=False)
    metrics = trainer.evaluate(test_loader, tag="test")
    logger.info(f"test metrics: {metrics}")
    Path(cfg.training.output_dir, "test_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
