#!/usr/bin/env python
"""Run Faster R-CNN + BioT5 over the training split to create pseudo-reports."""
from __future__ import annotations

import argparse

import torch

from medivlm.data import build_dataloader
from medivlm.models import MediVLM
from medivlm.models.pseudo_labeler import BioT5PseudoReporter
from medivlm.training import generate_pseudo_reports
from medivlm.utils import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    loader = build_dataloader(cfg.data, args.split, shuffle=False)
    model = MediVLM(cfg)
    pseudo_labeler = BioT5PseudoReporter(
        biot5_model_name=cfg.pseudo_labeler.biot5_model_name,
        max_length=cfg.pseudo_labeler.max_length,
        num_beams=cfg.pseudo_labeler.num_beams,
    )
    generate_pseudo_reports(
        model=model,
        pseudo_labeler=pseudo_labeler,
        loader=loader,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
