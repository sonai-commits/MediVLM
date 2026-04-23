#!/usr/bin/env python
"""Evaluate a checkpoint against the test split with every metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from medivlm.data import build_dataloader
from medivlm.evaluation import (
    SeverityScorer,
    compute_bertscore,
    compute_nlg_metrics,
    compute_radgraph_f1,
    compute_ratescore,
)
from medivlm.models import MediVLM
from medivlm.utils import load_checkpoint, load_config
from medivlm.utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default=None)
    p.add_argument("--metrics", nargs="+",
                   default=["nlg", "bertscore", "severity"],
                   choices=["nlg", "bertscore", "radgraph", "ratescore", "severity"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    loader = build_dataloader(cfg.data, args.split, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MediVLM(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    # 1) generate reports
    refs, hyps, ids = [], [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.no_grad():
            gen = model.generate(images)
        hyps.extend(gen)
        refs.extend(batch["reports"])
        ids.extend(batch["ids"])

    # 2) metrics
    results = {}
    if "nlg" in args.metrics:
        results.update(compute_nlg_metrics(refs, hyps))
    if "bertscore" in args.metrics:
        try:
            results.update(compute_bertscore(refs, hyps, device="cuda" if torch.cuda.is_available() else "cpu"))
        except Exception as exc:
            logger.warning(f"BERTScore skipped: {exc}")
    if "radgraph" in args.metrics:
        try:
            results.update(compute_radgraph_f1(refs, hyps))
        except Exception as exc:
            logger.warning(f"RadGraph-F1 skipped: {exc}")
    if "ratescore" in args.metrics:
        try:
            results.update(compute_ratescore(refs, hyps))
        except Exception as exc:
            logger.warning(f"RaTEScore skipped: {exc}")

    # 3) severity scores (per hypothesis)
    severities = []
    if "severity" in args.metrics:
        # fit on the *training* split so the score is comparable across runs
        train_loader = build_dataloader(cfg.data, "train", shuffle=False)
        train_corpus = []
        for batch in train_loader:
            train_corpus.extend(batch["reports"])
        scorer = SeverityScorer(
            seed_terms=cfg.severity.seed_terms,
            use_nltk_sentiment=cfg.severity.use_nltk_sentiment,
            normalize_across_corpus=cfg.severity.normalize_across_corpus,
        ).fit(train_corpus)
        severities = scorer.batch_score(hyps)
        results["mean_severity"] = float(sum(severities) / max(len(severities), 1))

    logger.info(f"metrics: {results}")
    out_path = args.output or str(Path(cfg.training.output_dir) / f"eval_{args.split}.json")
    payload = {
        "metrics": results,
        "predictions": [
            {"id": i, "reference": r, "hypothesis": h, "severity": s}
            for i, r, h, s in zip(ids, refs, hyps, severities or [None] * len(hyps))
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
