#!/usr/bin/env python
"""Generate a radiology report + severity score for a single image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from medivlm.data.transforms import build_image_transform
from medivlm.evaluation import SeverityScorer
from medivlm.models import MediVLM
from medivlm.utils import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True, help="Path to the medical image")
    p.add_argument("--severity-corpus", default=None,
                   help="Optional .txt file (one report per line) used to fit "
                        "the TF-IDF severity scorer. If omitted, the score will "
                        "be unnormalised.")
    p.add_argument("--num-beams", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MediVLM(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    transform = build_image_transform(cfg.image_encoder.image_size, train=False)
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        reports = model.generate(image, num_beams=args.num_beams)
    report = reports[0]

    severity = None
    if args.severity_corpus and Path(args.severity_corpus).exists():
        corpus = Path(args.severity_corpus).read_text().splitlines()
        scorer = SeverityScorer(
            seed_terms=cfg.severity.seed_terms,
            use_nltk_sentiment=cfg.severity.use_nltk_sentiment,
            normalize_across_corpus=True,
        ).fit(corpus)
        severity = scorer.score(report)

    print(json.dumps({"report": report, "severity": severity}, indent=2))


if __name__ == "__main__":
    main()
