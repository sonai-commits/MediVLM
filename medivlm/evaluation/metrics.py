"""BLEU-1..4, METEOR, ROUGE-L implementations.

We use NLTK for BLEU + METEOR and ``rouge_score`` for ROUGE-L. All
scores are corpus-level, matching the usual radiology-report
leaderboard convention (R2Gen, X-RGen etc.).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ROUGE_AVAILABLE = False


def _ensure_nltk() -> None:
    needed = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, pkg in needed:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def _tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())


def compute_bleu(references: Sequence[str], hypotheses: Sequence[str]) -> Dict[str, float]:
    _ensure_nltk()
    smooth = SmoothingFunction().method1
    ref_toks = [[_tokenize(r)] for r in references]
    hyp_toks = [_tokenize(h) for h in hypotheses]
    scores = {}
    for n in (1, 2, 3, 4):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"BLEU-{n}"] = float(
            corpus_bleu(ref_toks, hyp_toks, weights=weights, smoothing_function=smooth)
        )
    return scores


def compute_meteor(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    _ensure_nltk()
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(meteor_score([_tokenize(ref)], _tokenize(hyp)))
    return float(sum(scores) / max(len(scores), 1))


def compute_rouge_l(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if not _ROUGE_AVAILABLE:
        raise ImportError("Install rouge-score: pip install rouge-score")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(r.lower(), h.lower())["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    return float(sum(vals) / max(len(vals), 1))


def compute_nlg_metrics(
    references: Sequence[str], hypotheses: Sequence[str]
) -> Dict[str, float]:
    out = {}
    out.update(compute_bleu(references, hypotheses))
    out["METEOR"] = compute_meteor(references, hypotheses)
    out["ROUGE-L"] = compute_rouge_l(references, hypotheses)
    return out
