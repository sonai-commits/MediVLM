"""BERTScore wrapper (Zhang et al. 2019)."""
from __future__ import annotations

from typing import Dict, List, Sequence

try:
    from bert_score import score as _bert_score
    _AVAILABLE = True
except ImportError:  
    _AVAILABLE = False


def compute_bertscore(
    references: Sequence[str],
    hypotheses: Sequence[str],
    model_type: str = "distilbert-base-uncased",
    lang: str = "en",
    device: str = None,
) -> Dict[str, float]:
    if not _AVAILABLE:
        raise ImportError("Install bert-score: pip install bert-score")
    P, R, F1 = _bert_score(
        cands=list(hypotheses),
        refs=list(references),
        model_type=model_type,
        lang=lang,
        verbose=False,
        device=device,
        rescale_with_baseline=False,
    )
    return {
        "BERTScore-P": float(P.mean()),
        "BERTScore-R": float(R.mean()),
        "BERTScore": float(F1.mean()),
    }
