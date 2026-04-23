"""RaTEScore wrapper (Zhao et al. 2024).

RaTEScore is a medical entity-aware semantic similarity metric. The
canonical implementation is distributed via the ``RaTEScore`` package.
This wrapper treats it as an optional dependency — same convention as
``radgraph`` / ``bert_score``.
"""
from __future__ import annotations

from typing import Dict, Sequence

try:
    # Official package: https://github.com/wjhou/RaTEScore / pip install RaTEScore
    from RaTEScore import RaTEScore
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def compute_ratescore(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> Dict[str, float]:
    if not _AVAILABLE:
        raise ImportError(
            "Install RaTEScore (`pip install RaTEScore`) to compute this metric. "
            "See https://github.com/wjhou/RaTEScore"
        )
    scorer = RaTEScore()
    scores = scorer.compute_score(list(hypotheses), list(references))
    mean = sum(scores) / max(len(scores), 1)
    return {"RaTEScore": float(mean)}
