"""RadGraph-F1 metric (Yu et al. 2023).

RadGraph-F1 measures the F1 of entity + relation graph overlap between
reference and candidate radiology reports. It depends on an external
package (``radgraph``) which ships a fine-tuned PubMedBERT model for
entity/relation extraction.

This wrapper keeps the dependency optional: if ``radgraph`` is not
installed, importing this module still works but calling the function
will raise a error. The Allen-NLP API mirrors the `radgraph` reference implementation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

try:
    from radgraph import F1RadGraph
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def compute_radgraph_f1(
    references: Sequence[str],
    hypotheses: Sequence[str],
    reward_level: str = "partial",
    device: int = 0,
) -> Dict[str, float]:
    if not _AVAILABLE:
        raise ImportError(
            "Install radgraph (`pip install radgraph`) and its model weights to "
            "compute RadGraph-F1. See https://pypi.org/project/radgraph/"
        )
    scorer = F1RadGraph(reward_level=reward_level, cuda=device)
    mean_f1, individual_f1, _ = scorer(hyps=list(hypotheses), refs=list(references))
    return {"RadGraph-F1": float(mean_f1)}
