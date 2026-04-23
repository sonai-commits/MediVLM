"""Evaluation entry points — lazy so you don't pay nltk/bert-score/radgraph tax
unless you actually call the metric."""
from importlib import import_module
from typing import Any

__all__ = [
    "compute_nlg_metrics",
    "compute_bleu",
    "compute_rouge_l",
    "compute_meteor",
    "compute_bertscore",
    "compute_radgraph_f1",
    "compute_ratescore",
    "SeverityScorer",
]

_TABLE = {
    "compute_nlg_metrics": (".metrics", "compute_nlg_metrics"),
    "compute_bleu": (".metrics", "compute_bleu"),
    "compute_rouge_l": (".metrics", "compute_rouge_l"),
    "compute_meteor": (".metrics", "compute_meteor"),
    "compute_bertscore": (".bertscore", "compute_bertscore"),
    "compute_radgraph_f1": (".radgraph", "compute_radgraph_f1"),
    "compute_ratescore": (".ratescore", "compute_ratescore"),
    "SeverityScorer": (".severity", "SeverityScorer"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _TABLE:
        mod_name, attr = _TABLE[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
