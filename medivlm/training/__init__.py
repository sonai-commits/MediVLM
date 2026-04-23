"""training-loop re-exports."""
from importlib import import_module
from typing import Any

__all__ = ["Trainer", "UnsupervisedTrainer", "generate_pseudo_reports", "combined_loss"]

_TABLE = {
    "Trainer":                 (".trainer", "Trainer"),
    "UnsupervisedTrainer":     (".unsupervised", "UnsupervisedTrainer"),
    "generate_pseudo_reports": (".unsupervised", "generate_pseudo_reports"),
    "combined_loss":           (".losses", "combined_loss"),
}


def __getattr__(name: str) -> Any:
    if name in _TABLE:
        mod_name, attr = _TABLE[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
