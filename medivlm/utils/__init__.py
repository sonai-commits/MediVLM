""" utility re-exports."""
from importlib import import_module
from typing import Any

from .config import MediVLMConfig, load_config
from .logger import get_logger

__all__ = [
    "MediVLMConfig",
    "load_config",
    "get_logger",
    "save_checkpoint",
    "load_checkpoint",
]


def __getattr__(name: str) -> Any:  
    if name in ("save_checkpoint", "load_checkpoint"):
        return getattr(import_module(".checkpoints", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
