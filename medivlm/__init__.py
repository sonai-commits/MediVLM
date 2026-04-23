"""MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images.
Goswami, Subedi, Chakraborty (Findings of EMNLP 2025).
https://aclanthology.org/2025.findings-emnlp.544/
"""
from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__all__ = ["MediVLM", "MediVLMConfig", "load_config"]


def __getattr__(name: str) -> Any:  # PEP 562
    if name == "MediVLM":
        return import_module(".models.medivlm", __name__).MediVLM
    if name == "MediVLMConfig":
        return import_module(".utils.config", __name__).MediVLMConfig
    if name == "load_config":
        return import_module(".utils.config", __name__).load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
