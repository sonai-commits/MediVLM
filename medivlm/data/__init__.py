"""dataset re-exports"""
from importlib import import_module
from typing import Any

__all__ = [
    "build_image_transform",
    "clean_report",
    "IuXrayDataset",
    "CasiaCxrDataset",
    "MimicCxrDataset",
    "build_dataset",
    "build_dataloader",
]

_TABLE = {
    "build_image_transform": (".transforms", "build_image_transform"),
    "clean_report":          (".transforms", "clean_report"),
    "IuXrayDataset":         (".iu_xray", "IuXrayDataset"),
    "CasiaCxrDataset":       (".casia_cxr", "CasiaCxrDataset"),
    "MimicCxrDataset":       (".mimic_cxr", "MimicCxrDataset"),
    "build_dataset":         (".factory", "build_dataset"),
    "build_dataloader":      (".factory", "build_dataloader"),
}


def __getattr__(name: str) -> Any:
    if name in _TABLE:
        mod_name, attr = _TABLE[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
