""" builds dataset and dataloader from a config."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.config import DataConfig
from .iu_xray import IuXrayDataset
from .casia_cxr import CasiaCxrDataset
from .mimic_cxr import MimicCxrDataset

_REGISTRY = {
    "iu_xray": IuXrayDataset,
    "casia_cxr": CasiaCxrDataset,
    "mimic_cxr": MimicCxrDataset,
}


def build_dataset(cfg: DataConfig, split: str) -> Dataset:
    key = cfg.dataset.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {cfg.dataset}. Known: {list(_REGISTRY)}")
    cls = _REGISTRY[key]
    return cls(
        root=cfg.root,
        ann_file=cfg.ann_file,
        split=split,
        image_size=cfg.image_size,
    )


def collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    reports = [b["report"] for b in batch]
    ids = [b["id"] for b in batch]
    return {"images": images, "reports": reports, "ids": ids}


def build_dataloader(cfg: DataConfig, split: str, shuffle: bool = None) -> DataLoader:
    ds = build_dataset(cfg, split)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
