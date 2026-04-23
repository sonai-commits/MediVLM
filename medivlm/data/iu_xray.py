"""IU X-Ray dataset (Demner-Fushman et al. 2016).

use the commonly distributed train/val/test split
(5.2K/0.7K/1.5K images, 2.8K/0.4K/0.8K reports, Table 5). Images come
as frontal + lateral views; per Chen et al. 2020 we pick up to two
views per report. The expected annotation JSON follows the widely used
format from R2Gen:

    {
        "train": [
            {"id": "CXR1000_1", "image_path": ["1000_IM-0001-4001.png",
                                               "1000_IM-0001-3001.png"],
             "report": "The heart is normal in size ..."},
            ...
        ],
        "val":  [...],
        "test": [...]
    }
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_image_transform, clean_report


class IuXrayDataset(Dataset):
    name = "iu_xray"

    def __init__(
        self,
        root: str,
        ann_file: str = "annotations.json",
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        max_views: int = 2,
        max_sentences: int = 4,
    ) -> None:
        self.root = Path(root)
        ann_path = self.root / ann_file if not os.path.isabs(ann_file) else Path(ann_file)
        with open(ann_path, "r") as f:
            ann = json.load(f)
        if split not in ann:
            raise KeyError(f"split '{split}' missing. Available: {list(ann.keys())}")
        self.samples = ann[split]
        self.split = split
        self.transform = transform or build_image_transform(image_size, train=(split == "train"))
        self.max_views = max_views
        self.max_sentences = max_sentences
        self.image_dir = self.root / "images"
        if not self.image_dir.exists():
            # distributions keep images alongside annotations.json
            self.image_dir = self.root

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, relpath: str) -> torch.Tensor:
        path = self.image_dir / relpath
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        paths: Sequence[str] = s.get("image_path", [s.get("image")])
        paths = list(paths)[: self.max_views] if self.max_views > 0 else list(paths)
        images = torch.stack([self._load_image(p) for p in paths], dim=0)   # (V, 3, H, W)
        # For MediVLM we feed one image at a time per instance. callers can decide whether to average per-view features or randomly pick one.
        image = images[0]
        report = clean_report(s.get("report", ""), max_sentences=self.max_sentences)
        return {
            "id": s.get("id", str(idx)),
            "image": image,
            "views": images,                 # keep all views for optional multi-view fusion
            "report": report,
        }
