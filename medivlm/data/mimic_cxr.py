"""MIMIC-CXR dataset (Johnson et al. 2019) - largest CXR-report corpus.

 369K train / 3.0K val / 5.2K test images with
222.8K / 1.8K / 3.3K reports respectively.

Expected annotation format (R2Gen / memory-driven transformer style):
    {
        "train": [
            {"id": "abc123", "image_path": ["path/to/frontal.jpg"],
             "report": "Findings: ..."},
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
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_image_transform, clean_report


class MimicCxrDataset(Dataset):
    name = "mimic_cxr"

    def __init__(
        self,
        root: str,
        ann_file: str = "annotations.json",
        split: str = "train",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        max_sentences: int = 4,
    ) -> None:
        self.root = Path(root)
        ann_path = self.root / ann_file if not os.path.isabs(ann_file) else Path(ann_file)
        with open(ann_path, "r") as f:
            ann = json.load(f)
        self.samples: List[Dict] = ann.get(split, [])
        self.split = split
        self.transform = transform or build_image_transform(image_size, train=(split == "train"))
        self.max_sentences = max_sentences
        self.image_dir = self.root / "images"
        if not self.image_dir.exists():
            self.image_dir = self.root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        paths = s.get("image_path", [s.get("image")])
        if isinstance(paths, list):
            path = paths[0]
        else:
            path = paths
        img = Image.open(self.image_dir / path).convert("RGB")
        image = self.transform(img)
        report = clean_report(s.get("report", ""), max_sentences=self.max_sentences)
        return {
            "id": s.get("id", str(idx)),
            "image": image,
            "report": report,
        }
