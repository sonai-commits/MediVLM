"""CASIA-CXR dataset (Metmer and Yang 2024) - French radiology reports.
paper uses the *Pneumonia* category (~1.8K train / 0.1K val / 0.1K
test, Table 5). Reports are in French. The annotation schema mirrors
the IU X-Ray one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_image_transform, clean_report


class CasiaCxrDataset(Dataset):
    name = "casia_cxr"

    def __init__(
        self,
        root: str,
        ann_file: str = "annotations.json",
        split: str = "train",
        category: str = "Pneumonia",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        max_sentences: int = 4,
    ) -> None:
        self.root = Path(root)
        ann_path = self.root / ann_file if not os.path.isabs(ann_file) else Path(ann_file)
        with open(ann_path, "r") as f:
            ann = json.load(f)
        self.samples = []
        split_entries = ann.get(split, [])
        for s in split_entries:
            if category and s.get("category", category) != category:
                continue
            self.samples.append(s)
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
        path = s.get("image_path", s.get("image"))
        if isinstance(path, list):
            path = path[0]
        img = Image.open(self.image_dir / path).convert("RGB")
        image = self.transform(img)
        report = clean_report(s.get("report", ""), max_sentences=self.max_sentences)
        return {
            "id": s.get("id", str(idx)),
            "image": image,
            "report": report,
        }
