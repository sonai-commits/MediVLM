"""Image transforms + report cleaning used by each dataset.
"""
from __future__ import annotations

import re
from typing import Any, Callable, List, Tuple


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int = 224, train: bool = True) -> Callable[[Any], Any]:
    """Returns a torchvision transform that outputs a tensor in [0, 1].

    Images are kept in [0, 1] because the detector expects that range.
    The image encoder re-normalises internally with CLIP mean/std.
    """
    from torchvision import transforms  # local import - avoid torch dep for clean_report
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.0),  # radiographs should keep L/R
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


_WS = re.compile(r"\s+")
_DEID = re.compile(r"\b(?:XXXX|___+|___)\b", re.IGNORECASE)
_TRAILING = re.compile(r"^(?:findings\s*:|impressions?\s*:|findings/impression\s*:)", re.IGNORECASE)


def clean_report(text: str, max_sentences: int = 4) -> str:
    """Normalise MIMIC-style / IU-Xray radiology reports.

    * strips whitespace and "Findings:" / "Impression:" prefixes
    * collapses deidentification placeholders ``XXXX`` / ``___`` to a generic token
    * trims to ``max_sentences`` sentences when positive.
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = _TRAILING.sub("", text).strip()
    text = _DEID.sub("xxxx", text)
    text = _WS.sub(" ", text)

    if max_sentences and max_sentences > 0:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s for s in sentences if s.strip()][: max_sentences]
        text = " ".join(sentences)
    return text.strip()
