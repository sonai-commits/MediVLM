"""Configuration dataclasses and YAML loader for MediVLM.

The defaults :
    image size = 224x224, patch size = 28x28, top-p=8 patches,
    GPT-2 decoder fine-tuned with lr=2e-5, batch=32, 30-50 epochs, AdamW.
    lambda1=1, lambda2=0.7, tau=0.07, max tokens = 77 (4 sentences).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DetectorConfig:
    """Faster R-CNN with ResNet-34 backbone, pretrained on MS-CXR."""
    backbone: str = "resnet34"
    num_anatomical_classes: int = 29  # MS-CXR has 29 anatomical structure labels
    weights_path: Optional[str] = None  # local path to MS-CXR finetuned weights
    min_size: int = 224
    max_size: int = 224
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    top_p_patches: int = 8  #top 8 confident patches
    patch_size: int = 28    #28x28 patches that get resized to CLIP input
    freeze: bool = True


@dataclass
class ImageEncoderConfig:
    """CLIP-ViT-L/14 as the visual encoder."""
    model_name: str = "openai/clip-vit-large-patch14"
    freeze: bool = True
    pos_mlp_hidden: int = 256   # MLP producing position embedding for (x,y,w,h)
    image_size: int = 224       # model expects 224x224 patches


@dataclass
class TextEncoderConfig:
    """ClinicalBERT as the text encoder (frozen)."""
    model_name: str = "medicalai/ClinicalBERT"  # ref: emilyalsentzer/Bio_ClinicalBERT
    freeze: bool = True
    max_length: int = 128


@dataclass
class ProjectionConfig:
    """Projection heads for visual and text features (contrastive alignment)."""
    proj_dim: int = 512
    temperature: float = 0.07
    dropout: float = 0.1


@dataclass
class FusionConfig:
    """Multi-head cross-attention fusion of aligned modalities."""
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    """Fine-tuned GPT-2 decoder (last 4 transformer blocks trainable per Figure 2)."""
    model_name: str = "gpt2"  # 12 layers, 768-d hidden, 50257 vocab
    trainable_blocks: int = 4
    max_length: int = 77          # ~4 sentences (Table 7)
    beam_size: int = 4
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0


@dataclass
class PseudoLabelerConfig:
    """BioT5 decoder fine-tuned with ClinicalBERT encoder output to produce pseudo-reports."""
    biot5_model_name: str = "QizhiPei/biot5-base"
    max_length: int = 77
    num_beams: int = 4


@dataclass
class SeverityConfig:
    """TF-IDF based severity scoring (Section 3.4 + A.1)."""
    seed_terms: Optional[list] = None  # supplied via YAML; falls back to defaults inside severity module
    use_nltk_sentiment: bool = True
    normalize_across_corpus: bool = True


@dataclass
class DataConfig:
    dataset: str = "iu_xray"  # iu_xray | casia_cxr | mimic_cxr
    root: str = "data/iu_xray"
    ann_file: str = "annotations.json"
    image_size: int = 224
    num_workers: int = 4
    batch_size: int = 32


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 2e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    lambda_ce: float = 1.0        
    lambda_contrast: float = 0.7  
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    eval_every: int = 1
    save_every: int = 1
    seed: int = 42
    mixed_precision: bool = True
    log_every: int = 50
    output_dir: str = "outputs"


@dataclass
class MediVLMConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    pseudo_labeler: PseudoLabelerConfig = field(default_factory=PseudoLabelerConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _update_dataclass(dc: Any, updates: Dict[str, Any]) -> Any:
    """Recursively update a dataclass instance from a nested dict."""
    if updates is None:
        return dc
    for key, value in updates.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> MediVLMConfig:
    """Load a YAML config on top of the defaults; apply optional overrides."""
    cfg = MediVLMConfig()
    if path is not None:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        _update_dataclass(cfg, user_cfg)
    if overrides:
        _update_dataclass(cfg, overrides)
    return cfg
