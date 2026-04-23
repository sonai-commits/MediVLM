from importlib import import_module
from typing import Any

__all__ = [
    "FasterRCNNDetector",
    "DetectorOutput",
    "CLIPImageEncoder",
    "ClinicalBertTextEncoder",
    "ProjectionHead",
    "contrastive_loss",
    "CrossAttentionFusion",
    "GPT2ReportDecoder",
    "BioT5PseudoReporter",
    "MediVLM",
]

_TABLE = {
    "FasterRCNNDetector":     (".detector", "FasterRCNNDetector"),
    "DetectorOutput":         (".detector", "DetectorOutput"),
    "CLIPImageEncoder":       (".image_encoder", "CLIPImageEncoder"),
    "ClinicalBertTextEncoder":(".text_encoder", "ClinicalBertTextEncoder"),
    "ProjectionHead":         (".projection", "ProjectionHead"),
    "contrastive_loss":       (".projection", "contrastive_loss"),
    "CrossAttentionFusion":   (".fusion", "CrossAttentionFusion"),
    "GPT2ReportDecoder":      (".decoder", "GPT2ReportDecoder"),
    "BioT5PseudoReporter":    (".pseudo_labeler", "BioT5PseudoReporter"),
    "MediVLM":                (".medivlm", "MediVLM"),
}


def __getattr__(name: str) -> Any:
    if name in _TABLE:
        mod_name, attr = _TABLE[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
