"""Faster R-CNN anatomical-region detector.

a Faster R-CNN detector with a ResNet-34 backbone,
pretrained on MS-CXR, produces the bounding boxes of anatomical regions.
We keep the detector frozen during MediVLM training. During forward, we
select the top-p confident region proposals (p=8) and return
their boxes,semantic labels and cropped image patches.

Implementation notes
--------------------
* torchvision does not ship a Faster R-CNN with ResNet-34 FPN out of the
  box, so we build one ourselves using a ResNet-34 backbone + FPN.
* With ``weights_path`` provided, the full detector state dict  is loaded. Otherwise the ResNet-34
  ImageNet weights are used as an initialization — users are expected
  to point to their MS-CXR fine-tuned checkpoint for proper results.
* The detector runs on the *original* image (not the CLIP-preprocessed
  one) so box coordinates correspond to pixel space.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

# MS-CXR anatomical region labels (29 classes), ordered per the MS-CXR release.
# See Boecking et al. 2022 - https://huggingface.co/datasets/microsoft/MS-CXR
MSCXR_CLASSES: Tuple[str, ...] = (
    "background",
    "right lung", "right upper lung zone", "right mid lung zone",
    "right lower lung zone", "right hilar structures",
    "right apical zone", "right costophrenic angle", "right hemidiaphragm",
    "left lung", "left upper lung zone", "left mid lung zone",
    "left lower lung zone", "left hilar structures",
    "left apical zone", "left costophrenic angle", "left hemidiaphragm",
    "trachea", "spine", "right clavicle", "left clavicle",
    "aortic arch", "mediastinum", "upper mediastinum",
    "svc", "cardiac silhouette", "cavoatrial junction",
    "right atrium", "carina", "abdomen",
)


@dataclass
class DetectorOutput:
    """One entry per image in the batch."""
    boxes: torch.Tensor          # (K, 4)  xyxy pixel coords
    labels: torch.Tensor         # (K,)   integer class ids
    label_names: List[str]       # (K,)   anatomical region strings
    scores: torch.Tensor         # (K,)
    patches: torch.Tensor        # (K, 3, patch_size, patch_size) cropped and resized
    norm_coords: torch.Tensor    # (K, 4) normalised (x, y, w, h) in [0, 1]


def _resnet34_fpn_backbone(
    pretrained: bool = True,
    trainable_layers: int = 3,
    out_channels: int = 256,
) -> BackboneWithFPN:
    """Build a ResNet-34 + FPN backbone (torchvision only ships the 50/101 variants with FPN)."""
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = resnet34(weights=weights)
    # Freeze the stem + early layers depending on ``trainable_layers``.
    # Named layers in a resnet: layer1..layer4 (and conv1, bn1 as stem).
    layer_names = ["layer4", "layer3", "layer2", "layer1"][: max(0, trainable_layers)]
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(any(name.startswith(ln) for ln in layer_names))

    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    # ResNet-34 channel dims at each stage
    in_channels_list = [64, 128, 256, 512]
    return BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )


class FasterRCNNDetector(nn.Module):
    """Wraps torchvision's FasterRCNN with a ResNet-34 FPN backbone."""

    def __init__(
        self,
        num_anatomical_classes: int = len(MSCXR_CLASSES),
        weights_path: Optional[str] = None,
        min_size: int = 224,
        max_size: int = 224,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        top_p_patches: int = 8,
        patch_size: int = 28,
        freeze: bool = True,
        class_names: Sequence[str] = MSCXR_CLASSES,
    ) -> None:
        super().__init__()
        backbone = _resnet34_fpn_backbone(pretrained=True, trainable_layers=3)
        anchor_generator = AnchorGenerator(
            sizes=tuple((size,) for size in (16, 32, 64, 128, 256)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_anatomical_classes,
            min_size=min_size,
            max_size=max_size,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
        )
        self.top_p_patches = top_p_patches
        self.patch_size = patch_size
        self.class_names = tuple(class_names)

        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.detector.load_state_dict(state, strict=False)
            if missing or unexpected:
                #for partial or DDP-wrapped checkpoints.
                import warnings
                warnings.warn(
                    f"Detector loaded with missing={len(missing)} unexpected={len(unexpected)} keys"
                )

        self._frozen = bool(freeze)
        if freeze:
            self.requires_grad_(False)
            self.eval()

    # torchvision FasterRCNN needs targets when in training mode.
    # Keep the detector pinned to eval when frozen, regardless of the
    # parent module's train()/eval() calls.
    def train(self, mode: bool = True):  # type: ignore[override]
        if getattr(self, "_frozen", False):
            return super().train(False)
        return super().train(mode)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> List[DetectorOutput]:
        """images: (B, 3, H, W) float tensor normalised to [0, 1]."""
        assert images.ndim == 4, "detector expects (B, 3, H, W)"
        # torchvision FasterRCNN wants a list of 3xHxW tensors (unbatched).
        image_list = [images[i] for i in range(images.size(0))]

        grad_ctx = torch.no_grad() if not any(p.requires_grad for p in self.detector.parameters()) \
                   else torch.enable_grad()
        with grad_ctx:
            detections = self.detector(image_list)

        outputs: List[DetectorOutput] = []
        for det, img in zip(detections, image_list):
            outputs.append(self._post_process(det, img))
        return outputs

    def _post_process(
        self, det: dict, image: torch.Tensor
    ) -> DetectorOutput:
        boxes = det["boxes"].detach()          # (K, 4) xyxy
        scores = det["scores"].detach()        # (K,)
        labels = det["labels"].detach().long() # (K,)

        if boxes.numel() == 0:
            return self._fallback_output(image)

        top_k = min(self.top_p_patches, boxes.size(0))
        top_idx = torch.topk(scores, top_k, largest=True).indices
        boxes = boxes[top_idx]
        scores = scores[top_idx]
        labels = labels[top_idx]

        patches = self._crop_and_resize(image, boxes)
        norm_coords = self._normalise_coords(boxes, image.shape[-2:])
        label_names = [self.class_names[i] if i < len(self.class_names) else f"cls_{int(i)}"
                       for i in labels.tolist()]

        # Pad to top_p_patches so downstream shapes are deterministic.
        patches, norm_coords, labels, scores, label_names = self._pad_to_top_p(
            patches, norm_coords, labels, scores, label_names, image
        )
        return DetectorOutput(
            boxes=boxes,
            labels=labels,
            label_names=label_names,
            scores=scores,
            patches=patches,
            norm_coords=norm_coords,
        )

    # ------------------------------------------------------------------
    def _fallback_output(self, image: torch.Tensor) -> DetectorOutput:
        """If the detector returns no regions, feed the whole image as one patch.

        This prevents the downstream pipeline from crashing on very easy
        or very out-of-distribution images.
        """
        H, W = image.shape[-2:]
        full = torch.tensor([[0.0, 0.0, W, H]], device=image.device)
        patch = F.interpolate(
            image.unsqueeze(0), size=(self.patch_size, self.patch_size),
            mode="bilinear", align_corners=False,
        )
        patches = patch.expand(self.top_p_patches, -1, -1, -1).clone()
        norm_coords = self._normalise_coords(full.expand(self.top_p_patches, -1), (H, W))
        labels = torch.zeros(self.top_p_patches, dtype=torch.long, device=image.device)
        scores = torch.zeros(self.top_p_patches, device=image.device)
        return DetectorOutput(
            boxes=full.expand(self.top_p_patches, -1),
            labels=labels, label_names=["whole image"] * self.top_p_patches,
            scores=scores, patches=patches, norm_coords=norm_coords,
        )

    def _pad_to_top_p(self, patches, norm_coords, labels, scores, label_names, image):
        if patches.size(0) == self.top_p_patches:
            return patches, norm_coords, labels, scores, label_names
        pad = self.top_p_patches - patches.size(0)
        pad_patch = F.interpolate(
            image.unsqueeze(0), size=(self.patch_size, self.patch_size),
            mode="bilinear", align_corners=False,
        ).expand(pad, -1, -1, -1)
        H, W = image.shape[-2:]
        pad_box = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=image.device).expand(pad, -1)
        pad_labels = torch.zeros(pad, dtype=torch.long, device=image.device)
        pad_scores = torch.zeros(pad, device=image.device)
        patches = torch.cat([patches, pad_patch], dim=0)
        norm_coords = torch.cat([norm_coords, pad_box], dim=0)
        labels = torch.cat([labels, pad_labels], dim=0)
        scores = torch.cat([scores, pad_scores], dim=0)
        label_names = list(label_names) + ["padding"] * pad
        return patches, norm_coords, labels, scores, label_names

    def _crop_and_resize(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Crop each bounding box from ``image`` and resize to (patch_size, patch_size)."""
        H, W = image.shape[-2:]
        patches = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(W, int(x2)); y2 = min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                crop = image.unsqueeze(0)
            else:
                crop = image[:, y1:y2, x1:x2].unsqueeze(0)
            crop = F.interpolate(
                crop, size=(self.patch_size, self.patch_size),
                mode="bilinear", align_corners=False,
            )
            patches.append(crop.squeeze(0))
        return torch.stack(patches, dim=0)

    def _normalise_coords(
        self, boxes: torch.Tensor, image_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """Return (x, y, w, h) normalised to [0, 1] from xyxy boxes."""
        H, W = image_hw
        x1, y1, x2, y2 = boxes.unbind(-1)
        xn = x1 / max(W, 1)
        yn = y1 / max(H, 1)
        wn = (x2 - x1).clamp(min=0) / max(W, 1)
        hn = (y2 - y1).clamp(min=0) / max(H, 1)
        return torch.stack([xn, yn, wn, hn], dim=-1).clamp(0.0, 1.0)
