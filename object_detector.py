from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn


@dataclass(frozen=True)
class DetectionResult:
    label: str
    score: float
    box: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ObjectDetector:
    """CPU-friendly detector for common COCO objects.

    This is used mainly to verify object presence and crop a target region
    before answering attribute questions like color.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.device = torch.device("cpu")
        self.threshold = threshold
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = list(weights.meta["categories"])
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image, target_object: str | None = None) -> list[DetectionResult]:
        tensor = TF.to_tensor(image).to(self.device)
        outputs = self.model([tensor])[0]

        detections: list[DetectionResult] = []
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            score_value = float(score.item())
            if score_value < self.threshold:
                continue
            label_name = self._label_name(int(label.item()))
            if target_object and not self._matches_target(label_name, target_object):
                continue
            detections.append(
                DetectionResult(
                    label=label_name,
                    score=score_value,
                    box=tuple(float(value.item()) for value in box),
                )
            )

        detections.sort(key=lambda item: (item.score, item.area), reverse=True)
        return detections

    def crop_target(self, image: Image.Image, target_object: str | None) -> Image.Image | None:
        detections = self.detect(image, target_object=target_object)
        if not detections:
            return None

        detection = detections[0]
        x1, y1, x2, y2 = detection.box
        width, height = image.size
        left = max(0, int(x1))
        top = max(0, int(y1))
        right = min(width, int(x2))
        bottom = min(height, int(y2))

        if right <= left or bottom <= top:
            return None

        return image.crop((left, top, right, bottom))

    def is_present(self, image: Image.Image, target_object: str | None) -> bool:
        return bool(self.detect(image, target_object=target_object))

    def _label_name(self, label_id: int) -> str:
        if 0 <= label_id < len(self.categories):
            return self.categories[label_id].lower()
        return str(label_id)

    @staticmethod
    def _matches_target(label_name: str, target_object: str) -> bool:
        label_name = label_name.lower()
        target_object = target_object.lower().strip()

        aliases = {
            "car": {"car", "police car", "taxi", "sedan", "auto"},
            "vehicle": {"car", "truck", "bus", "train", "motorcycle", "bicycle"},
            "police car": {"car", "vehicle"},
            "truck": {"truck"},
            "bus": {"bus"},
            "person": {"person"},
            "man": {"person"},
            "woman": {"person"},
            "bike": {"bicycle", "motorcycle"},
            "bicycle": {"bicycle"},
            "motorcycle": {"motorcycle"},
            "dog": {"dog"},
            "cat": {"cat"},
        }

        for key, values in aliases.items():
            if key in target_object:
                return label_name in values

        return target_object in label_name


@lru_cache(maxsize=1)
def load_default_detector() -> ObjectDetector:
    return ObjectDetector()
