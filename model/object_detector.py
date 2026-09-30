from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


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
    """Accurate object & landscape detector with strict presence verification and neat red mark localization."""

    # Extended aliases mapping user queries to COCO labels
    ALIASES = {
        "person": {
            "person", "people", "man", "woman", "boy", "girl", "child",
            "kid", "farmer", "worker", "someone", "anyone", "human", "guy", "lady"
        },
        "cat": {
            "cat", "cats", "kitten", "kitty", "leopard", "cheetah", "tiger",
            "lion", "feline", "jaguar", "panther", "cougar"
        },
        "dog": {"dog", "dogs", "puppy", "pup", "hound", "wolf"},
        "bird": {"bird", "birds", "eagle", "hawk", "duck", "pigeon", "parrot", "sparrow", "owl"},
        "horse": {"horse", "horses", "pony", "donkey", "mule"},
        "sheep": {"sheep", "lamb", "goat"},
        "cow": {"cow", "cows", "cattle", "bull", "ox"},
        "car": {"car", "cars", "automobile", "auto", "taxi", "sedan", "cab"},
        "vehicle": {"car", "truck", "bus", "train", "motorcycle", "bicycle", "vehicle"},
        "truck": {"truck", "trucks", "lorry", "pickup"},
        "bus": {"bus", "buses"},
        "bicycle": {"bicycle", "bicycles", "bike", "bikes", "cycle"},
        "motorcycle": {"motorcycle", "motorcycles", "motorbike", "scooter"},
        "backpack": {"backpack", "bag", "schoolbag"},
        "umbrella": {"umbrella", "parasol"},
        "handbag": {"handbag", "purse"},
        "bottle": {"bottle", "bottles"},
        "cup": {"cup", "mug", "glass"},
        "chair": {"chair", "chairs", "seat", "stool"},
        "couch": {"couch", "sofa"},
        "bed": {"bed"},
        "dining table": {"table", "desk"},
        "laptop": {"laptop", "computer", "pc"},
        "cell phone": {"phone", "cellphone", "mobile", "smartphone"},
        "book": {"book", "books"},
        "clock": {"clock", "watch"},
    }

    # Landscape & open-vocabulary concepts handled via CLIP spatial localization
    LANDSCAPE_CONCEPTS = {
        "mountain", "mountains", "hill", "hills", "valley",
        "river", "lake", "pond", "sea", "ocean", "water", "stream", "waterfall",
        "sky", "cloud", "clouds",
        "field", "grass", "meadow", "lawn", "crop", "crops", "rice field",
        "tree", "trees", "forest", "jungle", "woods", "plant", "plants",
        "road", "path", "street", "trail", "ground",
        "sun", "sunset", "sunrise",
        "building", "house", "hut", "roof", "fence", "wall",
        "hat", "shirt", "clothes", "shoe", "shoes"
    }

    def __init__(self, threshold: float = 0.55) -> None:
        self.device = torch.device("cpu")
        self.threshold = threshold

        # Prefer MobileNet-V3 FPN for fast CPU inference
        try:
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_fpn,
            )
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.categories = list(weights.meta["categories"])
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(self.device)
        except Exception:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.categories = list(weights.meta["categories"])
            self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image, target_object: str | None = None) -> list[DetectionResult]:
        orig_w, orig_h = image.size

        # Cap max dimension to 640px during detection for fast CPU inference
        max_dim = max(orig_w, orig_h)
        if max_dim > 640:
            scale = 640.0 / max_dim
            proc_w, proc_h = int(orig_w * scale), int(orig_h * scale)
            detect_image = image.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
        else:
            scale = 1.0
            detect_image = image

        tensor = TF.to_tensor(detect_image).to(self.device)
        outputs = self.model([tensor])[0]

        detections: list[DetectionResult] = []
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            score_value = float(score.item())
            if score_value < self.threshold:
                continue
            label_name = self._label_name(int(label.item()))
            if target_object and not self._matches_target(label_name, target_object):
                continue

            # Rescale box back to original image size
            x1, y1, x2, y2 = [float(v.item()) / scale for v in box]
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            detections.append(
                DetectionResult(
                    label=label_name,
                    score=score_value,
                    box=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda item: (item.score, item.area), reverse=True)
        return detections

    def extract_target_from_question(self, question: str) -> str | None:
        """Extract the queried object or landscape concept from the question."""
        q = question.lower().strip().rstrip("?").strip()

        patterns = [
            r"(?:is there|are there|can you see|do you see|find|locate|show|where is|where are|look for)\s+(?:any|a|an|the|some)?\s*([a-zA-Z\s]+?)(?:\s+in\s+|\s+on\s+|\s+at\s+|$)",
            r"(?:what color is|what colour is|what is the color of)\s+(?:the|a|an)?\s*([a-zA-Z\s]+)",
            r"(?:how many)\s+([a-zA-Z\s]+?)(?:\s+are|\s+is|\s+in|\s+on|$)",
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                concept = m.group(1).strip()
                concept = re.sub(r"\b(this|the|image|picture|photo)\b", "", concept).strip()
                if concept:
                    for canonical, syns in self.ALIASES.items():
                        if concept in syns or any(s in concept for s in syns):
                            return canonical
                    return concept

        words = set(re.findall(r"\b[a-zA-Z]+\b", q))
        for canonical, syns in self.ALIASES.items():
            if words.intersection(syns):
                return canonical

        for cat in self.LANDSCAPE_CONCEPTS:
            if cat in words or cat in q:
                return cat

        for cat in self.categories:
            cat_lower = cat.lower()
            if cat_lower in words or cat_lower in q:
                return cat_lower

        return None

    def _locate_with_clip(
        self,
        image: Image.Image,
        target: str,
        clip_model,
        threshold: float = 0.52,
    ) -> list[DetectionResult]:
        """Locate open-vocabulary/landscape features using global verification and neat spatial region scoring."""
        if clip_model is None:
            return []

        w, h = image.size

        # 1. Global image presence verification
        global_prompts = [
            f"a photo of a {target}",
            f"a photo with no {target} present",
            "a photo of something else entirely",
        ]
        try:
            global_ranked = clip_model.rank_answers(image, global_prompts)
            global_target_score = next((r.score for r in global_ranked if target in r.answer.lower()), 0.0)
            # If target presence on the full image is weak or outranked by negative prompts, reject!
            if global_target_score < 0.40 or not global_ranked[0].answer.lower().startswith(f"a photo of a {target}"):
                return []
        except Exception:
            return []

        # 2. Neat multi-scale candidate spatial regions (3x3 grid & quadrants)
        candidate_boxes = [
            # 3x3 fine grid cells for neat localized boxes
            (0, 0, int(w * 0.40), int(h * 0.40)),
            (int(w * 0.30), 0, int(w * 0.70), int(h * 0.40)),
            (int(w * 0.60), 0, w, int(h * 0.40)),
            (0, int(h * 0.30), int(w * 0.45), int(h * 0.70)),
            (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)),
            (int(w * 0.55), int(h * 0.30), w, int(h * 0.70)),
            (0, int(h * 0.60), int(w * 0.45), h),
            (int(w * 0.30), int(h * 0.60), int(w * 0.70), h),
            (int(w * 0.55), int(h * 0.60), w, h),
            # Combined quadrants for larger landscape features
            (0, 0, w, int(h * 0.50)),
            (0, int(h * 0.50), w, h),
            (0, 0, int(w * 0.60), int(h * 0.60)),
            (int(w * 0.40), 0, w, int(h * 0.60)),
        ]

        scored_boxes = []
        for box in candidate_boxes:
            crop = image.crop(box)
            try:
                ranked = clip_model.rank_answers(crop, global_prompts)
                target_score = next((r.score for r in ranked if target in r.answer.lower()), 0.0)
                if target_score >= threshold and ranked[0].answer.lower().startswith(f"a photo of a {target}"):
                    # Prefer tighter boxes when score is comparable
                    area_penalty = 1.0 - (0.15 * ((box[2] - box[0]) * (box[3] - box[1])) / (w * h))
                    scored_boxes.append((target_score * area_penalty, target_score, box))
            except Exception:
                continue

        if not scored_boxes:
            return []

        scored_boxes.sort(key=lambda item: item[0], reverse=True)
        _, best_score, best_box = scored_boxes[0]
        return [DetectionResult(label=target, score=best_score, box=best_box)]

    def draw_red_marks(
        self,
        image: Image.Image,
        detections: list[DetectionResult],
        box_color: str = "#FF1E1E",
        line_width: int = 4,
    ) -> Image.Image:
        """Draw neat, prominent red bounding boxes, corner accents, and labels on detected objects."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for det in detections:
            x1, y1, x2, y2 = det.box

            # 1. Main bold red rectangle
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

            # 2. Corner brackets for modern, clean visual emphasis
            corner_len = min(22.0, (x2 - x1) / 3, (y2 - y1) / 3)
            # Top-left
            draw.line([(x1, y1), (x1 + corner_len, y1)], fill=box_color, width=line_width + 2)
            draw.line([(x1, y1), (x1, y1 + corner_len)], fill=box_color, width=line_width + 2)
            # Top-right
            draw.line([(x2, y1), (x2 - corner_len, y1)], fill=box_color, width=line_width + 2)
            draw.line([(x2, y1), (x2, y1 + corner_len)], fill=box_color, width=line_width + 2)
            # Bottom-left
            draw.line([(x1, y2), (x1 + corner_len, y2)], fill=box_color, width=line_width + 2)
            draw.line([(x1, y2), (x1, y2 - corner_len)], fill=box_color, width=line_width + 2)
            # Bottom-right
            draw.line([(x2, y2), (x2 - corner_len, y2)], fill=box_color, width=line_width + 2)
            draw.line([(x2, y2), (x2, y2 - corner_len)], fill=box_color, width=line_width + 2)

            # 3. Red badge label with white text
            label_text = f" {det.label.upper()} {det.score:.0%} "
            badge_h = 22
            badge_w = len(label_text) * 8 + 4

            badge_y1 = max(0.0, y1 - badge_h)
            badge_y2 = badge_y1 + badge_h
            badge_x1 = x1
            badge_x2 = min(float(image.width), x1 + badge_w)

            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=box_color)
            draw.text((badge_x1 + 3, badge_y1 + 4), label_text, fill="white")

        return annotated

    def locate_and_mark(
        self,
        image: Image.Image,
        question: str,
        clip_model=None,
    ) -> tuple[Image.Image | None, list[DetectionResult], str | None]:
        """Locate queried object or landscape concept and render a red mark annotation."""
        target = self.extract_target_from_question(question)

        # 1. If target is in landscape/open-vocabulary concepts, prioritize CLIP spatial localization
        if target and (target in self.LANDSCAPE_CONCEPTS or not any(self._matches_target(cat, target) for cat in self.categories)):
            if clip_model is not None:
                clip_dets = self._locate_with_clip(image, target, clip_model)
                if clip_dets:
                    marked = self.draw_red_marks(image, clip_dets)
                    return marked, clip_dets, target
            return None, [], target

        # 2. Try standard COCO object detector
        detections = []
        if target:
            raw_dets = self.detect(image, target_object=target)
            detections = [d for d in raw_dets if self._matches_target(d.label, target)]

        # 3. If COCO detector found nothing, try CLIP spatial localization as backup
        if not detections and target and clip_model is not None:
            clip_dets = self._locate_with_clip(image, target, clip_model)
            if clip_dets:
                marked = self.draw_red_marks(image, clip_dets)
                return marked, clip_dets, target

        # If nothing found with high confidence, return empty - NEVER show unrelated objects!
        if not detections:
            return None, [], target

        marked_image = self.draw_red_marks(image, detections)
        return marked_image, detections, target

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

    def _matches_target(self, label_name: str, target_object: str) -> bool:
        label_name = label_name.lower().strip()
        target_object = target_object.lower().strip()

        if target_object == label_name:
            return True

        for key, values in self.ALIASES.items():
            if key == target_object or target_object in values:
                if label_name == key or label_name in values:
                    return True

        return target_object in label_name or label_name in target_object


@lru_cache(maxsize=1)
def load_default_detector() -> ObjectDetector:
    return ObjectDetector()
