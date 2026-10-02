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
    """High-accuracy object & landscape detector with discriminative contrastive grounding and neat red spotlighting."""

    # Extended aliases mapping user queries to COCO labels
    ALIASES = {
        "person": {
            "person", "people", "man", "woman", "boy", "girl", "child",
            "kid", "farmer", "worker", "someone", "anyone", "human", "guy",
            "lady", "player", "runner", "driver", "passenger", "pedestrian", "officer", "police"
        },
        "cat": {
            "cat", "cats", "kitten", "kitty", "leopard", "cheetah", "tiger",
            "lion", "feline", "jaguar", "panther", "cougar"
        },
        "dog": {"dog", "dogs", "puppy", "pup", "hound", "wolf", "canine"},
        "animal": {
            "animal", "animals", "wildlife", "beast", "pet", "creature",
            "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird"
        },
        "bird": {"bird", "birds", "eagle", "hawk", "duck", "pigeon", "parrot", "sparrow", "owl", "swan", "goose"},
        "horse": {"horse", "horses", "pony", "donkey", "mule", "stallion"},
        "sheep": {"sheep", "lamb", "goat", "ram"},
        "cow": {"cow", "cows", "cattle", "bull", "ox", "calf"},
        "elephant": {"elephant", "elephants"},
        "bear": {"bear", "bears"},
        "zebra": {"zebra", "zebras"},
        "giraffe": {"giraffe", "giraffes"},
        "car": {"car", "cars", "automobile", "auto", "taxi", "sedan", "cab", "suv", "police car", "vehicle"},
        "vehicle": {"car", "truck", "bus", "train", "motorcycle", "bicycle", "bike", "vehicle", "vehicles", "airplane", "boat"},
        "truck": {"truck", "trucks", "lorry", "pickup", "van"},
        "bus": {"bus", "buses"},
        "bicycle": {"bicycle", "bicycles", "bike", "bikes", "cycle", "cycling"},
        "motorcycle": {"motorcycle", "motorcycles", "motorbike", "scooter", "moped"},
        "airplane": {"airplane", "aeroplane", "plane", "aircraft", "jet"},
        "boat": {"boat", "boats", "ship", "yacht", "vessel", "canoe", "kayak", "ferry"},
        "backpack": {"backpack", "bag", "schoolbag", "rucksack"},
        "umbrella": {"umbrella", "parasol"},
        "handbag": {"handbag", "purse", "tote bag"},
        "bottle": {"bottle", "bottles"},
        "cup": {"cup", "mug", "glass", "tumbler"},
        "chair": {"chair", "chairs", "seat", "stool", "armchair"},
        "couch": {"couch", "sofa", "settee"},
        "bed": {"bed", "beds", "mattress"},
        "dining table": {"table", "desk", "dining table"},
        "laptop": {"laptop", "computer", "pc", "notebook"},
        "cell phone": {"phone", "cellphone", "mobile", "smartphone"},
        "book": {"book", "books"},
        "clock": {"clock", "watch", "timer"},
        "food": {
            "food", "fruit", "snack", "meal", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "pizza", "donut", "cake", "hot dog"
        },
    }

    # Landscape & open-vocabulary concepts handled via CLIP discriminative spatial localization
    LANDSCAPE_CONCEPTS = {
        "mountain", "mountains", "hill", "hills", "ridge", "peak", "valley",
        "river", "lake", "pond", "sea", "ocean", "water", "stream", "waterfall",
        "sky", "cloud", "clouds",
        "field", "grass", "meadow", "lawn", "crop", "crops", "rice field",
        "tree", "trees", "forest", "jungle", "woods", "plant", "plants",
        "road", "path", "street", "trail", "ground", "rock", "rocks",
        "sun", "sunset", "sunrise",
        "building", "house", "hut", "roof", "fence", "wall", "bridge",
        "hat", "shirt", "clothes", "shoe", "shoes"
    }

    # Semantic negative distractor suites to prevent false grounding
    LANDSCAPE_PROMPT_SUITES = {
        "mountain": [
            "a photo of mountains, forested mountain slopes, hills, or mountain ridge",
            "a photo of a waterfall or rushing cascading water",
            "a photo of a calm lake, pond, or river water",
            "a photo of the open sky or white clouds",
        ],
        "mountains": [
            "a photo of mountains, forested mountain slopes, hills, or mountain ridge",
            "a photo of a waterfall or rushing cascading water",
            "a photo of a calm lake, pond, or river water",
            "a photo of the open sky or white clouds",
        ],
        "waterfall": [
            "a photo of a waterfall, cascade, or rushing water falling over rocks",
            "a photo of a calm flat lake, pond, or river",
            "a photo of mountains or hills",
            "a photo of the open sky",
        ],
        "lake": [
            "a photo of a calm lake, pond, or body of water",
            "a photo of a rushing waterfall or stream",
            "a photo of high mountains or hills",
            "a photo of tall green trees or forest",
            "a photo of the open sky",
        ],
        "river": [
            "a photo of a flowing river, creek, or stream",
            "a photo of a waterfall or cascade",
            "a photo of high mountains or hills",
            "a photo of tall green trees or forest",
            "a photo of the open sky",
        ],
        "water": [
            "a photo of water, lake, river, or pond",
            "a photo of tall green trees or forest",
            "a photo of high mountain peaks or ridge",
            "a photo of the open sky",
            "a photo of dry land, rocks, or soil",
        ],
        "tree": [
            "a photo of tall green pine trees or forest",
            "a photo of high mountain peaks or rocky ridge",
            "a photo of a lake, river, or water",
            "a photo of the open sky or clouds",
            "a photo of rocks or ground",
        ],
        "trees": [
            "a photo of tall green pine trees or forest",
            "a photo of high mountain peaks or rocky ridge",
            "a photo of a lake, river, or water",
            "a photo of the open sky or clouds",
            "a photo of rocks or ground",
        ],
        "forest": [
            "a photo of a dense green forest, woods, or many trees",
            "a photo of high mountain peaks or rocky ridge",
            "a photo of a lake, river, or water",
            "a photo of the open sky",
        ],
        "sky": [
            "a photo of the open blue sky or white clouds",
            "a photo of high mountain peaks or ridge",
            "a photo of tall green trees or forest",
            "a photo of a lake, river, or water",
            "a photo of the ground or rocks",
        ],
        "cloud": [
            "a photo of white puffy clouds in the sky",
            "a photo of clear blue sky with no clouds",
            "a photo of high mountains or hills",
            "a photo of trees or forest",
            "a photo of water or ground",
        ],
        "clouds": [
            "a photo of white puffy clouds in the sky",
            "a photo of clear blue sky with no clouds",
            "a photo of high mountains or hills",
            "a photo of trees or forest",
            "a photo of water or ground",
        ],
        "field": [
            "a photo of an open green field, meadow, or grassland",
            "a photo of dense forest or trees",
            "a photo of high mountains or hills",
            "a photo of water, lake, or river",
            "a photo of the open sky",
        ],
        "grass": [
            "a photo of green grass, lawn, or meadow",
            "a photo of trees or forest",
            "a photo of high mountains or hills",
            "a photo of water or rocks",
        ],
        "road": [
            "a photo of a paved asphalt road, highway, or street",
            "a photo of a dirt path or green grass",
            "a photo of buildings or houses",
            "a photo of trees or nature",
        ],
        "building": [
            "a photo of a building, house, roof, or architecture",
            "a photo of trees or nature",
            "a photo of a road or street",
            "a photo of the open sky",
        ],
    }

    def __init__(self, threshold: float = 0.45) -> None:
        self.device = torch.device("cpu")
        self.threshold = threshold

        # Prefer MobileNet-V3 FPN for fast CPU inference (<0.4s)
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
            scale_x = float(proc_w) / float(orig_w)
            scale_y = float(proc_h) / float(orig_h)
        else:
            scale_x = 1.0
            scale_y = 1.0
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

            # Exact coordinate rescaling with sub-pixel preservation
            x1 = max(0.0, min(float(orig_w), float(box[0].item()) / scale_x))
            y1 = max(0.0, min(float(orig_h), float(box[1].item()) / scale_y))
            x2 = max(0.0, min(float(orig_w), float(box[2].item()) / scale_x))
            y2 = max(0.0, min(float(orig_h), float(box[3].item()) / scale_y))

            detections.append(
                DetectionResult(
                    label=label_name,
                    score=score_value,
                    box=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda item: (item.score, item.area), reverse=True)
        return detections

    def extract_target_from_question(self, question: str, answer: str | None = None) -> str | None:
        """Extract the queried object or landscape concept from question (or fallback to answer)."""
        q = question.lower().strip().rstrip("?").strip()

        patterns = [
            r"(?:is there|are there|can you see|do you see|find|locate|show|where is|where are|look for)\s+(?:any|a|an|the|some)?\s*([a-zA-Z\s]+?)(?:\s+in\s+|\s+on\s+|\s+at\s+|$)",
            r"(?:what color is|what colour is|what is the color of)\s+(?:the|a|an)?\s*([a-zA-Z\s]+)",
            r"(?:how many)\s+([a-zA-Z\s]+?)(?:\s+are|\s+is|\s+in|\s+on|$)",
            r"(?:what is the)\s+([a-zA-Z\s]+?)(?:\s+doing|\s+wearing|\s+in|$)",
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                concept = m.group(1).strip()
                concept = re.sub(r"\b(this|the|image|picture|photo|there|here)\b", "", concept).strip()
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

        # Fallback to answer if question was open-ended (e.g. "what is in the picture?")
        if answer:
            ans_clean = answer.lower().strip()
            for canonical, syns in self.ALIASES.items():
                if any(s in ans_clean for s in syns):
                    return canonical
            for cat in self.categories:
                if cat.lower() in ans_clean:
                    return cat.lower()
            for cat in self.LANDSCAPE_CONCEPTS:
                if cat in ans_clean:
                    return cat

        return None

    def _get_discriminative_prompts(self, target: str) -> list[str]:
        t = target.lower().strip()
        if t in self.LANDSCAPE_PROMPT_SUITES:
            return self.LANDSCAPE_PROMPT_SUITES[t]
        for key, suite in self.LANDSCAPE_PROMPT_SUITES.items():
            if key in t or t in key:
                return suite

        # Generic discriminative suite with realistic scene distractors
        return [
            f"a photo of a {target}",
            "a photo of green trees or forest vegetation",
            "a photo of calm water, a lake, or river",
            "a photo of a rushing waterfall",
            "a photo of high mountain peaks or ridge",
            "a photo of the open sky or clouds",
            "a photo of dry land, rocks, or soil",
        ]

    def _locate_with_clip(
        self,
        image: Image.Image,
        target: str,
        clip_model,
        threshold: float = 0.35,
    ) -> list[DetectionResult]:
        """Locate landscape features using competitive discriminative prompts & multi-scale spatial tiling."""
        if clip_model is None:
            return []

        w, h = image.size
        candidate_prompts = self._get_discriminative_prompts(target)
        target_prompt = candidate_prompts[0]

        # Candidate spatial regions tailored for natural scenes & landscapes
        candidate_boxes = [
            # Top Center Peaks & Full Horizon
            (int(w * 0.15), 0, int(w * 0.85), int(h * 0.45)),
            (0, 0, w, int(h * 0.45)),
            (0, 0, int(w * 0.65), int(h * 0.48)),
            (int(w * 0.35), 0, w, int(h * 0.48)),
            (0, 0, w, int(h * 0.35)),
            (0, int(h * 0.10), w, int(h * 0.55)),
            # Left & Right upper halves
            (0, 0, int(w * 0.55), int(h * 0.60)),
            (int(w * 0.45), 0, w, int(h * 0.60)),
            # Lower & foreground landscape bands (lakes, rivers, paths, fields)
            (0, int(h * 0.35), w, h),
            (0, int(h * 0.50), w, h),
            (0, int(h * 0.20), int(w * 0.50), int(h * 0.80)),
            (int(w * 0.50), int(h * 0.20), w, int(h * 0.80)),
            (int(w * 0.25), int(h * 0.20), int(w * 0.75), int(h * 0.75)),
            # Fine 3x3 grid cells
            (0, 0, int(w * 0.40), int(h * 0.40)),
            (int(w * 0.30), 0, int(w * 0.70), int(h * 0.40)),
            (int(w * 0.60), 0, w, int(h * 0.40)),
            (0, int(h * 0.30), int(w * 0.45), int(h * 0.70)),
            (int(w * 0.28), int(h * 0.28), int(w * 0.72), int(h * 0.72)),
            (int(w * 0.55), int(h * 0.30), w, int(h * 0.70)),
            (0, int(h * 0.60), int(w * 0.45), h),
            (int(w * 0.30), int(h * 0.60), int(w * 0.70), h),
            (int(w * 0.55), int(h * 0.60), w, h),
        ]

        scored_candidates = []
        for box in candidate_boxes:
            crop = image.crop(box)
            try:
                ranked = clip_model.rank_answers(crop, candidate_prompts)
                if not ranked:
                    continue

                # CRITICAL ACCURACY RULE:
                # The target prompt MUST be the undisputed #1 winning class for this crop!
                # If water or waterfall or trees beat mountain on this crop, it is NOT a mountain!
                if ranked[0].answer != target_prompt:
                    continue

                target_score = ranked[0].score
                if target_score < threshold:
                    continue

                second_score = ranked[1].score if len(ranked) > 1 else 0.0
                margin = target_score - second_score

                # Score metric combines absolute confidence and margin over competing classes
                quality_metric = (target_score * 0.7) + (margin * 0.3)
                scored_candidates.append((quality_metric, target_score, box))
            except Exception:
                continue

        if not scored_candidates:
            return []

        # Sort by best discriminative quality metric
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_score, best_box = scored_candidates[0]

        return [DetectionResult(label=target, score=best_score, box=best_box)]

    def draw_red_marks(
        self,
        image: Image.Image,
        detections: list[DetectionResult],
        box_color: str = "#FF1E1E",
        line_width: int = 4,
    ) -> Image.Image:
        """Draw prominent red bounding boxes, corner accents, and labels on detected objects."""
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
        answer: str | None = None,
    ) -> tuple[Image.Image | None, list[DetectionResult], str | None]:
        """Locate queried object or landscape concept and render a high-precision red mark annotation."""
        target = self.extract_target_from_question(question, answer=answer)

        # 1. If target is in landscape/open-vocabulary concepts, prioritize CLIP discriminative localization
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

        # 3. If COCO detector found nothing, try CLIP discriminative spatial localization as backup
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
