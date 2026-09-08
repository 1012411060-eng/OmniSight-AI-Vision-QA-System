from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from model.blip_model import BlipAnswer, BlipVQAModel
from model.clip_model import ClipVQAModel
from model.object_detector import ObjectDetector
from model.ocr_model import OCRModel, OCRResult, OCRTextLine


@dataclass(frozen=True)
class QuestionAnalysis:
    intent: str
    supported: bool
    reason: str
    answer_style: str
    candidates: list[str]
    target_object: str | None = None


SUPPORTED_VISUAL_HINTS = (
    "image",
    "picture",
    "photo",
    "scene",
    "shown",
    "see",
    "visible",
    "look",
    "looks",
    "wear",
    "wearing",
    "hold",
    "holding",
    "color",
    "colour",
    "count",
    "many",
    "where",
    "what is",
    "what are",
    "who is",
    "who are",
    "doing",
    "happening",
    "compare",
    "difference",
    "describe",
    "detail",
    "complex",
)

NON_VISUAL_TOPICS = (
    "capital",
    "weather",
    "meaning",
    "define",
    "translation",
    "translate",
    "math",
    "calculate",
    "date",
    "time",
    "history",
    "president",
    "law",
    "salary",
    "price",
)

COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "orange",
    "gray",
    "grey",
    "pink",
    "purple",
}

OBJECT_FALLBACKS = (
    "police car",
    "car",
    "vehicle",
    "truck",
    "bus",
    "person",
    "man",
    "woman",
    "bike",
    "bicycle",
    "motorcycle",
    "dog",
    "cat",
)

COLOR_NAME_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (220, 20, 60),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green": (34, 139, 34),
    "blue": (30, 144, 255),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
}


@dataclass(frozen=True)
class VQAResult:
    answer: str
    confidence: float
    ranked_answers: list[dict[str, object]]
    analysis: QuestionAnalysis
    explanation: str | None
    strategy: str


@dataclass(frozen=True)
class ColorCluster:
    name: str
    ratio: float
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ColorSummary:
    count: int
    names: list[str]
    clusters: list[ColorCluster]


def analyze_question(question: str) -> QuestionAnalysis:
    normalized = question.strip().lower()

    if not normalized:
        return QuestionAnalysis(
            intent="invalid",
            supported=False,
            reason="Please type a question about the image.",
            answer_style="short answer",
            candidates=[],
        )

    if len(normalized.split()) < 2:
        return QuestionAnalysis(
            intent="invalid",
            supported=False,
            reason="That question is too short to analyze. Try asking about an object, color, count, or activity in the image.",
            answer_style="short answer",
            candidates=[],
        )

    if _looks_like_ocr_question(normalized):
        return QuestionAnalysis(
            intent="ocr_text",
            supported=True,
            reason="Detected a text, brand, logo, or visible price question.",
            answer_style="short answer",
            candidates=[],
        )
    if _looks_non_visual(normalized):
        return QuestionAnalysis(
            intent="unsupported",
            supported=False,
            reason="That looks like a non-visual question. I can answer questions about what is visible in the image.",
            answer_style="short answer",
            candidates=[],
        )

    if _looks_like_yes_no(normalized):
        return QuestionAnalysis(
            intent="yes_no",
            supported=True,
            reason="Detected a yes/no style question.",
            answer_style="yes/no",
            candidates=["yes", "no"],
        )

    if _looks_like_color_count(normalized):
        target = _extract_target_object(normalized)
        return QuestionAnalysis(
            intent="color_count",
            supported=True,
            reason="Detected a color-count question.",
            answer_style="count",
            candidates=[],
            target_object=target,
        )

    if "how many" in normalized or "number of" in normalized or normalized.startswith("count "):
        return QuestionAnalysis(
            intent="count",
            supported=True,
            reason="Detected a counting question.",
            answer_style="count",
            candidates=[str(i) for i in range(0, 11)],
        )

    if any(token in normalized for token in ("what color", "what colour", "color", "colour")):
        target = _extract_target_object(normalized)
        return QuestionAnalysis(
            intent="color",
            supported=True,
            reason="Detected a color question.",
            answer_style="color",
            candidates=[
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "brown",
                "orange",
                "gray",
                "pink",
                "purple",
            ],
            target_object=target,
        )

    if _looks_like_complex_question(normalized):
        target = _extract_target_object(normalized)
        return QuestionAnalysis(
            intent="complex",
            supported=True,
            reason="Detected a complex visual question.",
            answer_style="short answer",
            candidates=[],
            target_object=target,
        )

    if "where" in normalized:
        return QuestionAnalysis(
            intent="location",
            supported=True,
            reason="Detected a location question.",
            answer_style="location",
            candidates=[
                "indoors",
                "outdoors",
                "on the street",
                "in a room",
                "in a kitchen",
                "in a park",
                "at the beach",
                "in an office",
            ],
        )

    if "doing" in normalized or "happening" in normalized or "doing?" in normalized:
        return QuestionAnalysis(
            intent="activity",
            supported=True,
            reason="Detected an activity question.",
            answer_style="activity",
            candidates=[
                "standing",
                "sitting",
                "walking",
                "running",
                "eating",
                "talking",
                "driving",
                "riding a bike",
                "playing",
                "sleeping",
            ],
        )

    if _looks_like_identity_question(normalized):
        return QuestionAnalysis(
            intent="object",
            supported=True,
            reason="Detected an object-identification question.",
            answer_style="object",
            candidates=[
                "person",
                "animal",
                "vehicle",
                "food",
                "indoor scene",
                "outdoor scene",
                "object",
                "text",
                "sport",
            ],
        )

    if _looks_like_visual_general(normalized):
        return QuestionAnalysis(
            intent="open_visual",
            supported=True,
            reason="Detected a general visual question.",
            answer_style="short answer",
            candidates=[
                "person",
                "animal",
                "vehicle",
                "food",
                "indoor scene",
                "outdoor scene",
                "object",
                "text",
                "sport",
            ],
        )

    return QuestionAnalysis(
        intent="unsupported",
        supported=False,
        reason="I could not map that question to something visible in the image. Try asking about what is shown, its color, count, location, or activity.",
        answer_style="short answer",
        candidates=[],
    )



def _looks_like_brand_or_price(question: str) -> bool:
    return bool(
        any(term in question for term in ("brand", "logo", "price", "cost", "estimate", "how much is", "how much does"))
    )
def build_candidate_answers(question: str) -> list[str]:
    return analyze_question(question).candidates or [
        "person",
        "animal",
        "vehicle",
        "food",
        "indoor scene",
        "outdoor scene",
        "object",
        "text",
        "sport",
    ]


def answer_question(
    clip_model: ClipVQAModel,
    blip_model: BlipVQAModel,
    detector: ObjectDetector,
    ocr_model: OCRModel,
    image: Image.Image,
    question: str,
    candidates: Sequence[str],
    top_k: int = 5,
) -> VQAResult:
    analysis = analyze_question(question)
    blip_available = getattr(blip_model, "available", True)

    if not analysis.supported:
        return VQAResult(
            answer=analysis.reason,
            confidence=0.0,
            ranked_answers=[],
            analysis=analysis,
            explanation=None,
            strategy="unsupported",
        )

    if analysis.intent == "ocr_text":
        return _answer_with_ocr(ocr_model, image, question, analysis)

    if analysis.intent == "complex":
        if not blip_available:
            ranked = clip_model.rank_answers(image=image, candidates=build_candidate_answers(question), top_k=top_k)
            if not ranked:
                return VQAResult(
                    answer="No answer candidates were provided.",
                    confidence=0.0,
                    ranked_answers=[],
                    analysis=analysis,
                    explanation=None,
                    strategy="clip_rank",
                )
            top_answer = ranked[0]
            return VQAResult(
                answer=top_answer.answer,
                confidence=top_answer.score,
                ranked_answers=[{"answer": item.answer, "score": item.score} for item in ranked],
                analysis=analysis,
                explanation="BLIP was unavailable, so the app fell back to CLIP ranking for the complex question.",
                strategy="clip_rank_fallback",
            )

        prompt = _compose_complex_prompt(question, analysis.target_object)
        answer = blip_model.answer(image=image, question=prompt)
        compact = _briefen_answer(answer.answer)
        return VQAResult(
            answer=compact,
            confidence=answer.confidence,
            ranked_answers=[{"answer": compact, "score": answer.confidence}],
            analysis=analysis,
            explanation="Used a more descriptive prompt with a short caption-style context for a harder visual question.",
            strategy="complex_blip",
        )

    if analysis.intent == "color_count":
        cropped_image, detected_object = _prepare_target_crop(detector, image, analysis.target_object)
        summary = _estimate_dominant_colors(cropped_image or image)
        names = summary.names or ["unknown"]
        color_list = _format_color_list(summary.clusters)

        if analysis.target_object and detected_object is None:
            answer = _format_missing_object_answer(analysis.target_object)
            return VQAResult(
                answer=answer,
                confidence=0.95,
                ranked_answers=[{"answer": answer, "score": 0.95}],
                analysis=analysis,
                explanation="The detector could not find the target object, so the app did not estimate colors on a missing object.",
                strategy="detector_presence",
            )

        if analysis.target_object and detected_object is not None:
            answer = f"On the {analysis.target_object}, I can see about {summary.count} prominent colors: {color_list}."
        else:
            answer = f"I can see about {summary.count} prominent colors: {color_list}."

        ranked = [
            {"answer": f"{cluster.name} ({cluster.ratio:.0%})", "score": cluster.ratio}
            for cluster in summary.clusters
        ]
        return VQAResult(
            answer=answer,
            confidence=1.0,
            ranked_answers=ranked,
            analysis=analysis,
            explanation="Estimated the dominant colors directly from the detected object crop when available.",
            strategy="color_count_crop" if cropped_image is not None else "color_count",
        )

    if analysis.intent == "color":
        cropped_image, detection = _prepare_target_crop(detector, image, analysis.target_object)
        if detection is None and analysis.target_object:
            answer = _format_missing_object_answer(analysis.target_object)
            return VQAResult(
                answer=answer,
                confidence=0.95,
                ranked_answers=[{"answer": answer, "score": 0.95}],
                analysis=analysis,
                explanation="The detector could not find the target object before answering the color question.",
                strategy="detector_presence",
            )

        if not blip_available:
            ranked = clip_model.rank_answers(image=cropped_image or image, candidates=candidates, top_k=top_k)
            if not ranked:
                return VQAResult(
                    answer="No answer candidates were provided.",
                    confidence=0.0,
                    ranked_answers=[],
                    analysis=analysis,
                    explanation=None,
                    strategy="clip_rank",
                )
            top_answer = ranked[0]
            explanation = "BLIP was unavailable, so the app fell back to CLIP ranking for the color question."
            return VQAResult(
                answer=top_answer.answer,
                confidence=top_answer.score,
                ranked_answers=[{"answer": item.answer, "score": item.score} for item in ranked],
                analysis=analysis,
                explanation=explanation,
                strategy="clip_rank_fallback",
            )

        direct = blip_model.answer(image=cropped_image or image, question=question)
        direct = _normalize_color_answer(direct, candidates)
        explanation = "Used BLIP on a detected object crop for the color question." if cropped_image is not None else "Used BLIP direct answering for a more specific visual attribute question."
        return VQAResult(
            answer=direct.answer,
            confidence=direct.confidence,
            ranked_answers=[{"answer": direct.answer, "score": direct.confidence}],
            analysis=analysis,
            explanation=explanation,
            strategy="detector_crop" if cropped_image is not None else "blip_direct",
        )

    if analysis.intent in {"yes_no", "count"}:
        if not blip_available:
            ranked = clip_model.rank_answers(image=image, candidates=candidates, top_k=top_k)
            if not ranked:
                return VQAResult(
                    answer="No answer candidates were provided.",
                    confidence=0.0,
                    ranked_answers=[],
                    analysis=analysis,
                    explanation=None,
                    strategy="clip_rank",
                )
            top_answer = ranked[0]
            return VQAResult(
                answer=top_answer.answer,
                confidence=top_answer.score,
                ranked_answers=[{"answer": item.answer, "score": item.score} for item in ranked],
                analysis=analysis,
                explanation="BLIP was unavailable, so the app fell back to CLIP ranking for this short-answer question.",
                strategy="clip_rank_fallback",
            )

        direct = blip_model.answer(image=image, question=question)
        if analysis.intent == "yes_no":
            normalized = direct.answer.strip().lower()
            if normalized not in {"yes", "no"}:
                if any(word in question.lower() for word in ("not ", "without", "no ")):
                    direct = BlipAnswer(answer="no", confidence=direct.confidence)
                else:
                    direct = BlipAnswer(answer="yes", confidence=direct.confidence)
        return VQAResult(
            answer=direct.answer,
            confidence=direct.confidence,
            ranked_answers=[{"answer": direct.answer, "score": direct.confidence}],
            analysis=analysis,
            explanation="Used BLIP direct answering for a question that needs a yes/no or count response.",
            strategy="blip_direct",
        )

    if not clip_model.available:
        direct = blip_model.answer(image=image, question=question)
        return VQAResult(
            answer=direct.answer,
            confidence=direct.confidence,
            ranked_answers=[{"answer": direct.answer, "score": direct.confidence}],
            analysis=analysis,
            explanation="CLIP was unavailable, so the app fell back to BLIP direct answering.",
            strategy="blip_fallback",
        )

    ranked = clip_model.rank_answers(image=image, candidates=candidates, top_k=top_k)
    if not ranked:
        return VQAResult(
            answer="No answer candidates were provided.",
            confidence=0.0,
            ranked_answers=[],
            analysis=analysis,
            explanation=None,
            strategy="clip_rank",
        )

    top_answer = ranked[0]
    return VQAResult(
        answer=top_answer.answer,
        confidence=top_answer.score,
        ranked_answers=[{"answer": item.answer, "score": item.score} for item in ranked],
        analysis=analysis,
        explanation="Used CLIP ranking over a targeted candidate set.",
        strategy="clip_rank",
    )




def _compose_complex_prompt(question: str, target_object: str | None) -> str:
    if target_object:
        return f"Question about the image: {question}. Focus on the {target_object} if visible. Answer briefly and directly."
    return f"Question about the image: {question}. Answer briefly and directly."
def _answer_with_ocr(ocr_model: OCRModel, image: Image.Image, question: str, analysis: QuestionAnalysis) -> VQAResult:
    ocr = ocr_model.extract_text(image)
    if not ocr.available:
        return VQAResult(
            answer=ocr.note or "OCR is not available in this environment.",
            confidence=0.0,
            ranked_answers=[],
            analysis=analysis,
            explanation=None,
            strategy="ocr_unavailable",
        )

    if not ocr.lines:
        return VQAResult(
            answer="I could not read any visible text in the image.",
            confidence=0.0,
            ranked_answers=[],
            analysis=analysis,
            explanation="OCR did not find readable text, so brand or price could not be confirmed.",
            strategy="ocr_empty",
        )

    text_blob = _ocr_visible_text(ocr.lines)
    if _looks_like_price_question(question):
        price_line = _find_price_line(ocr.lines)
        if price_line:
            return VQAResult(
                answer=f"I can read a visible price: {price_line.text}.",
                confidence=price_line.confidence,
                ranked_answers=[{"answer": price_line.text, "score": price_line.confidence}],
                analysis=analysis,
                explanation="Used OCR to read a visible price-like text from the image.",
                strategy="ocr_price",
            )
        return VQAResult(
            answer=f"I can read visible text, but I cannot confirm a clear price from it: {text_blob}",
            confidence=0.35,
            ranked_answers=[{"answer": line.text, "score": line.confidence} for line in ocr.lines[:5]],
            analysis=analysis,
            explanation="OCR found text, but not a reliable price string.",
            strategy="ocr_price_unclear",
        )

    if _looks_like_brand_question(question):
        brand = _infer_brand_from_text(ocr.lines)
        if brand:
            return VQAResult(
                answer=f"The visible text suggests {brand}.",
                confidence=0.9,
                ranked_answers=[{"answer": brand, "score": 0.9}] + [{"answer": line.text, "score": line.confidence} for line in ocr.lines[:4]],
                analysis=analysis,
                explanation="Used OCR text plus brand matching to infer the likely brand.",
                strategy="ocr_brand",
            )
        return VQAResult(
            answer=f"I can read visible text: {text_blob}",
            confidence=0.4,
            ranked_answers=[{"answer": line.text, "score": line.confidence} for line in ocr.lines[:5]],
            analysis=analysis,
            explanation="OCR found visible text, but no clear brand name was confirmed.",
            strategy="ocr_brand_unclear",
        )

    return VQAResult(
        answer=f"I can read visible text: {text_blob}",
        confidence=0.5,
        ranked_answers=[{"answer": line.text, "score": line.confidence} for line in ocr.lines[:5]],
        analysis=analysis,
        explanation="Used OCR to extract visible text from the image.",
        strategy="ocr_text",
    )


def _ocr_visible_text(lines: Sequence[OCRTextLine]) -> str:
    return " | ".join(line.text for line in lines[:6])


def _looks_like_brand_question(question: str) -> bool:
    return bool(any(term in question for term in ("brand", "logo", "which company", "what company", "whose logo", "brand name")))


def _looks_like_price_question(question: str) -> bool:
    return bool(any(term in question for term in ("price", "cost", "how much is", "how much does", "how much are", "amount")))


def _looks_like_ocr_question(question: str) -> bool:
    return _looks_like_brand_question(question) or _looks_like_price_question(question) or any(term in question for term in ("what does it say", "what is written", "visible text", "read the text", "text in the image"))


def _infer_brand_from_text(lines: Sequence[OCRTextLine]) -> str | None:
    normalized = " ".join(line.text.lower() for line in lines)
    for brand in sorted(BRAND_HINTS, key=len, reverse=True):
        if brand in normalized:
            return brand.title() if brand != "h&m" else "H&M"
    return None


def _find_price_line(lines: Sequence[OCRTextLine]) -> OCRTextLine | None:
    for line in lines:
        if PRICE_PATTERN.search(line.text):
            return line
    return None


def _briefen_answer(answer: str, max_words: int = 18) -> str:
    cleaned = re.sub(r"\s+", " ", answer.strip())
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def _estimate_dominant_colors(image: Image.Image, max_colors: int = 10) -> ColorSummary:
    rgb = image.convert("RGB")
    if max(rgb.size) > 640:
        rgb.thumbnail((640, 640))

    pixels = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if pixels.size == 0:
        cluster = ColorCluster(name="unknown", ratio=1.0, rgb=(0, 0, 0))
        return ColorSummary(count=1, names=[cluster.name], clusters=[cluster])

    unique_pixels = np.unique(pixels, axis=0)
    k = max(1, min(max_colors, len(unique_pixels), 10))
    if k == 1:
        color = tuple(int(v) for v in unique_pixels[0])
        cluster = ColorCluster(name=_nearest_color_name(color), ratio=1.0, rgb=color)
        return ColorSummary(count=1, names=[cluster.name], clusters=[cluster])

    sample = pixels
    if len(sample) > 30000:
        indices = np.random.choice(len(sample), 30000, replace=False)
        sample = sample[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.8)
    _, labels, centers = cv2.kmeans(
        sample.astype(np.float32),
        k,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS,
    )

    counts = np.bincount(labels.flatten(), minlength=k)
    total = float(counts.sum()) or 1.0

    clusters: list[ColorCluster] = []
    for index, center in enumerate(centers):
        ratio = float(counts[index]) / total
        rgb_value = tuple(int(round(value)) for value in center)
        clusters.append(ColorCluster(name=_nearest_color_name(rgb_value), ratio=ratio, rgb=rgb_value))

    merged = _merge_similar_color_clusters(clusters, ratio_threshold=0.015)
    merged.sort(key=lambda item: item.ratio, reverse=True)
    names = [cluster.name for cluster in merged]
    return ColorSummary(count=len(names), names=names, clusters=merged)


def _format_color_list(clusters: Sequence[ColorCluster], max_items: int = 10) -> str:
    items = [f"{cluster.name} ({cluster.ratio:.0%})" for cluster in clusters[:max_items]]
    return ", ".join(items)


def _merge_similar_color_clusters(clusters: list[ColorCluster], ratio_threshold: float = 0.03) -> list[ColorCluster]:
    merged: dict[str, ColorCluster] = {}
    for cluster in clusters:
        if cluster.ratio < ratio_threshold and merged and cluster.name in merged:
            continue
        existing = merged.get(cluster.name)
        if existing is None or cluster.ratio > existing.ratio:
            merged[cluster.name] = cluster
    return list(merged.values())


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    rgb_array = np.array(rgb, dtype=np.float32)
    best_name = "unknown"
    best_distance = float("inf")
    for name, reference in COLOR_NAME_PALETTE.items():
        distance = float(np.linalg.norm(rgb_array - np.array(reference, dtype=np.float32)))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def _prepare_target_crop(
    detector: ObjectDetector,
    image: Image.Image,
    target_object: str | None,
) -> tuple[Image.Image | None, object | None]:
    if not target_object:
        return None, None

    detections = detector.detect(image, target_object=target_object)
    if not detections:
        return None, None

    detection = detections[0]
    x1, y1, x2, y2 = detection.box
    width, height = image.size
    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(width, int(x2))
    bottom = min(height, int(y2))

    if right <= left or bottom <= top:
        return None, detection

    return image.crop((left, top, right, bottom)), detection


def _format_missing_object_answer(target_object: str) -> str:
    cleaned = _clean_target_object(target_object)
    return f"There is no {cleaned} in the image."


def _normalize_color_answer(answer: BlipAnswer, candidates: Sequence[str]) -> BlipAnswer:
    text = _normalize_text(answer.answer)
    tokens = set(text.split())
    if tokens & COLOR_WORDS:
        return answer

    for candidate in candidates:
        candidate_lc = candidate.lower()
        if candidate_lc in COLOR_WORDS:
            return BlipAnswer(answer=candidate, confidence=answer.confidence)

    return answer


def _extract_target_object(question: str) -> str | None:
    patterns = [
        r"(?:what is the|what's the|what is)\s+color of the\s+([a-z0-9\s-]+?)(?:\s+in\s+the\s+image|\s+in\s+this\s+image)?(?:\?|$)",
        r"(?:what is the|what's the|what is)\s+colour of the\s+([a-z0-9\s-]+?)(?:\s+in\s+the\s+image|\s+in\s+this\s+image)?(?:\?|$)",
        r"(?:what is the|what's the|what is)\s+color of\s+([a-z0-9\s-]+?)(?:\s+in\s+the\s+image|\s+in\s+this\s+image)?(?:\?|$)",
        r"(?:what is the|what's the|what is)\s+colour of\s+([a-z0-9\s-]+?)(?:\s+in\s+the\s+image|\s+in\s+this\s+image)?(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            target = _clean_target_object(match.group(1))
            if target:
                return target

    for fallback in OBJECT_FALLBACKS:
        if fallback in question:
            return fallback

    return None


def _clean_target_object(target: str) -> str:
    cleaned = _normalize_text(target)
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned)
    cleaned = re.sub(r"\b(in|on|at)\s+(the|this)?\s*image\b", "", cleaned)
    cleaned = re.sub(r"\b(in|on|at)\s+(the|this)?\s*picture\b", "", cleaned)
    cleaned = re.sub(r"\b(in|on|at)\s+(the|this)?\s*photo\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,")
    return cleaned


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _looks_like_yes_no(question: str) -> bool:
    return bool(
        re.match(r"^(is|are|was|were|do|does|did|can|could|should|has|have|had|will|would|may|might)\b", question)
        or question.startswith("is ")
        or question.startswith("are ")
    )


def _looks_like_color_count(question: str) -> bool:
    return bool(
        "how many color" in question
        or "how many colours" in question
        or "how many colors" in question
        or re.search(r"how many\s+colors?\s+(are|are there|can you see|do you see)", question)
    )


def _looks_like_complex_question(question: str) -> bool:
    if len(question.split()) >= 10:
        return True
    if any(token in question for token in ("compare", "difference", "describe", "detail", "why", "how does", "what is happening")):
        return True
    return False


def _looks_like_identity_question(question: str) -> bool:
    return bool(
        "what is in the image" in question
        or "what is in this image" in question
        or "what is shown" in question
        or "what are in the image" in question
        or "what is this" in question
        or "what are these" in question
        or "what is it" in question
        or question.startswith("what is")
        or question.startswith("what are")
        or question.startswith("who is")
        or question.startswith("who are")
    )


def _looks_like_visual_general(question: str) -> bool:
    if any(hint in question for hint in SUPPORTED_VISUAL_HINTS):
        return True
    return question.startswith("what") or question.startswith("who") or question.startswith("where")


def _looks_non_visual(question: str) -> bool:
    if any(topic in question for topic in NON_VISUAL_TOPICS):
        return True
    if "why" in question or "how do i" in question or "how to" in question:
        return True
    if "in the world" in question or "universe" in question:
        return True
    return False
