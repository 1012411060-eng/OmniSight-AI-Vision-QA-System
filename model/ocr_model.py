from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from PIL import Image

from utils.preprocess import prepare_for_ocr

try:
    import easyocr
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None


@dataclass(frozen=True)
class OCRTextLine:
    text: str
    confidence: float
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class OCRResult:
    lines: list[OCRTextLine]
    combined_text: str
    available: bool
    note: str | None = None


class OCRModel:
    """Lightweight OCR wrapper for visible text extraction.

    Uses EasyOCR when installed. The preprocessing is tuned for screenshots,
    logos, receipts, and high-contrast text in photos.
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or ["en"]
        self.available = easyocr is not None
        self.reader = easyocr.Reader(self.languages, gpu=False) if self.available else None

    def extract_text(self, image: Image.Image) -> OCRResult:
        if not self.available:
            return OCRResult(
                lines=[],
                combined_text="",
                available=False,
                note="OCR is unavailable because easyocr is not installed.",
            )

        variants = [prepare_for_ocr(image), image.convert("RGB")]
        collected: list[OCRTextLine] = []
        seen: set[str] = set()

        for variant in variants:
            array = np.array(variant)
            results = self.reader.readtext(
                array,
                detail=1,
                paragraph=False,
                text_threshold=0.4,
                low_text=0.25,
                link_threshold=0.3,
                contrast_ths=0.1,
                adjust_contrast=0.7,
            )
            for item in results:
                if len(item) != 3:
                    continue
                box, text, confidence = item
                cleaned = _clean_ocr_text(text)
                if not cleaned:
                    continue
                signature = cleaned.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                collected.append(
                    OCRTextLine(
                        text=cleaned,
                        confidence=float(confidence),
                        box=tuple(tuple(float(v) for v in point) for point in box),
                    )
                )

        collected.sort(key=lambda item: item.confidence, reverse=True)
        combined = " | ".join(line.text for line in collected[:8])
        return OCRResult(lines=collected, combined_text=combined, available=True)

    def extract_combined_text(self, image: Image.Image) -> str:
        return self.extract_text(image).combined_text


@lru_cache(maxsize=1)
def load_default_ocr_model() -> OCRModel:
    return OCRModel()


def _clean_ocr_text(text: str) -> str:
    return " ".join(str(text).strip().split())
