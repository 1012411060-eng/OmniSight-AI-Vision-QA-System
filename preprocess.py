from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_image(uploaded_file: BinaryIO) -> Image.Image:
    """Load an uploaded image and normalize orientation."""
    raw_bytes = uploaded_file.read()
    image = Image.open(BytesIO(raw_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def prepare_for_ocr(image: Image.Image) -> Image.Image:
    """Create a text-friendly version of the image for OCR."""
    base = ImageOps.exif_transpose(image.convert("RGB"))
    base = ImageOps.autocontrast(base)

    width, height = base.size
    max_dim = max(width, height)
    if max_dim < 1400:
        scale = 2
        base = base.resize((width * scale, height * scale), Image.Resampling.LANCZOS)

    base = ImageEnhance.Contrast(base).enhance(1.6)
    base = ImageEnhance.Sharpness(base).enhance(1.8)
    base = base.filter(ImageFilter.UnsharpMask(radius=1.3, percent=180, threshold=3))
    return base
