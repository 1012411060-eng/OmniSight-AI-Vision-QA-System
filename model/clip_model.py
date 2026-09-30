from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from optimum.onnxruntime import ORTModelForZeroShotImageClassification
except ImportError:  
    ORTModelForZeroShotImageClassification = None


@dataclass(frozen=True)
class CandidateScore:
    answer: str
    score: float


class ClipVQAModel:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", backend: str = "onnx") -> None:
        self.device = torch.device("cpu")
        self.model_name = model_name
        self.backend = backend if backend in {"onnx", "torch"} else "torch"
        self.available = True
        self.load_error: str | None = None
        self.processor = None
        self.model = None

        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = self._load_model()
        except Exception:  # pragma: no cover - defensive runtime guard
            self.available = False
            self.load_error = (
                f"CLIP could not be loaded from '{model_name}'. "
                "Check your internet connection, Hugging Face cache, or local folder conflicts."
            )
            self.processor = None
            self.model = None

    def _load_model(self):
        if self.backend == "onnx" and ORTModelForZeroShotImageClassification is not None:
            return ORTModelForZeroShotImageClassification.from_pretrained(self.model_name, export=True)
        model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        try:
            import torch.ao.quantization as ao_quant
            model = ao_quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass
        return model

    @torch.inference_mode()
    def rank_answers(
        self,
        image: Image.Image,
        candidates: Sequence[str],
        top_k: int = 5,
    ) -> list[CandidateScore]:
        if not self.available or self.processor is None or self.model is None:
            return []
        if not candidates:
            return []

        if max(image.size) > 768:
            image = image.copy()
            image.thumbnail((768, 768), Image.Resampling.BICUBIC)

        prompts = [self._candidate_prompt(answer) for answer in candidates]
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        if self.backend == "onnx" and ORTModelForZeroShotImageClassification is not None:
            inputs = {key: value.cpu().numpy() for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = torch.as_tensor(outputs.logits_per_image).squeeze(0)
        else:
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image.squeeze(0)

        probabilities = torch.softmax(logits, dim=-1)

        ranked = [
            CandidateScore(answer=candidates[i], score=float(probabilities[i]))
            for i in torch.argsort(probabilities, descending=True).tolist()
        ]
        return ranked[:top_k]

    @staticmethod
    def _candidate_prompt(answer: str) -> str:
        return f"a photo of {answer.strip()}"
