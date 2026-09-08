from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor


@dataclass(frozen=True)
class BlipAnswer:
    answer: str
    confidence: float


class BlipVQAModel:
    """Open-ended VQA model for natural language answers.

    This uses BLIP for image-question answering on CPU.
    The confidence score is a lightweight generation-based estimate,
    not a calibrated probability.
    """

    def __init__(self, model_name: str = "Salesforce/blip-vqa-base") -> None:
        self.device = torch.device("cpu")
        self.available = True
        self.load_error: str | None = None
        self.processor = None
        self.model = None

        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.available = False
            self.load_error = (
                f"BLIP could not be loaded from '{model_name}'. "
                "Check your internet connection, Hugging Face cache, or local folder conflicts."
            )
            self.processor = None
            self.model = None

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str) -> BlipAnswer:
        if not self.available or self.processor is None or self.model is None:
            return BlipAnswer(answer=self.load_error or "BLIP is unavailable.", confidence=0.0)

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )
        answer = self.processor.tokenizer.decode(generated.sequences[0], skip_special_tokens=True).strip()
        confidence = self._estimate_confidence(generated.sequences[0], generated.scores)
        return BlipAnswer(answer=answer or "No answer generated.", confidence=confidence)

    @torch.inference_mode()
    def answer_many(
        self,
        image: Image.Image,
        question: str,
        top_k: int = 5,
    ) -> list[BlipAnswer]:
        if not self.available or self.processor is None or self.model is None:
            return [BlipAnswer(answer=self.load_error or "BLIP is unavailable.", confidence=0.0)]

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )

        answers: list[BlipAnswer] = []
        seen: set[str] = set()
        for sequence_index, sequence in enumerate(generated.sequences):
            answer = self.processor.tokenizer.decode(sequence, skip_special_tokens=True).strip()
            normalized = answer.lower()
            if not answer or normalized in seen:
                continue
            seen.add(normalized)
            confidence = self._estimate_confidence(sequence, generated.scores, sequence_index=sequence_index)
            answers.append(BlipAnswer(answer=answer, confidence=confidence))

        if not answers:
            answers.append(BlipAnswer(answer="No answer generated.", confidence=0.0))

        return answers[:top_k]

    @torch.inference_mode()
    def describe(self, image: Image.Image) -> str:
        if not self.available or self.processor is None or self.model is None:
            return self.load_error or "BLIP is unavailable."

        prompt = "Describe the image in one short sentence."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(**inputs, max_new_tokens=24)
        caption = self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return caption or "No description generated."

    def _estimate_confidence(
        self,
        sequence: torch.Tensor,
        scores: list[torch.Tensor],
        sequence_index: int = 0,
    ) -> float:
        if not scores:
            return 0.0

        prompt_length = sequence.shape[-1] - len(scores)
        step_probabilities: list[float] = []

        for step_index, step_scores in enumerate(scores):
            step_tensor = step_scores[sequence_index]
            token_id = int(sequence[prompt_length + step_index].item())
            probabilities = torch.softmax(step_tensor, dim=-1)
            step_probabilities.append(float(probabilities[token_id]))

        if not step_probabilities:
            return 0.0

        return float(sum(step_probabilities) / len(step_probabilities))
