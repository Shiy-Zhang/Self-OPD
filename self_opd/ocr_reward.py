"""PaddleOCR-based reward used as the public Self-OPD example."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("PADDLE_DEVICE", "cpu")
os.environ.setdefault("FLAGS_use_cuda", "0")

import numpy as np
import paddle
import torch
from Levenshtein import distance
from paddleocr import PaddleOCR
from PIL import Image

paddle.device.set_device("cpu")


def extract_target_text(prompt: str) -> str:
    """Return the first double-quoted string from an OCR prompt."""
    parts = prompt.split('"')
    if len(parts) < 3:
        raise ValueError(f'OCR prompt must contain target text in double quotes: {prompt!r}')
    target = parts[1].strip()
    if not target:
        raise ValueError("OCR target text cannot be empty")
    return target


def _local_model_paths() -> dict[str, str]:
    """Resolve optional local PaddleOCR weights from environment variables."""
    explicit = {
        "det_model_dir": os.environ.get("SELF_OPD_OCR_DET_MODEL"),
        "rec_model_dir": os.environ.get("SELF_OPD_OCR_REC_MODEL"),
        "cls_model_dir": os.environ.get("SELF_OPD_OCR_CLS_MODEL"),
    }
    if any(explicit.values()):
        missing = [name for name, value in explicit.items() if not value]
        if missing:
            raise ValueError(f"All OCR model paths must be set together; missing {missing}")
        return {name: str(Path(value).expanduser()) for name, value in explicit.items()}
    return {}


def _collect_recognized_text(result) -> str:
    """Extract text from PaddleOCR 2.x or 3.x result objects."""
    if not result:
        return ""
    first = result[0] if isinstance(result, list) else result
    payload = getattr(first, "json", first)
    if callable(payload):
        payload = payload()
    if isinstance(payload, dict):
        payload = payload.get("res", payload)
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [1.0] * len(texts))
        return "".join(text for text, score in zip(texts, scores) if float(score) > 0)

    lines = first if isinstance(first, list) else []
    return "".join(
        item[1][0]
        for item in lines
        if isinstance(item, (list, tuple))
        and len(item) > 1
        and isinstance(item[1], (list, tuple))
        and len(item[1]) > 1
        and float(item[1][1]) > 0
    )


class OCRScorer:
    """Character-level OCR reward in ``[0, 1]``.

    PaddleOCR runs on CPU so reward inference does not consume training GPU
    memory. Without explicit model paths, PaddleOCR downloads its public English
    models into its standard cache on first use.
    """

    def __init__(self) -> None:
        model_paths = _local_model_paths()
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                show_log=False,
                **model_paths,
            )
            self._uses_predict_api = False
        except (TypeError, ValueError):
            if model_paths:
                raise ValueError(
                    "SELF_OPD_OCR_* model directories use the PaddleOCR 2.x API. "
                    "For PaddleOCR 3.x, configure model locations through PaddleOCR itself."
                )
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self._uses_predict_api = True

    @torch.no_grad()
    def __call__(self, images, prompts: list[str]) -> list[float]:
        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have the same length")

        rewards: list[float] = []
        for image, prompt in zip(images, prompts):
            target = extract_target_text(prompt)
            normalized_target = target.replace(" ", "").lower()
            if isinstance(image, Image.Image):
                image = np.asarray(image)

            try:
                result = self.ocr.predict(image) if self._uses_predict_api else self.ocr.ocr(image, cls=False)
                recognized = _collect_recognized_text(result).replace(" ", "").lower()
                edit_distance = (
                    0 if normalized_target in recognized
                    else min(distance(recognized, normalized_target), len(normalized_target))
                )
            except Exception as error:
                warnings.warn(f"OCR scoring failed: {error}", stacklevel=2)
                edit_distance = len(normalized_target)

            rewards.append(1.0 - edit_distance / len(normalized_target))

        return rewards
