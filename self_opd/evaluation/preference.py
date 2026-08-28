"""PickScore and HPSv2 scorers used by the paper evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _feature_tensor(value, names: tuple[str, ...]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError(f"Could not extract a feature tensor from {type(value).__name__}")


class PickScoreScorer:
    """Compute the raw PickScore logit used in the paper tables."""

    def __init__(
        self,
        device: str,
        dtype: torch.dtype = torch.float32,
        processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_id: str = "yuvalkirstain/PickScore_v1",
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(
            processor_id, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).eval().to(self.device, dtype=dtype)

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have the same length")
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {
            key: value.to(
                self.device,
                dtype=self.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in image_inputs.items()
        }
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        image_features = _feature_tensor(
            self.model.get_image_features(**image_inputs),
            ("image_embeds", "pooler_output"),
        )
        text_features = _feature_tensor(
            self.model.get_text_features(**text_inputs),
            ("text_embeds", "pooler_output"),
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = self.model.logit_scale.exp() * (image_features * text_features).sum(dim=-1)
        return scores.float().cpu().tolist()


class HPSv2Scorer:
    """Compute HPSv2.1 with the official OpenCLIP checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str) -> None:
        import open_clip

        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"HPSv2 checkpoint not found: {checkpoint}. Pass --hpsv2-checkpoint."
            )
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, device="cpu"
        )
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have the same length")
        image_inputs = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device)
        text_inputs = self.tokenizer(prompts).to(self.device)
        image_features = self.model.encode_image(image_inputs)
        text_features = self.model.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).float().cpu().tolist()
