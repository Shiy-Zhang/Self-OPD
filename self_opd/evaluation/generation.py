"""Deterministic image generation for evaluation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .common import (
    EvaluationSet,
    batched,
    generated_images_are_valid,
    image_paths,
    write_manifest,
)


def load_pipeline(
    base_model: str,
    lora_path: str | None,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool,
):
    """Load SD3.5 and optionally merge a PEFT LoRA adapter."""
    from diffusers import StableDiffusion3Pipeline

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    if lora_path and lora_path.lower() != "none":
        from peft import PeftModel

        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            lora_path,
            local_files_only=local_files_only,
        ).merge_and_unload()
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.inference_mode()
def generate_dataset(
    pipeline,
    dataset: EvaluationSet,
    output_root: str | Path,
    generation: dict[str, Any],
    batch_size: int,
    overwrite: bool = False,
) -> list[Path]:
    output_root = Path(output_root)
    directory = output_root / "images" / dataset.name
    manifest = output_root / "manifests" / f"{dataset.name}.json"
    if not overwrite and generated_images_are_valid(directory, manifest, dataset, generation):
        print(f"Reusing {len(dataset.examples)} {dataset.name} images from {directory}")
        return image_paths(directory, len(dataset.examples))

    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Existing images in {directory} do not match this run. "
            "Use --overwrite-images or provide a different --output-dir."
        )
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                path.unlink()

    prompts = dataset.prompts
    device = str(pipeline.device)
    for start, prompt_batch in tqdm(
        batched(prompts, batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=f"Generate {dataset.name}",
    ):
        generators = [
            torch.Generator(device=device).manual_seed(generation["seed"] + index)
            for index in range(start, start + len(prompt_batch))
        ]
        images = pipeline(
            prompt=prompt_batch,
            num_inference_steps=generation["num_inference_steps"],
            guidance_scale=generation["guidance_scale"],
            height=generation["height"],
            width=generation["width"],
            max_sequence_length=generation["max_sequence_length"],
            generator=generators,
        ).images
        for offset, image in enumerate(images):
            image.save(directory / f"{start + offset:06d}.png")
    write_manifest(manifest, dataset, generation)
    return image_paths(directory, len(dataset.examples))
