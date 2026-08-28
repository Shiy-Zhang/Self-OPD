#!/usr/bin/env python3
"""Generate images with SD3.5-Medium and a Self-OPD LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_PROMPT = (
    "A realistic photograph of a fast food drive-thru menu board at dusk, "
    'featuring a bold and colorful advertisement that reads "Try Our New Burger" '
    "with an appetizing image of the burger below, set against the backdrop of a "
    "busy suburban street."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora-path",
        default="ShiyiZhang/Self-OPD",
        help="Path or Hugging Face Hub ID of a LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base model path or Hugging Face model ID",
    )
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", action="append", help="Prompt; repeat for multiple prompts")
    prompts.add_argument("--prompt-file", help="UTF-8 text file with one prompt per line")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompt_file}")
        return prompts
    return [DEFAULT_PROMPT]


def resolve_device_and_dtype(device_name: str, dtype_name: str):
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if device == "auto":
        device = "cpu"
    if dtype_name == "auto":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype_name]
    return device, dtype


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from diffusers import StableDiffusion3Pipeline
    from peft import PeftModel

    prompts = load_prompts(args)
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        args.lora_path,
        local_files_only=args.local_files_only,
    ).merge_and_unload()
    pipeline = pipeline.to(device)

    records = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + start + index)
            for index in range(len(batch))
        ]
        images = pipeline(
            prompt=batch,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            max_sequence_length=128,
            generator=generators,
        ).images
        for index, (prompt, image) in enumerate(zip(batch, images), start=start):
            filename = f"{index:05d}.png"
            image.save(output_dir / filename)
            records.append({"index": index, "prompt": prompt, "image": filename})

    metadata = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "samples": records,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} images to {output_dir}")


if __name__ == "__main__":
    main()
