#!/usr/bin/env python3
"""Run the complete Self-OPD evaluation protocol on a model or saved images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from self_opd.evaluation.common import (  # noqa: E402
    EvaluationSet,
    batched,
    generated_images_are_valid,
    image_paths,
    load_geneval,
    load_text_prompts,
    open_images,
    write_json,
    write_jsonl,
)
from self_opd.evaluation.generation import generate_dataset, load_pipeline  # noqa: E402


METRICS = {"geneval", "ocr", "pickscore", "hpsv2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-path", help="LoRA directory or Hub ID; omit for the base model")
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base model directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Any of: all, geneval, ocr, pickscore, hpsv2",
    )
    parser.add_argument("--geneval-data", default=str(REPO_ROOT / "data/geneval/test_metadata.jsonl"))
    parser.add_argument("--ocr-data", default=str(REPO_ROOT / "data/ocr/test.txt"))
    parser.add_argument("--drawbench-data", default=str(REPO_ROOT / "data/drawbench/test.txt"))
    parser.add_argument("--pickapic-data", default=str(REPO_ROOT / "data/pickapic/test.txt"))
    parser.add_argument("--geneval-images", help="Existing images in GenEval prompt order")
    parser.add_argument("--ocr-images", help="Existing images in OCR prompt order")
    parser.add_argument("--drawbench-images", help="Existing images in DrawBench prompt order")
    parser.add_argument("--pickapic-images", help="Existing images in Pick-a-Pic prompt order")
    parser.add_argument("--score-only", action="store_true", help="Never generate missing images")
    parser.add_argument("--generate-only", action="store_true", help="Generate images without scoring")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit per dataset; 0 uses all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument(
        "--geneval-detector-config",
        default=os.environ.get("GENEVAL_DETECTOR_CONFIG"),
        help="MMDetection Mask2Former config (or GENEVAL_DETECTOR_CONFIG)",
    )
    parser.add_argument(
        "--geneval-detector-checkpoint",
        default=os.environ.get("GENEVAL_DETECTOR_CHECKPOINT"),
        help="GenEval Mask2Former checkpoint (or GENEVAL_DETECTOR_CHECKPOINT)",
    )
    parser.add_argument(
        "--geneval-classnames",
        default=str(REPO_ROOT / "data/geneval/object_names.txt"),
    )
    parser.add_argument("--geneval-clip-model", default="ViT-L-14")
    parser.add_argument("--geneval-clip-pretrained", default="openai")
    parser.add_argument("--pickscore-processor", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--pickscore-model", default="yuvalkirstain/PickScore_v1")
    parser.add_argument(
        "--hpsv2-checkpoint",
        default=os.environ.get("HPSV2_CHECKPOINT"),
        help="HPS_v2.1_compressed.pt (or HPSV2_CHECKPOINT)",
    )
    args = parser.parse_args()
    if args.score_only and args.generate_only:
        parser.error("--score-only and --generate-only are mutually exclusive")
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")
    return args


def resolve_metrics(values: list[str]) -> set[str]:
    requested = {
        item.strip().lower()
        for value in values
        for item in value.split(",")
        if item.strip()
    }
    if "all" in requested:
        requested = set(METRICS)
    unknown = requested - METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one metric is required")
    return requested


def resolve_device_dtype(device_name: str, dtype_name: str) -> tuple[str, torch.dtype]:
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


def score_in_batches(
    name: str,
    scorer: Callable,
    paths: list[Path],
    prompts: list[str],
    batch_size: int,
) -> list[float]:
    scores = []
    paired = list(zip(paths, prompts))
    for _, batch in tqdm(
        batched(paired, batch_size),
        total=(len(paired) + batch_size - 1) // batch_size,
        desc=name,
    ):
        batch_paths = [item[0] for item in batch]
        batch_prompts = [item[1] for item in batch]
        images = open_images(batch_paths)
        try:
            scores.extend(float(value) for value in scorer(images, batch_prompts))
        finally:
            for image in images:
                image.close()
    return scores


def save_scores(
    output_dir: Path,
    metric: str,
    dataset: EvaluationSet,
    paths: list[Path],
    scores: list[float],
) -> None:
    write_jsonl(
        output_dir / "scores" / f"{metric}_{dataset.name}.jsonl",
        (
            {
                "index": index,
                "prompt": example.prompt,
                "image": str(path),
                "score": score,
            }
            for index, (example, path, score) in enumerate(
                zip(dataset.examples, paths, scores)
            )
        ),
    )


def main() -> None:
    args = parse_args()
    metrics = resolve_metrics(args.metrics)
    device, dtype = resolve_device_dtype(args.device, args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needs_same_image_sets = bool(metrics & {"pickscore", "hpsv2"})
    datasets: dict[str, EvaluationSet] = {}
    if "geneval" in metrics or needs_same_image_sets:
        datasets["geneval"] = load_geneval(args.geneval_data, args.max_samples)
    if "ocr" in metrics or needs_same_image_sets:
        datasets["ocr"] = load_text_prompts(
            args.ocr_data, "ocr", args.max_samples, require_quoted_text=True
        )
    if "pickscore" in metrics:
        datasets["drawbench"] = load_text_prompts(
            args.drawbench_data, "drawbench", args.max_samples
        )
    if "hpsv2" in metrics:
        datasets["pickapic"] = load_text_prompts(
            args.pickapic_data, "pickapic", args.max_samples
        )

    generation = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "max_sequence_length": args.max_sequence_length,
    }
    external_directories = {
        "geneval": args.geneval_images,
        "ocr": args.ocr_images,
        "drawbench": args.drawbench_images,
        "pickapic": args.pickapic_images,
    }
    paths_by_dataset: dict[str, list[Path]] = {}
    missing = []
    for name, dataset in datasets.items():
        external = external_directories[name]
        if external:
            paths_by_dataset[name] = image_paths(
                external,
                len(dataset.examples),
                allow_extra=bool(args.max_samples),
            )
        else:
            generated_dir = output_dir / "images" / name
            manifest = output_dir / "manifests" / f"{name}.json"
            if generated_images_are_valid(generated_dir, manifest, dataset, generation):
                paths_by_dataset[name] = image_paths(generated_dir, len(dataset.examples))
            else:
                missing.append(name)

    if missing:
        if args.score_only:
            raise FileNotFoundError(f"Missing complete image sets: {', '.join(missing)}")
        pipeline = load_pipeline(
            args.base_model, args.lora_path, device, dtype, args.local_files_only
        )
        for name in missing:
            paths_by_dataset[name] = generate_dataset(
                pipeline,
                datasets[name],
                output_dir,
                generation,
                args.generation_batch_size,
                overwrite=args.overwrite_images,
            )
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.generate_only:
        print(f"Generated evaluation images in {output_dir / 'images'}")
        return

    results: dict = {
        "protocol": {
            "same_test_images": "Unweighted mean of the GenEval-set mean and OCR-set mean",
            "separate_test_set": {
                "pickscore": "Mean over held-out DrawBench prompts",
                "hpsv2": "Mean over held-out Pick-a-Pic prompts",
            },
        },
        "generation": generation,
        "datasets": {name: len(dataset.examples) for name, dataset in datasets.items()},
    }

    if "geneval" in metrics:
        if not args.geneval_detector_config or not args.geneval_detector_checkpoint:
            raise ValueError(
                "GenEval requires --geneval-detector-config and "
                "--geneval-detector-checkpoint (or the corresponding environment variables)."
            )
        from self_opd.evaluation.geneval import GenevalScorer, aggregate_geneval

        scorer = GenevalScorer(
            detector_config=args.geneval_detector_config,
            detector_checkpoint=args.geneval_detector_checkpoint,
            classnames=args.geneval_classnames,
            device=device,
            clip_architecture=args.geneval_clip_model,
            clip_pretrained=args.geneval_clip_pretrained,
        )
        strict, continuous = [], []
        dataset = datasets["geneval"]
        indexed = list(zip(paths_by_dataset["geneval"], dataset.examples))
        for _, batch in tqdm(
            batched(indexed, args.score_batch_size),
            total=(len(indexed) + args.score_batch_size - 1) // args.score_batch_size,
            desc="GenEval",
        ):
            images = open_images(item[0] for item in batch)
            try:
                batch_strict, batch_continuous = scorer(
                    images, [item[1].metadata for item in batch]
                )
            finally:
                for image in images:
                    image.close()
            strict.extend(batch_strict)
            continuous.extend(batch_continuous)
        results["geneval"] = aggregate_geneval(
            [example.metadata for example in dataset.examples], strict, continuous
        )
        write_jsonl(
            output_dir / "scores/geneval.jsonl",
            (
                {
                    "index": index,
                    "prompt": example.prompt,
                    "image": str(path),
                    "tag": example.metadata.get("tag"),
                    "strict": strict_score,
                    "continuous": continuous_score,
                }
                for index, (example, path, strict_score, continuous_score) in enumerate(
                    zip(dataset.examples, paths_by_dataset["geneval"], strict, continuous)
                )
            ),
        )
        del scorer
        write_json(output_dir / "results.json", results)

    if "ocr" in metrics:
        from self_opd.ocr_reward import OCRScorer

        scorer = OCRScorer()
        dataset = datasets["ocr"]
        scores = score_in_batches(
            "OCR", scorer, paths_by_dataset["ocr"], dataset.prompts, args.score_batch_size
        )
        results["ocr"] = {"accuracy": float(np.mean(scores)), "count": len(scores)}
        save_scores(output_dir, "ocr", dataset, paths_by_dataset["ocr"], scores)
        del scorer
        write_json(output_dir / "results.json", results)

    preference_results = {}
    for metric in ("pickscore", "hpsv2"):
        if metric not in metrics:
            continue
        if metric == "pickscore":
            from self_opd.evaluation.preference import PickScoreScorer

            scorer = PickScoreScorer(
                device=device,
                dtype=torch.float32,
                processor_id=args.pickscore_processor,
                model_id=args.pickscore_model,
                local_files_only=args.local_files_only,
            )
        else:
            if not args.hpsv2_checkpoint:
                raise ValueError(
                    "HPSv2 requires --hpsv2-checkpoint or HPSV2_CHECKPOINT."
                )
            from self_opd.evaluation.preference import HPSv2Scorer

            scorer = HPSv2Scorer(args.hpsv2_checkpoint, device)

        separate_dataset = "drawbench" if metric == "pickscore" else "pickapic"
        means = {}
        for name in ("geneval", "ocr", separate_dataset):
            dataset = datasets[name]
            scores = score_in_batches(
                f"{metric} ({name})",
                scorer,
                paths_by_dataset[name],
                dataset.prompts,
                args.score_batch_size,
            )
            means[name] = float(np.mean(scores))
            save_scores(output_dir, metric, dataset, paths_by_dataset[name], scores)
        preference_results[metric] = {
            "same_test_images": float(np.mean([means["geneval"], means["ocr"]])),
            "separate_test_set": means[separate_dataset],
            "separate_dataset": separate_dataset,
            "by_dataset": means,
        }
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results["preference"] = preference_results
        write_json(output_dir / "results.json", results)

    table = {}
    if "geneval" in results:
        table.update(
            geneval_strict=results["geneval"]["strict"],
            geneval_continuous=results["geneval"]["continuous"],
        )
    if "ocr" in results:
        table["ocr"] = results["ocr"]["accuracy"]
    for metric, values in preference_results.items():
        table[f"{metric}_same_test_images"] = values["same_test_images"]
        table[f"{metric}_separate_test_set"] = values["separate_test_set"]
    results["table"] = table
    write_json(output_dir / "results.json", results)
    print(f"Evaluation complete: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
