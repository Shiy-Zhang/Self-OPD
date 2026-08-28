"""Shared dataset, image-generation, and result helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class EvaluationExample:
    prompt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvaluationSet:
    name: str
    examples: list[EvaluationExample]

    @property
    def prompts(self) -> list[str]:
        return [example.prompt for example in self.examples]


def load_geneval(path: str | Path, limit: int = 0) -> EvaluationSet:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            metadata = json.loads(line)
            prompt = metadata.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            examples.append(EvaluationExample(prompt.strip(), metadata))
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No GenEval examples found in {path}")
    return EvaluationSet("geneval", examples)


def load_text_prompts(
    path: str | Path,
    name: str,
    limit: int = 0,
    require_quoted_text: bool = False,
) -> EvaluationSet:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if not prompt:
                continue
            if require_quoted_text:
                parts = prompt.split('"')
                if len(parts) < 3 or not parts[1].strip():
                    continue
            examples.append(EvaluationExample(prompt, {}))
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No prompts found in {path}")
    return EvaluationSet(name, examples)


def dataset_fingerprint(dataset: EvaluationSet) -> str:
    payload = json.dumps(
        [asdict(example) for example in dataset.examples],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_paths(
    directory: str | Path,
    expected_count: int,
    allow_extra: bool = False,
) -> list[Path]:
    directory = Path(directory)
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    ) if directory.is_dir() else []
    if len(paths) < expected_count or (len(paths) > expected_count and not allow_extra):
        raise ValueError(
            f"Expected {expected_count} images in {directory}, found {len(paths)}. "
            "Image filenames must sort in prompt order."
        )
    return paths[:expected_count]


def generated_images_are_valid(
    directory: str | Path,
    manifest_path: str | Path,
    dataset: EvaluationSet,
    generation: dict[str, Any],
) -> bool:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        image_paths(directory, len(dataset.examples))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("dataset_fingerprint") == dataset_fingerprint(dataset)
        and manifest.get("generation") == generation
        and manifest.get("count") == len(dataset.examples)
    )


def write_manifest(
    path: str | Path,
    dataset: EvaluationSet,
    generation: dict[str, Any],
) -> None:
    payload = {
        "dataset": dataset.name,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "count": len(dataset.examples),
        "generation": generation,
        "samples": [
            {
                "index": index,
                "prompt": example.prompt,
                "image": f"{index:06d}.png",
                "metadata": example.metadata,
            }
            for index, example in enumerate(dataset.examples)
        ],
    }
    write_json(path, payload)


def open_images(paths: Iterable[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def batched(items: list[Any], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
