"""Reward registry and weighted reward composition for Self-OPD."""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

RewardFunction = Callable[[Any, list[str], list[dict]], tuple[list[float], dict]]
RewardFactory = Callable[[torch.device], RewardFunction]

_REWARD_REGISTRY: dict[str, RewardFactory] = {}
_LOADED_MODULES: set[str] = set()


def register_reward(name: str) -> Callable[[RewardFactory], RewardFactory]:
    """Register a reward factory under ``name``."""

    def decorator(factory: RewardFactory) -> RewardFactory:
        if name in _REWARD_REGISTRY:
            raise ValueError(f"Reward '{name}' is already registered")
        _REWARD_REGISTRY[name] = factory
        return factory

    return decorator


def available_rewards() -> tuple[str, ...]:
    return tuple(sorted(_REWARD_REGISTRY))


def load_reward_modules(modules: str | None = None) -> None:
    """Import comma-separated modules that register additional rewards."""
    modules = modules if modules is not None else os.environ.get("SELF_OPD_REWARD_MODULES", "")
    for module_name in (item.strip() for item in modules.split(",")):
        if module_name and module_name not in _LOADED_MODULES:
            importlib.import_module(module_name)
            _LOADED_MODULES.add(module_name)


def _images_to_numpy(images: Any) -> Any:
    if not isinstance(images, torch.Tensor):
        return images
    array = (images.detach().float() * 255).round().clamp(0, 255).to(torch.uint8)
    return array.cpu().numpy().transpose(0, 2, 3, 1)


@register_reward("ocr")
def create_ocr_reward(device: torch.device) -> RewardFunction:
    """Build the included CPU PaddleOCR reward example."""
    del device
    from self_opd.ocr_reward import OCRScorer

    scorer = OCRScorer()
    # Training submits multiple batches to a thread pool. PaddleOCR/OpenCV is
    # not thread-safe, so serialize access to the shared scorer instance.
    lock = threading.Lock()

    def score(images, prompts, metadata):
        del metadata
        with lock:
            values = scorer(_images_to_numpy(images), prompts)
        return [float(value) for value in values], {}

    return score


def build_multi_reward(
    device: torch.device,
    reward_weights: Mapping[str, float],
) -> Callable:
    """Build a weighted composition of all configured rewards.

    Each registered reward is reported separately. ``avg`` is their weighted
    sum and is used for trajectory-level grouping. Branch-level Self-OPD also
    receives the individual values, so it can combine multiple reward models.
    """
    if not reward_weights:
        raise ValueError("At least one reward must be configured")

    load_reward_modules()

    unknown = set(reward_weights) - set(_REWARD_REGISTRY)
    if unknown:
        choices = ", ".join(available_rewards()) or "<none>"
        raise KeyError(f"Unknown rewards {sorted(unknown)}; available: {choices}")

    functions = {
        name: _REWARD_REGISTRY[name](device) for name in reward_weights
    }

    def score(images, prompts, metadata, ref_images=None, only_strict=True):
        del ref_images, only_strict
        details: dict[str, list[float]] = {}
        weighted_total: np.ndarray | None = None

        for name, weight in reward_weights.items():
            values, _ = functions[name](images, prompts, metadata)
            values_array = np.asarray(values, dtype=np.float32)
            details[name] = values_array.tolist()
            contribution = float(weight) * values_array
            weighted_total = contribution if weighted_total is None else weighted_total + contribution

        assert weighted_total is not None
        details["avg"] = weighted_total.tolist()
        return details, {}

    return score
