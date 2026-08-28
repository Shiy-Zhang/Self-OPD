"""Public Self-OPD configuration with OCR as the reward example."""

import importlib.util
import os
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("base.py")
_SPEC = importlib.util.spec_from_file_location("self_opd_base_config", _BASE_PATH)
base = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(base)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def default():
    """Default SD3.5-Medium training configuration."""
    config = base.get_config()

    config.pretrained.model = os.environ.get(
        "SELF_OPD_BASE_MODEL", "stabilityai/stable-diffusion-3.5-medium"
    )
    config.dataset = os.environ.get(
        "SELF_OPD_DATASET", str(_REPO_ROOT / "data" / "ocr")
    )
    config.run_name = "self-opd-sd35m"
    config.prompt_fn = "text_file"
    config.reward_fn = {"ocr": 1.0}

    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    num_branches = int(os.environ.get("SELF_OPD_NUM_BRANCHES", "8"))
    num_images_per_prompt = int(os.environ.get("SELF_OPD_NUM_IMAGES_PER_PROMPT", "16"))
    if num_processes <= 0 or num_branches <= 0 or num_images_per_prompt <= 0:
        raise ValueError(
            "WORLD_SIZE, SELF_OPD_NUM_BRANCHES, and "
            "SELF_OPD_NUM_IMAGES_PER_PROMPT must be positive"
        )
    if num_images_per_prompt % num_processes:
        raise ValueError("SELF_OPD_NUM_IMAGES_PER_PROMPT must be divisible by WORLD_SIZE")

    train_batch_size = num_images_per_prompt // num_processes
    denominator = num_processes * train_batch_size
    if 256 % denominator != 0:
        raise ValueError(
            "WORLD_SIZE * train_batch_size must divide 256; adjust "
            "SELF_OPD_NUM_BRANCHES or the preset"
        )

    config.self_opd.num_branches = num_branches
    config.sample.num_image_per_prompt = num_images_per_prompt
    config.sample.train_batch_size = train_batch_size
    config.train.batch_size = train_batch_size
    config.sample.num_batches_per_epoch = 256 // denominator
    if config.sample.num_batches_per_epoch % 2:
        raise ValueError("num_batches_per_epoch must be even")
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2

    config.train.lora_path = os.environ.get("SELF_OPD_WARM_LORA") or None
    output_dir = os.environ.get("SELF_OPD_OUTPUT_DIR")
    if output_dir:
        config.save_dir = output_dir
        config.logdir = output_dir
    else:
        config.save_dir = "logs/self-opd-sd35m"

    return config
