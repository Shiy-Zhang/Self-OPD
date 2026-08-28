"""Shared data and checkpoint helpers for Self-OPD training."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.data import Dataset, Sampler

from self_opd.diffusers_patch.sd3_text_encoding import encode_prompt


class TextPromptDataset(Dataset):
    """Read prompts from a line-delimited ``<split>.txt`` file."""

    def __init__(self, dataset_dir: str, split: str = "train") -> None:
        path = os.path.join(dataset_dir, f"{split}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt split not found: {path}")
        with open(path, encoding="utf-8") as handle:
            self.prompts = [line.strip() for line in handle if line.strip()]
        if not self.prompts:
            raise ValueError(f"Prompt split is empty: {path}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        return {"prompt": self.prompts[index], "metadata": {}}

    @staticmethod
    def collate_fn(examples: list[dict]) -> tuple[list[str], list[dict]]:
        return (
            [example["prompt"] for example in examples],
            [example["metadata"] for example in examples],
        )


class DistributedKRepeatSampler(Sampler):
    """Repeat each sampled prompt K times and shard batches across workers."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        total_samples = num_replicas * batch_size
        if total_samples % k:
            raise ValueError(
                f"num_replicas * batch_size ({total_samples}) must be divisible by k ({k})"
            )
        self.unique_samples = total_samples // k

    def __iter__(self):
        while True:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator)[
                : self.unique_samples
            ].tolist()
            repeated = [index for index in indices for _ in range(self.k)]
            permutation = torch.randperm(len(repeated), generator=generator).tolist()
            shuffled = [repeated[index] for index in permutation]
            start = self.rank * self.batch_size
            yield shuffled[start : start + self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def compute_text_embeddings(
    prompts,
    text_encoders,
    tokenizers,
    max_sequence_length: int,
    device,
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompts, max_sequence_length
        )
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.asarray(prompts)
    _, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    reward_groups = np.split(grouped, np.cumsum(counts)[:-1])
    standard_deviations = np.asarray([np.std(group) for group in reward_groups])
    return float(np.mean(standard_deviations == 0)), float(standard_deviations.mean())


def create_generator(prompts: list[str], base_seed: int) -> list[torch.Generator]:
    generators = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode()).digest()
        prompt_seed = int.from_bytes(digest[:4], "big")
        generators.append(torch.Generator().manual_seed((base_seed + prompt_seed) % 2**31))
    return generators


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def save_checkpoint(
    save_dir,
    transformer,
    global_step,
    accelerator,
    ema,
    trainable_parameters,
    config,
) -> None:
    checkpoint_dir = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}", "lora")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not accelerator.is_main_process:
        return
    if config.train.ema:
        ema.copy_ema_to(trainable_parameters, store_temp=True)
    unwrap_model(transformer, accelerator).save_pretrained(checkpoint_dir)
    if config.train.ema:
        ema.copy_temp_to(trainable_parameters)
