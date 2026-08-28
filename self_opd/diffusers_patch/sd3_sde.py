# Adapted from https://github.com/kvablack/ddpo-pytorch.
"""Stochastic transition for the SD3 flow-matching scheduler."""

from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor


def sde_step(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    noise_level: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one reverse-SDE step and return sample, mean, and noise scale."""
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_indices = [scheduler.index_for_timestep(value) for value in timestep]
    next_indices = [index + 1 for index in step_indices]
    shape = (-1, *([1] * (sample.ndim - 1)))
    sigma = scheduler.sigmas[step_indices].to(sample.device, sample.dtype).view(shape)
    sigma_next = scheduler.sigmas[next_indices].to(sample.device, sample.dtype).view(shape)
    sigma_max = scheduler.sigmas[1].item()
    delta = sigma_next - sigma

    denominator = 1 - torch.where(sigma == 1, sigma_max, sigma)
    noise_scale = torch.sqrt(sigma / denominator) * noise_level
    prev_sample_mean = (
        sample * (1 + noise_scale.square() / (2 * sigma) * delta)
        + model_output
        * (1 + noise_scale.square() * (1 - sigma) / (2 * sigma))
        * delta
    )
    if prev_sample is None:
        noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + noise_scale * torch.sqrt(-delta) * noise
    return prev_sample, prev_sample_mean, noise_scale
