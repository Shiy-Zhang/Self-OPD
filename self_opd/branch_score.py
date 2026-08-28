"""Branch sampling and all-branch target construction for Self-OPD."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from self_opd.diffusers_patch.sd3_sde import sde_step


@dataclass
class BranchSamplingConfig:
    """Runtime settings for on-policy branch sampling."""

    num_branches: int = 8
    score_weights: dict[str, float] | None = None
    cfg_scale: float = 4.5
    do_cfg: bool = True
    branch_weight_clip: float = 1.0
    direction_aware: bool = True


@torch.no_grad()
def _ode_rollout_to_x0(
    transformer,
    scheduler,
    latent: torch.Tensor,
    start_step: int,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    cfg_scale: float,
    do_cfg: bool,
    autocast,
) -> torch.Tensor:
    """Complete a latent to x0 with deterministic ODE steps."""
    current = latent
    for step in range(start_step, timesteps.shape[0]):
        timestep = timesteps[step]
        with autocast():
            if do_cfg:
                prediction = transformer(
                    hidden_states=torch.cat([current] * 2),
                    timestep=torch.cat([timestep.expand(current.shape[0])] * 2),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                unconditional, conditional = prediction.chunk(2)
                prediction = unconditional + cfg_scale * (conditional - unconditional)
            else:
                prediction = transformer(
                    hidden_states=current,
                    timestep=timestep.expand(current.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
        current, _, _ = sde_step(
            scheduler,
            prediction.float(),
            timestep.expand(current.shape[0]),
            current.float(),
            noise_level=0.0,
        )
    return current


@torch.no_grad()
def _decode_and_score(
    latents: torch.Tensor,
    pipeline,
    prompts: list[str],
    metadata: list[dict],
    score_fn: Callable,
) -> dict[str, torch.Tensor]:
    """Decode clean latents and return each reward as a tensor."""
    scaled = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    scaled = scaled.to(dtype=pipeline.vae.dtype)
    images = pipeline.vae.decode(scaled, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type="pt")
    rewards, _ = score_fn(images, prompts, metadata, only_strict=True)
    return {
        name: torch.as_tensor(values, device=latents.device).float()
        for name, values in rewards.items()
        if name != "avg"
    }


def _sample_one_step_branches(
    scheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    latent: torch.Tensor,
    noise_level: float,
    num_branches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K stochastic next latents around one shared policy mean."""
    _, policy_mean, std_dev = sde_step(
        scheduler,
        model_output.float(),
        timestep,
        latent.float(),
        noise_level=noise_level,
    )
    step_indices = [scheduler.index_for_timestep(value) for value in timestep]
    sigma = scheduler.sigmas[step_indices].view(-1, *([1] * (latent.ndim - 1)))
    sigma_next = scheduler.sigmas[[index + 1 for index in step_indices]].view(
        -1, *([1] * (latent.ndim - 1))
    )
    sqrt_negative_dt = torch.sqrt(-(sigma_next - sigma))

    branches = [
        policy_mean
        + std_dev
        * sqrt_negative_dt
        * torch.randn(latent.shape, device=latent.device, dtype=torch.float32)
        for _ in range(num_branches)
    ]
    stacked = torch.stack(branches, dim=1)
    flattened = stacked.reshape(latent.shape[0] * num_branches, *latent.shape[1:])
    return flattened, policy_mean


@torch.no_grad()
def sample_branches_and_score(
    *,
    transformer,
    pipeline,
    scheduler,
    x_j: torch.Tensor,
    j: int,
    timesteps: torch.Tensor,
    embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    prompts: list[str],
    metadata: list[dict],
    score_fn: Callable,
    noise_level: float,
    cfg: BranchSamplingConfig,
    autocast,
) -> dict:
    """Sample K SDE branches, roll them to x0, and evaluate all rewards."""
    batch_size = x_j.shape[0]
    num_branches = cfg.num_branches
    timestep = timesteps[j].expand(batch_size)

    with autocast():
        if cfg.do_cfg:
            prediction = transformer(
                hidden_states=torch.cat([x_j] * 2),
                timestep=torch.cat([timestep] * 2),
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + cfg.cfg_scale * (conditional - unconditional)
        else:
            prediction = transformer(
                hidden_states=x_j,
                timestep=timestep,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

    branch_latents, policy_mean = _sample_one_step_branches(
        scheduler, prediction, timestep, x_j, noise_level, num_branches
    )

    if cfg.do_cfg:
        unconditional, conditional = embeds.chunk(2)
        unconditional_pooled, conditional_pooled = pooled_embeds.chunk(2)
        branch_embeds = torch.cat(
            [
                unconditional.repeat_interleave(num_branches, 0),
                conditional.repeat_interleave(num_branches, 0),
            ]
        )
        branch_pooled = torch.cat(
            [
                unconditional_pooled.repeat_interleave(num_branches, 0),
                conditional_pooled.repeat_interleave(num_branches, 0),
            ]
        )
    else:
        branch_embeds = embeds.repeat_interleave(num_branches, dim=0)
        branch_pooled = pooled_embeds.repeat_interleave(num_branches, dim=0)

    branch_x0 = _ode_rollout_to_x0(
        transformer,
        scheduler,
        branch_latents,
        j + 1,
        timesteps,
        branch_embeds,
        branch_pooled,
        cfg.cfg_scale,
        cfg.do_cfg,
        autocast,
    )

    ode_next, _, _ = sde_step(
        scheduler, prediction.float(), timestep, x_j.float(), noise_level=0.0
    )
    ode_x0 = _ode_rollout_to_x0(
        transformer,
        scheduler,
        ode_next,
        j + 1,
        timesteps,
        embeds,
        pooled_embeds,
        cfg.cfg_scale,
        cfg.do_cfg,
        autocast,
    )

    repeated_prompts = [prompt for prompt in prompts for _ in range(num_branches)]
    repeated_metadata = [item for item in metadata for _ in range(num_branches)]
    flat_scores = _decode_and_score(
        branch_x0, pipeline, repeated_prompts, repeated_metadata, score_fn
    )
    ode_scores = _decode_and_score(ode_x0, pipeline, prompts, metadata, score_fn)

    return {
        "branch_x_jp1": branch_latents.view(
            batch_size, num_branches, *branch_latents.shape[1:]
        ),
        "per_scorer": {
            name: values.view(batch_size, num_branches)
            for name, values in flat_scores.items()
        },
        "ode_per_scorer": ode_scores,
        "policy_prev_sample_mean": policy_mean,
    }


def select_all_branches(
    branch_package: dict,
    policy_mean: torch.Tensor,
    *,
    cfg: BranchSamplingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build signed, direction-aware weights for every sampled branch."""
    per_reward = branch_package["per_scorer"]
    ode_per_reward = branch_package["ode_per_scorer"]
    branch_targets = branch_package["branch_x_jp1"]
    batch_size, num_branches = branch_targets.shape[:2]
    device = branch_targets.device

    weights = cfg.score_weights or {
        name: 1.0 / len(per_reward) for name in per_reward
    }
    branch_rewards = torch.zeros(batch_size, num_branches, device=device)
    ode_rewards = torch.zeros(batch_size, device=device)
    for name, values in per_reward.items():
        weight = float(weights.get(name, 0.0))
        branch_rewards += weight * values.float()
        ode_rewards += weight * ode_per_reward[name].float()

    reward_std = branch_rewards.std(dim=1, keepdim=True).clamp(min=1e-4)
    advantages = (branch_rewards - ode_rewards.unsqueeze(1)) / reward_std
    best_indices = advantages.max(dim=1).indices

    gather_index = best_indices.view(batch_size, 1, 1, 1, 1).expand(
        -1, 1, *branch_targets.shape[2:]
    )
    best_target = torch.gather(branch_targets, 1, gather_index).squeeze(1)
    best_direction = torch.nn.functional.normalize(
        (best_target - policy_mean.detach()).flatten(1), dim=1
    )

    cosine_similarity = torch.zeros(batch_size, num_branches, device=device)
    for branch_index in range(num_branches):
        direction = torch.nn.functional.normalize(
            (branch_targets[:, branch_index] - policy_mean.detach()).flatten(1), dim=1
        )
        cosine_similarity[:, branch_index] = (direction * best_direction).sum(dim=1)

    if cfg.direction_aware:
        negative = (advantages < 0).float()
        direction_coefficient = (
            negative * 0.5 * (1.0 - cosine_similarity) + (1.0 - negative)
        )
    else:
        direction_coefficient = torch.ones_like(advantages)

    raw_weights = advantages * direction_coefficient
    branch_weights = raw_weights.clamp(
        -cfg.branch_weight_clip, cfg.branch_weight_clip
    )
    diagnostics = {
        "cos_sim": cosine_similarity,
        "dir_coeff": direction_coefficient,
        "raw_weight": raw_weights,
        "r_branches": branch_rewards,
        "r_ode": ode_rewards,
        "best_idx": best_indices,
    }
    return branch_targets, branch_weights, advantages, diagnostics
