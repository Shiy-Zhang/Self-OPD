"""Train Self-OPD with one or more registered reward models."""

import os
import sys
import contextlib
import datetime
import json
from collections import defaultdict
from concurrent import futures
from datetime import timedelta
from functools import partial

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import tqdm as _tqdm
import wandb
from absl import app, flags
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusion3Pipeline
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader

from scripts.training_utils import (
    DistributedKRepeatSampler,
    TextPromptDataset,
    calculate_zero_std_ratio,
    compute_text_embeddings,
    create_generator,
    save_checkpoint,
)
from self_opd.branch_score import (
    BranchSamplingConfig,
    sample_branches_and_score,
    select_all_branches,
)
from self_opd.diffusers_patch.sd3_pipeline import sample_trajectory
from self_opd.diffusers_patch.sd3_sde import sde_step
from self_opd.ema import EMAModuleWrapper
from self_opd.rewards import build_multi_reward
from self_opd.stat_tracking import PerPromptStatTracker


tqdm = partial(_tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "configs/base.py", "Path to a Self-OPD training configuration."
)

logger = get_logger(__name__)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return getattr(model, "_orig_mod", model)


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name + "_" + unique_id) if config.run_name else unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    _tps = getattr(config.self_opd, "timesteps_per_step", 0)
    ts_per_batch = int(_tps) if _tps and _tps < num_train_timesteps else num_train_timesteps

    run_dir = os.path.join(config.logdir, config.run_name)
    accelerator_config = ProjectConfiguration(
        project_dir=run_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * ts_per_batch,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=90))],
    )
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "self-opd"),
            mode=os.environ.get("WANDB_MODE", "disabled"),
            name=config.run_name,
            config=config.to_dict(),
        )
    set_seed(config.seed, device_specific=True)

    logger.info("Loading the diffusion pipeline")
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1, disable=not accelerator.is_local_main_process,
        leave=False, desc="Timestep", dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer,
                config.train.lora_path,
                is_trainable=True,
            )
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if not transformer_trainable_parameters:
        raise RuntimeError(
            "The transformer has no trainable parameters. Check the LoRA configuration "
            "or load a warm-start adapter with is_trainable=True."
        )
    trainable_parameter_count = sum(parameter.numel() for parameter in transformer_trainable_parameters)
    logger.info("Trainable transformer parameters: %s", f"{trainable_parameter_count:,}")
    ema = EMAModuleWrapper(
        transformer_trainable_parameters, decay=0.9, update_step_interval=8,
        device=accelerator.device,
    )

    # Branches are sampled on-policy from the current transformer.
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    reward_fn = build_multi_reward(accelerator.device, config.reward_fn)

    if config.prompt_fn != "text_file":
        raise NotImplementedError(f"Unsupported prompt source: {config.prompt_fn}")
    ds_cls = TextPromptDataset

    train_dataset = ds_cls(config.dataset, "train")
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=1,
        collate_fn=ds_cls.collate_fn,
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader
    )

    so = config.self_opd
    branch_config = BranchSamplingConfig(
        num_branches=int(so.num_branches),
        score_weights=dict(so.score_weights) if so.score_weights else dict(config.reward_fn),
        cfg_scale=float(config.sample.guidance_scale),
        do_cfg=bool(config.train.cfg),
        branch_weight_clip=float(so.branch_weight_clip),
        direction_aware=bool(so.direction_aware) if "direction_aware" in so else True,
    )

    executor = futures.ThreadPoolExecutor(max_workers=8)
    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    total_samples = 0
    samples_per_epoch = (
        config.sample.num_batches_per_epoch
        * config.sample.train_batch_size
        * accelerator.num_processes
    )

    while True:
        pipeline.transformer.eval()
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_checkpoint(config.save_dir, transformer, global_step, accelerator,
                            ema, transformer_trainable_parameters, config)

        pipeline.transformer.eval()
        samples = []
        for i in tqdm(range(config.sample.num_batches_per_epoch),
                      desc=f"Epoch {epoch}: sampling",
                      disable=not accelerator.is_local_main_process, position=0):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers,
                max_sequence_length=128, device=accelerator.device,
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256,
                truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)

            generator = (
                create_generator(prompts, base_seed=epoch * 10000 + i)
                if config.sample.same_latent else None
            )
            with autocast():
                with torch.no_grad():
                    images, latents = sample_trajectory(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution, width=config.resolution,
                        noise_level=config.sample.trajectory_noise_level,
                        generator=generator,
                    )
            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)

            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "prompts_text": prompts,
                "prompt_metadata": prompt_metadata,
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "rewards": rewards,
            })

        for sample in samples:
            rewards, _ = sample["rewards"].result()
            sample["rewards"] = {
                k: torch.as_tensor(v, device=accelerator.device).float()
                for k, v in rewards.items()
            }

        # Stash text-only fields before tensor concat (they aren't tensors).
        prompts_text_per_batch = [s.pop("prompts_text") for s in samples]
        metadata_per_batch = [s.pop("prompt_metadata") for s in samples]

        samples_concat = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {sk: torch.cat([s[k][sk] for s in samples], dim=0) for sk in samples[0][k]}
            for k in samples[0].keys()
        }
        # Flat lists
        prompts_text_flat = [p for batch in prompts_text_per_batch for p in batch]
        metadata_flat = [m for batch in metadata_per_batch for m in batch]

        samples_concat["rewards"]["ori_avg"] = samples_concat["rewards"]["avg"]
        samples_concat["rewards"]["avg"] = samples_concat["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = {k: accelerator.gather(v).cpu().numpy() for k, v in samples_concat["rewards"].items()}
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples_concat["prompt_ids"]).cpu().numpy()
            decoded_prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(decoded_prompts, gathered_rewards["avg"])
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(decoded_prompts, gathered_rewards)
            if accelerator.is_main_process:
                gs, tpn = stat_tracker.get_stats()
                wandb.log(
                    {
                        "group_size": gs,
                        "trained_prompt_num": tpn,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            adv_np = gathered_rewards["avg"]
            advantages = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-4)

        advantages = torch.as_tensor(advantages)
        samples_concat["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        del samples_concat["rewards"], samples_concat["prompt_ids"]

        # Mask out zero-advantage prompts and re-balance to a multiple of num_batches.
        mask = (samples_concat["advantages"].abs().sum(dim=1) != 0)
        nb = config.sample.num_batches_per_epoch
        true_count = mask.sum().item()
        if true_count == 0:
            # All prompts had zero variance in this epoch (e.g. all OCR scores 0).
            # Skip the training step rather than crash on a 0-element reshape.
            if accelerator.is_main_process:
                print(f"[self-opd] epoch {epoch}: all advantages zero, skipping training")
            epoch += 1
            continue
        if true_count % nb != 0:
            false_idx = torch.where(~mask)[0]
            n_change = nb - (true_count % nb)
            if len(false_idx) >= n_change:
                ridx = torch.randperm(len(false_idx))[:n_change]
                mask[false_idx[ridx]] = True

        kept_indices = torch.where(mask)[0].tolist()
        samples_concat = {k: v[mask] for k, v in samples_concat.items()}
        prompts_text_flat = [prompts_text_flat[i] for i in kept_indices]
        metadata_flat = [metadata_flat[i] for i in kept_indices]

        total_batch_size, num_timesteps = samples_concat["timesteps"].shape
        assert num_timesteps == config.sample.num_steps
        if total_batch_size < nb:
            if accelerator.is_main_process:
                print(f"[self-opd] epoch {epoch}: only {total_batch_size} kept samples (<{nb} batches), skipping")
            epoch += 1
            continue

        # ---------------- TRAINING ----------------
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples_concat = {k: v[perm] for k, v in samples_concat.items()}
            perm_cpu = perm.cpu().tolist()
            prompts_text_flat = [prompts_text_flat[i] for i in perm_cpu]
            metadata_flat = [metadata_flat[i] for i in perm_cpu]

            per_batch_size = total_batch_size // config.sample.num_batches_per_epoch
            samples_batched = {
                k: v.reshape(-1, per_batch_size, *v.shape[1:])
                for k, v in samples_concat.items()
            }
            samples_batched_list = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)

            for i, sample in tqdm(list(enumerate(samples_batched_list)),
                                  desc=f"Epoch {epoch}.{inner_epoch}: training",
                                  position=0, disable=not accelerator.is_local_main_process):
                batch_prompts = prompts_text_flat[i * per_batch_size:(i + 1) * per_batch_size]
                batch_metadata = metadata_flat[i * per_batch_size:(i + 1) * per_batch_size]

                if config.train.cfg:
                    embeds = torch.cat([
                        train_neg_prompt_embeds[:len(sample["prompt_embeds"])],
                        sample["prompt_embeds"],
                    ])
                    pooled_embeds = torch.cat([
                        train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])],
                        sample["pooled_prompt_embeds"],
                    ])
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                # Choose timesteps to train on.  Optionally subsample.
                all_train_timesteps = list(range(num_train_timesteps))
                if so.timesteps_per_step and so.timesteps_per_step < num_train_timesteps:
                    g = torch.Generator(device="cpu").manual_seed(global_step + i)
                    tw = getattr(so, "timestep_weights", [])
                    if tw:
                        w = torch.tensor(tw[:num_train_timesteps], dtype=torch.float32)
                        w = w / w.sum()
                        chosen = torch.multinomial(w, int(so.timesteps_per_step), replacement=False, generator=g).tolist()
                    else:
                        chosen = torch.randperm(num_train_timesteps, generator=g)[: int(so.timesteps_per_step)].tolist()
                    train_timesteps = sorted(chosen)
                else:
                    train_timesteps = all_train_timesteps
                if so.timestep_subset:
                    train_timesteps = [j for j in train_timesteps if j in set(so.timestep_subset)]

                for j in tqdm(train_timesteps, desc="Timestep", position=1,
                              leave=False, disable=not accelerator.is_local_main_process):
                    with accelerator.accumulate(transformer):
                        x_j_cur = sample["latents"][:, j].float()      # [B, C, H, W]
                        t_j_cur = sample["timesteps"][:, j]
                        B_cur = x_j_cur.shape[0]

                        # ---------- Student forward (with grad) -> mu_theta ----------
                        with autocast():
                            if config.train.cfg:
                                noise_pred = transformer(
                                    hidden_states=torch.cat([sample["latents"][:, j]] * 2),
                                    timestep=torch.cat([sample["timesteps"][:, j]] * 2),
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    return_dict=False,
                                )[0]
                                u, c = noise_pred.chunk(2)
                                noise_pred = u + config.sample.guidance_scale * (c - u)
                            else:
                                noise_pred = transformer(
                                    hidden_states=sample["latents"][:, j],
                                    timestep=sample["timesteps"][:, j],
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    return_dict=False,
                                )[0]

                        # The student prediction remains attached for the regression loss.
                        branch_noise_level = float(so.branch_noise_level)
                        _, mu_theta, std_dev_t = sde_step(
                            pipeline.scheduler, noise_pred.float(), t_j_cur,
                            x_j_cur, noise_level=branch_noise_level,
                        )
                        step_idx = [pipeline.scheduler.index_for_timestep(t) for t in t_j_cur]
                        prev_step_idx = [s + 1 for s in step_idx]
                        sigma_j = pipeline.scheduler.sigmas[step_idx].view(-1, 1, 1, 1).to(x_j_cur.device)
                        sigma_jp1 = pipeline.scheduler.sigmas[prev_step_idx].view(-1, 1, 1, 1).to(x_j_cur.device)
                        abs_dt = (sigma_j - sigma_jp1).abs().clamp(min=1e-8)
                        denom = (2 * (std_dev_t ** 2) * abs_dt).clamp(min=1e-6)

                        # ---------- Branch sampling & scoring (on-policy, no grad) ----------
                        target_transformer = unwrap_model(transformer, accelerator)
                        with torch.no_grad():
                            full_timesteps = pipeline.scheduler.timesteps.to(accelerator.device)
                            branch_pkg = sample_branches_and_score(
                                transformer=target_transformer,
                                pipeline=pipeline,
                                scheduler=pipeline.scheduler,
                                x_j=sample["latents"][:, j],
                                j=j,
                                timesteps=full_timesteps,
                                embeds=embeds,
                                pooled_embeds=pooled_embeds,
                                prompts=batch_prompts,
                                metadata=batch_metadata,
                                score_fn=reward_fn,
                                noise_level=branch_noise_level,
                                cfg=branch_config,
                                autocast=autocast,
                            )

                        # All-branch, advantage-weighted target construction.
                        branch_targets, per_branch_weight, adv, diag = select_all_branches(
                            branch_pkg, mu_theta.detach(), cfg=branch_config,
                        )
                        branch_targets = branch_targets.float()  # [B, K, C, H, W]
                        B_cur, K_cur = branch_targets.shape[:2]

                        # For each branch k: adv_k * dir_k * ||mu_theta - x_jp1_k||^2 / denom
                        per_branch_loss = torch.zeros(B_cur, K_cur, device=accelerator.device)
                        for k in range(K_cur):
                            diff_sq_k = (mu_theta - branch_targets[:, k].detach()) ** 2
                            mse_k = (diff_sq_k / denom).mean(dim=(1, 2, 3))  # [B]
                            per_branch_loss[:, k] = per_branch_weight[:, k] * mse_k

                        loss = per_branch_loss.mean()

                        # ---------- Logging ----------
                        info["loss"].append(loss)
                        info["advantage_best_mean"].append(adv.max(dim=1).values.mean())
                        info["advantage_mean"].append(adv.mean())
                        info["pos_branch_ratio"].append((adv > 0).float().mean())
                        info["collision_rate"].append(((diag["r_branches"] - diag["r_ode"].unsqueeze(1)).abs() < 1e-4).float().mean())
                        info["best_sde_minus_ode"].append((diag["r_branches"].max(dim=1).values - diag["r_ode"]).mean())
                        info["weight_abs_mean"].append(per_branch_weight.abs().mean())

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        if accelerator.sync_gradients and config.train.ema:
                            ema.step(transformer_trainable_parameters, global_step)
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info_avg = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info_avg = accelerator.reduce(info_avg, reduction="mean")
                        info_avg.update({"epoch": epoch, "inner_epoch": inner_epoch, "total_samples": total_samples})
                        if accelerator.is_main_process:
                            wandb.log(info_avg, step=global_step)
                            print(f"[STEP] gstep={global_step} " + " ".join(f"{k}={float(v):.5f}" for k, v in info_avg.items() if k not in ("epoch", "inner_epoch")), flush=True)
                        global_step += 1
                        info = defaultdict(list)

        total_samples += samples_per_epoch
        epoch += 1


if __name__ == "__main__":
    app.run(main)
