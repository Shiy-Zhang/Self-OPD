"""Base configuration shared by Self-OPD training examples."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.run_name = "self-opd"
    config.seed = 42
    config.logdir = "logs"
    config.save_dir = "logs"
    config.save_freq = 60
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.use_lora = True
    config.resolution = 512
    config.dataset = ""
    config.prompt_fn = ""
    config.per_prompt_stat_tracking = True

    config.pretrained = ConfigDict()
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"

    config.sample = ConfigDict()
    config.sample.num_steps = 10
    config.sample.guidance_scale = 4.5
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 16
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.trajectory_noise_level = 0.7

    config.train = ConfigDict()
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 8
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.cfg = True
    config.train.ema = True
    config.train.learning_rate = 3e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.lora_path = None

    config.reward_fn = ConfigDict()

    config.self_opd = ConfigDict()
    config.self_opd.num_branches = 8
    config.self_opd.timesteps_per_step = 2
    config.self_opd.timestep_subset = []
    config.self_opd.timestep_weights = [10, 9, 8, 7, 6, 4, 2, 1, 0]
    config.self_opd.branch_noise_level = 0.7
    config.self_opd.score_weights = None
    config.self_opd.branch_weight_clip = 1.0
    config.self_opd.direction_aware = True

    return config
