"""Per-prompt reward normalization."""

from __future__ import annotations

import numpy as np


class PerPromptStatTracker:
    def __init__(self, global_std: bool = False) -> None:
        self.global_std = global_std
        self.stats: dict[str, list] = {}
        self.history_prompts: set[str] = set()

    def update(self, prompts, rewards) -> np.ndarray:
        prompts = np.asarray(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)
        advantages = np.zeros_like(rewards)
        unique_prompts = np.unique(prompts)

        for prompt in unique_prompts:
            prompt_rewards = rewards[prompts == prompt]
            self.stats.setdefault(prompt, []).extend(prompt_rewards.tolist())
            self.history_prompts.add(prompt)

        global_std = np.std(rewards, axis=0, keepdims=True) + 1e-4
        for prompt in unique_prompts:
            history = np.asarray(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(history, axis=0, keepdims=True)
            std = global_std if self.global_std else np.std(history, axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self) -> tuple[float, int]:
        average_group_size = (
            sum(len(values) for values in self.stats.values()) / len(self.stats)
            if self.stats
            else 0.0
        )
        return average_group_size, len(self.history_prompts)

    def clear(self) -> None:
        self.stats.clear()
