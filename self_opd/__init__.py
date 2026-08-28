"""Public Self-OPD training components."""

from __future__ import annotations


__all__ = [
    "BranchSamplingConfig",
    "available_rewards",
    "build_multi_reward",
    "load_reward_modules",
    "register_reward",
    "sample_branches_and_score",
    "select_all_branches",
]


def __getattr__(name: str):
    if name in {"BranchSamplingConfig", "sample_branches_and_score", "select_all_branches"}:
        from . import branch_score

        return getattr(branch_score, name)
    if name in {"available_rewards", "build_multi_reward", "load_reward_modules", "register_reward"}:
        from . import rewards

        return getattr(rewards, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
