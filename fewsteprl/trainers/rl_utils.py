"""Distributed K-repeat sampling and per-prompt advantage normalization.

- DistributedKRepeatSampler: each prompt appears exactly K times in the global batch.
- PerPromptStatTracker: per-prompt reward normalization into advantages.

Adapted from DiffusionNFT (arXiv 2509.16117) and flow_grpo's stat_tracking.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DistributedKRepeatSampler(Sampler):
    """Infinite batch sampler giving each prompt exactly K samples globally.

    Each iteration picks ``M = num_replicas * batch_size // K`` prompts, repeats
    each K times, shuffles, and gives each rank ``batch_size`` indices.
    ``num_replicas * batch_size`` must be divisible by K. Use as a DataLoader
    ``batch_sampler`` with ``num_workers=0`` (no prefetch, so resume is exact).
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        """
        Args:
            dataset_size: Number of prompts in the dataset.
            batch_size: Per-rank batch size.
            k: Samples per prompt.
            num_replicas: World size.
            rank: This process's rank.
            seed: Base seed; the shuffle uses ``seed + epoch``.
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"total_samples ({self.total_samples} = {num_replicas} * {batch_size}) "
            f"must be divisible by k ({k})"
        )
        self.m = self.total_samples // self.k  # number of unique prompts per iteration
        self.epoch = 0

    def __iter__(self):
        """Yield one batch of indices for this rank (infinite generator)."""
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_order = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_order]
            # Each rank takes batch_size consecutive elements.
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch: int) -> None:
        """Set the shuffle epoch; call before each ``next()`` for a fresh partition.

        The generator re-reads ``self.epoch`` on every resume.
        """
        self.epoch = epoch


class PerPromptStatTracker:
    """Per-prompt reward normalization: ``adv = (r - mean_p) / (std_p + 1e-4)``.

    Statistics accumulate over the prompt's reward history until :meth:`clear`.
    """

    def __init__(self, global_std: bool = False):
        """
        Args:
            global_std: Use the std over the whole batch instead of per prompt.
        """
        self.global_std = global_std
        self.stats: dict[str, Any] = {}
        self.history_prompts: set[int] = set()

    def update(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        """Return per-prompt normalized advantages, same shape as ``rewards``.

        Args:
            prompts: Prompt strings [N].
            rewards: Reward scores [N] or [N, T].
        """
        prompts_arr = np.array(prompts)
        rewards_arr = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts_arr)
        advantages = np.zeros_like(rewards_arr)

        for prompt in unique:
            prompt_rewards = rewards_arr[prompts_arr == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist() if prompt_rewards.ndim == 1 else prompt_rewards.flatten().tolist())
            self.history_prompts.add(hash(prompt))

        # self.stats values stay lists so .extend() works across update() calls.
        for prompt in unique:
            history = np.array(self.stats[prompt])
            prompt_rewards = rewards_arr[prompts_arr == prompt]
            mean = np.mean(history, axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards_arr, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(history, axis=0, keepdims=True) + 1e-4
            advantages[prompts_arr == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self) -> tuple[float, int]:
        """Return ``(average_group_size, number_of_unique_prompts_seen)``."""
        avg_group_size = (
            sum(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in self.stats.values())
            / max(len(self.stats), 1)
        )
        return avg_group_size, len(self.history_prompts)

    def get_mean_of_top_rewards(self, top_percentage: float) -> float:
        """Mean over prompts of each prompt's top ``top_percentage`` (0-100) rewards.

        Returns 0.0 if there is no data.
        """
        if not self.stats:
            return 0.0
        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            rewards = np.array(prompt_rewards) if isinstance(prompt_rewards, list) else prompt_rewards
            if rewards.size == 0:
                continue
            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue
            lower_bound = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound)
            top_rewards = rewards[rewards >= threshold]
            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        return float(np.mean(per_prompt_top_means)) if per_prompt_top_means else 0.0

    def clear(self) -> None:
        """Clear all accumulated statistics."""
        self.stats = {}

    def state_dict(self) -> dict[str, Any]:
        """Serialize tracker state for checkpoint saving."""
        serialized_stats = {}
        for k, v in self.stats.items():
            if isinstance(v, np.ndarray):
                serialized_stats[k] = v.tolist()
            elif isinstance(v, list):
                serialized_stats[k] = v
            else:
                serialized_stats[k] = [v]
        return {
            "stats": serialized_stats,
            "history_prompts": list(self.history_prompts),
            "global_std": self.global_std,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore tracker state from checkpoint."""
        self.stats = {k: v for k, v in state.get("stats", {}).items()}
        self.history_prompts = set(state.get("history_prompts", []))
        logger.info(
            f"  PerPromptStatTracker restored: {len(self.stats)} prompts, "
            f"{len(self.history_prompts)} unique seen"
        )
