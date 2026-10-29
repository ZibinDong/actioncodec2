"""Balanced sampling across LeRobot roots."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

__all__ = ["BalancedReplacementSampler"]


class BalancedReplacementSampler(Sampler[int]):
    """Uniform root, then uniform episode, then uniform step -- with replacement."""

    def __init__(
        self,
        entry_roots: Sequence[int],
        entry_offsets: Sequence[int],
        *,
        num_samples: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        root_weights: Sequence[float] | None = None,
        epoch: int = 0,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must lie inside a positive world_size")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        roots = np.asarray(entry_roots, dtype=np.int64)
        offsets = np.asarray(entry_offsets, dtype=np.int64)
        if offsets.ndim != 1 or len(offsets) != len(roots) + 1:
            raise ValueError("entry_offsets must hold one boundary per entry plus a total")
        counts = np.diff(offsets)
        if np.any(counts <= 0):
            raise ValueError("every entry must contribute at least one sample position")

        # Entries regrouped by root, flattened once so a draw is two array
        # lookups rather than a Python-level branch per sample.
        order = np.argsort(roots, kind="stable")
        self._entries_by_root = order
        present, first = np.unique(roots[order], return_index=True)
        self._root_start = np.append(first, len(order)).astype(np.int64)
        self._root_size = np.diff(self._root_start)
        self._roots = present

        if root_weights is None:
            weights = np.full(len(present), 1.0 / len(present))
        else:
            weights = np.asarray(root_weights, dtype=np.float64)
            if len(weights) != len(present):
                raise ValueError(
                    f"root_weights has {len(weights)} entries but {len(present)} roots "
                    "contribute samples"
                )
            if np.any(weights < 0) or not weights.sum() > 0:
                raise ValueError("root_weights must be non-negative and not all zero")
            weights = weights / weights.sum()
        self._weights = weights

        self._offsets = offsets
        self._counts = counts
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        # Ranks and passes get their own stream.  The multipliers are large
        # primes so no two (epoch, rank) pairs collide inside a run.
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + self.epoch * 9_176_309 + self.rank) % (2**63)
        )
        n = self.num_samples
        chosen = rng.choice(len(self._weights), size=n, p=self._weights)
        within = rng.integers(0, self._root_size[chosen])
        entries = self._entries_by_root[self._root_start[chosen] + within]
        steps = rng.integers(0, self._counts[entries])
        positions = self._offsets[entries] + steps
        return iter(positions.tolist())
