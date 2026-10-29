'Spatial Set-BPE fitting implementations.'

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np

from ..native_backend import require_native, resolve_backend
from .spatial import SpatialVocabulary

__all__ = ["train_spatial"]


class _SparseSpatialTrainer:
    def __init__(
        self,
        codes: np.ndarray,
        weights: np.ndarray,
        bins: Sequence[int],
    ) -> None:
        self.bins = tuple(int(value) for value in bins)
        self.dimension = len(self.bins)
        self.records = np.asarray(codes, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.int64)
        if self.records.ndim != 2 or self.records.shape[1] != self.dimension:
            raise ValueError("codes must have shape [U, D] matching the bin counts")
        if self.weights.shape != (self.records.shape[0],):
            raise ValueError("weights must have one entry per record")
        counts = np.asarray(self.bins, dtype=np.int64)
        if np.any(self.records < -1) or np.any(self.records >= counts[None, :]):
            raise ValueError("codes must contain -1 or an in-range cell index")
        if np.any(np.all(self.records < 0, axis=1)):
            raise ValueError("every record must contain at least one active coordinate")
        self.count = int(self.records.shape[0])
        self.offsets = np.concatenate(([0], np.cumsum(np.asarray(self.bins, dtype=np.int64))))
        self.atom_count = int(self.offsets[-1])

        self.occurrence: list[np.ndarray] = []
        self.support: list[int] = []
        self.anchor: list[int] = []
        self.frequency: list[int] = []
        for index in range(self.dimension):
            column = self.records[:, index]
            for value in range(self.bins[index]):
                members = np.flatnonzero(column == value).astype(np.int64)
                self._append_token(members, 1 << index, index)
        self._scratch = np.zeros(self.count, dtype=bool)
        self.merges: list[tuple[int, int]] = []
        self.statistics = {
            "sparse_updates": 0,
            "sparse_heap_rebuilds": 0,
            "sparse_pair_types_peak": 0,
        }

        self.member = np.where(
            self.records >= 0,
            self.offsets[:-1][None, :] + self.records,
            -1,
        )
        self._slot_pairs = tuple(
            (first, second)
            for first in range(self.dimension)
            for second in range(first + 1, self.dimension)
        )

        self._sparse_counts: dict[int, int] | None = None
        self._sparse_heap: list[tuple[int, int, int]] = []

    def _append_token(self, members: np.ndarray, support: int, anchor: int) -> int:
        token = len(self.occurrence)
        self.occurrence.append(members)
        self.support.append(support)
        self.anchor.append(anchor)
        self.frequency.append(int(self.weights[members].sum()) if members.size else 0)
        return token

    def _intersect(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.size > right.size:
            left, right = right, left
        if left.size == 0:
            return left
        self._scratch[left] = True
        result = right[self._scratch[right]]
        self._scratch[left] = False
        return result

    @staticmethod
    def _sparse_key(left: int, right: int) -> int:
        low, high = (left, right) if left < right else (right, left)
        return (int(low) << 32) | int(high)

    @staticmethod
    def _sparse_pair(key: int) -> tuple[int, int]:
        return key >> 32, key & 0xFFFFFFFF

    def build_sparse_index(self) -> None:
        """Materialize exact counters only for currently observed pairs once."""
        if self._sparse_counts is not None:
            return
        width = len(self.occurrence)
        counts = np.zeros(width * width, dtype=np.int64)
        for first, second in self._slot_pairs:
            left = self.member[:, first]
            right = self.member[:, second]
            valid = (left >= 0) & (right >= 0)
            if not valid.any():
                continue
            low = np.minimum(left[valid], right[valid])
            high = np.maximum(left[valid], right[valid])
            np.add.at(counts, low * width + high, self.weights[valid])
        observed = np.flatnonzero(counts)
        sparse: dict[int, int] = {}
        heap: list[tuple[int, int, int]] = []
        for packed in observed:
            left, right = divmod(int(packed), width)
            key = self._sparse_key(left, right)
            value = int(counts[packed])
            sparse[key] = value
            heap.append((-value, left, right))
        heapq.heapify(heap)
        self._sparse_counts = sparse
        self._sparse_heap = heap
        self.statistics["sparse_pair_types_peak"] = len(sparse)

    def sparse_best_pair(self) -> tuple[int, int, int] | None:
        """Return the exact winner from the observed-pair max heap."""
        if self._sparse_counts is None:
            self.build_sparse_index()
        assert self._sparse_counts is not None
        while self._sparse_heap:
            negative, left, right = heapq.heappop(self._sparse_heap)
            value = self._sparse_counts.get(self._sparse_key(left, right), 0)
            if value == -negative:
                return left, right, value
        return None

    def _update_sparse_index(
        self,
        left: int,
        right: int,
        child: int,
        rows: np.ndarray,
        members: np.ndarray,
        mass: int,
    ) -> None:
        if self._sparse_counts is None:
            return
        counts = self._sparse_counts
        heap = self._sparse_heap
        updates = 0

        def change(key: int, delta: int) -> None:
            nonlocal updates
            new = counts.get(key, 0) + delta
            if new < 0:
                pair = self._sparse_pair(key)
                raise RuntimeError(f"sparse spatial pair counter became negative for {pair}")
            if new:
                counts[key] = new
                pair = self._sparse_pair(key)
                heapq.heappush(heap, (-new, pair[0], pair[1]))
            else:
                counts.pop(key, None)
            updates += 1

        change(self._sparse_key(left, right), -mass)
        other_mass = np.zeros(len(self.occurrence), dtype=np.int64)
        weights = self.weights[members]
        for slot in range(rows.shape[1]):
            values = rows[:, slot]
            valid = (values >= 0) & (values != left) & (values != right)
            if valid.any():
                np.add.at(other_mass, values[valid], weights[valid])
        for raw_other in np.flatnonzero(other_mass):
            other = int(raw_other)
            value = int(other_mass[other])
            change(self._sparse_key(left, other), -value)
            change(self._sparse_key(right, other), -value)
            change(self._sparse_key(child, other), value)

        self.statistics["sparse_updates"] += updates
        self.statistics["sparse_pair_types_peak"] = max(
            self.statistics["sparse_pair_types_peak"], len(counts)
        )
        if len(heap) > 2 * len(counts) + 65_536:
            self._sparse_heap = [(-value, *self._sparse_pair(key)) for key, value in counts.items()]
            heapq.heapify(self._sparse_heap)
            self.statistics["sparse_heap_rebuilds"] += 1

    def merge(self, left: int, right: int) -> int:
        """Apply one exact incremental merge update."""
        members = self._intersect(self.occurrence[left], self.occurrence[right])
        mass = int(self.weights[members].sum()) if members.size else 0
        rows = self.member[members] if members.size else self.member[:0].copy()
        self.occurrence[left] = np.setdiff1d(self.occurrence[left], members, assume_unique=True)
        self.occurrence[right] = np.setdiff1d(self.occurrence[right], members, assume_unique=True)
        self.frequency[left] -= mass
        self.frequency[right] -= mass
        token = self._append_token(
            members,
            self.support[left] | self.support[right],
            min(self.anchor[left], self.anchor[right]),
        )
        self._update_sparse_index(left, right, token, rows, members, mass)

        if members.size:
            rows[rows == right] = -1
            rows[rows == left] = token
            self.member[members] = rows
        self.merges.append((left, right))
        return token


def _train_spatial_python(
    codes: np.ndarray,
    weights: np.ndarray,
    bins: Sequence[int],
    *,
    vocab_size: int,
    progress: bool = False,
) -> tuple[SpatialVocabulary, dict[str, object]]:
    trainer = _SparseSpatialTrainer(codes, weights, bins)
    if vocab_size < trainer.atom_count:
        raise ValueError(
            f"vocab_size {vocab_size} is below the {trainer.atom_count} primitive atoms"
        )
    total_weight = int(trainer.weights.sum())
    active_width = np.sum(trainer.records >= 0, axis=1, dtype=np.int64)
    initial = int(np.dot(trainer.weights, active_width))
    length = initial
    trainer.build_sparse_index()
    for step in range(vocab_size - trainer.atom_count):
        winner = trainer.sparse_best_pair()
        if winner is None:
            break
        left, right, frequency = winner
        trainer.merge(left, right)
        length -= frequency
        if progress and step % 256 == 0:
            print(
                f"  merge {step:>5}  tokens/action {length / total_weight:.4f}",
                flush=True,
            )
    vocabulary = SpatialVocabulary(tuple(int(value) for value in bins), tuple(trainer.merges))
    report = {
        "records": int(trainer.count),
        "weight": total_weight,
        "atoms": trainer.atom_count,
        "merges": len(trainer.merges),
        "vocab_size": vocabulary.vocab_size,
        "tokens_per_action_initial": initial / total_weight,
        "tokens_per_action": length / total_weight,
        "compression": initial / length,
        "sparse_updates": trainer.statistics["sparse_updates"],
        "sparse_heap_rebuilds": trainer.statistics["sparse_heap_rebuilds"],
        "sparse_pair_types_peak": trainer.statistics["sparse_pair_types_peak"],
    }
    return vocabulary, report


def train_spatial(
    codes: np.ndarray,
    weights: np.ndarray,
    bins: Sequence[int],
    *,
    vocab_size: int,
    backend: str = "auto",
    progress: bool = False,
    **kwargs: object,
) -> tuple[SpatialVocabulary, dict[str, object]]:
    'Fit spatial Set-BPE, preferring the native kernel when it is built.'
    if resolve_backend(backend) == "python":
        kwargs.pop("threads", None)
        return _train_spatial_python(
            codes, weights, bins, vocab_size=vocab_size, progress=progress, **kwargs
        )
    values = np.ascontiguousarray(codes, dtype=np.int32)
    masses = np.ascontiguousarray(weights, dtype=np.int64)
    merges, report = require_native().train_spatial(
        values,
        masses,
        [int(count) for count in bins],
        int(vocab_size),
        int(kwargs.get("threads", 0)),
    )
    vocabulary = SpatialVocabulary(
        tuple(int(count) for count in bins),
        tuple((int(left), int(right)) for left, right in merges),
    )
    return vocabulary, dict(report)
