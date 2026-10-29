"""Runtime spatial Set-BPE vocabulary and lossless row serialization."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .._validation import active_dimensions
from ..native_backend import native_kernel
from ..configuration import _integer
from ..native_serialization import NativeCacheMixin

__all__ = ["SpatialVocabulary", "aggregate_records"]


def aggregate_records(
    codes: np.ndarray, bins: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Lossless micro-aggregation of identical quantized actions."""
    values = np.asarray(codes, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("codes must have shape [U, D]")
    if bins is not None:
        dimensions = tuple(int(count) for count in bins)
        if len(dimensions) != values.shape[1]:
            raise ValueError("bins must have one entry per code coordinate")
        counts = np.asarray(dimensions, dtype=np.int64)
        valid = values >= 0
        if np.any(values < -1) or np.any(values >= counts[None, :]):
            raise ValueError("codes must contain -1 or an in-range cell index")
        cardinality = int(np.prod(np.asarray(dimensions, dtype=object)))
        if np.all(valid) and cardinality <= np.iinfo(np.int64).max:
            keys = np.ravel_multi_index(values.T, dimensions)
            unique_keys, counts = np.unique(keys, return_counts=True)
            unique = np.stack(np.unravel_index(unique_keys, dimensions), axis=1)
            return unique.astype(np.int64, copy=False), counts.astype(np.int64)
    unique, counts = np.unique(values, axis=0, return_counts=True)
    return unique, counts.astype(np.int64)


@dataclass
class SpatialVocabulary(NativeCacheMixin):
    """Atoms, merge rules and the decoder metadata they induce."""

    bins: tuple[int, ...]
    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.bins or any(count < 1 for count in self.bins):
            raise ValueError("every dimension needs at least one bin")
        self.offsets = np.concatenate(([0], np.cumsum(np.asarray(self.bins, dtype=np.int64))))
        self.atom_count = int(self.offsets[-1])
        self.vocab_size = self.atom_count + len(self.merges)
        dimension = len(self.bins)
        support = np.zeros(self.vocab_size, dtype=np.int64)
        expansion = np.full((self.vocab_size, dimension), -1, dtype=np.int64)
        for index, count in enumerate(self.bins):
            for value in range(count):
                token = int(self.offsets[index]) + value
                support[token] = 1 << index
                expansion[token, index] = value
        for rank, (left, right) in enumerate(self.merges):
            left = _integer(left, "spatial merge left")
            right = _integer(right, "spatial merge right")
            token = self.atom_count + rank
            if left >= token or right >= token:
                raise ValueError("a merge rule must reference already-created tokens")
            if support[left] & support[right]:
                raise ValueError("a merge rule joins two overlapping supports")
            support[token] = support[left] | support[right]
            merged = np.maximum(expansion[left], expansion[right])
            expansion[token] = merged
        self.support = support
        self.expansion = expansion
        self.anchor = np.asarray(
            [int(mask & -mask).bit_length() - 1 for mask in support], dtype=np.int64
        )
        self.full_support = (1 << dimension) - 1
        self._rules: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(self.merges):
            key = (min(left, right), max(left, right))
            self._rules[key] = (self.atom_count + rank, rank)

    @property
    def dimension(self) -> int:
        return len(self.bins)

    def atom(self, dimension: int, value: int) -> int:
        return int(self.offsets[dimension]) + int(value)

    def encode(
        self,
        code: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
    ) -> list[int]:
        """Deterministic online Set-BPE encoding (Algorithm 8)."""
        compact = [int(value) for value in code]
        if active_dims is None:
            if len(compact) != self.dimension:
                raise ValueError(f"a dense quantized action must have {self.dimension} slots")
            pairs = enumerate(compact)
        else:
            dimensions = active_dimensions(active_dims, self.dimension)
            if len(dimensions) != len(compact):
                raise ValueError("active_dims must have one entry per compact coordinate")
            pairs = zip(dimensions, compact, strict=True)
        tokens = set()
        for index, value in pairs:
            if value == -1:
                if active_dims is not None:
                    raise ValueError("compact coordinates must not contain -1")
                continue
            if not 0 <= value < self.bins[index]:
                raise ValueError(f"bin {value} is out of range for dimension {index}")
            tokens.add(self.atom(index, value))
        if not tokens:
            raise ValueError("a quantized action must contain at least one active coordinate")
        constructor = native_kernel("SetBPE")
        if constructor is not None:
            codec = getattr(self, "_compiled_spatial", None)
            if codec is None:
                codec = self._compiled_spatial = constructor(self.bins, 1, self.merges, ())
            dense = np.full((1, 1, self.dimension), -1, dtype=np.int64)
            for atom in tokens:
                dimension = int(self.anchor[atom])
                dense[0, 0, dimension] = atom - self.offsets[dimension]
            return codec.encode_batch(dense)[0]
        heap: list[tuple[int, int, int, int]] = []
        ordered = sorted(tokens)
        for position, left in enumerate(ordered):
            for right in ordered[position + 1 :]:
                rule = self._rules.get((left, right))
                if rule is not None:
                    heapq.heappush(heap, (rule[1], rule[0], left, right))
        while heap:
            _, child, left, right = heapq.heappop(heap)
            if left not in tokens or right not in tokens:
                continue
            tokens.discard(left)
            tokens.discard(right)
            for other in tokens:
                key = (min(child, other), max(child, other))
                rule = self._rules.get(key)
                if rule is not None:
                    heapq.heappush(heap, (rule[1], rule[0], key[0], key[1]))
            tokens.add(child)
        return self.serialize(tokens)

    def serialize(self, tokens: Iterable[int]) -> list[int]:
        """Canonical order for autoregressive supervision."""
        return sorted(tokens, key=lambda token: (int(self.anchor[token]), int(token)))

    def decode(
        self,
        tokens: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Expand tokens to a dense row, using ``-1`` for absent coordinates."""
        kernel = native_kernel("spatial_expand")
        if kernel is not None:
            dims = active_dimensions(active_dims, self.dimension)
            return kernel(
                [int(token) for token in tokens],
                self.expansion,
                self.support,
                sum(1 << d for d in dims),
            )
        code = np.full(self.dimension, -1, dtype=np.int64)
        covered = 0
        for token in tokens:
            value = int(token)
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"token {value} is outside the vocabulary")
            if covered & int(self.support[value]):
                raise ValueError("token supports overlap; this is not a valid action")
            covered |= int(self.support[value])
            local = self.expansion[value]
            code = np.where(local >= 0, local, code)
        if active_dims is None:
            expected = self.full_support
        else:
            dimensions = active_dimensions(active_dims, self.dimension)
            expected = sum(1 << dimension for dimension in dimensions)
        if covered != expected:
            raise ValueError("token set does not exactly cover the requested active dimensions")
        return code
