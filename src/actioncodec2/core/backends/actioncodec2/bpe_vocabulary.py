'Runtime plain-BPE vocabulary over the row-major flattened cell stream.'

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._validation import active_dimensions
from .native_serialization import NativeCacheMixin, decode_native, encode_native

__all__ = ["BPEVocabulary"]


@dataclass
class BPEVocabulary(NativeCacheMixin):
    """Primitive atoms plus ordered, adjacency-based merge rules."""

    bins: tuple[int, ...]
    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.bins or any(count < 1 for count in self.bins):
            raise ValueError("every dimension needs at least one bin")
        self.offsets = np.concatenate(([0], np.cumsum(np.asarray(self.bins, dtype=np.int64))))
        self.atom_count = int(self.offsets[-1])
        self.vocab_size = self.atom_count + len(self.merges)
        atom_dims = np.repeat(
            np.arange(len(self.bins), dtype=np.int64), np.asarray(self.bins, dtype=np.int64)
        )
        # Every token expands to a fixed run of atoms.  Storing the expansions
        # once keeps decoding and grammar construction table lookups instead of
        # recursive walks.
        expansions: list[tuple[int, ...]] = [(token,) for token in range(self.atom_count)]
        for rank, (left, right) in enumerate(self.merges):
            token = self.atom_count + rank
            if not 0 <= left < token or not 0 <= right < token:
                raise ValueError("a merge rule must reference already-created tokens")
            expansions.append(expansions[left] + expansions[right])
        self.expansions = tuple(expansions)
        self.token_dims = tuple(
            tuple(int(atom_dims[atom]) for atom in expansion) for expansion in expansions
        )
        self.token_cells = np.asarray([len(expansion) for expansion in expansions], dtype=np.int64)
        self._rules = {
            (int(left), int(right)): (self.atom_count + rank, rank)
            for rank, (left, right) in enumerate(self.merges)
        }

    @property
    def dimension(self) -> int:
        return len(self.bins)

    def atom(self, dimension: int, value: int) -> int:
        return int(self.offsets[dimension]) + int(value)

    def _flatten(self, values: np.ndarray, dimensions: tuple[int, ...]) -> list[int]:
        """Serialize dense ``[T, D]`` cells row-major over the active support."""
        counts = np.asarray(self.bins, dtype=np.int64)
        if np.any(values < -1) or np.any(values >= counts[None, :]):
            raise ValueError("cells contain an invalid active or absent index")
        support = values >= 0
        if np.any(support != support[:1]):
            raise ValueError("active dimensions must remain constant within a trajectory")
        present = tuple(int(dim) for dim in np.flatnonzero(support[0]))
        if present != dimensions:
            raise ValueError("cells do not activate exactly the requested dimensions")
        if not present:
            raise ValueError("every control step needs at least one active coordinate")
        active = values[:, list(present)]
        base = np.asarray([int(self.offsets[dim]) for dim in present], dtype=np.int64)
        return (active + base[None, :]).reshape(-1).tolist()

    def encode(self, cells: np.ndarray) -> list[int]:
        'Apply the merges greedily in creation-rank order to one trajectory.'
        values = np.asarray(cells, dtype=np.int64)
        if values.ndim != 2 or len(values) < 1 or values.shape[1] < 1:
            raise ValueError("cells must have non-empty shape [T, D]")
        dimensions = tuple(int(dim) for dim in np.flatnonzero(values[0] >= 0))
        stream = self._flatten(values, dimensions)
        result = encode_native(self, values[None])
        if result is not None:
            return result[0]
        count = len(stream)
        previous = list(range(-1, count - 1))
        following = list(range(1, count + 1))
        following[-1] = -1
        alive = [True] * count
        rules = self._rules
        heap: list[tuple[int, int]] = []
        for index in range(count - 1):
            rule = rules.get((stream[index], stream[index + 1]))
            if rule is not None:
                heap.append((rule[1], index))
        heapq.heapify(heap)
        while heap:
            rank, index = heapq.heappop(heap)
            if not alive[index]:
                continue
            follower = following[index]
            if follower < 0 or not alive[follower]:
                continue
            rule = rules.get((stream[index], stream[follower]))
            if rule is None or rule[1] != rank:
                continue
            stream[index] = rule[0]
            alive[follower] = False
            after = following[follower]
            following[index] = after
            if after >= 0:
                previous[after] = index
                successor = rules.get((stream[index], stream[after]))
                if successor is not None:
                    heapq.heappush(heap, (successor[1], index))
            before = previous[index]
            if before >= 0:
                predecessor = rules.get((stream[before], stream[index]))
                if predecessor is not None:
                    heapq.heappush(heap, (predecessor[1], before))
        return [stream[index] for index in range(count) if alive[index]]

    def decode(
        self,
        tokens: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Expand tokens to dense cells, writing ``-1`` in absent coordinates."""
        dimensions = active_dimensions(active_dims, self.dimension)
        result = decode_native(self, [[int(token) for token in tokens]], dimensions)
        if result is not None:
            return result[0]
        atoms: list[int] = []
        for index, raw_token in enumerate(tokens):
            token = int(raw_token)
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"tokens[{index}] is outside the vocabulary")
            atoms.extend(self.expansions[token])
        width = len(dimensions)
        if not atoms or len(atoms) % width:
            raise ValueError("token stream does not cover a whole number of routed control steps")
        values = np.asarray(atoms, dtype=np.int64).reshape(-1, width)
        starts = np.asarray([int(self.offsets[dim]) for dim in dimensions], dtype=np.int64)
        stops = np.asarray([int(self.offsets[dim + 1]) for dim in dimensions], dtype=np.int64)
        if np.any(values < starts[None, :]) or np.any(values >= stops[None, :]):
            raise ValueError("token stream does not follow the routed dimension order")
        cells = np.full((len(values), self.dimension), -1, dtype=np.int64)
        cells[:, list(dimensions)] = values - starts[None, :]
        return cells
