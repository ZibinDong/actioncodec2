"""Runtime rooted-temporal Set-BPE vocabulary and lossless serialization."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._validation import active_dimensions
from ..native_serialization import NativeCacheMixin, decode_native, encode_native
from .spatial import SpatialVocabulary

__all__ = ["TemporalVocabulary"]

Rule = tuple[int, int, int]
Pattern = tuple[tuple[int, int], ...]


def _active_support(active_dims: Sequence[int] | None, dimension: int) -> int:
    dims = active_dimensions(active_dims, dimension)
    return sum(1 << value for value in dims)


def _validate_episode(values: np.ndarray, spatial: SpatialVocabulary) -> int:
    if values.ndim != 2 or values.shape[1] != spatial.dimension or len(values) < 1:
        raise ValueError(
            f"every episode must have non-empty shape [T, {spatial.dimension}], got {values.shape}"
        )
    counts = np.asarray(spatial.bins, dtype=np.int64)
    if np.any(values < -1) or np.any(values >= counts[None, :]):
        raise ValueError("episodes must contain -1 or an in-range cell index")
    support = values >= 0
    if np.any(np.sum(support, axis=1) == 0):
        raise ValueError("every timestep needs at least one active coordinate")
    if np.any(support != support[:1]):
        raise ValueError("active dimensions must remain constant within an episode")
    return sum(1 << int(value) for value in np.flatnonzero(support[0]))


@dataclass
class TemporalVocabulary(NativeCacheMixin):
    """A spatial vocabulary plus rooted, forward-only temporal merge rules."""

    spatial: SpatialVocabulary
    lookahead: int
    merges: tuple[Rule, ...]

    def __post_init__(self) -> None:
        if self.lookahead < 1:
            raise ValueError("lookahead must be positive")
        self.patterns: list[Pattern] = [((0, token),) for token in range(self.spatial.vocab_size)]
        self.spans: list[int] = [1] * self.spatial.vocab_size
        self.root: list[int] = list(range(self.spatial.vocab_size))
        self.support_masks: list[tuple[int, ...]] = [
            (int(self.spatial.support[token]),) for token in range(self.spatial.vocab_size)
        ]
        self._rules: dict[Rule, tuple[int, int]] = {}
        for rank, rule in enumerate(self.merges):
            left, right, offset = (int(value) for value in rule)
            child = self.spatial.vocab_size + rank
            if left >= child or right >= child or left < 0 or right < 0:
                raise ValueError("a temporal merge must reference already-created tokens")
            pattern, masks = _merge_pattern(
                self.patterns[left],
                self.patterns[right],
                offset,
                self.lookahead,
                self.spatial,
            )
            self.patterns.append(pattern)
            self.spans.append(len(masks))
            self.root.append(self.root[left])
            self.support_masks.append(masks)
            key = (left, right, offset)
            if key in self._rules:
                raise ValueError("duplicate temporal merge rule")
            self._rules[key] = (child, rank)

    @property
    def vocab_size(self) -> int:
        return self.spatial.vocab_size + len(self.merges)

    def encode(self, codes: np.ndarray) -> list[int]:
        """Encode a dense episode; ``-1`` slots are semantically absent."""
        values = np.asarray(codes, dtype=np.int64)
        _validate_episode(values, self.spatial)
        result = encode_native(self, values[None])
        if result is not None:
            return result[0]

        at: list[dict[int, int]] = [dict() for _ in range(values.shape[0])]
        tokens: list[int] = []
        anchors: list[int] = []
        active: list[bool] = []
        for anchor, row in enumerate(values):
            for token in self.spatial.encode(row):
                instance = len(tokens)
                tokens.append(token)
                anchors.append(anchor)
                active.append(True)
                at[anchor][token] = instance

        frontier: list[tuple[int, int, int, int, int, int]] = []

        def offer(left_instance: int, right_instance: int) -> None:
            left = tokens[left_instance]
            right = tokens[right_instance]
            offset = anchors[right_instance] - anchors[left_instance]
            rule = self._rules.get((left, right, offset))
            if rule is not None:
                child, rank = rule
                heapq.heappush(
                    frontier,
                    (
                        rank,
                        anchors[left_instance],
                        int(self.spatial.anchor[self.root[child]]),
                        left_instance,
                        right_instance,
                        child,
                    ),
                )

        for anchor, left_items in enumerate(at):
            for offset in range(1, min(self.lookahead, len(at) - anchor)):
                for left_instance in left_items.values():
                    for right_instance in at[anchor + offset].values():
                        offer(left_instance, right_instance)

        while frontier:
            _, anchor, _, left_instance, right_instance, child = heapq.heappop(frontier)
            if not active[left_instance] or not active[right_instance]:
                continue
            if anchors[left_instance] != anchor:
                continue
            rule = (
                tokens[left_instance],
                tokens[right_instance],
                anchors[right_instance] - anchor,
            )
            resolved = self._rules.get(rule)
            if resolved is None or resolved[0] != child:
                continue

            active[left_instance] = False
            active[right_instance] = False
            del at[anchor][tokens[left_instance]]
            del at[anchors[right_instance]][tokens[right_instance]]
            new_instance = len(tokens)
            tokens.append(child)
            anchors.append(anchor)
            active.append(True)
            at[anchor][child] = new_instance

            for offset in range(1, self.lookahead):
                future = anchor + offset
                if future < len(at):
                    for other in at[future].values():
                        offer(new_instance, other)
                past = anchor - offset
                if past >= 0:
                    for other in at[past].values():
                        offer(other, new_instance)

        ordered: list[tuple[int, int, int]] = []
        for anchor, instances in enumerate(at):
            for token, instance in instances.items():
                if active[instance]:
                    ordered.append((anchor, int(self.spatial.anchor[self.root[token]]), token))
        ordered.sort()
        return [token for _, _, token in ordered]

    def decode(
        self,
        stream: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Decode ids using the route's requested active-dimension support."""
        if not stream:
            return np.empty((0, self.spatial.dimension), dtype=np.int64)
        dims = active_dimensions(active_dims, self.spatial.dimension)
        result = decode_native(self, [[int(token) for token in stream]], dims)
        if result is not None:
            return result[0]
        full = _active_support(active_dims, self.spatial.dimension)
        coverage: list[int] = [0]
        spatial_tokens: list[list[int]] = [[]]
        front = 0
        for raw_token in stream:
            token = int(raw_token)
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"temporal token {token} is outside the vocabulary")
            while front < len(coverage) and coverage[front] == full:
                front += 1
            if front == len(coverage):
                coverage.append(0)
                spatial_tokens.append([])

            uncovered = full & ~coverage[front]
            expected_anchor = int(uncovered & -uncovered).bit_length() - 1
            actual_anchor = int(self.spatial.anchor[self.root[token]])
            if actual_anchor != expected_anchor:
                raise ValueError(
                    "temporal stream violates root order at the earliest incomplete step"
                )

            for offset, spatial_token in self.patterns[token]:
                step = front + offset
                while step >= len(coverage):
                    coverage.append(0)
                    spatial_tokens.append([])
                support = int(self.spatial.support[spatial_token])
                if coverage[step] & support:
                    raise ValueError("temporal stream contains overlapping spatial support")
                coverage[step] |= support
                spatial_tokens[step].append(spatial_token)

        if any(mask != full for mask in coverage):
            raise ValueError("temporal stream ended with an incomplete control step")
        return np.stack(
            [self.spatial.decode(tokens, active_dims=active_dims) for tokens in spatial_tokens]
        )


def _merge_pattern(
    left: Pattern,
    right: Pattern,
    offset: int,
    lookahead: int,
    spatial: SpatialVocabulary,
) -> tuple[Pattern, tuple[int, ...]]:
    if not 1 <= offset < lookahead:
        raise ValueError("a temporal merge offset must be in [1, lookahead)")
    shifted = tuple((time + offset, token) for time, token in right)
    pattern = tuple(sorted((*left, *shifted)))
    span = 1 + max(time for time, _ in pattern)
    if span > lookahead:
        raise ValueError("temporal merge exceeds the lookahead horizon")
    masks = [0] * span
    roots = 0
    for time, token in pattern:
        support = int(spatial.support[token])
        if masks[time] & support:
            raise ValueError("temporal merge joins overlapping spatial supports")
        masks[time] |= support
        roots += int(time == 0)
    if roots != 1:
        raise ValueError("a rooted temporal token must have exactly one time-zero spatial token")
    return pattern, tuple(masks)
