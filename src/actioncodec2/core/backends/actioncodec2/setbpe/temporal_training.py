'Temporal Set-BPE fitting implementations.'

from __future__ import annotations

import heapq
from array import array
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from ..native_backend import require_native, resolve_backend
from .spatial import SpatialVocabulary
from .temporal import Pattern, Rule, TemporalVocabulary, _merge_pattern, _validate_episode

__all__ = ["train_temporal"]


class _SparseTemporalTrainer:
    """Mutable sparse local graph with exact shifted self-rule certification."""

    def __init__(
        self,
        episodes: Sequence[np.ndarray],
        spatial: SpatialVocabulary,
        lookahead: int,
        max_vocab_size: int,
    ) -> None:
        if lookahead < 1:
            raise ValueError("lookahead must be positive")
        self.spatial = spatial
        self.lookahead = lookahead
        self.max_vocab_size = max_vocab_size
        self.patterns: list[Pattern] = [((0, token),) for token in range(spatial.vocab_size)]
        self.spans: list[int] = [1] * spatial.vocab_size
        self.masks: list[tuple[int, ...]] = [
            (int(spatial.support[token]),) for token in range(spatial.vocab_size)
        ]
        self._mask_table = np.zeros((max_vocab_size, lookahead), dtype=np.uint64)
        self._mask_table[: spatial.vocab_size, 0] = spatial.support.astype(np.uint64, copy=False)
        self.merges: list[Rule] = []
        self.tokens = array("I")
        self.anchors = array("I")
        self.slots = array("I")
        total_anchors = sum(len(episode) + lookahead for episode in episodes)
        self.member = np.full((total_anchors, spatial.dimension), -1, dtype=np.int32)
        self.occupied = np.zeros(total_anchors, dtype=np.uint64)
        self._slot_mask = (1 << spatial.dimension) - 1
        self.occurrences: list[dict[int, int]] = [dict() for _ in range(spatial.vocab_size)]
        self.episode_count = len(episodes)
        self.action_count = sum(len(episode) for episode in episodes)
        self.initial_count = 0
        self.counts: dict[int, int] = {}
        self.self_versions: dict[int, int] = {}
        self.heap: list[tuple[int, int, int, int, int, bool]] = []
        self.self_cache: dict[int, tuple[int, list[tuple[int, int]]]] = {}
        self._legal_table = np.zeros((lookahead, max_vocab_size, max_vocab_size), dtype=np.uint8)
        for offset in range(1, lookahead):
            self._legal_table[offset, : spatial.vocab_size, : spatial.vocab_size] = 2
        self.statistics = Counter()
        self._active_count = 0

        self._load_episodes(episodes)
        self.counts = self._initial_pair_counts()
        self.heap = self._initial_heap()
        self.statistics["initial_edge_types"] = len(self.counts)
        self.statistics["initial_edges"] = sum(self.counts.values())

    def _load_episodes(self, episodes: Sequence[np.ndarray]) -> None:
        """Populate sparse token instances from spatially encoded episodes."""

        base = 0
        encoding_cache: dict[bytes, tuple[int, ...]] = {}
        for episode in episodes:
            values = np.asarray(episode, dtype=np.int64)
            _validate_episode(values, self.spatial)
            needed = base + len(values) + self.lookahead
            for time, row in enumerate(values):
                cache_key = row.tobytes()
                encoded = encoding_cache.get(cache_key)
                if encoded is None:
                    encoded = tuple(self.spatial.encode(row))
                    encoding_cache[cache_key] = encoded
                anchor = base + time
                for token in encoded:
                    self._new_instance(token, anchor)
                    self.initial_count += 1
            base = needed

    def _initial_pair_counts(self) -> dict[int, int]:
        """Count all initially adjacent temporal token pairs."""
        base_vocab = self.spatial.vocab_size
        dense_size = base_vocab * base_vocab * self.lookahead
        if dense_size <= 1 << 25:
            dense_counts = np.zeros(dense_size, dtype=np.int64)
            for offset in range(1, self.lookahead):
                left_rows = self.member[:-offset]
                right_rows = self.member[offset:]
                for left_slot in range(self.spatial.dimension):
                    left = left_rows[:, left_slot]
                    for right_slot in range(self.spatial.dimension):
                        right = right_rows[:, right_slot]
                        valid = (left >= 0) & (right >= 0)
                        if not valid.any():
                            continue
                        packed = (
                            left[valid].astype(np.int64) * base_vocab + right[valid]
                        ) * self.lookahead + offset
                        np.add.at(dense_counts, packed, 1)
            initial_counts: dict[int, int] = {}
            for raw_key in np.flatnonzero(dense_counts):
                pair, offset = divmod(int(raw_key), self.lookahead)
                left, right = divmod(pair, base_vocab)
                packed_key = (left * self.max_vocab_size + right) * self.lookahead + offset
                initial_counts[packed_key] = int(dense_counts[raw_key])
            return initial_counts
        return self._initial_pair_counts_sparse()

    def _initial_pair_counts_sparse(self) -> dict[int, int]:
        """Count initial pairs without allocating a dense pair table."""
        fallback: Counter[int] = Counter()
        for anchor, left_row in enumerate(self.member):
            left_tokens = left_row[left_row >= 0]
            for offset in range(1, self.lookahead):
                target = anchor + offset
                if target >= len(self.member):
                    break
                right_tokens = self.member[target]
                right_tokens = right_tokens[right_tokens >= 0]
                for raw_left in left_tokens:
                    for raw_right in right_tokens:
                        packed_key = (
                            int(raw_left) * self.max_vocab_size + int(raw_right)
                        ) * self.lookahead + offset
                        fallback[packed_key] += 1
        return dict(fallback)

    def _initial_heap(self) -> list[tuple[int, int, int, int, int, bool]]:
        heap: list[tuple[int, int, int, int, int, bool]] = []
        for packed_key, count in self.counts.items():
            pair, offset = divmod(packed_key, self.lookahead)
            left, right = divmod(pair, self.max_vocab_size)
            heap.append((-count, left, right, offset, 0, False))
        heapq.heapify(heap)
        return heap

    def _new_instance(self, token: int, anchor: int) -> int:
        while token >= len(self.occurrences):
            self.occurrences.append({})
        if anchor in self.occurrences[token]:
            raise RuntimeError("a temporal type cannot occur twice at one anchor")
        occupied = int(self.occupied[anchor])
        available = self._slot_mask ^ occupied
        if available == 0:
            raise RuntimeError("temporal anchor exceeded the spatial token-width bound")
        lowest = available & -available
        slot = lowest.bit_length() - 1
        self.occupied[anchor] = occupied | lowest
        instance = len(self.tokens)
        self.tokens.append(token)
        self.anchors.append(anchor)
        self.slots.append(slot)
        self.member[anchor, slot] = token
        self._active_count += 1
        self.occurrences[token][anchor] = instance
        return instance

    def _initialize_legality(self, child: int) -> None:
        """Fill all legality entries involving one newly created token."""
        count = child + 1
        spans = np.asarray(self.spans[:count], dtype=np.int64)
        masks = self._mask_table[:count]
        child_masks = self._mask_table[child]
        child_span = self.spans[child]
        for offset in range(1, self.lookahead):
            overlap = np.zeros(count, dtype=bool)
            for relative in range(self.lookahead - offset):
                overlap |= (child_masks[offset + relative] & masks[:, relative]) != 0
            legal = (offset + spans <= self.lookahead) & ~overlap
            self._legal_table[offset, child, :count] = np.where(legal, 2, 1)

            overlap.fill(False)
            for relative in range(min(child_span, self.lookahead - offset)):
                overlap |= (masks[:, offset + relative] & child_masks[relative]) != 0
            legal = (offset + child_span <= self.lookahead) & ~overlap
            self._legal_table[offset, :count, child] = np.where(legal, 2, 1)

    def _entry(
        self, packed_key: int, priority: int, certified: bool
    ) -> tuple[int, int, int, int, int, bool]:
        pair, offset = divmod(packed_key, self.lookahead)
        left, right = divmod(pair, self.max_vocab_size)
        return (
            -priority,
            left,
            right,
            offset,
            self.self_versions.get(packed_key, 0) if left == right else 0,
            certified,
        )

    def _select(self, key: Rule) -> list[tuple[int, int]]:
        left, right, offset = key
        left_occurrences = self.occurrences[left]
        right_occurrences = self.occurrences[right]
        if left != right:
            return [
                (left_instance, right_occurrences[anchor + offset])
                for anchor, left_instance in left_occurrences.items()
                if anchor + offset in right_occurrences
            ]
        selected: list[tuple[int, int]] = []
        consumed: set[int] = set()
        for anchor in sorted(left_occurrences):
            target = anchor + offset
            if anchor in consumed or target in consumed or target not in right_occurrences:
                continue
            selected.append((left_occurrences[anchor], right_occurrences[target]))
            consumed.add(anchor)
            consumed.add(target)
        return selected

    def best(self) -> tuple[Rule, list[tuple[int, int]]] | None:
        while self.heap:
            negative, left, right, offset, version, certified = heapq.heappop(self.heap)
            self.statistics["heap_pops"] += 1
            key = (left, right, offset)
            packed_key = (left * self.max_vocab_size + right) * self.lookahead + offset
            current_count = self.counts.get(packed_key, 0)
            if current_count <= 0:
                continue
            if left == right:
                current_version = self.self_versions.get(packed_key, 0)
                stale = current_version != version
            else:
                stale = current_count != -negative
            if stale:
                heapq.heappush(self.heap, self._entry(packed_key, current_count, False))
                continue
            if left == right and not certified:
                matches = self._select(key)
                self.self_cache[packed_key] = (version, matches)
                heapq.heappush(self.heap, self._entry(packed_key, len(matches), True))
                self.statistics["self_certifications"] += 1
                continue
            if certified:
                cached = self.self_cache.get(packed_key)
                if cached is None or cached[0] != version:
                    continue
                matches = cached[1]
            else:
                matches = self._select(key)
            gain = -negative
            if len(matches) != gain:
                raise RuntimeError("sparse temporal counter disagrees with exact matching")
            return key, matches
        return None

    def _accumulate_incident(
        self,
        instances: Sequence[int] | set[int],
        multiplier: int,
        deltas: Counter[int],
    ) -> None:
        """Vectorize all local edges incident to a changed instance set."""
        grouped: dict[int, list[int]] = {}
        for instance in instances:
            grouped.setdefault(self.tokens[instance], []).append(self.anchors[instance])
        width = len(self.patterns)
        total_anchors = self.member.shape[0]
        vocab_width = self.max_vocab_size
        lookahead = self.lookahead
        for token, raw_anchors in grouped.items():
            anchors = np.asarray(raw_anchors, dtype=np.int64)
            for offset in range(1, self.lookahead):
                future_anchors = anchors[anchors + offset < total_anchors] + offset
                if future_anchors.size:
                    values = self.member[future_anchors].reshape(-1)
                    values = values[values >= 0]
                    if values.size:
                        counts = np.bincount(values, minlength=width)
                        neighbors = np.flatnonzero(counts)
                        neighbors = neighbors[self._legal_table[offset, token, neighbors] == 2]
                        for raw_neighbor in neighbors:
                            neighbor = int(raw_neighbor)
                            packed_key = (token * vocab_width + neighbor) * lookahead + offset
                            deltas[packed_key] += multiplier * int(counts[neighbor])
                past_anchors = anchors[anchors >= offset] - offset
                if past_anchors.size:
                    values = self.member[past_anchors].reshape(-1)
                    values = values[values >= 0]
                    if values.size:
                        counts = np.bincount(values, minlength=width)
                        neighbors = np.flatnonzero(counts)
                        neighbors = neighbors[self._legal_table[offset, neighbors, token] == 2]
                        for raw_neighbor in neighbors:
                            neighbor = int(raw_neighbor)
                            packed_key = (neighbor * vocab_width + token) * lookahead + offset
                            deltas[packed_key] += multiplier * int(counts[neighbor])

    def _correct_internal_edges(
        self,
        instances: Sequence[int] | set[int],
        correction: int,
        deltas: Counter[int],
    ) -> None:
        """Incident aggregation sees changed-to-changed edges from both ends."""
        by_anchor: dict[int, list[int]] = {}
        for instance in instances:
            by_anchor.setdefault(self.anchors[instance], []).append(self.tokens[instance])
        vocab_width = self.max_vocab_size
        lookahead = self.lookahead
        for anchor, left_tokens in by_anchor.items():
            for offset in range(1, self.lookahead):
                right_tokens = by_anchor.get(anchor + offset)
                if right_tokens is None:
                    continue
                for left in left_tokens:
                    for right in right_tokens:
                        if self._legal_table[offset, left, right] == 2:
                            packed_key = (left * vocab_width + right) * lookahead + offset
                            deltas[packed_key] += correction

    def _apply_deltas(self, deltas: Counter[int], selected: Rule) -> None:
        'Update counts while deferring heap repair for decreasing keys.'
        counts = self.counts
        self_versions = self.self_versions
        heap = self.heap
        updates = 0
        vocab_width = self.max_vocab_size
        lookahead = self.lookahead
        selected_key = (selected[0] * vocab_width + selected[1]) * lookahead + selected[2]
        for packed_key, delta in deltas.items():
            previous = counts.get(packed_key, 0)
            count = previous + delta
            if count < 0:
                raise RuntimeError("temporal edge counter became negative")
            # Only shifted self-rules need generations: for distinct token
            # types raw edge count equals matching cardinality, so an
            # unchanged count leaves an old heap entry exact.
            pair = packed_key // lookahead
            is_self = pair % (vocab_width + 1) == 0
            if is_self:
                version = self_versions.get(packed_key, 0) + 1
                self_versions[packed_key] = version
                self.self_cache.pop(packed_key, None)
            else:
                version = 0
            if count:
                counts[packed_key] = count
                if previous == 0 or packed_key == selected_key:
                    pair, offset = divmod(packed_key, lookahead)
                    left, right = divmod(pair, vocab_width)
                    heapq.heappush(heap, (-count, left, right, offset, version, False))
            else:
                counts.pop(packed_key, None)
            updates += abs(delta)
        self.statistics["counter_updates"] += updates

    def commit(self, key: Rule, matches: Sequence[tuple[int, int]]) -> None:
        left, right, offset = key
        pattern, masks = _merge_pattern(
            self.patterns[left],
            self.patterns[right],
            offset,
            self.lookahead,
            self.spatial,
        )
        child = len(self.patterns)
        self.patterns.append(pattern)
        self.spans.append(len(masks))
        self.masks.append(masks)
        self.occurrences.append({})
        self.merges.append(key)
        self._mask_table[child, : len(masks)] = masks
        self._initialize_legality(child)

        removed = {instance for pair in matches for instance in pair}
        if len(removed) != 2 * len(matches):
            raise RuntimeError("selected temporal matches overlap")
        deltas: Counter[int] = Counter()
        self._accumulate_incident(removed, -1, deltas)
        self._correct_internal_edges(removed, 1, deltas)

        for instance in removed:
            token = self.tokens[instance]
            anchor = self.anchors[instance]
            self.member[anchor, self.slots[instance]] = -1
            self._active_count -= 1
            self.occupied[anchor] = int(self.occupied[anchor]) & ~(1 << self.slots[instance])
            del self.occurrences[token][anchor]

        children = [
            self._new_instance(child, self.anchors[left_instance]) for left_instance, _ in matches
        ]
        self._accumulate_incident(children, 1, deltas)
        self._correct_internal_edges(children, -1, deltas)
        self._apply_deltas(deltas, key)

    @property
    def active_count(self) -> int:
        return self._active_count


def _prepare_episodes(
    episodes: Iterable[np.ndarray], spatial: SpatialVocabulary
) -> Sequence[np.ndarray]:
    if getattr(episodes, "_actioncodec2_validated", False):
        if not isinstance(episodes, Sequence) or len(episodes) == 0:
            raise ValueError("validated temporal corpus must be a non-empty sequence")
        return episodes
    prepared: list[np.ndarray] = []
    for episode in episodes:
        values = np.asarray(episode)
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError("episodes must contain integer cell indices")
        _validate_episode(values, spatial)
        prepared.append(np.ascontiguousarray(values))
    if not prepared or sum(len(episode) for episode in prepared) == 0:
        raise ValueError("temporal training requires at least one non-empty episode")
    return prepared


def _train_temporal_python(
    episodes: Iterable[np.ndarray],
    spatial: SpatialVocabulary,
    *,
    vocab_size: int,
    lookahead: int,
    progress: bool = False,
) -> tuple[TemporalVocabulary, dict[str, object]]:
    """Fit exact rooted temporal BPE with sparse observed-edge counters."""
    prepared = _prepare_episodes(episodes, spatial)
    if vocab_size < spatial.vocab_size:
        raise ValueError(
            f"vocab_size {vocab_size} is below the {spatial.vocab_size} spatial tokens"
        )
    trainer = _SparseTemporalTrainer(prepared, spatial, lookahead, vocab_size)
    length = trainer.initial_count
    for step in range(vocab_size - spatial.vocab_size):
        winner = trainer.best()
        if winner is None:
            break
        key, matches = winner
        if not matches:
            break
        trainer.commit(key, matches)
        length -= len(matches)
        if progress and step % 64 == 0:
            print(
                f"  temporal merge {step:>5}  tokens/action {length / trainer.action_count:.4f}",
                flush=True,
            )
    if length != trainer.active_count:
        raise RuntimeError("temporal token accounting disagrees with active tiling")
    vocabulary = TemporalVocabulary(spatial, lookahead, tuple(trainer.merges))
    report: dict[str, object] = {
        "episodes": trainer.episode_count,
        "actions": trainer.action_count,
        "lookahead": lookahead,
        "spatial_vocab_size": spatial.vocab_size,
        "merges": len(trainer.merges),
        "vocab_size": vocabulary.vocab_size,
        "tokens_per_action_initial": trainer.initial_count / trainer.action_count,
        "tokens_per_action": length / trainer.action_count,
        "compression": trainer.initial_count / length,
        "trainer": "sparse_exact",
        **{key: int(value) for key, value in trainer.statistics.items()},
    }
    return vocabulary, report


def _flatten_instances_native(
    episodes: Sequence[np.ndarray],
    spatial: SpatialVocabulary,
    lookahead: int,
    threads: int,
) -> tuple[Iterator[tuple[np.ndarray, np.ndarray]], int, int]:
    """Spatially encode bounded chunks without a corpus-sized final copy."""
    if getattr(episodes, "_actioncodec2_validated", False):
        return _flatten_cell_corpus(episodes, spatial, lookahead, threads)
    actions = sum(len(episode) for episode in episodes)
    total_anchors = actions + len(episodes) * lookahead
    max_batch_actions = 1_000_000
    merges = np.asarray(spatial.merges, dtype=np.int32).reshape(-1, 2)

    def generate() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        pending: list[np.ndarray] = []
        pending_anchors: list[np.ndarray] = []
        pending_actions = 0
        base = 0

        def encode() -> tuple[np.ndarray, np.ndarray]:
            codes = np.concatenate(pending, axis=0).astype(np.int32, copy=False)
            row_anchors = np.concatenate(pending_anchors)
            tokens, anchors = require_native().spatial_encode_instances(
                np.ascontiguousarray(codes),
                np.ascontiguousarray(row_anchors),
                [int(count) for count in spatial.bins],
                np.ascontiguousarray(merges),
                int(threads),
            )
            return np.asarray(tokens, dtype=np.int32), np.asarray(anchors, dtype=np.int64)

        for episode in episodes:
            length = len(episode)
            pending.append(episode)
            pending_anchors.append(np.arange(base, base + length, dtype=np.int64))
            pending_actions += length
            base += length + lookahead
            if pending_actions >= max_batch_actions:
                yield encode()
                pending.clear()
                pending_anchors.clear()
                pending_actions = 0
        if pending:
            yield encode()

    return generate(), total_anchors, actions


def _flatten_cell_corpus(
    episodes: Sequence[np.ndarray],
    spatial: SpatialVocabulary,
    lookahead: int,
    threads: int,
) -> tuple[Iterator[tuple[np.ndarray, np.ndarray]], int, int]:
    """Encode contiguous quantizer chunks while vectorizing episode gaps."""
    actions = sum(len(codes) for codes in episodes.chunks if codes is not None)
    total_anchors = actions + len(episodes) * lookahead
    merges = np.asarray(spatial.merges, dtype=np.int32).reshape(-1, 2)
    chunks = episodes.chunks
    offsets_by_chunk = episodes.offsets

    def generate() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        base = 0
        for chunk_index, (codes, offsets) in enumerate(zip(chunks, offsets_by_chunk, strict=True)):
            if codes is None:
                raise RuntimeError("cell corpus chunk has already been consumed")
            lengths = np.diff(offsets)
            row_anchors = (
                base
                + np.arange(len(codes), dtype=np.int64)
                + np.repeat(np.arange(len(lengths), dtype=np.int64) * lookahead, lengths)
            )
            tokens, anchors = require_native().spatial_encode_instances(
                np.ascontiguousarray(codes),
                row_anchors,
                [int(count) for count in spatial.bins],
                np.ascontiguousarray(merges),
                int(threads),
            )
            episodes.release_chunk(chunk_index)
            yield np.asarray(tokens, dtype=np.int32), np.asarray(anchors, dtype=np.int64)
            base += len(codes) + len(lengths) * lookahead

    return generate(), total_anchors, actions


def train_temporal(
    episodes: Iterable[np.ndarray],
    spatial: SpatialVocabulary,
    *,
    vocab_size: int,
    lookahead: int,
    backend: str = "auto",
    progress: bool = False,
    threads: int = 0,
) -> tuple[TemporalVocabulary, dict[str, object]]:
    'Fit rooted temporal Set-BPE on top of a frozen spatial vocabulary.'
    prepared = _prepare_episodes(episodes, spatial)
    if resolve_backend(backend) == "python":
        return _train_temporal_python(
            prepared,
            spatial,
            vocab_size=vocab_size,
            lookahead=lookahead,
            progress=progress,
        )
    episode_count = len(prepared)
    instance_chunks, total_anchors, actions = _flatten_instances_native(
        prepared, spatial, lookahead, threads
    )
    merges, report = require_native().train_temporal(
        instance_chunks,
        int(total_anchors),
        np.ascontiguousarray(spatial.support, dtype=np.uint64),
        np.ascontiguousarray(spatial.anchor, dtype=np.int32),
        int(spatial.dimension),
        int(lookahead),
        int(vocab_size),
        int(actions),
        int(threads),
    )
    vocabulary = TemporalVocabulary(
        spatial, int(lookahead), tuple(tuple(int(value) for value in rule) for rule in merges)
    )
    payload = dict(report)
    payload["episodes"] = episode_count
    return vocabulary, payload
