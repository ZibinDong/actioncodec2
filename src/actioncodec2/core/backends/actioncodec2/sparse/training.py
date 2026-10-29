"""ActionCodec2Sparse vocabulary training: optional spatial stage, then certified temporal merges."""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ..native_backend import native_kernel, resolve_backend
from ..setbpe.spatial import aggregate_records
from ..setbpe.spatial_training import train_spatial
from .layout import SparseLayout, sparsify_chunks
from .vocabulary import Instances, SparseVocabulary, TokenTables, rule_key

MIN_PAIR_COUNT = 2
CERTIFY_BATCH = 32


def _unpack(key: int) -> tuple[int, int, int]:
    return key >> 40, (key >> 16) & 0xFFFFFF, key & 0xFFFF


class _Stat:
    __slots__ = (
        "retired",
        "raw",
        "version",
        "exact",
        "exact_version",
        "added",
        "added_at_exact",
        "dirty_epoch",
        "live",
        "queued_version",
        "queued_count",
    )

    def __init__(self) -> None:
        self.retired = self.live = False
        self.raw = self.version = self.exact_version = self.added = self.added_at_exact = 0
        self.dirty_epoch = self.queued_version = self.queued_count = 0
        self.exact = -1

    @property
    def certified(self) -> bool:
        return self.exact >= 0 and self.exact_version == self.version

    @property
    def bound(self) -> int:
        """Upper bound on the exact count: certified counts grow only by inserted pairs."""
        if self.certified:
            return self.exact
        if self.exact >= 0:
            return min(self.raw, self.exact + self.added - self.added_at_exact)
        return self.raw


class ReferenceTrainer:
    """Python reference of the native temporal trainer (same interface and merges)."""

    def __init__(
        self, spatial_support, dims, lookahead, vocab_size, allow_zero_offset, min_count=2
    ):
        self.tables = TokenTables([int(s) for s in spatial_support], dims - 1, lookahead)
        self.dims, self.vocab_size = dims, vocab_size
        self.allow_zero, self.min_count = allow_zero_offset, min_count
        self.stats: dict[int, _Stat] = defaultdict(_Stat)
        self.heap: list[tuple[int, int, int]] = []
        self._merges: list[tuple[int, int, int]] = []
        self.epoch = 0
        self.counters = dict(dry_runs=0, rejected_by_validity=0, applied=0)

    def set_instances(self, anchors, tokens, frames) -> None:
        self.state = Instances(self.tables, int(frames), self.dims)
        self.occurrence: dict[int, list[int]] = defaultdict(list)
        for anchor, token in zip(
            np.asarray(anchors).tolist(), np.asarray(tokens).tolist(), strict=True
        ):
            if not 0 <= token < self.tables.base:
                raise ValueError("token outside spatial vocab")
            self.occurrence[token].append(self.state.add(token, anchor))
        for values in self.occurrence.values():
            values.sort(key=lambda i: self.state.root[i])
        self.initial_instances = len(self.state.tok)

    def merges(self) -> list[tuple[int, int, int]]:
        return list(self._merges)

    def _queue(self, key: int, stat: _Stat, count: int) -> None:
        heapq.heappush(self.heap, (-count, key, stat.version))
        stat.live, stat.queued_version, stat.queued_count = True, stat.version, count

    def _pairs(self, left: int, right: int, offset: int):
        st = self.state
        for a in self.occurrence[left]:
            if not st.alive[a]:
                continue
            f = st.root[a] + offset
            if f >= st.frames:
                continue
            b = int(st.member[f, self.tables.rdim[right]])
            if b >= 0 and b != a and st.alive[b] and st.tok[b] == right and st.root[b] == f:
                yield a, b

    def _count_exact(self, key: int) -> int:
        left, right, offset = _unpack(key)
        consumed, count = set(), 0
        for a, b in self._pairs(left, right, offset):
            if left == right and a in consumed:
                continue
            if not self.state.valid(a, b, offset):
                continue
            if left == right:
                consumed.add(b)
            count += 1
        return count

    def _take(self):
        count, key, version = heapq.heappop(self.heap)
        stat = self.stats.get(key)
        if stat is None or stat.retired or not stat.live or version != stat.queued_version:
            return None
        stat.live = False
        if version == stat.version:
            return -count, key, stat
        if stat.bound > 0:
            self._queue(key, stat, stat.bound)
        return None

    def _current_certified(self, entry) -> bool:
        _, key, version = entry
        stat = self.stats.get(key)
        return (
            stat is not None
            and not stat.retired
            and stat.live
            and version == stat.queued_version == stat.version
            and stat.certified
        )

    def _certify(self, keys: list[int]) -> None:
        self.counters["dry_runs"] += len(keys)
        exact = [self._count_exact(key) for key in keys]
        for key, count in zip(keys, exact, strict=True):
            stat = self.stats[key]
            stat.exact, stat.exact_version, stat.added_at_exact = count, stat.version, stat.added
            if count >= self.min_count:
                self._queue(key, stat, count)

    def _apply(self, left: int, right: int, offset: int, child: int) -> list[int]:
        st, delta = self.state, defaultdict(int)
        pairs = list(self._pairs(left, right, offset))
        tok = st.tok
        for a, b in pairs:
            if not st.alive[a] or not st.alive[b]:
                continue
            if not st.valid(a, b, offset):
                self.counters["rejected_by_validity"] += 1
                continue
            for x, y, o in st.partners(a, self.allow_zero):
                delta[rule_key(tok[x], tok[y], o)] -= 1
            for x, y, o in st.partners(b, self.allow_zero):
                if a not in (x, y):
                    delta[rule_key(tok[x], tok[y], o)] -= 1
            root = st.root[a]
            st.kill(a)
            st.kill(b)
            merged = st.add(child, root)
            self.occurrence[child].append(merged)
            for x, y, o in st.partners(merged, self.allow_zero):
                delta[rule_key(tok[x], tok[y], o)] += 1
            self.counters["applied"] += 1
        dirty = []
        for key, change in delta.items():
            stat = self.stats[key]
            stat.raw += change
            stat.added += max(change, 0)
            stat.version += 1
            if stat.dirty_epoch != self.epoch:
                stat.dirty_epoch = self.epoch
                dirty.append(key)
        return dirty

    def train(self, progress: bool = False) -> dict:
        del progress
        st = self.state
        for i in range(len(st.tok)):
            for a, b, o in st.partners(i, self.allow_zero):
                if a == i:
                    self.stats[rule_key(st.tok[a], st.tok[b], o)].raw += 1
        initial_types = len(self.stats)
        for key, stat in list(self.stats.items()):
            self._queue(key, stat, stat.raw)
        while len(self.tables) < self.vocab_size and self.heap:
            taken = self._take()
            if taken is None:
                continue
            count, key, stat = taken
            if not stat.certified:
                batch = [key]
                while self.heap and len(batch) < CERTIFY_BATCH:
                    if self._current_certified(self.heap[0]):
                        break
                    following = self._take()
                    if following is not None:
                        batch.append(following[1])
                self._certify(batch)
                continue
            if count != stat.exact or stat.exact < self.min_count:
                continue
            left, right, offset = _unpack(key)
            if self.tables.compose(left, right, offset) is None:
                stat.exact = 0
                continue
            child = self.tables.add_merge(left, right, offset)
            self._merges.append((left, right, offset))
            stat.retired = True
            self.epoch += 1
            for dirty in self._apply(left, right, offset, child):
                other = self.stats[dirty]
                if not other.retired and other.bound > 0:
                    if not other.live or other.bound > other.queued_count:
                        self._queue(dirty, other, other.bound)
        return dict(
            initial_instances=self.initial_instances,
            final_instances=sum(st.alive),
            initial_rule_types=initial_types,
            merges=len(self._merges),
            **self.counters,
        )


def _spatial_instances(vocabulary: SparseVocabulary, frames: np.ndarray):
    bins = np.asarray(vocabulary.layout.sparse_bins, dtype=np.int64) + 1
    if float(np.prod(bins.astype(float))) < 2.0**62:
        keys = np.ravel_multi_index((frames.astype(np.int64) + 1).T, bins)
        unique, inverse = np.unique(keys, return_inverse=True)
        rows = np.stack(np.unravel_index(unique, bins), axis=1) - 1
    else:
        rows, inverse = np.unique(frames.astype(np.int64), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    nonempty = np.flatnonzero(np.any(rows >= 0, axis=1))
    counts = np.zeros(len(rows), dtype=np.int64)
    codec = vocabulary._codec()
    if codec is not None:
        encoded_counts, flat = codec.spatial_encode(np.asarray(rows[nonempty], dtype=np.int32))
        counts[nonempty] = encoded_counts
        flat = np.asarray(flat, dtype=np.int32)
    else:
        parts = [vocabulary.spatial.encode(rows[i].tolist()) for i in nonempty]
        counts[nonempty] = [len(p) for p in parts]
        flat = np.asarray([t for p in parts for t in sorted(p)], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    per_frame = counts[inverse]
    total = int(per_frame.sum())
    position = np.arange(total) - np.repeat(np.cumsum(per_frame) - per_frame, per_frame)
    return per_frame, flat[np.repeat(offsets[:-1][inverse], per_frame) + position]


def fit_sparse_vocabulary(
    layout: SparseLayout,
    chunks: Sequence[np.ndarray],
    *,
    spatial_vocab_size: int,
    vocab_size: int,
    lookahead: int,
    backend: str = "auto",
    threads: int = 0,
    progress: bool = False,
) -> tuple[SparseVocabulary, dict]:
    """Fit spatial then temporal sparse Set-BPE on independent dense chunks ``[(T_i, D)]``."""
    backend = resolve_backend(backend)
    started = time.perf_counter()
    frames, starts = sparsify_chunks(layout, chunks)
    atoms = sum(layout.sparse_bins)
    if spatial_vocab_size < atoms:
        raise ValueError(f"spatial_vocab_size must be at least the {atoms} sparse atoms")
    spatial_merges, spatial_report = (), {}
    if spatial_vocab_size > atoms:
        records = frames[np.any(frames >= 0, axis=1)]
        codes, weights = aggregate_records(records, layout.sparse_bins)
        spatial, spatial_report = train_spatial(
            codes,
            weights,
            layout.sparse_bins,
            vocab_size=spatial_vocab_size,
            backend=backend,
            threads=threads,
            progress=progress,
        )
        spatial_merges = spatial.merges
    spatial_done = time.perf_counter()
    vocabulary = SparseVocabulary(layout, spatial_merges, (), lookahead)
    per_frame, tokens = _spatial_instances(vocabulary, frames)
    chunk = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    anchors = np.repeat(np.arange(len(frames), dtype=np.int64) + chunk * lookahead, per_frame)
    support = [int(s) for s in vocabulary.spatial.support]
    arguments = (support, layout.dims + 1, lookahead, vocab_size, True, MIN_PAIR_COUNT)
    if backend == "native":
        trainer = native_kernel("SparseTrainer")(
            *arguments, threads, bool(getattr(layout, "dense", False))
        )
    else:
        trainer = ReferenceTrainer(*arguments)
    trainer.set_instances(anchors, tokens.astype(np.int32), len(frames) + len(chunks) * lookahead)
    del anchors
    temporal_report = dict(trainer.train(progress))
    vocabulary = SparseVocabulary(layout, spatial_merges, trainer.merges(), lookahead)
    report = dict(
        chunks=len(chunks),
        frames=len(frames),
        sparse_atoms=atoms,
        atoms_per_frame=float(np.count_nonzero(frames >= 0) / len(frames)),
        spatial_vocab_size=vocabulary.spatial.vocab_size,
        vocab_size=vocabulary.vocab_size,
        spatial_instances=len(tokens),
        spatial=dict(spatial_report),
        temporal=temporal_report,
        seconds=dict(spatial=spatial_done - started, total=time.perf_counter() - started),
    )
    return vocabulary, report
