"""Fitted ActionCodec2Sparse token layer: spatial Set-BPE per sparse frame plus temporal rules."""

from __future__ import annotations

import heapq
import json
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np

from ..native_backend import native_kernel, resolve_backend
from ..configuration import _integer
from ..setbpe.spatial import SpatialVocabulary
from .layout import SparseLayout

SPARSE_BPE_NAME = "sparse_bpe.json"
FORMAT = "actioncodec2_sparse_bpe"
FORMAT_VERSION = 1
MAX_LOOKAHEAD = 32


def _ctz(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


class TokenTables:
    """Per-token geometry shared by the encoder, decoder, trainer and grammar."""

    def __init__(self, support: Sequence[int], e_dim: int, lookahead: int) -> None:
        if not 1 <= lookahead <= MAX_LOOKAHEAD:
            raise ValueError(f"lookahead must be in [1, {MAX_LOOKAHEAD}]")
        self.e_dim, self.lookahead, self.base = e_dim, lookahead, len(support)
        self.masks: list[tuple[int, ...]] = []
        self.patterns: list[tuple[tuple[int, int], ...]] = []
        self.span: list[int] = []
        self.rdim: list[int] = []
        self.emask: list[int] = []
        self.first_from: list[tuple[int, ...]] = []
        for token, mask in enumerate(support):
            if not mask:
                raise ValueError("spatial token without support")
            self._push(((0, token),), (int(mask),))

    def _push(self, pattern, masks) -> None:
        span = max(o for o, m in enumerate(masks) if m) + 1
        masks = tuple(masks[:span])
        self.masks.append(masks)
        self.patterns.append(tuple(pattern))
        self.span.append(span)
        self.rdim.append(_ctz(masks[0]))
        self.emask.append(sum(1 << o for o, m in enumerate(masks) if m >> self.e_dim & 1))
        first, following = [], -1
        for o in range(span - 1, -1, -1):
            if masks[o]:
                following = o
            first.append(following)
        self.first_from.append(tuple(reversed(first)))

    def __len__(self) -> int:
        return len(self.span)

    def mask(self, token: int, offset: int) -> int:
        masks = self.masks[token]
        return masks[offset] if offset < len(masks) else 0

    def compose(self, left: int, right: int, offset: int) -> tuple[int, ...] | None:
        """Union support of ``right`` rooted ``offset`` frames after ``left``, or ``None``."""
        if not 0 <= offset < self.lookahead:
            return None
        span = max(self.span[left], offset + self.span[right])
        if span > self.lookahead:
            return None
        out = [self.mask(left, o) for o in range(span)]
        for o in range(self.span[right]):
            if out[o + offset] & self.mask(right, o):
                return None
            out[o + offset] |= self.mask(right, o)
        return tuple(out)

    def add_merge(self, left: int, right: int, offset: int) -> int:
        masks = self.compose(left, right, offset)
        if masks is None:
            raise ValueError("illegal temporal merge")
        pattern = sorted(
            self.patterns[left] + tuple((o + offset, t) for o, t in self.patterns[right])
        )
        self._push(pattern, masks)
        return len(self) - 1


def rule_key(left: int, right: int, offset: int) -> int:
    return (left << 40) | (right << 16) | offset


class Instances:
    """Mutable rooted token instances over one frame axis (reference implementation)."""

    def __init__(self, tables: TokenTables, frames: int, dims: int) -> None:
        self.tb, self.frames, self.dims = tables, frames, dims
        self.tok: list[int] = []
        self.root: list[int] = []
        self.alive: list[bool] = []
        self.member = np.full((frames, dims), -1, dtype=np.int64)

    def key(self, i: int) -> tuple[int, int, int, int]:
        e = self.tb.emask[self.tok[i]]
        root = self.root[i]
        return (root, 1 if e else 0, root + _ctz(e) if e else 0, self.tb.rdim[self.tok[i]])

    def add(self, token: int, root: int) -> int:
        self.tok.append(token)
        self.root.append(root)
        self.alive.append(True)
        index = len(self.tok) - 1
        self._paint(index, index)
        return index

    def _paint(self, i: int, value: int) -> None:
        token, root = self.tok[i], self.root[i]
        for o, mask in enumerate(self.tb.masks[token]):
            while mask:
                d = _ctz(mask)
                mask &= mask - 1
                if value >= 0 and self.member[root + o, d] >= 0:
                    raise RuntimeError("instance overlap")
                self.member[root + o, d] = value

    def kill(self, i: int) -> None:
        self.alive[i] = False
        self._paint(i, -1)

    def _first(self, a: int, b: int) -> bool:
        return (self.root[a], self.tb.rdim[self.tok[a]]) < (self.root[b], self.tb.rdim[self.tok[b]])

    def partners(self, i: int, allow_zero: bool):
        """Ordered candidate pairs ``(a, b, offset)`` involving ``i`` in the lookahead window."""
        tb, L, root = self.tb, self.tb.lookahead, self.root[i]
        lo, hi = max(0, root - L + 1), min(self.frames - 1, root + L - 1)
        for f in range(lo, hi + 1):
            for d in range(self.dims):
                j = int(self.member[f, d])
                if j < 0 or j == i:
                    continue
                tj, o = self.tok[j], f - self.root[j]
                k = max(0, lo - self.root[j])
                if tb.first_from[tj][k] != o or _ctz(tb.mask(tj, o)) != d:
                    continue
                a, b = (i, j) if self._first(i, j) else (j, i)
                off = self.root[b] - self.root[a]
                if off >= L or (off == 0 and not allow_zero):
                    continue
                if max(tb.span[self.tok[a]], off + tb.span[self.tok[b]]) > L:
                    continue
                yield a, b, off

    def valid(self, a: int, b: int, off: int) -> bool:
        """Whether merging ``a`` (rooted first) with ``b`` keeps the stream decodable."""
        tb, ta, tbb = self.tb, self.tok[a], self.tok[b]
        span = max(tb.span[ta], off + tb.span[tbb])
        emask = tb.emask[ta] | (tb.emask[tbb] << off)
        root = self.root[a]
        merged = (root, 1 if emask else 0, root + _ctz(emask) if emask else 0, tb.rdim[ta])
        for o in range(span):
            mask = tb.mask(ta, o) | (tb.mask(tbb, o - off) if o >= off else 0)
            if not mask:
                continue
            u = root + o
            h = int(self.member[u, tb.e_dim])
            if h >= 0 and h not in (a, b) and not self.key(h) > merged:
                return False
            if emask >> o & 1:
                for d in range(self.dims):
                    j = int(self.member[u, d])
                    if j >= 0 and j not in (a, b) and not self.key(j) < merged:
                        return False
        return True

    def serialize(self) -> list[int]:
        live = sorted((i for i, alive in enumerate(self.alive) if alive), key=self.key)
        return [self.tok[i] for i in live]


class SparseVocabulary:
    """Fitted sparse token layer over one physical profile's dense cells."""

    def __init__(self, layout: SparseLayout, spatial_merges, temporal_merges, lookahead: int):
        self.layout = layout
        self.lookahead = _integer(lookahead, "lookahead", positive=True)
        self.spatial = SpatialVocabulary(layout.sparse_bins, tuple(map(tuple, spatial_merges)))
        self.temporal_merges = tuple(
            tuple(_integer(v, "temporal merge value") for v in m) for m in temporal_merges
        )
        self.tables = TokenTables([int(s) for s in self.spatial.support], layout.e_dim, lookahead)
        self._rank = {}
        for rank, (left, right, offset) in enumerate(self.temporal_merges):
            if max(left, right) >= len(self.tables):
                raise ValueError("a temporal merge must reference already-created tokens")
            self.tables.add_merge(left, right, offset)
            if self._rank.setdefault(rule_key(left, right, offset), rank) != rank:
                raise ValueError("duplicate temporal rule")
        self._native = None

    @property
    def vocab_size(self) -> int:
        return len(self.tables)

    def __getstate__(self):

        return {**vars(self), "_native": None}

    def _codec(self):
        constructor = native_kernel("SparseCodec")
        if constructor is not None and self._native is None:
            self._native = constructor(
                list(self.layout.bins),
                self.layout.native_zero,
                [tuple(m) for m in self.spatial.merges],
                self.lookahead,
                list(self.temporal_merges),
            )
        return self._native if constructor is not None else None

    def _cells(self, cells) -> np.ndarray:
        values = np.asarray(cells)
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("cells must have shape (B, T, D) with B, T >= 1")
        if values.shape[2] != self.layout.dims or values.dtype.kind not in "iu":
            raise ValueError(f"cells must be integers with {self.layout.dims} slots")
        if np.any(values < -1) or np.any(values >= np.asarray(self.layout.bins)):
            raise ValueError("cell outside the primitive range")
        return values

    def encode_batch(self, cells, *, backend: str = "auto", threads: int = 0) -> list[list[int]]:
        """Encode independent dense chunks ``(B, T, D)`` (``-1`` = inactive) into token rows."""
        values = self._cells(cells)
        if resolve_backend(backend) == "native" and self._codec() is not None:
            return self._codec().encode_batch(np.asarray(values, dtype=np.int32), threads)
        return [self._encode_reference(chunk) for chunk in values]

    def _encode_reference(self, cells: np.ndarray) -> list[int]:
        sparse = self.layout.sparsify(cells)
        state = Instances(self.tables, len(sparse) + self.lookahead, self.layout.dims + 1)
        for t, row in enumerate(sparse):
            if np.any(row >= 0):
                for token in self.spatial.encode(row.tolist()):
                    state.add(int(token), t)
        heap = []

        def offer(i):
            for a, b, off in state.partners(i, True):
                rank = self._rank.get(rule_key(state.tok[a], state.tok[b], off))
                if rank is not None:
                    item = (rank, state.root[a], self.tables.rdim[state.tok[a]], state.root[b])
                    heapq.heappush(heap, (*item, a, b, off))

        for i in range(len(state.tok)):
            offer(i)
        while heap:
            rank, *_, a, b, off = heapq.heappop(heap)
            if not state.alive[a] or not state.alive[b] or not state.valid(a, b, off):
                continue
            root = state.root[a]
            state.kill(a)
            state.kill(b)
            offer(state.add(self.tables.base + rank, root))
        return state.serialize()

    def decode_batch(
        self, rows: Sequence[Sequence[int]], horizon: int, active: Sequence[bool]
    ) -> np.ndarray:
        """Token rows to dense cells ``(B, horizon, D)``; inactive slots are ``-1``."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        flags = [bool(a) for a in active]
        if len(flags) != self.layout.dims:
            raise ValueError(f"active must have {self.layout.dims} flags")
        checked = []
        for row in rows:
            if any(
                isinstance(t, (bool, np.bool_))
                or not isinstance(t, Integral)
                or not 0 <= t < self.vocab_size
                for t in row
            ):
                raise ValueError("token ids must be integers within the sparse vocabulary")
            checked.append([int(t) for t in row])
        codec = self._codec()
        if codec is not None:
            return np.asarray(codec.decode_batch(checked, horizon, flags), dtype=np.int64)
        out = np.empty((len(checked), horizon, self.layout.dims), dtype=np.int64)
        for index, row in enumerate(checked):
            out[index] = self.layout.densify(self._frames_reference(row, horizon, flags), flags)
        return out

    def _frames_reference(self, row: Sequence[int], horizon: int, active) -> np.ndarray:
        tb, e_bit = self.tables, 1 << self.layout.e_dim
        allowed = e_bit | sum(1 << d for d, flag in enumerate(active) if flag)
        cover = [0] * (horizon + self.lookahead)
        sparse = np.full((horizon, self.layout.dims + 1), -1, dtype=np.int64)
        front = 0
        for token in row:
            if not 0 <= token < len(tb):
                raise ValueError("token outside vocab")
            if front >= horizon:
                raise ValueError("token after the last open frame")

            own: dict[int, int] = {}
            for offset, spatial in tb.patterns[token]:
                u, mask = front + offset, int(self.spatial.support[spatial])
                if u >= horizon:
                    raise ValueError("token exceeds horizon")
                if mask & ~allowed:
                    raise ValueError("token touches an inactive slot")
                if own.get(u, 0) & mask:
                    raise ValueError("overlapping token supports")
                if mask & e_bit and u == horizon - 1:
                    raise ValueError("end-of-frame atom on the last frame")
                own[u] = own.get(u, 0) | mask
                expansion = self.spatial.expansion[spatial]
                sparse[u] = np.where(expansion >= 0, expansion, sparse[u])
            for u, mask in own.items():
                if cover[u] & mask:
                    raise ValueError("overlapping token supports")
                if cover[u] & e_bit:
                    raise ValueError("token touches a closed frame")
                cover[u] |= mask
            while front < horizon and cover[front] & e_bit:
                front += 1
        return sparse

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "layout": {"bins": list(self.layout.bins), "zero": list(self.layout.zero)},
            "lookahead": self.lookahead,
            "spatial_merges": [list(m) for m in self.spatial.merges],
            "temporal_merges": [list(m) for m in self.temporal_merges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SparseVocabulary:
        if payload.get("format") != FORMAT or payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"not an {FORMAT} v{FORMAT_VERSION} vocabulary")
        layout = SparseLayout(tuple(payload["layout"]["bins"]), tuple(payload["layout"]["zero"]))
        return cls(
            layout, payload["spatial_merges"], payload["temporal_merges"], payload["lookahead"]
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        path = Path(directory) / SPARSE_BPE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> SparseVocabulary:
        path = Path(directory) / SPARSE_BPE_NAME
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
