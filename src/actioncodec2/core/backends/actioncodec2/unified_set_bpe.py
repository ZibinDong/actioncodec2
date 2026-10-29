"""Dense single-stage spatial/temporal Set-BPE using AC2Sparse's merge trainer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from ._validation import active_dimensions
from .bpe import BPE_NAME, _dense_cells, _read_payload, _rules
from .cache import BoundedCache
from .grammar import _PrefixGrammar
from .sparse.training import fit_sparse_vocabulary
from .sparse.vocabulary import SparseVocabulary


@dataclass(frozen=True)
class DenseSetLayout:
    'Keep every active physical cell and append an end-of-frame atom.'

    bins: tuple[int, ...]
    dense = True

    @property
    def dims(self):
        return len(self.bins)

    @property
    def e_dim(self):
        return self.dims

    @property
    def zero(self):
        return (-2,) * self.dims

    @property
    def native_zero(self):
        return [-2] * self.dims

    @property
    def sparse_bins(self):
        return self.bins + (1,)

    def sparsify(self, cells):
        cells = np.asarray(cells, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != self.dims or not len(cells):
            raise ValueError("cells must be a nonempty dense trajectory")
        if np.any(cells < -1) or np.any(cells >= np.asarray(self.bins)):
            raise ValueError("cell outside the primitive range")
        if np.any((cells >= 0) != (cells[0] >= 0)) or not np.any(cells[0] >= 0):
            raise ValueError("active slots must remain fixed and nonempty")
        out = np.full((len(cells), self.dims + 1), -1, dtype=np.int64)
        out[:, : self.dims] = cells
        out[:-1, self.dims] = 0
        return out

    def densify(self, frames, active):
        cells = np.asarray(frames, dtype=np.int64)[:, : self.dims].copy()
        flags = np.asarray(active, dtype=bool)
        if np.any(cells[:, flags] < 0) or np.any(cells[:, ~flags] >= 0):
            raise ValueError("dense stream is missing an active cell")
        return cells


class UnifiedSetBPESerializer:
    'Single-stage Set-BPE over complete physical cells and frame markers.'

    scheme = "unified_set_bpe"

    def __init__(self, *, bins: Sequence[int], lookahead: int, merges):
        self.bins = tuple(int(v) for v in bins)
        self.lookahead = int(lookahead)
        self.merges = _rules(merges, 3, "merges")
        self.layout = DenseSetLayout(self.bins)
        self.vocabulary = SparseVocabulary(self.layout, (), self.merges, self.lookahead)
        self.vocab_size = self.vocabulary.vocab_size
        physical_mask = (1 << len(self.bins)) - 1
        self.token_cell_counts = np.asarray(
            [
                sum((int(mask) & physical_mask).bit_count() for mask in masks)
                for masks in self.vocabulary.tables.masks
            ],
            dtype=np.int64,
        )

    def configure_encoding(self, *, spatial_search=False, temporal_search=False, **kwargs):
        if spatial_search or temporal_search:
            raise ValueError("packing search is not available for unified Set-BPE")
        return self

    def encode_batch(self, cells, *, active_dims=None):
        dense, _ = _dense_cells(np.asarray(cells), self.bins, active_dims, ndim=3)
        return self.vocabulary.encode_batch(dense)

    def encode(self, cells, *, active_dims=None):
        dense, _ = _dense_cells(np.asarray(cells), self.bins, active_dims)
        return self.vocabulary.encode_batch(dense[None])[0]

    def decode_batch(self, streams, *, active_dims=None):
        dims = active_dimensions(active_dims, len(self.bins))
        rows, horizons = [], []
        for row in streams:
            checked = []
            for token in row:
                if (
                    isinstance(token, bool)
                    or not isinstance(token, Integral)
                    or not 0 <= int(token) < self.vocab_size
                ):
                    raise ValueError("token outside vocabulary")
                checked.append(int(token))
            count = sum(int(self.token_cell_counts[token]) for token in checked)
            horizon, remainder = divmod(count, len(dims))
            if not checked or not horizon or remainder:
                raise ValueError("incomplete unified Set-BPE stream")
            rows.append(checked)
            horizons.append(horizon)
        if not rows or len(set(horizons)) != 1:
            raise ValueError("streams must have the same nonempty horizon")
        flags = [d in dims for d in range(len(self.bins))]
        return self.vocabulary.decode_batch(rows, horizons[0], flags)

    def decode(self, tokens, *, active_dims=None):
        return self.decode_batch([tokens], active_dims=active_dims)[0]

    def to_dict(self):
        return {"merges": [list(rule) for rule in self.merges]}

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, bins, lookahead):
        payload = _read_payload(directory)
        if set(payload) != {"merges"}:
            raise ValueError("unified bpe.json requires exactly merges")
        return cls(bins=bins, lookahead=lookahead, merges=payload["merges"])

    def save_pretrained(self, directory: str | Path):
        path = Path(directory) / BPE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


class UnifiedSetBPEGrammar(_PrefixGrammar):
    'Dense frame-coverage grammar for unified Set-BPE.'

    def __init__(self, serializer, *, horizon, active_dims=None):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon, self.vocab_size = horizon, serializer.vocab_size
        dims = active_dimensions(active_dims, len(serializer.bins))
        self._full = sum(1 << d for d in dims)
        self._e_bit = 1 << len(serializer.bins)
        self._e_atom = sum(serializer.bins)
        self._patterns = serializer.vocabulary.tables.masks
        self._lookahead = serializer.lookahead
        self._state_cache, self._mask_cache = BoundedCache(256), BoundedCache(64)
        self._empty_state = self._initial_state()
        self._native = None

    def _compiled(self):
        return None

    def _initial_state(self):
        return (0,) * self.horizon

    def _covered_steps(self, state):
        for index, mask in enumerate(state):
            needed = self._full | (self._e_bit if index < self.horizon - 1 else 0)
            if mask != needed:
                return index
        return self.horizon

    def _state_is_complete(self, state):
        return self._covered_steps(state) == self.horizon

    def _advance(self, state, token):
        if not 0 <= token < self.vocab_size:
            return None
        front = self._covered_steps(state)
        if front == self.horizon:
            return None
        masks = self._patterns[token]
        if front + len(masks) > self.horizon:
            return None
        updated = list(state)
        allowed = self._full | self._e_bit
        for offset, mask in enumerate(masks):
            mask = int(mask)
            if not mask:
                continue
            index = front + offset
            if mask & ~allowed or mask & updated[index] or updated[index] & self._e_bit:
                return None
            if index == self.horizon - 1 and mask & self._e_bit:
                return None
            updated[index] |= mask
            if mask & self._e_bit and updated[index] & self._full != self._full:
                return None
        return tuple(updated)

    def _python_mask(self, state):
        return np.asarray(
            [self._advance(state, token) is not None for token in range(self.vocab_size)]
        )

    def completion_token(self, state, zero_cells, bins):
        front = self._covered_steps(state)
        missing = self._full & ~state[front]
        if not missing:
            return self._e_atom
        dim = (missing & -missing).bit_length() - 1
        return sum(bins[:dim]) + int(zero_cells[dim])


def fit_unified_set_bpe(config, cells, *, threads, backend="native", progress=True):
    """Fit AC2Sparse's one-stage Set-BPE on complete dense-cell chunks."""
    from .training import build_primitives

    primitives = build_primitives(config)
    layout = DenseSetLayout(tuple(primitives.bins))
    vocab, report = fit_sparse_vocabulary(
        layout,
        cells,
        spatial_vocab_size=sum(layout.sparse_bins),
        vocab_size=config.train.bpe.vocab_size,
        lookahead=config.set_bpe.lookahead,
        backend=backend,
        threads=threads,
        progress=progress,
    )
    serializer = UnifiedSetBPESerializer(
        bins=primitives.bins, lookahead=config.set_bpe.lookahead, merges=vocab.temporal_merges
    )
    return serializer, report
