"""Constrained-generation grammars for the two ActionCodec2 BPE schemes."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from ._validation import active_dimensions
from .bpe import BPESerializer, SetBPESerializer
from .cache import BoundedCache, cached_prefix_state
from .native_backend import native_kernel


def _coverage_metadata(serializer, dims):
    """Share immutable vocabulary tables across horizon-cache evictions."""
    cache = getattr(serializer, "_coverage_metadata", None)
    if cache is None:
        cache = serializer._coverage_metadata = BoundedCache(32)
    metadata = cache.get(dims)
    if metadata is not None:
        return metadata
    metadata = {"compiled": None}
    if serializer.scheme == "set_bpe":
        full = sum(1 << d for d in dims)
        metadata["roots"] = np.asarray(
            [serializer.spatial.anchor[serializer.root[t]] for t in range(serializer.vocab_size)],
            dtype=np.int64,
        )
        patterns = tuple(
            tuple((offset, int(serializer.spatial.support[token])) for offset, token in pattern)
            for pattern in serializer.patterns
        )
        metadata["patterns"] = patterns
        metadata["scope"] = np.asarray(
            [all(support & ~full == 0 for _, support in pattern) for pattern in patterns],
            dtype=bool,
        )
    else:
        phase = np.zeros((len(dims), serializer.vocab_size), dtype=bool)
        for index in range(len(dims)):
            for token, expansion in enumerate(serializer.vocabulary.token_dims):
                phase[index, token] = all(
                    d == dims[(index + offset) % len(dims)] for offset, d in enumerate(expansion)
                )
        metadata["phase"] = phase
        metadata["lengths"] = np.asarray(serializer.token_cell_counts, dtype=np.int64)
    cache[dims] = metadata
    return metadata


class _PrefixGrammar:
    """Shared public queries over scheme-specific prefix state transitions."""

    def _state(self, tokens: Sequence[int]) -> tuple[int, ...] | None:
        checked = []
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, Integral):
                return None
            checked.append(int(token))
        return cached_prefix_state(
            tuple(checked), self._state_cache, self._empty_state, self._advance
        )

    def _next_token_mask_from_state(self, state: tuple[int, ...]) -> np.ndarray:
        """Return a defensive copy of one cached coverage mask."""
        cached = self._mask_cache.get(state)
        if cached is None:
            compiled = self._compiled()
            cached = compiled.mask(state) if compiled is not None else self._python_mask(state)
            self._mask_cache[state] = cached
        return cached.copy()

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        """Return a boolean vocabulary mask for the next generated token."""
        state = self._state(tokens)
        if state is None:
            raise ValueError("token prefix is not a legal ActionCodec2 stream")
        return self._next_token_mask_from_state(state)

    def is_complete(self, tokens: Sequence[int]) -> bool:
        """Whether ``tokens`` is exactly one complete trajectory of this horizon."""
        state = self._state(tokens)
        return state is not None and self._state_is_complete(state)


class ActionCodec2Grammar(_PrefixGrammar):
    'Prefix grammar for one exact, fixed-horizon rooted token stream.'

    def __init__(
        self,
        serializer: SetBPESerializer,
        *,
        horizon: int,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self.vocab_size = serializer.vocab_size
        dims = active_dimensions(active_dims, serializer.spatial.dimension)
        self._full = sum(1 << dimension for dimension in dims)
        self._metadata = _coverage_metadata(serializer, dims)
        self._root_anchor = self._metadata["roots"]
        self._patterns = self._metadata["patterns"]
        self._in_scope = self._metadata["scope"]
        self._mask_cache = BoundedCache(64)
        self._state_cache = BoundedCache(256)
        self._empty_state = self._initial_state()
        self._native = None
        self._compiled()

    def _compiled(self):
        constructor = native_kernel("SetGrammar")
        if constructor is None:
            return None
        if self._native is None:
            prototype = self._metadata["compiled"]
            if prototype is None:
                prototype = self._metadata["compiled"] = constructor(
                    1, self._full, self._root_anchor.tolist(), self._patterns
                )
            self._native = prototype.clone(self.horizon)
        return self._native

    def _advance(self, state: tuple[int, ...], token: int) -> tuple[int, ...] | None:
        compiled = self._compiled()
        if compiled is not None:
            return compiled.advance(state, token)
        if not 0 <= token < self.vocab_size or not self._in_scope[token]:
            return None
        coverage = list(state)
        front = next(
            (index for index, mask in enumerate(coverage) if mask != self._full), self.horizon
        )
        if front == self.horizon:
            return None
        uncovered = self._full & ~coverage[front]
        expected_anchor = int(uncovered & -uncovered).bit_length() - 1
        if int(self._root_anchor[token]) != expected_anchor:
            return None
        for offset, support in self._patterns[token]:
            step = front + offset
            if step >= self.horizon or coverage[step] & support:
                return None
            coverage[step] |= support
        return tuple(coverage)

    def _initial_state(self) -> tuple[int, ...]:
        """Return the empty coverage state for incremental internal validation."""
        return (0,) * self.horizon

    def _covered_steps(self, state: tuple[int, ...]) -> int:
        return next((i for i, mask in enumerate(state) if mask != self._full), self.horizon)

    def _state_is_complete(self, state: tuple[int, ...]) -> bool:
        """Whether an incrementally consumed state covers the whole trajectory."""
        return all(mask == self._full for mask in state)

    def _python_mask(self, state: tuple[int, ...]) -> np.ndarray:
        if self._state_is_complete(state):
            return np.zeros(self.vocab_size, dtype=bool)
        front = next(index for index, mask in enumerate(state) if mask != self._full)
        uncovered = self._full & ~state[front]
        expected_anchor = int(uncovered & -uncovered).bit_length() - 1
        mask = (self._root_anchor == expected_anchor) & self._in_scope
        for token in np.flatnonzero(mask):
            for offset, support in self._patterns[int(token)]:
                step = front + offset
                if step >= self.horizon or state[step] & support:
                    mask[token] = False
                    break
        return mask


class ActionCodec2BPEGrammar(_PrefixGrammar):
    'Prefix grammar for one exact, fixed-horizon plain-BPE token stream.'

    def __init__(
        self,
        serializer: BPESerializer,
        *,
        horizon: int,
        active_dims: Sequence[int] | None = None,
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self.vocab_size = serializer.vocab_size
        dims = active_dimensions(active_dims, serializer.vocabulary.dimension)
        self._width = len(dims)
        self._cells = horizon * self._width
        self._metadata = _coverage_metadata(serializer, dims)
        self._token_cells = self._metadata["lengths"]
        self._phase = self._metadata["phase"]
        self._mask_cache = BoundedCache(64)
        self._state_cache = BoundedCache(256)
        self._empty_state = self._initial_state()

        self._active_dims = dims
        self._native = None
        self._compiled()

    def _compiled(self):
        constructor = native_kernel("PlainGrammar")
        if constructor is None:
            return None
        if self._native is None:
            prototype = self._metadata["compiled"]
            if prototype is None:
                prototype = self._metadata["compiled"] = constructor(
                    1, self._token_cells.tolist(), self._active_dims, self._phase
                )
            self._native = prototype.clone(self.horizon)
        return self._native

    def _initial_state(self) -> tuple[int, ...]:
        return (0,)

    def _covered_steps(self, state: tuple[int, ...]) -> int:
        return state[0] // self._width

    def _state_is_complete(self, state: tuple[int, ...]) -> bool:
        return state[0] == self._cells

    def _advance(self, state: tuple[int, ...], token: int) -> tuple[int, ...] | None:
        compiled = self._compiled()
        if compiled is not None:
            return compiled.advance(state, token)
        if not 0 <= token < self.vocab_size:
            return None
        covered = state[0]
        length = int(self._token_cells[token])
        if covered + length > self._cells:
            return None
        if not self._phase[covered % self._width, token]:
            return None
        return (covered + length,)

    def _python_mask(self, state: tuple[int, ...]) -> np.ndarray:
        covered = state[0]
        if covered == self._cells:
            return np.zeros(self.vocab_size, dtype=bool)
        return self._phase[covered % self._width] & (self._token_cells <= self._cells - covered)
