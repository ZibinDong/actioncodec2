"""Prefix grammar for generating one ActionCodec2Sparse chunk."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .vocabulary import SparseVocabulary


SparseState = tuple[int, tuple[int, ...]]


class SparseGrammar:
    """Legal next tokens of one ``horizon``-frame chunk over the active slots."""

    def __init__(self, vocabulary: SparseVocabulary, horizon: int, active: Sequence[bool]):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        layout, tables = vocabulary.layout, vocabulary.tables
        self.horizon, self.lookahead = horizon, vocabulary.lookahead
        self.vocab_size = vocabulary.vocab_size
        self.masks = np.zeros((len(tables), self.lookahead), dtype=np.int64)
        for token, masks in enumerate(tables.masks):
            self.masks[token, : len(masks)] = masks
        self.e_bit = 1 << layout.e_dim
        allowed = self.e_bit | sum(1 << d for d, flag in enumerate(active) if flag)
        self.in_scope = np.all((self.masks & ~allowed) == 0, axis=1)
        self.span = np.asarray(tables.span, dtype=np.int64)
        self.holds_e = (self.masks & self.e_bit) != 0
        self.closes_first = self.holds_e[:, 0]
        self.absolute = sum(
            1 << d for d, zero in enumerate(layout.zero) if zero is None and active[d]
        )

    def initial(self) -> SparseState:
        return 0, (0,) * (self.horizon + self.lookahead)

    def front(self, state: SparseState) -> int:
        return state[0]

    def mask(self, state: SparseState) -> np.ndarray:
        """Boolean ``(V,)`` mask of tokens legal after ``state``."""
        front, cover = state
        if front >= self.horizon:
            return np.zeros(len(self.masks), dtype=bool)
        window = np.asarray(cover[front : front + self.lookahead], dtype=np.int64)
        legal = self.in_scope & (front + self.span <= self.horizon)
        legal &= np.all((self.masks & window[None, :]) == 0, axis=1)
        closed = (window & self.e_bit) != 0
        legal &= ~np.any((self.masks != 0) & closed[None, :], axis=1)
        last = self.horizon - 1 - front
        if last < self.lookahead:
            legal &= ~self.holds_e[:, last]
        if front == 0 and (cover[0] & self.absolute) != self.absolute:
            stated = (cover[0] | self.masks[:, 0]) & self.absolute
            legal &= ~self.closes_first | (stated == self.absolute)
        return legal

    def advance(self, state: SparseState, token: int) -> SparseState | None:
        """State after ``token``, or ``None`` when it is illegal."""
        if not 0 <= token < len(self.masks) or not self.mask(state)[token]:
            return None
        front, cover = state
        updated = list(cover)
        for offset in range(self.lookahead):
            updated[front + offset] |= int(self.masks[token, offset])
        while front < self.horizon and updated[front] & self.e_bit:
            front += 1
        return front, tuple(updated)

    def is_complete(self, state: SparseState) -> bool:
        """Whether the chunk may end here (every active absolute slot stated)."""
        return (state[1][0] & self.absolute) == self.absolute

    def accepts(self, tokens: Sequence[int]) -> bool:
        state = self.initial()
        for token in tokens:
            state = self.advance(state, int(token))
            if state is None:
                return False
        return self.is_complete(state)
