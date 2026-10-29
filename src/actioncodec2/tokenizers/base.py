"""The one interface every action tokenizer in this repo presents to a VLA."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "ChunkTokenizer",
    "ChunkGrammar",
    "FixedLengthGrammar",
    "QuantileNormalizer",
]


class ChunkGrammar(ABC):
    """Which token ids may legally continue a partially generated chunk."""

    @abstractmethod
    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        """Boolean ``[vocab_size]`` mask of ids that may follow ``tokens``."""

    @abstractmethod
    def is_complete(self, tokens: Sequence[int]) -> bool:
        """Whether ``tokens`` forms exactly one complete chunk."""


class FixedLengthGrammar(ChunkGrammar):
    """Every vocabulary id is legal for exactly ``length`` positions."""

    def __init__(self, length: int, vocab_size: int) -> None:
        self.length = int(length)
        self.vocab_size = int(vocab_size)
        if self.length <= 0 or self.vocab_size <= 0:
            raise ValueError("fixed token length and vocabulary must be positive")
        self._open = np.ones(self.vocab_size, dtype=bool)
        self._closed = np.zeros(self.vocab_size, dtype=bool)

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        return self._open.copy() if len(tokens) < self.length else self._closed.copy()

    def is_complete(self, tokens: Sequence[int]) -> bool:
        return len(tokens) == self.length


class ChunkTokenizer(ABC):
    """Fixed-horizon action chunk <-> token ids, in raw action units."""

    #: Short stable identifier used in configs and checkpoints.
    name: str = "abstract"
    #: True when decoding reproduces the encoder's own quantization exactly.
    is_exact: bool = True
    #: True when a chunk cannot be decoded without the action immediately
    #: before it.  Only a streaming higher-order codec needs one; every other
    #: backend answers False, so a caller never has to ask whether the question
    #: applies to the tokenizer it was handed.
    requires_previous_action: bool = False

    time_horizon: int
    action_dim: int
    vocab_size: int

    @property
    @abstractmethod
    def max_tokens_per_chunk(self) -> int:
        """Worst-case token count for one chunk; sizes the generation budget."""

    @abstractmethod
    def encode(self, chunk: np.ndarray) -> list[int]:
        """Encode one ``[time_horizon, action_dim]`` chunk."""

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> np.ndarray:
        """Decode one token stream back to ``[time_horizon, action_dim]``."""

    def encode_batch(
        self, chunks: np.ndarray, previous_actions: np.ndarray | None = None
    ) -> list[list[int]]:
        """Encode chunks [B, T, D]. Stateless tokenizers require previous_actions=None."""
        if previous_actions is not None:
            raise TypeError(
                f"{type(self).__name__} decodes a chunk on its own and cannot use a "
                "previous action; pass None"
            )
        values = self._check_chunks(chunks)
        return [self.encode(chunk) for chunk in values]

    def decode_batch(self, streams: Sequence[Sequence[int]]) -> np.ndarray:
        return np.stack([self.decode(stream) for stream in streams])

    @property
    def grammar(self) -> ChunkGrammar | None:
        """A grammar for constrained decoding, when the format has one."""
        return None

    def chunk_at(self, actions: np.ndarray, start: int) -> np.ndarray:
        """One horizon-long chunk, tail-padded by repeating the final action."""
        values = np.asarray(actions, dtype=np.float32)
        window = values[start : start + self.time_horizon]
        if len(window) < self.time_horizon:
            padding = np.repeat(values[-1][None, :], self.time_horizon - len(window), axis=0)
            window = np.concatenate([window, padding], axis=0)
        return window

    def rollout_previous_actions(
        self, actions: np.ndarray, *, execution_stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return boundary actions and a mask of valid chunk starts."""
        return (
            np.zeros((len(actions), self.action_dim), dtype=np.float32),
            np.ones(len(actions), dtype=bool),
        )

    # ---------------------------------------------------------------- shared

    def _check_chunk(self, chunk: np.ndarray) -> np.ndarray:
        values = np.asarray(chunk, dtype=np.float32)
        expected = (self.time_horizon, self.action_dim)
        if values.shape != expected:
            raise ValueError(f"a chunk must have shape {expected}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("chunk contains non-finite values")
        return values

    def _check_chunks(self, chunks: np.ndarray) -> np.ndarray:
        values = np.asarray(chunks, dtype=np.float32)
        expected = (self.time_horizon, self.action_dim)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError(f"chunks must have shape [B, {expected[0]}, {expected[1]}]")
        if not np.isfinite(values).all():
            raise ValueError("chunks must contain finite values")
        return values

    def reconstruction_error(self, chunks: np.ndarray) -> dict[str, float]:
        """Round-trip a batch and report the error it actually costs."""
        values = self._check_chunks(chunks)
        decoded = self.decode_batch(self.encode_batch(values))
        error = np.abs(decoded - values)
        return {
            "mean_absolute_error": float(error.mean()),
            "max_absolute_error": float(error.max()),
            "tokens_per_action": float(
                sum(len(stream) for stream in self.encode_batch(values))
                / (values.shape[0] * self.time_horizon)
            ),
        }

    # ------------------------------------------------------------ (de)serial

    @abstractmethod
    def to_json(self) -> dict[str, object]:
        ...

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")


class QuantileNormalizer:
    """Map per-coordinate bounds to [-1, 1], clipping values outside the bounds."""

    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        if self.low.shape != self.high.shape or self.low.ndim != 1:
            raise ValueError("normalizer bounds must be matching vectors")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
            raise ValueError("normalizer bounds must be finite")
        span = self.high - self.low
        if np.any(span <= 0):
            raise ValueError("every normalizer coordinate needs positive span")
        self._scale = 2.0 / span

    @classmethod
    def fit(
        cls, actions: np.ndarray, *, quantile: float = 0.01
    ) -> QuantileNormalizer:
        values = np.asarray(actions, dtype=np.float32).reshape(-1, np.shape(actions)[-1])
        low = np.quantile(values, quantile, axis=0)
        high = np.quantile(values, 1.0 - quantile, axis=0)
        # A constant coordinate (a gripper pinned to one value in a subset)
        # would otherwise produce a zero span.
        degenerate = high - low <= 0
        low = np.where(degenerate, low - 1.0, low)
        high = np.where(degenerate, high + 1.0, high)
        return cls(low, high)

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        return np.clip((values - self.low) * self._scale - 1.0, -1.0, 1.0)

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32)
        return (values + 1.0) / self._scale + self.low

    def to_json(self) -> dict[str, object]:
        return {"low": self.low.tolist(), "high": self.high.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> QuantileNormalizer:
        return cls(payload["low"], payload["high"])
