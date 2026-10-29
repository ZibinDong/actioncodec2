"""FAST behind the common chunk-tokenizer interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from actioncodec2.tokenizers.base import ChunkGrammar, ChunkTokenizer, QuantileNormalizer

__all__ = ["FastTokenizer", "FastGrammar", "FastUnconstrainedGrammar"]

DEFAULT_SOURCE = "physical-intelligence/fast"


def _bytes_to_unicode_inverse() -> dict[str, int]:
    """Inverse of the GPT-2 / ``tokenizers`` ByteLevel reversible alphabet."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    values = list(printable)
    codepoints = list(printable)
    extra = 0
    for value in range(256):
        if value not in printable:
            values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(values, codepoints, strict=True)}


def _utf8_prefix_state(
    raw: bytes, *, minimum_symbol: int, maximum_symbol: int
) -> tuple[int, bytes] | None:
    """Split ``raw`` into (complete in-range codepoints, incomplete suffix)."""
    try:
        decoded = raw.decode("utf-8")
        if any(not minimum_symbol <= ord(value) <= maximum_symbol for value in decoded):
            return None
        return len(decoded), b""
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(raw):
            return None
        prefix = raw[: error.start].decode("utf-8")
        if any(not minimum_symbol <= ord(value) <= maximum_symbol for value in prefix):
            return None
        return len(prefix), raw[error.start :]


class FastGrammar(ChunkGrammar):
    """Grammar for a complete FAST action chunk."""

    def __init__(self, tokenizer: FastTokenizer) -> None:
        self.vocab_size = tokenizer.vocab_size
        self.target_symbols = tokenizer.time_horizon * tokenizer.action_dim
        processor = tokenizer.processor
        # For a length-T orthonormal DCT of actions in [-1, 1] no coefficient
        # magnitude exceeds sqrt(T); one quantization unit of headroom covers
        # rounding.
        limit = int(np.ceil(processor.scale * np.sqrt(tokenizer.time_horizon))) + 1
        self.minimum_symbol = max(0, -limit - int(processor.min_token))
        self.maximum_symbol = limit - int(processor.min_token)
        symbols = [
            chr(value).encode("utf-8")
            for value in range(self.minimum_symbol, self.maximum_symbol + 1)
        ]
        self.max_tokens = self.target_symbols * max(map(len, symbols))
        self._valid_pending = {
            encoded[:width]
            for encoded in symbols
            for width in range(1, len(encoded))
        }
        inverse = _bytes_to_unicode_inverse()
        vocabulary = processor.bpe_tokenizer.get_vocab()
        pieces: list[bytes] = [b""] * self.vocab_size
        for piece, token_id in vocabulary.items():
            if 0 <= token_id < self.vocab_size:
                pieces[token_id] = bytes(inverse[character] for character in piece)
        self._pieces = pieces
        self._cache: dict[tuple[int, bytes], np.ndarray] = {}

    def _state(self, tokens: Sequence[int]) -> tuple[int, bytes] | None:
        if any(not 0 <= int(token) < self.vocab_size or not self._pieces[int(token)]
               for token in tokens):
            return None
        raw = b"".join(self._pieces[int(token)] for token in tokens)
        return _utf8_prefix_state(
            raw, minimum_symbol=self.minimum_symbol, maximum_symbol=self.maximum_symbol
        )

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        state = self._state(tokens)
        if state is None:
            raise ValueError("token prefix is not a legal FAST chunk")
        count, pending = state
        if count >= self.target_symbols and not pending:
            return np.zeros(self.vocab_size, dtype=bool)
        cached = self._cache.get((count, pending))
        if cached is None:
            cached = np.zeros(self.vocab_size, dtype=bool)
            for token_id, piece in enumerate(self._pieces):
                if not piece:
                    continue
                result = _utf8_prefix_state(
                    pending + piece,
                    minimum_symbol=self.minimum_symbol,
                    maximum_symbol=self.maximum_symbol,
                )
                if result is None:
                    continue
                complete, suffix = result
                if suffix and suffix not in self._valid_pending:
                    continue
                total = count + complete
                if total < self.target_symbols or (total == self.target_symbols and not suffix):
                    cached[token_id] = True
            self._cache[(count, pending)] = cached
        return cached.copy()

    def is_complete(self, tokens: Sequence[int]) -> bool:
        return self._state(tokens) == (self.target_symbols, b"")


class FastUnconstrainedGrammar(ChunkGrammar):
    """Allow FAST ids without masking; stop on a complete or malformed stream, or at the byte-count limit."""

    def __init__(self, strict: FastGrammar) -> None:
        self.strict = strict
        self.vocab_size = strict.vocab_size
        self.maximum = strict.max_tokens
        self._open = np.ones(self.vocab_size, dtype=bool)
        self._closed = np.zeros(self.vocab_size, dtype=bool)

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        return self._closed.copy() if self.is_complete(tokens) else self._open.copy()

    def is_complete(self, tokens: Sequence[int]) -> bool:
        if len(tokens) >= self.maximum:
            return True
        state = self.strict._state(tokens)
        return state is None or state == (self.strict.target_symbols, b"")


class FastTokenizer(ChunkTokenizer):
    """The official FAST+ processor plus this repo's normalization boundary."""

    name = "fast"
    is_exact = True

    def __init__(
        self,
        processor,
        normalizer: QuantileNormalizer,
        *,
        time_horizon: int,
        action_dim: int,
        source: str = DEFAULT_SOURCE,
        revision: str = "",
        constrained_decoding: bool = True,
        artifact: str | Path | None = None,
    ) -> None:
        self.processor = processor
        self.normalizer = normalizer
        self.time_horizon = int(time_horizon)
        self.action_dim = int(action_dim)
        self.vocab_size = int(processor.vocab_size)
        self.source = source
        self.revision = revision
        self.constrained_decoding = bool(constrained_decoding)
        self.artifact = None if artifact is None else str(artifact)
        if self.time_horizon <= 0 or self.action_dim <= 0 or self.vocab_size <= 0:
            raise ValueError("FAST horizon, dimension and vocabulary must be positive")
        self._strict_grammar = FastGrammar(self)
        self._grammar = (
            self._strict_grammar
            if self.constrained_decoding
            else FastUnconstrainedGrammar(self._strict_grammar)
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        normalizer: QuantileNormalizer,
        *,
        time_horizon: int,
        action_dim: int,
        source: str = DEFAULT_SOURCE,
        revision: str = "",
        constrained_decoding: bool = True,
    ) -> FastTokenizer:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True
        )
        return cls(
            processor,
            normalizer,
            time_horizon=time_horizon,
            action_dim=action_dim,
            source=source,
            revision=revision,
            constrained_decoding=constrained_decoding,
            artifact=path,
        )

    @property
    def max_tokens_per_chunk(self) -> int:
        """Upper bound on the byte-level token count of a complete chunk."""
        return self._strict_grammar.max_tokens

    @property
    def grammar(self) -> ChunkGrammar:
        return self._grammar

    def encode(self, chunk: np.ndarray) -> list[int]:
        return self.encode_batch(self._check_chunk(chunk)[None, ...])[0]

    def encode_batch(
        self, chunks: np.ndarray, previous_actions: np.ndarray | None = None
    ) -> list[list[int]]:
        if previous_actions is not None:
            raise TypeError(
                f"{type(self).__name__} decodes a chunk on its own and cannot use a "
                "previous action; pass None"
            )
        values = self.normalizer(self._check_chunks(chunks)).astype(np.float32)
        encoded = [[int(token) for token in row] for row in self.processor(values)]
        for row in encoded:
            if not row or not self._strict_grammar.is_complete(row):
                raise AssertionError("the official FAST encoder emitted an incomplete chunk")
            if min(row) < 0 or max(row) >= self.vocab_size:
                raise AssertionError("the official FAST encoder emitted an out-of-range token")
        return encoded

    def decode(self, tokens: Sequence[int]) -> np.ndarray:
        return self.decode_batch([tokens])[0]

    def decode_batch(self, streams: Sequence[Sequence[int]]) -> np.ndarray:
        rows = [[int(token) for token in stream] for stream in streams]
        if not rows:
            raise ValueError("FAST decoding needs at least one token stream")

        # The policy is normally grammar-constrained, but checkpoints are also
        # evaluated without constrained/"soft" decoding.  The official FAST
        # processor catches malformed BPE streams and substitutes normalized
        # zeros; after inverse normalization that would become the corpus
        # midpoint, not the safe zero action the rollout contract asks for.
        # Detect structure here and leave malformed rows as *raw-unit* zeros.
        complete = []
        for row in rows:
            try:
                complete.append(self._strict_grammar.is_complete(row))
            except (IndexError, KeyError, TypeError, ValueError):
                complete.append(False)
        result = np.zeros(
            (len(rows), self.time_horizon, self.action_dim), dtype=np.float32
        )
        valid_indices = [index for index, good in enumerate(complete) if good]
        if not valid_indices:
            return result
        valid_rows = [rows[index] for index in valid_indices]
        decoded = np.asarray(
            self.processor.decode(
                valid_rows, time_horizon=self.time_horizon, action_dim=self.action_dim
            ),
            dtype=np.float32,
        )
        expected = (len(valid_rows), self.time_horizon, self.action_dim)
        if decoded.shape != expected or not np.isfinite(decoded).all():
            raise ValueError("the official FAST decoder returned an invalid chunk")
        result[valid_indices] = self.normalizer.inverse(decoded)
        return result

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "time_horizon": self.time_horizon,
            "action_dim": self.action_dim,
            "normalizer": self.normalizer.to_json(),
            "source": self.source,
            "revision": self.revision,
            "constrained_decoding": self.constrained_decoding,
            "vocab_size": self.vocab_size,
        }
