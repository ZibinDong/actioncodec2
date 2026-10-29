"""One ActionCodec2Sparse physical profile: unchanged physical stage plus sparse token layer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np

from .._validation import active_dimensions
from ..cache import BoundedCache
from ..configuration import CONFIG_NAME, ActionCodec2Config, ConfigError
from ..hf import save_huggingface_metadata
from ..physical_codec import PhysicalCodec
from ..primitives import PRIMITIVES_NAME, PrimitiveStore
from .grammar import SparseGrammar
from .layout import SparseLayout
from .vocabulary import SparseVocabulary

PROFILE_PROCESSOR_CLASS = "ActionCodec2SparseProcessor"


class ActionCodec2SparseProcessor:
    """Fitted sparse profile over canonical first- or second-order profile actions."""

    def __init__(
        self,
        config: ActionCodec2Config,
        primitives: PrimitiveStore,
        vocabulary: SparseVocabulary,
    ) -> None:
        if config.set_bpe.scheme != "sparse_set_bpe":
            raise ConfigError("ActionCodec2Sparse profiles require set_bpe.scheme=sparse_set_bpe")
        if vocabulary.layout != SparseLayout.from_config(config, primitives):
            raise ValueError("sparse layout does not match the physical primitives")
        if vocabulary.lookahead != config.set_bpe.lookahead:
            raise ValueError("sparse vocabulary lookahead does not match config.yaml")
        self.config, self.primitives, self.vocabulary = config, primitives, vocabulary
        self.physical = PhysicalCodec(config, primitives)
        self.requires_previous_first_order = self.physical.requires_previous_first_order
        self.action_dim = config.physical.input_dim
        self.bins = primitives.bins
        self._grammar_cache = BoundedCache(8)

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.vocab_size

    @property
    def layout(self) -> SparseLayout:
        return self.vocabulary.layout

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> ActionCodec2SparseProcessor:
        """Load ``config.yaml``, ``primitives.npz`` and ``sparse_bpe.json``."""
        root = Path(directory)
        config = ActionCodec2Config.from_pretrained(root)
        primitives = PrimitiveStore.from_pretrained(root, input_dim=config.physical.input_dim)
        return cls(config, primitives, SparseVocabulary.from_pretrained(root))

    def save_pretrained(self, directory: str | Path) -> list[str]:
        root = Path(directory)
        self.config.save_pretrained(root)
        self.primitives.save_pretrained(root)
        files = [root / CONFIG_NAME, root / PRIMITIVES_NAME, self.vocabulary.save_pretrained(root)]
        files.extend(
            save_huggingface_metadata(
                root, processor_class=PROFILE_PROCESSOR_CLASS, vocab_size=self.vocab_size
            )
        )
        return [str(path) for path in files]

    @property
    def artifact_fingerprint(self) -> str:
        """Content hash over config, primitive axes and the ordered sparse rules."""
        digest = hashlib.sha256()
        digest.update(
            json.dumps(self.config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        )
        for axis in self.primitives.values:
            values = np.ascontiguousarray(axis)
            digest.update(str(values.dtype).encode())
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
        digest.update(
            json.dumps(self.vocabulary.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        )
        return digest.hexdigest()

    def _active(self, dims: tuple[int, ...]) -> np.ndarray:
        flags = np.zeros(self.action_dim, dtype=bool)
        flags[list(dims)] = True
        return flags

    def _dense_previous_first_order(
        self, previous_first_order: np.ndarray | None, dims: tuple[int, ...], batch: int
    ) -> np.ndarray | None:
        active = set(dims)
        for component in self.config.physical.components.values():
            present = active.intersection(component.dims)
            if (
                component.primitive_order == 2
                and component.space == "so3"
                and present
                and present != set(component.dims)
            ):
                raise ValueError(
                    "second-order SO(3) requires all three active component dimensions"
                )
        if previous_first_order is None:
            if any(
                c.primitive_order == 2 and active.intersection(c.dims)
                for c in self.config.physical.components.values()
            ):
                raise ValueError(
                    f"second-order tokenization requires previous_first_order with shape {(batch, len(dims))} in canonical input units"
                )
            return (
                np.zeros((batch, self.action_dim), dtype=np.float32)
                if self.requires_previous_first_order
                else None
            )
        values = np.asarray(previous_first_order, dtype=np.float32)
        if values.shape != (batch, len(dims)) or not np.all(np.isfinite(values)):
            raise ValueError(f"previous_first_order must be finite with shape {(batch, len(dims))}")
        dense = np.zeros((batch, self.action_dim), dtype=np.float32)
        dense[:, dims] = values
        return dense

    def quantize_batch(
        self, actions: np.ndarray, *, active_dims=None, previous_first_order=None
    ) -> np.ndarray:
        """Compact ``(B,T,D_active)`` actions to dense cells ``(B,T,D)``, ``-1`` inactive."""
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 3 or len(values) < 1 or values.shape[1] < 1:
            raise ValueError("actions must have non-empty shape [B, T, D_active]")
        dims = active_dimensions(active_dims, self.action_dim)
        if len(dims) != values.shape[2]:
            raise ValueError("active_dims must have one entry per action coordinate")
        dense = np.zeros((len(values), values.shape[1], self.action_dim), dtype=np.float32)
        dense[:, :, dims] = values
        previous = self._dense_previous_first_order(previous_first_order, dims, len(values))
        cells = np.asarray(
            self.physical.encode_batch(dense, previous_first_order=previous), dtype=np.int64
        )
        cells[:, :, ~self._active(dims)] = -1
        return cells

    def encode_batch(
        self,
        actions: np.ndarray,
        *,
        active_dims: Sequence[int] | None = None,
        previous_first_order: np.ndarray | None = None,
        threads: int = 0,
    ) -> list[list[int]]:
        """Encode independent ``(B,T,D_active)`` chunks into ragged local token rows."""
        cells = self.quantize_batch(
            actions, active_dims=active_dims, previous_first_order=previous_first_order
        )
        return self.vocabulary.encode_batch(cells, threads=threads)

    def cells_from_tokens(
        self, rows: Sequence[Sequence[int]], *, horizon: int, active_dims=None
    ) -> np.ndarray:
        """Losslessly recover dense cells ``(B,horizon,D)`` with ``-1`` for inactive slots."""
        dims = active_dimensions(active_dims, self.action_dim)
        return self.vocabulary.decode_batch(rows, horizon, self._active(dims))

    def _filled(self, cells: np.ndarray) -> np.ndarray:
        zero = [0 if z is None else z for z in self.layout.zero]
        return np.where(cells >= 0, cells, np.asarray(zero, dtype=np.int64))

    def decode_batch(
        self,
        rows: Sequence[Sequence[int]],
        *,
        horizon: int,
        active_dims=None,
        previous_first_order: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode rows to ``(B,horizon,D_active)`` reconstructed profile actions."""
        dims = active_dimensions(active_dims, self.action_dim)
        cells = self.cells_from_tokens(rows, horizon=horizon, active_dims=dims)
        previous = self._dense_previous_first_order(previous_first_order, dims, len(cells))
        decoded = self.physical.decode_batch(self._filled(cells), previous_first_order=previous)
        return np.take(decoded, dims, axis=2)

    def final_first_order(
        self,
        rows: Sequence[Sequence[int]],
        *,
        horizon: int,
        previous_first_order: np.ndarray,
        active_dims=None,
        executed_steps: int | None = None,
    ) -> np.ndarray:
        """Next compact ``(B,D_active)`` boundary after ``executed_steps`` codec frames."""
        dims = active_dimensions(active_dims, self.action_dim)
        cells = self.cells_from_tokens(rows, horizon=horizon, active_dims=dims)
        if executed_steps is not None:
            if (
                isinstance(executed_steps, bool)
                or not isinstance(executed_steps, Integral)
                or not 1 <= executed_steps <= cells.shape[1]
            ):
                raise ValueError("executed_steps must be within the chunk horizon")
            cells = cells[:, :executed_steps]
        previous = self._dense_previous_first_order(previous_first_order, dims, len(cells))
        result = self.physical.final_first_order(self._filled(cells), previous_first_order=previous)
        return np.take(result, dims, axis=1)

    def grammar(self, horizon: int, *, active_dims=None) -> SparseGrammar:
        """Prefix grammar of one ``horizon``-frame chunk over the active slots."""
        dims = active_dimensions(active_dims, self.action_dim)
        key = (int(horizon), dims)
        grammar = self._grammar_cache.get(key)
        if grammar is None:
            grammar = SparseGrammar(self.vocabulary, horizon, self._active(dims))
            self._grammar_cache[key] = grammar
        return grammar
