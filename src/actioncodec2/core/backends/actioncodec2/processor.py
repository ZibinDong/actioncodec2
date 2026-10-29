'Public ActionCodec2 processor and constrained-generation grammars.'

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from transformers.processing_utils import ProcessorMixin

from ._validation import active_dimensions
from .bpe import BPE_NAME, BPESerializer, SetBPESerializer, serializer_class
from .cache import BoundedCache
from .configuration import CONFIG_NAME, MODEL_TYPE, ActionCodec2Config
from .grammar import ActionCodec2BPEGrammar as ActionCodec2BPEGrammar
from .grammar import ActionCodec2Grammar as ActionCodec2Grammar
from .hf import (
    resolve_pretrained_directory,
    save_huggingface_metadata,
    upload_pretrained_directory,
)
from .physical_codec import PhysicalCodec
from .primitives import PRIMITIVES_NAME, PrimitiveStore


class ActionCodec2Processor(ProcessorMixin):
    'Hugging Face-style processor for continuous robot action trajectories.'

    model_type = MODEL_TYPE
    name = "actioncodec2"
    is_exact = False
    set_bpe_is_lossless = True
    model_input_names = ["input_ids"]
    attributes = []

    def __init__(
        self,
        config: ActionCodec2Config,
        primitives: PrimitiveStore,
        serializer: SetBPESerializer | BPESerializer,
    ) -> None:
        if serializer.scheme != config.set_bpe.scheme:
            raise ValueError("serializer scheme does not match config.yaml")
        if tuple(serializer.bins) != tuple(primitives.bins):
            raise ValueError("BPE bins do not match primitives.npz")
        if serializer.lookahead != config.set_bpe.lookahead:
            raise ValueError("BPE lookahead does not match config.yaml")
        # ProcessorMixin is designed around modality sub-processors. This
        # processor is atomic, so it deliberately owns and serializes its three
        # artifact objects directly instead of invoking ProcessorMixin.__init__.
        self.config = config
        self.primitives = primitives
        self.serializer = serializer
        self._grammar_cache = BoundedCache(8)
        self.physical = PhysicalCodec(config, primitives)
        self.action_dim = config.physical.input_dim
        self.bins = primitives.bins
        self.vocab_size = serializer.vocab_size
        self.lookahead = serializer.lookahead
        self.requires_previous_first_order = self.physical.requires_previous_first_order
        zero_cells = []
        for dimension, axis in enumerate(primitives.values):
            matches = np.flatnonzero(np.asarray(axis) == 0)
            if not len(matches):
                raise ValueError(
                    "ActionCodec2 artifacts require an exact zero primitive in every axis; "
                    f"dimension {dimension} has none"
                )
            zero_cells.append(int(matches[0]))
        self._zero_cells = np.asarray(zero_cells, dtype=np.int64)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> ActionCodec2Processor:
        'Load a local directory or Hugging Face Hub snapshot.'
        directory = resolve_pretrained_directory(pretrained_model_name_or_path, **kwargs)
        config = ActionCodec2Config.from_pretrained(directory)
        primitives = PrimitiveStore.from_pretrained(directory, input_dim=config.physical.input_dim)
        serializer = serializer_class(config).from_pretrained(
            directory,
            bins=primitives.bins,
            lookahead=config.set_bpe.lookahead,
        )
        return cls(config, primitives, serializer)

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Save numerical artifacts plus standard AutoProcessor metadata."""
        if not push_to_hub and kwargs:
            raise TypeError(f"unused save_pretrained keyword(s): {', '.join(sorted(kwargs))}")
        directory = Path(save_directory)
        self.config.save_pretrained(directory)
        self.primitives.save_pretrained(directory)
        self.serializer.save_pretrained(directory)
        metadata = save_huggingface_metadata(directory)
        files = [
            directory / CONFIG_NAME,
            directory / PRIMITIVES_NAME,
            directory / BPE_NAME,
            *metadata,
        ]
        if push_to_hub:
            upload_pretrained_directory(directory, **kwargs)
        return [str(path) for path in files]

    def to_dict(self, legacy_serialization: bool = True) -> dict[str, object]:
        """Return lightweight processor metadata, never large primitive/BPE arrays."""
        return {
            "processor_class": self.__class__.__name__,
            "model_type": self.model_type,
            "action_dim": self.action_dim,
            "vocab_size": self.vocab_size,
        }

    @property
    def artifact_fingerprint(self) -> str:
        """Content hash over config, primitive axes, and ordered BPE rules."""
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
            json.dumps(self.serializer.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        )
        return digest.hexdigest()

    def __call__(
        self, actions: np.ndarray, *, previous_first_order: np.ndarray | None = None
    ) -> dict[str, list[list[int]]]:
        'Return ragged ``input_ids`` for ``[T,D]`` or ``[B,T,D]`` actions.'
        values = np.asarray(actions)
        if values.ndim == 2:
            values = values[None]
            if previous_first_order is not None:
                previous_first_order = np.asarray(previous_first_order)[None]
        return {"input_ids": self.encode_batch(values, previous_first_order=previous_first_order)}

    def quantize_batch(
        self, actions: np.ndarray, *, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Quantize ``[B,T,D]`` using a ``[B,D]`` first-order boundary if required."""
        return self.physical.encode_batch(actions, previous_first_order=previous_first_order)

    def _dense_previous_first_order(
        self, previous_first_order: np.ndarray | None, dims: tuple[int, ...], batch: int
    ) -> np.ndarray | None:
        """Expand a compact ``[B,D_active]`` boundary, validating active slots."""
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

    def tokens_from_cells(
        self,
        cells: np.ndarray,
        *,
        active_dims: Sequence[int] | None = None,
    ) -> list[int]:
        """Apply the lossless tokenizer to one compact or dense cell trajectory."""
        return self.serializer.encode(cells, active_dims=active_dims)

    def configure_encoding(
        self,
        *,
        spatial_search: bool = False,
        temporal_search: bool = False,
        search_window: int = 32,
        search_beam: int = 4,
    ) -> ActionCodec2Processor:
        'Opt into lossless packing search; both stages default to disabled.'
        self.serializer.configure_encoding(
            spatial_search=spatial_search,
            temporal_search=temporal_search,
            search_window=search_window,
            search_beam=search_beam,
        )
        return self

    def cells_from_tokens(
        self,
        tokens: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Losslessly recover dense cells with ``-1`` in absent dimensions."""
        return self.serializer.decode(tokens, active_dims=active_dims)

    def _fill_inactive_cells(self, cells: np.ndarray) -> np.ndarray:
        """Replace semantic absence with each physical axis's exact zero cell."""
        values = np.asarray(cells, dtype=np.int64)
        return np.where(values >= 0, values, self._zero_cells)

    def encode(
        self,
        actions: np.ndarray,
        *,
        active_dims: Sequence[int] | None = None,
        previous_first_order: np.ndarray | None = None,
    ) -> list[int]:
        'Encode ``[T,D_active]`` with an optional compact ``[D_active]`` boundary.'
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or len(values) < 1 or values.shape[1] < 1:
            raise ValueError("actions must have non-empty shape [T, D_active]")
        dims = (
            tuple(range(values.shape[1]))
            if active_dims is None and values.shape[1] < self.action_dim
            else active_dimensions(active_dims, self.action_dim)
        )
        if len(dims) != values.shape[1]:
            raise ValueError("active_dims must have one entry per action coordinate")
        previous = None if previous_first_order is None else np.asarray(previous_first_order)[None]
        return self.encode_batch(values[None], active_dims=dims, previous_first_order=previous)[0]

    def encode_batch(
        self,
        actions: np.ndarray,
        *,
        active_dims: Sequence[int] | None = None,
        previous_first_order: np.ndarray | None = None,
    ) -> list[list[int]]:
        'Encode a dense or compact ``[B,T,D]`` batch into ragged streams.'
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 3 or len(values) < 1 or values.shape[1] < 1 or values.shape[2] < 1:
            raise ValueError("actions must have non-empty shape [B, T, D_active]")
        dims = active_dimensions(active_dims, self.action_dim)
        if len(dims) != values.shape[2]:
            raise ValueError("active_dims must have one entry per action coordinate")
        full_width = dims == tuple(range(self.action_dim))
        if full_width:
            dense = values
        else:
            dense = np.zeros((len(values), values.shape[1], self.action_dim), dtype=np.float32)
            dense[:, :, dims] = values
        previous = self._dense_previous_first_order(previous_first_order, dims, len(values))
        cells = self.quantize_batch(dense, previous_first_order=previous)
        if not full_width:
            cells = np.take(cells, dims, axis=2)
        return self.serializer.encode_batch(cells, active_dims=dims)

    def decode(
        self,
        tokens: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
        previous_first_order: np.ndarray | None = None,
    ) -> np.ndarray:
        'Decode one stream to ``[T,D_active]`` reconstructed actions.'
        previous = None if previous_first_order is None else np.asarray(previous_first_order)[None]
        return self.decode_batch([tokens], active_dims=active_dims, previous_first_order=previous)[
            0
        ]

    def decode_batch(
        self,
        streams: Sequence[Sequence[int]],
        *,
        active_dims: Sequence[int] | None = None,
        previous_first_order: np.ndarray | None = None,
    ) -> np.ndarray:
        'Decode equal-length streams into one batched physical array.'
        dims = active_dimensions(active_dims, self.action_dim)
        stacked_cells = self.serializer.decode_batch(streams, active_dims=dims)
        dense_cells = self._fill_inactive_cells(stacked_cells)
        previous = self._dense_previous_first_order(previous_first_order, dims, len(stacked_cells))
        decoded = self.physical.decode_batch(dense_cells, previous_first_order=previous)
        return np.take(decoded, dims, axis=2)

    def final_first_order(
        self,
        streams: Sequence[Sequence[int]],
        *,
        previous_first_order: np.ndarray,
        active_dims: Sequence[int] | None = None,
        executed_steps: int | None = None,
    ) -> np.ndarray:
        'Return the next compact ``[B,D_active]`` reconstructed boundary.'
        dims = active_dimensions(active_dims, self.action_dim)
        cells = [self.cells_from_tokens(stream, active_dims=dims) for stream in streams]
        if not cells or len({len(row) for row in cells}) != 1:
            raise ValueError("streams must be non-empty and have equal decoded horizons")
        if executed_steps is not None:
            if (
                isinstance(executed_steps, bool)
                or not isinstance(executed_steps, Integral)
                or not 1 <= executed_steps <= len(cells[0])
            ):
                raise ValueError("executed_steps must be within the decoded horizon")
            cells = [row[:executed_steps] for row in cells]
        previous = self._dense_previous_first_order(previous_first_order, dims, len(cells))
        result = self.physical.final_first_order(
            self._fill_inactive_cells(np.stack(cells)), previous_first_order=previous
        )
        return np.take(result, dims, axis=1)

    def eval(self) -> ActionCodec2Processor:
        """Return ``self``; the rule processor has no train/eval state."""
        return self

    def to(self, device: object) -> ActionCodec2Processor:
        """Return ``self``; all runtime arrays are CPU NumPy values."""
        del device
        return self

    def grammar(
        self,
        horizon: int,
        *,
        active_dims: Sequence[int] | None = None,
    ) -> ActionCodec2Grammar | ActionCodec2BPEGrammar:
        """Build a constrained grammar for one horizon and route support."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        dims = active_dimensions(active_dims, self.action_dim)
        key = (horizon, dims)
        grammar = self._grammar_cache.get(key)
        if grammar is None:
            if self.serializer.scheme == "unified_set_bpe":
                from .unified_set_bpe import UnifiedSetBPEGrammar

                grammar_class = UnifiedSetBPEGrammar
            else:
                grammar_class = (
                    ActionCodec2BPEGrammar
                    if isinstance(self.serializer, BPESerializer)
                    else ActionCodec2Grammar
                )
            grammar = grammar_class(self.serializer, horizon=horizon, active_dims=dims)
            self._grammar_cache[key] = grammar
        return grammar
