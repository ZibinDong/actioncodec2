"""Action-space-bound, arbitrary-horizon ActionCodec2 frontend."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from ...backends.actioncodec2 import (
    ActionCodec2Router,
    PresetConfig,
)
from ...backends.actioncodec2.native_backend import native_kernel
from ...backends.actioncodec2.semantic import resampled_horizon
from .._types import ActionIdxBatch, TokenIds
from ..base_action_tokenizer import BaseActionTokenizer
from ..layout import ActionSpaceMixin
from ..conversion import _numpy, _numpy_token_rows, restore_source_layout








class ActionCodec2Tokenizer(ActionSpaceMixin, BaseActionTokenizer):
    'ActionCodec2 frontend bound to physical tensor semantics, not a horizon.'

    encode_kwargs = frozenset({"fps", "current_state", "previous_first_order"})

    def __init__(self, backend: ActionCodec2Router, *, action_space: str) -> None:
        if not isinstance(backend, ActionCodec2Router):
            raise TypeError(
                f"ActionCodec2Tokenizer requires ActionCodec2Router, got {type(backend).__name__}"
            )
        preset = backend.preset(action_space)
        if preset.target_hz != 15.0:
            raise ValueError(
                f"ActionCodec2 frontend artifacts require target_hz=15; got {preset.target_hz}. "
                "Pass the source data rate via fps; use a physical profile calibrated at 15 Hz. "
                "Changing target_hz alone does not recalibrate physical bounds."
            )
        self.action_tokenizer = backend
        self._action_space = preset
        self._action_vocab_size = int(backend.vocab_size)
        self._new_action_tokens = list(backend.new_action_tokens)
        self.action_dim = preset.input_dim
        self._warned_default_fps = False
        self.requires_previous_first_order = bool(backend.second_order_boundaries(preset))
        self._encoded_action_dim = sum(
            len(component.active_dims) for component in preset.components
        )
        self.max_action_token_len = None
        self.fixed_action_token_len = None
        super().__init__(append_bar_sentinels=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        action_space: str | Path | PresetConfig | Mapping[str, object],
        **kwargs: object,
    ) -> ActionCodec2Tokenizer:
        'Load a local/Hub router and bind one explicit physical action space.'
        router = ActionCodec2Router.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls._from_router(router, action_space=action_space)

    @classmethod
    def _from_router(cls, router: ActionCodec2Router, *, action_space):
        """Bind an already loaded router using the public layout-selection rules."""
        if isinstance(action_space, (PresetConfig, Mapping, Path)):
            name = router.register_action_space(action_space).name
        elif isinstance(action_space, str):
            candidate = Path(action_space)
            name = (
                router.register_action_space(candidate).name
                if candidate.is_file()
                else action_space
            )
        else:
            raise TypeError(
                "action_space must be a registered name, PresetConfig, mapping, or YAML path"
            )
        return cls(router, action_space=name)

    def configure_encoding(
        self,
        *,
        spatial_search: bool = False,
        temporal_search: bool = False,
        search_window: int = 32,
        search_beam: int = 4,
    ) -> ActionCodec2Tokenizer:
        'Opt into native packing search; both independent stages default off.'
        self.action_tokenizer.configure_encoding(
            spatial_search=spatial_search,
            temporal_search=temporal_search,
            search_window=search_window,
            search_beam=search_beam,
        )
        return self


    def _describe_action_spaces(self, action_dim: int | None = None) -> str:
        return self.action_tokenizer.describe_action_spaces(
            selected=self.action_space.name, action_dim=action_dim
        )



    def print_action_spaces(
        self, *, action_dim: int | None = None, file: TextIO | None = None
    ) -> None:
        """Print all layouts or candidates matching ``action_dim`` without changing the binding."""
        description = self._describe_action_spaces(action_dim)
        print(description, file=sys.stdout if file is None else file)


    def for_action_space(self, name: str) -> ActionCodec2Tokenizer:
        """Return a new tokenizer sharing weights but bound to another option."""
        return type(self)(self.action_tokenizer, action_space=name)

    def _source_fps(self, fps, preset=None):
        preset = self.action_space if preset is None else preset
        if fps is None:
            if not self._warned_default_fps:
                warnings.warn(
                    f"fps was omitted; assuming {preset.source_hz:g} Hz from action_space "
                    f"{preset.name!r}. Pass the actual fps explicitly to avoid changing action timing.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_default_fps = True
            return preset.source_hz
        return fps


    def max_tokens_for_horizon(self, horizon: int, *, fps: float | None = None) -> int:
        'Return the token bound after converting a source chunk to the codec clock.'
        source_fps = self._source_fps(fps)
        internal = resampled_horizon(
            horizon, source_hz=source_fps, target_hz=self.action_space.target_hz
        )
        return internal * self._encoded_action_dim

    def grammar(self, horizon: int, *, fps: float | None = None):
        'Build a grammar for the codec steps implied by source chunk geometry.'
        source_fps = self._source_fps(fps)
        internal = resampled_horizon(
            horizon, source_hz=source_fps, target_hz=self.action_space.target_hz
        )
        return self.action_tokenizer.grammar(preset=self.action_space.name, horizon=internal)

    def __call__(
        self,
        action: torch.Tensor | np.ndarray,
        *,
        fps: float | None = None,
        current_state: torch.Tensor | np.ndarray | None = None,
        previous_first_order: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        encode_kwargs: dict | None = None,
    ) -> ActionIdxBatch:
        'Encode arbitrary-length ``(B,T,D)`` or ``(T,D)`` actions.'
        kwargs = self._check_encode_kwargs(encode_kwargs)
        direct = {
            "fps": fps,
            "current_state": current_state,
            "previous_first_order": previous_first_order,
        }
        duplicates = sorted(
            key for key, value in direct.items() if value is not None and key in kwargs
        )
        if duplicates:
            raise ValueError(f"encode arguments provided twice: {duplicates}")
        kwargs.update({key: value for key, value in direct.items() if value is not None})
        reference = kwargs.get("current_state")
        if self.action_space.input == "absolute" and reference is None:
            raise ValueError(
                f"action space {self.action_space.name!r} is absolute; encode requires "
                "current_state with shape [B, D] to preserve the first transition"
            )
        if self.action_space.input == "delta" and reference is not None:
            raise ValueError(
                f"action space {self.action_space.name!r} is delta; omit current_state"
            )
        values = torch.as_tensor(action)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        return self._encode_action_indices(values, encode_kwargs=kwargs)

    encode = __call__

    def decode(
        self,
        action_token_ids: Sequence[TokenIds],
        *,
        fps: float | None = None,
        current_state: torch.Tensor | np.ndarray | None = None,
        previous_first_order: Mapping[str, torch.Tensor | np.ndarray] | None = None,
    ) -> torch.Tensor:
        'Decode a batch, deriving sequence length from tokens and output rate.'
        return self.decode_token_ids_to_actions_batch(
            action_token_ids,
            action_dim=self.action_dim,
            decode_kwargs={
                "fps": fps,
                "current_state": current_state,
                "previous_first_order": previous_first_order,
            },
        )

    def decode_token_ids_to_actions_batch(
        self,
        action_token_ids: Sequence[TokenIds],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> torch.Tensor:
        'Decode integer rows to ``(B,T,D)``; the native router validates ranges.'
        if native_kernel("route_decode_batch") is None:
            return super().decode_token_ids_to_actions_batch(
                action_token_ids,
                time_horizon=time_horizon,
                action_dim=action_dim,
                decode_kwargs=decode_kwargs,
            )
        actions, _ = self._decode_action_indices(
            _numpy_token_rows(action_token_ids),
            time_horizon=time_horizon,
            action_dim=action_dim,
            decode_kwargs=decode_kwargs,
        )
        return torch.from_numpy(actions)


    def decode_prefix(
        self,
        action_token_ids,
        *,
        horizon: int,
        steps: int,
        fps: float | None = None,
        current_state=None,
        previous_first_order=None,
    ) -> torch.Tensor:
        'Decode a covered streaming prefix to ``(B,steps,D)`` on the 15 Hz grid.'
        if self.action_space.input == "absolute" and current_state is None:
            raise ValueError("absolute prefix decode requires current_state")
        if self.action_space.input == "delta" and current_state is not None:
            raise ValueError("delta prefix decode must omit current_state")
        decoded = self.action_tokenizer.decode_prefix_batch(
            [self._checked_decode_row(row).tolist() for row in action_token_ids],
            preset=self.action_space.name,
            horizon=horizon,
            steps=steps,
            fps=float(self._source_fps(fps)),
            current_state=_numpy(current_state),
            previous_first_order=self._numpy_boundary(previous_first_order),
        )
        return torch.from_numpy(self._restore_source_layout(decoded))

    def final_first_order(
        self,
        action_token_ids: Sequence[TokenIds],
        *,
        previous_first_order: Mapping[str, torch.Tensor | np.ndarray],
        executed_steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        'Return reconstructed component states after a canonical-step prefix.'
        return self.action_tokenizer.final_first_order(
            [self._checked_decode_row(row).tolist() for row in action_token_ids],
            preset=self.action_space.name,
            previous_first_order=self._numpy_boundary(previous_first_order),
            executed_steps=executed_steps,
        )

    def _encode_action_indices(
        self, action: torch.Tensor, encode_kwargs: dict | None = None
    ) -> ActionIdxBatch:
        """Encode finite ``(B,T,D)`` actions without fixing ``T`` at construction."""
        kwargs = dict(encode_kwargs or {})
        fps = float(self._source_fps(kwargs.get("fps")))
        current_state = _numpy(kwargs.get("current_state"))
        values = _numpy(action)
        if values.ndim != 3 or values.shape[2] != self.action_space.input_dim:
            raise ValueError(
                f"actions must have shape [B, T_source, {self.action_space.input_dim}]"
            )
        return self.action_tokenizer.encode_batch(
            values,
            preset=self.action_space.name,
            fps=fps,
            current_state=current_state,
            previous_first_order=self._numpy_boundary(kwargs.get("previous_first_order")),
        )

    def _decode_action_indices(
        self,
        action_indices: ActionIdxBatch,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, list[set[str]]]:
        """Decode rows to the selected preset's original ``(B,T,D)`` layout."""
        if time_horizon is not None:
            raise ValueError(
                "ActionCodec2 decode infers sequence length from tokens; "
                "do not provide time_horizon"
            )
        if action_dim is not None and int(action_dim) != self.action_dim:
            raise ValueError(
                f"action_dim={action_dim} disagrees with action space "
                f"{self.action_space.name!r} (D={self.action_dim})"
            )
        kwargs = dict(decode_kwargs or {})
        unknown = sorted(set(kwargs) - self.encode_kwargs)
        if unknown:
            raise ValueError(f"ActionCodec2Tokenizer does not consume decode_kwargs {unknown}")
        raw_fps = kwargs.get("fps")
        fps = float(self._source_fps(raw_fps))
        current_state = _numpy(kwargs.get("current_state"))
        if self.action_space.input == "absolute" and current_state is None:
            raise ValueError(
                f"action space {self.action_space.name!r} is absolute; decode requires "
                "current_state with shape [B, D] to restore its external reference"
            )
        if self.action_space.input == "delta" and current_state is not None:
            raise ValueError(
                f"action space {self.action_space.name!r} is delta; omit current_state "
                "to decode in the registered input semantics"
            )
        decoded = self.action_tokenizer.decode_batch(
            action_indices,
            preset=self.action_space.name,
            fps=fps,
            current_state=current_state,
            previous_first_order=self._numpy_boundary(kwargs.get("previous_first_order")),
        )
        actions = self._restore_source_layout(decoded)
        return actions, [set() for _ in range(actions.shape[0])]

    def _restore_source_layout(self, decoded: Mapping[str, np.ndarray]) -> np.ndarray:
        return restore_source_layout(self.action_space, decoded)
