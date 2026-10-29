'Required-preset routing over the two shared ActionCodec2 processors.'

from __future__ import annotations

from .presets import canonical_preset_name

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import yaml

from ._validation import optional_finite_array
from .batch_semantics import externalize_batch
from .cache import BoundedCache, cached_prefix_state
from .configuration import (
    ActionCodec2Config,
    ConfigError,
    _integer,
    _mapping,
    _reject_unknown,
    _required,
    _UniqueKeyLoader,
)
from .hf import (
    ROUTER_ALLOW_PATTERNS,
    ROUTER_CONFIG_NAME,
    ROUTER_MODEL_TYPE,
    resolve_pretrained_directory,
    save_huggingface_metadata,
    save_model_card,
)
from .native_backend import native_kernel
from .presets import (
    ComponentSpec,
    PresetConfig,
    format_action_spaces,
)
from .processor import ActionCodec2Processor
from .semantic import resampled_horizon
from .trajectory import ActiveTrajectory

if TYPE_CHECKING:
    from ...frontend.actioncodec2 import ActionCodec2

# The manifest pins token ordering independently of fitted profile weights.
ROUTER_CONFIG_VERSION = 4
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_PROFILES = {"eef", "joint"}
_PROFILE_WIDTHS = {"eef": 7, "joint": 8}
_ComponentStates = tuple[tuple[int, ...], ...]
_RoutedState = tuple[int, _ComponentStates]


@dataclass(frozen=True)
class ProfileSpec:
    """Token namespace metadata for one shared physical processor."""

    name: str
    token_offset: int
    token_budget: int
    artifact_fingerprint: str

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ConfigError(f"profile.name must match {_NAME.pattern!r}")
        _integer(self.token_offset, "profile.token_offset")
        _integer(self.token_budget, "profile.token_budget", positive=True)
        if re.fullmatch(r"[0-9a-f]{64}", self.artifact_fingerprint) is None:
            raise ConfigError("profile.artifact_fingerprint must be a SHA-256 hex digest")

    @classmethod
    def from_dict(cls, payload: object, *, path: str) -> ProfileSpec:
        values = _mapping(payload, path)
        allowed = {"name", "token_offset", "token_budget", "artifact_fingerprint"}
        _reject_unknown(values, allowed, path)
        return cls(
            name=str(_required(values, "name", path)),
            token_offset=_integer(_required(values, "token_offset", path), f"{path}.token_offset"),
            token_budget=_integer(
                _required(values, "token_budget", path),
                f"{path}.token_budget",
                positive=True,
            ),
            artifact_fingerprint=str(_required(values, "artifact_fingerprint", path)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _second_order_dims(config: ActionCodec2Config) -> frozenset[int]:
    """Profile-local dimensions whose primitives integrate a previous increment."""
    return frozenset(
        dimension
        for component in config.physical.components.values()
        if component.primitive_order == 2
        for dimension in component.dims
    )


def _second_order_boundaries(
    preset: PresetConfig, second_order_dims: Mapping[str, frozenset[int]]
) -> dict[str, int]:
    'Components of ``preset`` needing an explicit boundary, with their active widths.'
    return {
        component.name: len(component.active_dims)
        for component in preset.components
        if second_order_dims[component.space].intersection(component.active_dims)
    }


@dataclass(frozen=True)
class _ProfileRuntime:
    spec: ProfileSpec
    processor: ActionCodec2Processor


@dataclass(frozen=True)
class _Segment:
    component: ComponentSpec
    profile: _ProfileRuntime


def _schedule(fronts: Sequence[int], finished: Sequence[bool], time_major: bool) -> int | None:
    open_ = [i for i, done in enumerate(finished) if not done]
    if not open_:
        return None
    return min(open_, key=lambda i: (fronts[i], i)) if time_major else open_[0]


class RoutedActionCodec2Grammar:
    'Coverage grammar over a shared time grid and namespaced physical components.'

    def __init__(self, router: ActionCodec2Router, *, preset: str, horizon: int) -> None:
        horizon = _integer(horizon, "horizon", positive=True)
        self.segments = router._segments(router.preset(preset))
        self.horizon = horizon
        self.token_order = router.preset(preset).token_order
        self.vocab_size = router.vocab_size
        self._grammars = tuple(
            segment.profile.processor.grammar(horizon, active_dims=segment.component.active_dims)
            for segment in self.segments
        )
        self._native_grammars = tuple(
            getattr(grammar, "_native", None) for grammar in self._grammars
        )
        self._native_offsets = tuple(segment.profile.spec.token_offset for segment in self.segments)
        states = tuple(grammar._initial_state() for grammar in self._grammars)
        self._initial_state = (self._next_component(states), states)
        # Prefix states only; the empty prefix is answered from _initial_state.
        self._state_cache = BoundedCache(256)

    def _compiled_grammars(self):
        if native_kernel("route_advance") is None:
            return None
        if any(g is None for g in self._native_grammars):
            self._native_grammars = tuple(g._compiled() for g in self._grammars)
        return self._native_grammars if all(g is not None for g in self._native_grammars) else None

    def _next_component(self, states: _ComponentStates) -> int:
        fronts = [grammar._covered_steps(state) for grammar, state in zip(self._grammars, states, strict=True)]
        index = _schedule(fronts, [front >= self.horizon for front in fronts],
                          self.token_order == "time_major")
        return len(fronts) if index is None else index

    def _advance_routed_state(
        self,
        routed_state: _RoutedState,
        token: int,
    ) -> _RoutedState | None:
        segment_index, states = routed_state
        kernel = native_kernel("route_advance")
        compiled = self._compiled_grammars()
        if kernel is not None and compiled is not None:
            return kernel(
                compiled,
                self._native_offsets,
                self.token_order == "time_major",
                segment_index,
                states,
                token,
            )
        if segment_index == len(self.segments):
            return None
        segment = self.segments[segment_index]
        local_token = token - segment.profile.spec.token_offset
        if not 0 <= local_token < segment.profile.processor.vocab_size:
            return None
        next_state = self._grammars[segment_index]._advance(states[segment_index], local_token)
        if next_state is None:
            return None
        updated = (*states[:segment_index], next_state, *states[segment_index + 1 :])
        return self._next_component(updated), updated

    def _consume(self, tokens: Sequence[int]) -> _RoutedState | None:
        checked: list[int] = []
        for raw_token in tokens:
            if isinstance(raw_token, bool) or not isinstance(raw_token, (int, np.integer)):
                return None
            checked.append(int(raw_token))
        return cached_prefix_state(
            tuple(checked), self._state_cache, self._initial_state, self._advance_routed_state
        )

    def reserve_states(self, rows: int) -> None:
        'Size the prefix cache for ``rows`` streams advanced in lockstep.'
        self._state_cache.reserve(4 * _integer(rows, "rows", positive=True))

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        """Return a global namespace mask for the next preset token."""
        state = self._consume(tokens)
        if state is None:
            raise ValueError("token prefix is not legal for this routed preset")
        segment_index, states = state
        output = np.zeros(self.vocab_size, dtype=bool)
        if segment_index == len(self.segments):
            return output
        segment = self.segments[segment_index]
        mask = self._grammars[segment_index]._next_token_mask_from_state(states[segment_index])
        start = segment.profile.spec.token_offset
        output[start : start + len(mask)] = mask
        return output

    def is_complete(self, tokens: Sequence[int]) -> bool:
        """Whether the prefix completes every segment exactly once."""
        state = self._consume(tokens)
        return state is not None and state[0] == len(self.segments)

    def covered_steps(self, tokens: Sequence[int]) -> int:
        """Return the contiguous frame prefix covered by every component."""
        state = self._consume(tokens)
        if state is None:
            raise ValueError("token prefix is not legal for this routed preset")
        return min(
            grammar._covered_steps(local) for grammar, local in zip(self._grammars, state[1], strict=True)
        )

    def is_prefix_complete(self, tokens: Sequence[int], *, steps: int) -> bool:
        """Whether a legal token prefix fully covers the requested action frames."""
        if type(steps) is not int or not 1 <= steps <= self.horizon:
            raise ValueError("steps must be an integer within the grammar horizon")
        state = self._consume(tokens)
        return state is not None and all(
            grammar._covered_steps(local) >= steps
            for grammar, local in zip(self._grammars, state[1], strict=True)
        )

    def interleave(self, streams: Sequence[Sequence[int]]) -> list[int]:
        """Route per-component local token streams without changing their tokens."""
        kernel = native_kernel("route_interleave")
        compiled = self._compiled_grammars()
        if kernel is not None and compiled is not None:
            return kernel(compiled, self._native_offsets, self.token_order == "time_major", streams)
        if len(streams) != len(self.segments):
            raise ValueError("one token stream is required per component")
        state = self._initial_state
        cursors = [0] * len(streams)
        output = []
        while state[0] < len(streams):
            index = state[0]
            if cursors[index] == len(streams[index]):
                raise ValueError("component token stream ends before complete coverage")
            token = (
                int(streams[index][cursors[index]]) + self.segments[index].profile.spec.token_offset
            )
            state = self._advance_routed_state(state, token)
            if state is None:
                raise ValueError("component token stream violates the routed grammar")
            cursors[index] += 1
            output.append(token)
        if any(cursor != len(stream) for cursor, stream in zip(cursors, streams, strict=True)):
            raise ValueError("extra component tokens after complete coverage")
        return output

    def complete_prefix(self, tokens: Sequence[int], *, steps: int) -> list[int]:
        'Fill only unreturned future cells with zero-motion primitive atoms.'
        if type(steps) is not int or not 1 <= steps <= self.horizon:
            raise ValueError("steps must be an integer within the grammar horizon")
        if self.covered_steps(tokens) < steps:
            raise ValueError(f"action prefix does not cover the first {steps} frames")
        kernel = native_kernel("route_complete")
        compiled = self._compiled_grammars()
        if kernel is not None and compiled is not None:
            zero_atoms = [
                (
                    np.r_[0, np.cumsum(segment.profile.processor.bins[:-1])]
                    + segment.profile.processor._zero_cells
                ).tolist()
                for segment in self.segments
            ]
            return kernel(
                compiled,
                self._native_offsets,
                self.token_order == "time_major",
                [int(token) for token in tokens],
                steps,
                zero_atoms,
            )
        output = [int(token) for token in tokens]
        state = self._consume(output)
        while state[0] < len(self.segments):
            index, states = state
            segment = self.segments[index]
            processor = segment.profile.processor
            local = states[index]
            dims = segment.component.active_dims
            if processor.serializer.scheme == "unified_set_bpe":
                atom = self._grammars[index].completion_token(
                    local, processor._zero_cells, processor.bins
                )
                token = segment.profile.spec.token_offset + atom
                state = self._advance_routed_state(state, token)
                if state is None:
                    raise RuntimeError("zero-motion suffix atom violates the routed grammar")
                output.append(token)
                continue
            if processor.serializer.scheme == "bpe":
                dimension = dims[local[0] % len(dims)]
            else:
                full = sum(1 << d for d in dims)
                uncovered = next(full & ~mask for mask in local if mask != full)
                dimension = (uncovered & -uncovered).bit_length() - 1
            atom = sum(processor.serializer.bins[:dimension]) + int(
                processor._zero_cells[dimension]
            )
            token = segment.profile.spec.token_offset + atom
            state = self._advance_routed_state(state, token)
            if state is None:
                raise RuntimeError("zero-motion suffix atom violates the routed grammar")
            output.append(token)
        return output


class ActionCodec2Router:
    'Route heterogeneous dataset states into physical token namespaces.'

    model_type = ROUTER_MODEL_TYPE

    def __init__(
        self,
        profiles: Sequence[tuple[ProfileSpec, ActionCodec2Processor]],
        presets: Sequence[PresetConfig],
    ) -> None:
        runtimes: dict[str, _ProfileRuntime] = {}
        expected_offset = 0
        for spec, processor in profiles:
            if spec.name in runtimes:
                raise ConfigError(f"duplicate physical profile {spec.name!r}")
            if spec.token_offset != expected_offset:
                raise ConfigError("profile token namespaces must be contiguous and ordered")
            if processor.vocab_size > spec.token_budget:
                raise ConfigError(
                    f"profile {spec.name!r} uses {processor.vocab_size} tokens, "
                    f"exceeding budget {spec.token_budget}"
                )
            if processor.artifact_fingerprint != spec.artifact_fingerprint:
                raise ConfigError(
                    f"profile {spec.name!r} artifact fingerprint does not match its processor"
                )
            if processor.config.set_bpe.allow_cross_component_merge:
                raise ConfigError(
                    "allow_cross_component_merge=true is not supported by the v1 router; "
                    "a component-aware Set-BPE serializer and grammar are required"
                )
            runtimes[spec.name] = _ProfileRuntime(spec, processor)
            expected_offset += spec.token_budget
        if not runtimes or set(runtimes) - _PROFILES:
            raise ConfigError("router profiles must be a non-empty subset of joint and eef")
        for name, runtime in runtimes.items():
            if runtime.processor.action_dim != _PROFILE_WIDTHS[name]:
                raise ConfigError(
                    f"profile {name!r} must have width {_PROFILE_WIDTHS[name]}, "
                    f"got {runtime.processor.action_dim}"
                )
        self.profiles = MappingProxyType(runtimes)
        self._second_order_dims = {
            name: _second_order_dims(runtime.processor.config) for name, runtime in runtimes.items()
        }
        self._grammar_cache = BoundedCache(8)
        self._action_spaces: dict[str, PresetConfig] = {}
        for preset in presets:
            self.register_action_space(preset)
        self.vocab_size = expected_offset
        self._token_cells = np.full(self.vocab_size, -1, dtype=np.int64)
        for runtime in runtimes.values():
            start = runtime.spec.token_offset
            self._token_cells[start : start + runtime.processor.vocab_size] = (
                runtime.processor.serializer.token_cell_counts
            )

    @property
    def action_spaces(self) -> Mapping[str, PresetConfig]:
        """Read-only registered built-in and user-created action spaces."""
        return MappingProxyType(self._action_spaces)

    def __str__(self) -> str:
        return self.describe_action_spaces()

    def second_order_boundaries(self, preset: str | PresetConfig) -> dict[str, int]:
        'Map each component needing ``previous_first_order`` to its active width.'
        if not isinstance(preset, PresetConfig):
            preset = self.preset(preset)
        return _second_order_boundaries(preset, self._second_order_dims)

    def describe_action_spaces(
        self, *, selected: str | None = None, action_dim: int | None = None
    ) -> str:
        """Explain all layouts, or source-width candidates, using this artifact's state requirements."""
        return format_action_spaces(
            self.action_spaces,
            selected=selected,
            action_dim=action_dim,
            second_order_boundaries={
                name: self.second_order_boundaries(preset)
                for name, preset in self.action_spaces.items()
            },
        )

    @classmethod
    def from_processors(
        cls,
        processors: Mapping[str, ActionCodec2Processor],
        *,
        token_budgets: Mapping[str, int],
        presets: Sequence[PresetConfig],
    ) -> ActionCodec2Router:
        """Assign stable contiguous namespaces in canonical joint/eef order."""
        if set(processors) != set(token_budgets):
            raise ConfigError("processors and token_budgets must share keys")
        if not processors or set(processors) - _PROFILES:
            raise ConfigError("processors must contain a non-empty subset of joint and eef")
        offset = 0
        entries = []
        for name in ("joint", "eef"):
            if name not in processors:
                continue
            processor = processors[name]
            budget = _integer(token_budgets[name], f"token_budgets.{name}", positive=True)
            spec = ProfileSpec(
                name,
                offset,
                budget,
                processor.artifact_fingerprint,
            )
            entries.append((spec, processor))
            offset += budget
        return cls(entries, presets)

    def preset(self, name: str) -> PresetConfig:
        """Resolve one preset or fail with the available names."""
        try:
            return self._action_spaces[canonical_preset_name(name)]
        except KeyError as error:
            raise KeyError(
                f"unknown preset {name!r}; available: {sorted(self._action_spaces)}"
            ) from error

    def register_action_space(
        self,
        preset: PresetConfig | Mapping[str, object] | str | Path,
    ) -> PresetConfig:
        'Register a validated action-space config, mapping, or YAML file.'
        if isinstance(preset, Mapping):
            preset = PresetConfig.from_dict(preset)
        elif isinstance(preset, (str, Path)):
            preset = PresetConfig.from_yaml(preset)
        if not isinstance(preset, PresetConfig):
            raise TypeError(
                "register_action_space expects PresetConfig, a mapping, or a preset YAML path"
            )
        missing = {component.space for component in preset.components} - self.profiles.keys()
        if missing:
            raise ConfigError(
                f"action space {preset.name!r} requires missing profiles: {sorted(missing)}"
            )
        existing = self._action_spaces.get(preset.name)
        if existing is not None and existing != preset:
            raise ConfigError(
                f"action-space name {preset.name!r} is already registered differently"
            )
        self._action_spaces[preset.name] = preset
        return preset

    def _segments(self, preset: PresetConfig) -> tuple[_Segment, ...]:
        return tuple(
            _Segment(component, self.profiles[component.space]) for component in preset.components
        )

    def configure_encoding(
        self,
        *,
        spatial_search: bool = False,
        temporal_search: bool = False,
        search_window: int = 32,
        search_beam: int = 4,
    ) -> ActionCodec2Router:
        """Apply runtime packing options to all profiles without changing artifacts."""
        from .packing import EncodingSearch

        options = EncodingSearch(spatial_search, temporal_search, search_window, search_beam)
        for profile in self.profiles.values():
            serializer = profile.processor.serializer
            options.validate_codec(serializer.scheme, serializer.bins)
        for profile in self.profiles.values():
            profile.processor.configure_encoding(
                spatial_search=spatial_search,
                temporal_search=temporal_search,
                search_window=search_window,
                search_beam=search_beam,
            )
        return self

    def encode_batch(
        self,
        inputs: np.ndarray,
        *,
        preset: str,
        fps: float | None = None,
        current_state: np.ndarray | None = None,
        previous_first_order: Mapping[str, np.ndarray] | None = None,
    ) -> list[list[int]]:
        'Encode one regular-rate ``[B,T,D]`` tensor at ``fps``.'
        preset_config = self.preset(preset)
        self._validate_boundary_keys(previous_first_order, preset_config)
        canonical = preset_config._canonicalize_batch(
            inputs, source_hz=fps, current_state=current_state
        )
        batch = len(next(iter(canonical.values())))
        component_streams = []
        for segment in self._segments(preset_config):
            values = canonical[segment.component.name]
            streams = segment.profile.processor.encode_batch(
                values,
                active_dims=segment.component.active_dims,
                previous_first_order=None
                if previous_first_order is None
                else previous_first_order.get(segment.component.name),
            )
            component_streams.append(streams)
        steps = next(iter(canonical.values())).shape[1]
        grammar = self.grammar(preset=preset, horizon=steps)
        kernel = native_kernel("route_interleave_batch")
        compiled = grammar._compiled_grammars()
        if kernel is not None and compiled is not None:
            return kernel(
                compiled,
                grammar._native_offsets,
                grammar.token_order == "time_major",
                component_streams,
            )
        return [
            grammar.interleave([streams[row] for streams in component_streams])
            for row in range(batch)
        ]

    def route_trajectories(
        self,
        inputs: np.ndarray,
        *,
        preset: str,
        current_state: np.ndarray | None = None,
        fps: float | None = None,
        previous_first_order: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, list[ActiveTrajectory]]:
        'Expose canonical complete trajectories grouped by physical profile.'
        preset_config = self.preset(preset)
        self._validate_boundary_keys(previous_first_order, preset_config)
        canonical = preset_config.canonicalize(inputs, current_state=current_state, source_hz=fps)
        output: dict[str, list[ActiveTrajectory]] = {}
        for component in preset_config.components:
            trajectories = canonical[component.name]
            processor = self.profiles[component.space].processor
            previous = (
                None if previous_first_order is None else previous_first_order.get(component.name)
            )
            dense = processor._dense_previous_first_order(
                previous, component.active_dims, len(trajectories)
            )
            rows = (
                [None] * len(trajectories)
                if dense is None
                else np.take(dense, component.active_dims, axis=1)
            )
            output.setdefault(component.space, []).extend(
                ActiveTrajectory(t.values, t.active_dims, previous_first_order=p)
                for t, p in zip(trajectories, rows, strict=True)
            )
        return output

    def _validate_boundary_keys(
        self, previous_first_order: Mapping[str, np.ndarray] | None, preset: PresetConfig
    ) -> None:
        if previous_first_order is None:
            previous_first_order = {}
        if not isinstance(previous_first_order, Mapping):
            raise ValueError(
                "previous_first_order must map component names to canonical [B,D_active] boundaries"
            )
        unknown = set(previous_first_order) - {c.name for c in preset.components}
        if unknown:
            raise ValueError(f"unknown previous_first_order component(s): {sorted(unknown)}")
        required = {
            name: f"(B, {width})" for name, width in self.second_order_boundaries(preset).items()
        }
        missing = set(required) - previous_first_order.keys()
        if missing:
            raise ValueError(
                "second-order tokenization requires previous_first_order with arrays "
                f"{required}; missing {sorted(missing)}. Use canonical {preset.target_hz:g} Hz "
                "reconstructed increments. codec.first_order_template(batch_size) explicitly "
                "creates zeros; use zeros only if the preceding increment is known to be zero."
            )

    def final_first_order(
        self,
        streams: Sequence[Sequence[int]],
        *,
        preset: str,
        previous_first_order: Mapping[str, np.ndarray],
        executed_steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        'Return component ``[B,D_active]`` boundaries at the canonical clock.'
        preset_config = self.preset(preset)
        if previous_first_order is None:
            raise ValueError("final_first_order requires previous_first_order")
        self._validate_boundary_keys(previous_first_order, preset_config)
        segments = self._segments(preset_config)
        _, component_streams = self._decode_streams(streams, preset_config, segments)
        output = {}
        for index, segment in enumerate(segments):
            processor = segment.profile.processor
            if not processor.requires_previous_first_order:
                continue
            output[segment.component.name] = processor.final_first_order(
                component_streams[index],
                previous_first_order=previous_first_order.get(segment.component.name),
                active_dims=segment.component.active_dims,
                executed_steps=executed_steps,
            )
        return output

    def _split_stream(
        self, tokens: Sequence[int], *, preset: PresetConfig, steps: int
    ) -> list[list[int]]:
        grammar = self.grammar(preset=preset.name, horizon=steps)
        kernel = native_kernel("route_split")
        compiled = grammar._compiled_grammars()
        if kernel is not None and compiled is not None:
            checked = []
            for cursor, token in enumerate(tokens):
                if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
                    raise ValueError(f"tokens[{cursor}] must be an integer")
                checked.append(int(token))
            return kernel(
                compiled, grammar._native_offsets, grammar.token_order == "time_major", checked
            )
        state = grammar._initial_state
        output = [[] for _ in grammar.segments]
        for cursor, raw_token in enumerate(tokens):
            if isinstance(raw_token, bool) or not isinstance(raw_token, (int, np.integer)):
                raise ValueError(f"tokens[{cursor}] must be an integer")
            index = state[0]
            if index == len(grammar.segments):
                raise ValueError("Action token stream has extra tokens after complete coverage")
            segment = grammar.segments[index]
            state = grammar._advance_routed_state(state, int(raw_token))
            if state is None:
                raise ValueError(
                    f"Action token stream has invalid coverage for component {segment.component.name!r}; check the preset token_order and constrain generation with codec.grammar(H, fps=...)."
                )
            output[index].append(int(raw_token) - segment.profile.spec.token_offset)
        if state[0] != len(grammar.segments):
            raise ValueError("Incomplete action token stream; discard or regenerate this row")
        return output

    def _token_cell_count(self, raw_token: object, *, index: int) -> int:
        """Return the number of primitive cells represented by one routed token."""
        if isinstance(raw_token, bool) or not isinstance(raw_token, (int, np.integer)):
            raise ValueError(f"tokens[{index}] must be an integer")
        token = int(raw_token)
        if 0 <= token < self.vocab_size and self._token_cells[token] >= 0:
            return int(self._token_cells[token])
        raise ValueError(f"tokens[{index}] is outside the routed profile vocabularies")

    def _decoded_steps(self, tokens: Sequence[int], segments: Sequence[_Segment]) -> int:
        """Derive sequence length from the primitive-cell coverage in a routed stream."""
        if not tokens:
            raise ValueError("token stream must not be empty")
        cells_per_step = sum(len(segment.component.active_dims) for segment in segments)
        represented_cells = sum(
            self._token_cell_count(token, index=index) for index, token in enumerate(tokens)
        )
        steps, remainder = divmod(represented_cells, cells_per_step)
        if steps < 1 or remainder:
            raise ValueError(
                "Incomplete action token stream: "
                f"{cells_per_step - remainder} more coordinate values are needed to finish a step. "
                "Constrain generation with codec.grammar(H, fps=...).next_token_mask(prefix) "
                "and check is_complete(tokens) for the intended H before decoding. "
                "Use codec.batch_decode(..., errors='return') to isolate invalid rows; "
                "do not pad truncated tokens with zeros."
            )
        return steps

    def _batch_steps(
        self,
        streams: Sequence[Sequence[int]],
        segments: Sequence[_Segment],
    ) -> int:
        """Derive the common decoded length of a dense output batch."""
        if not streams:
            raise ValueError("streams must not be empty")
        lengths = [self._decoded_steps(stream, segments) for stream in streams]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"all token streams must decode to the same length, got {lengths}; use codec.batch_decode(...) for independent variable-length outputs"
            )
        return lengths[0]

    def decode_batch(
        self,
        streams: Sequence[Sequence[int]],
        *,
        preset: str,
        fps: float | None = None,
        current_state: np.ndarray | None = None,
        previous_first_order: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        'Decode rows to deltas, or absolute components with ``current_state``.'
        preset_config = self.preset(preset)
        self._validate_boundary_keys(previous_first_order, preset_config)
        segments = self._segments(preset_config)
        internal_steps, component_streams = self._decode_streams(streams, preset_config, segments)
        state_array = optional_finite_array(
            current_state, shape=(len(streams), preset_config.input_dim)
        )
        output_fps = preset_config.source_hz if fps is None else fps
        output_steps = resampled_horizon(
            internal_steps,
            source_hz=preset_config.target_hz,
            target_hz=output_fps,
        )
        output: dict[str, np.ndarray] = {}
        for segment_index, segment in enumerate(segments):
            decoded_batch = segment.profile.processor.decode_batch(
                component_streams[segment_index],
                active_dims=segment.component.active_dims,
                previous_first_order=None
                if previous_first_order is None
                else previous_first_order.get(segment.component.name),
            )
            output[segment.component.name] = externalize_batch(
                decoded_batch,
                segment.component,
                source_hz=preset_config.target_hz,
                target_hz=output_fps,
                target_length=output_steps,
                method=preset_config.interpolation,
                current_state=state_array,
            )
        return output

    def _decode_streams(self, streams, preset, segments):
        """Validate routed rows and return their common horizon and component batches."""
        kernel = native_kernel("route_decode_batch")
        if kernel is not None:
            key = next(
                (key for key in reversed(tuple(self._grammar_cache)) if key[0] == preset.name), None
            )
            grammar = self._grammar_cache.get(key)
            if grammar is None:
                grammar = self.grammar(preset=preset.name, horizon=1)
            compiled = grammar._compiled_grammars()
            if compiled is not None:
                width = sum(len(segment.component.active_dims) for segment in segments)
                return kernel(
                    compiled,
                    grammar._native_offsets,
                    grammar.token_order == "time_major",
                    streams,
                    self._token_cells,
                    width,
                )
        steps = self._batch_steps(streams, segments)
        splits = [self._split_stream(row, preset=preset, steps=steps) for row in streams]
        return steps, [[row[i] for row in splits] for i in range(len(segments))]

    def decode_prefix_batch(
        self,
        streams,
        *,
        preset: str,
        horizon: int,
        steps: int,
        fps: float | None = None,
        current_state=None,
        previous_first_order=None,
    ) -> dict[str, np.ndarray]:
        'Decode only fully covered execution frames on the native codec clock.'
        config = self.preset(preset)
        output_fps = config.source_hz if fps is None else fps
        if output_fps != config.target_hz:
            raise ValueError("prefix decode requires fps equal to the native codec clock")
        grammar = self.grammar(preset=preset, horizon=horizon)
        completed = [grammar.complete_prefix(row, steps=steps) for row in streams]
        decoded = self.decode_batch(
            completed,
            preset=preset,
            fps=output_fps,
            current_state=current_state,
            previous_first_order=previous_first_order,
        )
        return {name: values[:, :steps] for name, values in decoded.items()}

    def grammar(self, *, preset: str, horizon: int) -> RoutedActionCodec2Grammar:
        """Build the global constrained grammar for one preset and target horizon."""
        horizon = _integer(horizon, "horizon", positive=True)
        self.preset(preset)
        key = (preset, horizon)
        grammar = self._grammar_cache.get(key)
        if grammar is None:
            grammar = RoutedActionCodec2Grammar(self, preset=preset, horizon=horizon)
            self._grammar_cache[key] = grammar
        return grammar

    @property
    def new_action_tokens(self) -> list[str]:
        """Stable physical token names, including reserved profile budget slots."""
        return [
            f"<action_{name}_{index:04d}>"
            for name, runtime in self.profiles.items()
            for index in range(runtime.spec.token_budget)
        ]

    def save_pretrained(self, directory: str | Path) -> list[str]:
        """Save one manifest plus isolated physical processor subdirectories."""
        root = Path(directory)
        files = []
        for runtime in self.profiles.values():
            files.extend(runtime.processor.save_pretrained(root / "profiles" / runtime.spec.name))
        manifest = {
            "model_type": self.model_type,
            "config_version": ROUTER_CONFIG_VERSION,
            "profiles": [runtime.spec.to_dict() for runtime in self.profiles.values()],
            "presets": [preset.to_dict() for preset in self.action_spaces.values()],
        }
        path = root / ROUTER_CONFIG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        metadata = save_huggingface_metadata(
            root,
            model_type=self.model_type,
            processor_class=type(self).__name__,
        )
        card = save_model_card(root, str(self))
        return [str(path), str(card), *map(str, metadata), *files]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: object,
    ) -> ActionCodec2Router | ActionCodec2:
        'Load a router, or bind its public processor when given ``action_space``.'
        if "action_space" in kwargs:
            from ...frontend.actioncodec2 import ActionCodec2

            return ActionCodec2.from_pretrained(pretrained_model_name_or_path, **kwargs)
        root = resolve_pretrained_directory(
            pretrained_model_name_or_path,
            allow_patterns=ROUTER_ALLOW_PATTERNS,
            **kwargs,
        )
        specs, presets = cls.read_manifest(root)
        profiles = [
            (spec, ActionCodec2Processor.from_pretrained(root / "profiles" / spec.name))
            for spec in specs
        ]
        return cls(profiles, presets)

    @classmethod
    def read_manifest(
        cls, directory: str | Path
    ) -> tuple[tuple[ProfileSpec, ...], tuple[PresetConfig, ...]]:
        "Read one artifact's profile specs and action spaces, loading no arrays."
        path = Path(directory) / ROUTER_CONFIG_NAME
        payload = _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader), "router"
        )
        _reject_unknown(payload, {"model_type", "config_version", "profiles", "presets"}, "router")
        if payload.get("model_type") != ROUTER_MODEL_TYPE:
            raise ConfigError(f"router.model_type must be {ROUTER_MODEL_TYPE!r}")
        version = payload.get("config_version")
        if version not in (3, ROUTER_CONFIG_VERSION):
            raise ConfigError(
                f"unsupported router config_version {payload.get('config_version')!r}"
            )
        raw_profiles = payload.get("profiles")
        raw_presets = payload.get("presets")
        if not isinstance(raw_profiles, list) or not isinstance(raw_presets, list):
            raise ConfigError("router.profiles and router.presets must be lists")
        if version == 3:
            raw_presets = [
                dict(_mapping(raw, "router.preset"), token_order="time_major")
                for raw in raw_presets
            ]
        elif any("token_order" not in _mapping(raw, "router.preset") for raw in raw_presets):
            raise ConfigError("router config_version=4 requires explicit preset.token_order")
        return (
            tuple(
                ProfileSpec.from_dict(raw, path=f"router.profiles[{index}]")
                for index, raw in enumerate(raw_profiles)
            ),
            tuple(
                PresetConfig.from_dict(raw, path=f"router.presets[{index}]")
                for index, raw in enumerate(raw_presets)
            ),
        )

    @classmethod
    def describe_pretrained(
        cls,
        directory: str | Path,
        *,
        selected: str | None = None,
        action_dim: int | None = None,
    ) -> str:
        "Explain a saved artifact's layouts without loading primitives or BPE tables."
        root = Path(directory)
        specs, presets = cls.read_manifest(root)
        second_order_dims = {
            spec.name: _second_order_dims(
                ActionCodec2Config.from_pretrained(root / "profiles" / spec.name)
            )
            for spec in specs
        }
        return format_action_spaces(
            {preset.name: preset for preset in presets},
            selected=selected,
            action_dim=action_dim,
            second_order_boundaries={
                preset.name: _second_order_boundaries(preset, second_order_dims)
                for preset in presets
            },
        )

    def eval(self) -> ActionCodec2Router:
        """Return ``self`` because all constituent processors are stateless."""
        return self

    def to(self, device: object) -> ActionCodec2Router:
        """Return ``self`` because routed processors use CPU NumPy arrays."""
        del device
        return self
