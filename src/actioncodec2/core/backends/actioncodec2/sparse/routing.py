"""Action-space routing over ActionCodec2Sparse profiles."""

from __future__ import annotations

from ..presets import canonical_preset_name
from ..routing import _schedule

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from .._validation import optional_finite_array
from ..batch_semantics import externalize_batch
from ..cache import BoundedCache, cached_prefix_state
from ..configuration import ConfigError, _integer, _mapping, _reject_unknown, _UniqueKeyLoader
from ..hf import save_huggingface_metadata
from ..presets import ComponentSpec, PresetConfig, format_action_spaces
from ..routing import (
    _PROFILE_WIDTHS,
    _PROFILES,
    ProfileSpec,
    _ProfileRuntime,
    _second_order_boundaries,
    _second_order_dims,
)
from ..semantic import resampled_horizon
from ..trajectory import ActiveTrajectory
from .processor import ActionCodec2SparseProcessor

SPARSE_ROUTER_MODEL_TYPE = "actioncodec2_sparse_router"
SPARSE_ROUTER_CONFIG_NAME = "sparse_router_config.yaml"
SPARSE_ROUTER_CONFIG_VERSION = 1


RoutedState = tuple[tuple[bool, ...], tuple]


@dataclass(frozen=True)
class _Segment:
    component: ComponentSpec
    spec: ProfileSpec
    processor: ActionCodec2SparseProcessor




class _FrontTracker:
    def __init__(self, segment: _Segment, horizon: int) -> None:
        tables = segment.processor.vocabulary.tables
        self.emask, self.offset = tables.emask, segment.spec.token_offset
        self.vocab, self.horizon = len(tables), horizon
        self.front, self.closed = 0, 0

    def push(self, token: int) -> None:
        self.closed |= self.emask[token] << self.front
        while self.front < self.horizon and self.closed >> self.front & 1:
            self.front += 1


class RoutedSparseGrammar:
    """Constrained-generation grammar of one preset and codec-frame horizon."""

    def __init__(self, router: ActionCodec2SparseRouter, *, preset: str, horizon: int) -> None:
        self.horizon = _integer(horizon, "horizon", positive=True)
        self.segments = router._segments(router.preset(preset))
        self.time_major = router.preset(preset).token_order == "time_major"
        self.vocab_size, self.end_token = router.vocab_size, router.end_token
        self._grammars = tuple(
            s.processor.grammar(self.horizon, active_dims=s.component.active_dims)
            for s in self.segments
        )
        self._initial: RoutedState = (
            (False,) * len(self.segments),
            tuple(g.initial() for g in self._grammars),
        )
        self._cache = BoundedCache(256)

    def _scheduled(self, state: RoutedState) -> int | None:
        fronts = [g.front(s) for g, s in zip(self._grammars, state[1], strict=True)]
        return _schedule(fronts, state[0], self.time_major)

    def _advance(self, state: RoutedState, token: int) -> RoutedState | None:
        finished, states = state
        index = self._scheduled(state)
        if index is None:
            return None
        if token == self.end_token:
            if finished.count(False) < 2 or not self._grammars[index].is_complete(states[index]):
                return None
            return (*finished[:index], True, *finished[index + 1 :]), states
        local = token - self.segments[index].spec.token_offset
        updated = self._grammars[index].advance(states[index], local)
        if updated is None:
            return None
        return finished, (*states[:index], updated, *states[index + 1 :])

    def _consume(self, tokens: Sequence[int]) -> RoutedState | None:
        checked = []
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
                return None
            checked.append(int(token))
        return cached_prefix_state(tuple(checked), self._cache, self._initial, self._advance)

    def reserve_states(self, rows: int) -> None:
        """Size the prefix cache for ``rows`` streams advanced in lockstep."""
        self._cache.reserve(4 * _integer(rows, "rows", positive=True))

    def _mask_from_state(self, state: RoutedState) -> np.ndarray:
        output = np.zeros(self.vocab_size, dtype=bool)
        index = self._scheduled(state)
        if index is None:
            return output
        finished, states = state
        mask = self._grammars[index].mask(states[index])
        start = self.segments[index].spec.token_offset
        output[start : start + len(mask)] = mask
        if finished.count(False) >= 2 and self._grammars[index].is_complete(states[index]):
            output[self.end_token] = True
        return output

    def _state_complete(self, state: RoutedState) -> bool:
        finished, states = state
        open_ = [i for i, done in enumerate(finished) if not done]
        return len(open_) == 1 and self._grammars[open_[0]].is_complete(states[open_[0]])

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        """Boolean ``(vocab_size,)`` mask of legal next routed tokens."""
        state = self._consume(tokens)
        if state is None:
            raise ValueError("token prefix is not legal for this sparse preset")
        return self._mask_from_state(state)

    def is_complete(self, tokens: Sequence[int]) -> bool:
        """Whether the stream may end here: one component left and its chunk complete."""
        state = self._consume(tokens)
        return state is not None and self._state_complete(state)

    def covered_steps(self, tokens: Sequence[int]) -> int:
        """Leading codec frames that no further token can change."""
        state = self._consume(tokens)
        if state is None:
            raise ValueError("token prefix is not legal for this sparse preset")
        return min(
            self.horizon if done else grammar.front(local)
            for grammar, local, done in zip(self._grammars, state[1], state[0], strict=True)
        )

    def accepts(self, tokens: Sequence[int]) -> bool:
        return self.is_complete(tokens)


class ActionCodec2SparseRouter:
    """Route action-space tensors through ActionCodec2Sparse profile namespaces."""

    model_type = SPARSE_ROUTER_MODEL_TYPE

    def __init__(
        self,
        profiles: Sequence[tuple[ProfileSpec, ActionCodec2SparseProcessor]],
        presets: Sequence[PresetConfig],
    ) -> None:
        runtimes: dict[str, _ProfileRuntime] = {}
        offset = 0
        for spec, processor in profiles:
            if spec.name in runtimes or spec.name not in _PROFILES:
                raise ConfigError(f"invalid or duplicate sparse profile {spec.name!r}")
            if spec.token_offset != offset:
                raise ConfigError("profile token namespaces must be contiguous and ordered")
            if processor.vocab_size > spec.token_budget:
                raise ConfigError(
                    f"profile {spec.name!r} uses {processor.vocab_size} tokens, "
                    f"exceeding budget {spec.token_budget}"
                )
            if processor.artifact_fingerprint != spec.artifact_fingerprint:
                raise ConfigError(f"profile {spec.name!r} fingerprint does not match its artifact")
            if processor.action_dim != _PROFILE_WIDTHS[spec.name]:
                raise ConfigError(
                    f"profile {spec.name!r} must have width {_PROFILE_WIDTHS[spec.name]}"
                )
            runtimes[spec.name] = _ProfileRuntime(spec, processor)
            offset += spec.token_budget
        if not runtimes:
            raise ConfigError("a sparse router needs at least one profile")
        self.profiles = MappingProxyType(runtimes)
        self._second_order_dims = {
            name: _second_order_dims(runtime.processor.config) for name, runtime in runtimes.items()
        }
        self.end_token = offset
        self.vocab_size = offset + 1
        self._grammars = BoundedCache(8)
        self._action_spaces: dict[str, PresetConfig] = {}
        for preset in presets:
            self.register_action_space(preset)

    @classmethod
    def from_processors(
        cls,
        processors: Mapping[str, ActionCodec2SparseProcessor],
        *,
        token_budgets: Mapping[str, int],
        presets: Sequence[PresetConfig],
    ) -> ActionCodec2SparseRouter:
        """Assign stable contiguous namespaces in canonical joint/EEF order."""
        if set(processors) != set(token_budgets) or set(processors) - _PROFILES:
            raise ConfigError("processors and token_budgets must share joint/eef keys")
        entries, offset = [], 0
        for name in ("joint", "eef"):
            if name in processors:
                budget = _integer(token_budgets[name], f"token_budgets.{name}", positive=True)
                spec = ProfileSpec(name, offset, budget, processors[name].artifact_fingerprint)
                entries.append((spec, processors[name]))
                offset += budget
        return cls(entries, presets)

    @property
    def action_spaces(self) -> Mapping[str, PresetConfig]:
        return MappingProxyType(self._action_spaces)

    def preset(self, name: str) -> PresetConfig:
        try:
            return self._action_spaces[canonical_preset_name(name)]
        except KeyError as error:
            raise KeyError(
                f"unknown preset {name!r}; available: {sorted(self._action_spaces)}"
            ) from error

    def register_action_space(self, preset) -> PresetConfig:
        """Register a validated preset, mapping, or YAML path without changing token IDs."""
        if isinstance(preset, Mapping):
            preset = PresetConfig.from_dict(preset)
        elif isinstance(preset, (str, Path)):
            preset = PresetConfig.from_yaml(preset)
        if not isinstance(preset, PresetConfig):
            raise TypeError("register_action_space expects PresetConfig, a mapping, or a YAML path")
        missing = {c.space for c in preset.components} - self.profiles.keys()
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

    def second_order_boundaries(self, preset) -> dict[str, int]:
        """Map each component needing ``previous_first_order`` to its active width."""
        if not isinstance(preset, PresetConfig):
            preset = self.preset(preset)
        return _second_order_boundaries(preset, self._second_order_dims)

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

    @staticmethod
    def _boundary(previous_first_order, segment: _Segment):
        return (
            None
            if previous_first_order is None
            else previous_first_order.get(segment.component.name)
        )

    def describe_action_spaces(self, *, selected=None, action_dim=None) -> str:
        return format_action_spaces(
            self.action_spaces,
            selected=selected,
            action_dim=action_dim,
            codec_name="ActionCodec2Sparse",
            chunk_horizon_decode=True,
        )

    def __str__(self) -> str:
        return self.describe_action_spaces()

    def to(self, device: object) -> ActionCodec2SparseRouter:
        """Return ``self`` because routed processors use CPU NumPy arrays."""
        del device
        return self

    def _segments(self, preset: PresetConfig) -> tuple[_Segment, ...]:
        return tuple(
            _Segment(c, self.profiles[c.space].spec, self.profiles[c.space].processor)
            for c in preset.components
        )

    def grammar(self, *, preset: str, horizon: int) -> RoutedSparseGrammar:
        key = (preset, int(horizon))
        grammar = self._grammars.get(key)
        if grammar is None:
            grammar = RoutedSparseGrammar(self, preset=preset, horizon=horizon)
            self._grammars[key] = grammar
        return grammar

    def encode_batch(
        self,
        inputs,
        *,
        preset: str,
        fps=None,
        current_state=None,
        previous_first_order=None,
        threads: int = 0,
    ) -> list[list[int]]:
        """Encode ``(B,T,D)`` source chunks at ``fps`` into routed token rows."""
        config = self.preset(preset)
        self._validate_boundary_keys(previous_first_order, config)
        canonical = config._canonicalize_batch(inputs, source_hz=fps, current_state=current_state)
        segments = self._segments(config)
        streams = [
            s.processor.encode_batch(
                canonical[s.component.name],
                active_dims=s.component.active_dims,
                previous_first_order=self._boundary(previous_first_order, s),
                threads=threads,
            )
            for s in segments
        ]
        horizon = next(iter(canonical.values())).shape[1]
        time_major = config.token_order == "time_major"
        return [
            self._interleave([rows[i] for rows in streams], segments, horizon, time_major)
            for i in range(len(streams[0]))
        ]

    def _interleave(self, streams, segments, horizon, time_major) -> list[int]:
        trackers = [_FrontTracker(s, horizon) for s in segments]
        cursors, finished, output = [0] * len(streams), [False] * len(streams), []
        while True:
            index = _schedule([t.front for t in trackers], finished, time_major)
            if index is None:
                return output
            if cursors[index] == len(streams[index]):
                if finished.count(False) == 1:
                    return output
                finished[index] = True
                output.append(self.end_token)
                continue
            token = int(streams[index][cursors[index]])
            cursors[index] += 1
            trackers[index].push(token)
            output.append(token + segments[index].spec.token_offset)

    def _split(self, row, segments, horizon, time_major) -> list[list[int]]:
        trackers = [_FrontTracker(s, horizon) for s in segments]
        finished, output = [False] * len(segments), [[] for _ in segments]
        for position, raw in enumerate(row):
            if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
                raise ValueError(f"tokens[{position}] must be an integer")
            index = _schedule([t.front for t in trackers], finished, time_major)
            if index is None:
                raise ValueError("token after every component is finished")
            token = int(raw)
            if token == self.end_token:
                if finished.count(False) < 2:
                    raise ValueError("the end token cannot close the last open component")
                finished[index] = True
                continue
            local = token - segments[index].spec.token_offset
            if not 0 <= local < trackers[index].vocab:
                raise ValueError(
                    f"tokens[{position}] is outside component {segments[index].component.name!r}"
                )
            if trackers[index].front >= horizon:
                raise ValueError("token after the last open frame")
            trackers[index].push(local)
            output[index].append(local)
        return output

    def _split_rows(self, rows, config, horizon):
        horizon = _integer(horizon, "horizon", positive=True)
        rows = list(rows)
        if not rows:
            raise ValueError("rows must not be empty")
        segments = self._segments(config)
        time_major = config.token_order == "time_major"
        return segments, [self._split(row, segments, horizon, time_major) for row in rows]

    def decode_batch(
        self,
        rows,
        *,
        preset: str,
        horizon: int,
        fps=None,
        current_state=None,
        previous_first_order=None,
    ) -> dict[str, np.ndarray]:
        """Decode rows of one codec-frame ``horizon`` to ``{component: (B,T_out,D_c)}``."""
        config = self.preset(preset)
        self._validate_boundary_keys(previous_first_order, config)
        segments, splits = self._split_rows(rows, config, horizon)
        state = optional_finite_array(current_state, shape=(len(splits), config.input_dim))
        output_fps = config.source_hz if fps is None else fps
        steps = resampled_horizon(horizon, source_hz=config.target_hz, target_hz=output_fps)
        output = {}
        for index, segment in enumerate(segments):
            decoded = segment.processor.decode_batch(
                [split[index] for split in splits],
                horizon=horizon,
                active_dims=segment.component.active_dims,
                previous_first_order=self._boundary(previous_first_order, segment),
            )
            output[segment.component.name] = externalize_batch(
                decoded,
                segment.component,
                source_hz=config.target_hz,
                target_hz=output_fps,
                target_length=steps,
                method=config.interpolation,
                current_state=state,
            )
        return output

    def final_first_order(
        self,
        rows,
        *,
        preset: str,
        horizon: int,
        previous_first_order,
        executed_steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Component ``(B,D_active)`` boundaries after ``executed_steps`` codec frames."""
        config = self.preset(preset)
        if previous_first_order is None:
            raise ValueError("final_first_order requires previous_first_order")
        self._validate_boundary_keys(previous_first_order, config)
        segments, splits = self._split_rows(rows, config, horizon)
        output = {}
        for index, segment in enumerate(segments):
            if not segment.processor.requires_previous_first_order:
                continue
            output[segment.component.name] = segment.processor.final_first_order(
                [split[index] for split in splits],
                horizon=horizon,
                previous_first_order=self._boundary(previous_first_order, segment),
                active_dims=segment.component.active_dims,
                executed_steps=executed_steps,
            )
        return output

    def route_trajectories(
        self,
        inputs: np.ndarray,
        *,
        preset: str,
        current_state: np.ndarray | None = None,
        fps: float | None = None,
        previous_first_order: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, list[ActiveTrajectory]]:
        """Expose canonical complete trajectories grouped by physical profile."""
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

    @property
    def new_action_tokens(self) -> list[str]:
        """Stable token names: profile namespaces (budget slots included), then the end token."""
        names = [
            f"<action_{name}_{index:04d}>"
            for name, runtime in self.profiles.items()
            for index in range(runtime.spec.token_budget)
        ]
        return [*names, "<action_end>"]

    def save_pretrained(self, directory) -> list[str]:
        root = Path(directory)
        files = []
        for name, runtime in self.profiles.items():
            files.extend(runtime.processor.save_pretrained(root / "profiles" / name))
        manifest = {
            "model_type": self.model_type,
            "config_version": SPARSE_ROUTER_CONFIG_VERSION,
            "profiles": [runtime.spec.to_dict() for runtime in self.profiles.values()],
            "presets": [preset.to_dict() for preset in self.action_spaces.values()],
        }
        path = root / SPARSE_ROUTER_CONFIG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return [str(path), *files]

    @staticmethod
    def read_manifest(directory) -> tuple[tuple[ProfileSpec, ...], tuple[PresetConfig, ...]]:
        """Read profile specs and action spaces without loading numerical arrays."""
        path = Path(directory) / SPARSE_ROUTER_CONFIG_NAME
        payload = _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader), "router"
        )
        _reject_unknown(payload, {"model_type", "config_version", "profiles", "presets"}, "router")
        if payload.get("model_type") != SPARSE_ROUTER_MODEL_TYPE:
            raise ConfigError(f"router.model_type must be {SPARSE_ROUTER_MODEL_TYPE!r}")
        if payload.get("config_version") != SPARSE_ROUTER_CONFIG_VERSION:
            raise ConfigError(
                f"unsupported sparse router config_version {payload.get('config_version')!r}"
            )
        profiles, presets = payload.get("profiles"), payload.get("presets")
        if not isinstance(profiles, list) or not isinstance(presets, list):
            raise ConfigError("router.profiles and router.presets must be lists")
        return (
            tuple(
                ProfileSpec.from_dict(p, path=f"router.profiles[{i}]")
                for i, p in enumerate(profiles)
            ),
            tuple(
                PresetConfig.from_dict(p, path=f"router.presets[{i}]")
                for i, p in enumerate(presets)
            ),
        )

    @classmethod
    def from_pretrained(cls, directory) -> ActionCodec2SparseRouter:
        root = Path(directory)
        specs, presets = cls.read_manifest(root)
        profiles = [
            (spec, ActionCodec2SparseProcessor.from_pretrained(root / "profiles" / spec.name))
            for spec in specs
        ]
        return cls(profiles, presets)


def save_router_metadata(directory, **processor_config) -> tuple[Path, Path]:
    """Hugging Face discovery files for a saved sparse router or bound frontend."""
    return save_huggingface_metadata(
        directory,
        model_type="actioncodec2_sparse",
        processor_class="ActionCodec2Sparse",
        **processor_config,
    )
