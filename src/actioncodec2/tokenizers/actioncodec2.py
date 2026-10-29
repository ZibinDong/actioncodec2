"""G0.5 adapter for the public ActionCodec2 processor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import importlib
import math
from numbers import Integral
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .base import ChunkGrammar, ChunkTokenizer
from .hub_acceleration import _configure as _configure_threads


def _load_processor(path: str | Path, *, variant=None, revision=None, variant_fingerprint=None):
    from .hub_catalog import ActionCodec2

    root = Path(path).expanduser()
    if root.name == "profiles" and (root.parent / "processor_config.json").is_file():
        root = root.parent
    if root.is_dir() and not (root / "processor_config.json").is_file():
        raise FileNotFoundError(f"ActionCodec2 Hugging Face processor is missing: {root}")
    frontend = ActionCodec2.from_pretrained(
        str(root), variant=variant, revision=revision, local_files_only=root.is_dir(),
        variant_fingerprint=variant_fingerprint,
    )
    router = frontend.action_tokenizer
    router._catalog_origin = dict(frontend._variant_origin)
    return router


def _native_kernels():
    """This package's compiled kernels, or ``None`` under ``ACTIONCODEC2_NO_NATIVE=1``.

    Without them every stage runs as the artifact's bundled Python runtime,
    which produces the same tokens and decodes but far more slowly.
    """
    if os.environ.get("ACTIONCODEC2_NO_NATIVE"):
        return None
    try:
        from . import _actioncodec2_native as native
    except ImportError as error:
        raise ImportError(
            "the ActionCodec2 native kernels are not built; reinstall this package "
            "(uv pip install -e .) with a C++17 compiler available, or set "
            "ACTIONCODEC2_NO_NATIVE=1 to use the slower NumPy path"
        ) from error
    _configure_threads(native)
    return native




def _resampled_horizon(length: int, source_hz: float, target_hz: float) -> int:
    if length < 1 or source_hz <= 0 or target_hz <= 0:
        raise ValueError("horizon and control frequencies must be positive")
    return max(1, math.floor(length * target_hz / source_hz + 0.5))


def _token_ids(tokens, vocab_size: int, offset: int = 0) -> list[int] | None:
    checked = []
    for token in tokens:
        if type(token) is not int and (isinstance(token, bool) or not isinstance(token, Integral)):
            return None
        if not 0 <= token < vocab_size:
            return None
        checked.append(int(token) + offset)
    return checked


class ActionCodec2Grammar(ChunkGrammar):
    """Expose one profile's ``vocab_size`` rows from the router's global IDs."""

    def __init__(self, grammar, offset: int, vocab_size: int):
        self._grammar, self.offset, self.vocab_size = grammar, int(offset), int(vocab_size)

    def _global(self, tokens):
        return _token_ids(tokens, self.vocab_size, self.offset)

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray:
        checked = self._global(tokens)
        if checked is None:
            raise ValueError("tokens must be integers within the profile vocabulary")
        mask = np.asarray(self._grammar.next_token_mask(checked), dtype=bool)
        return mask[self.offset : self.offset + self.vocab_size].copy()

    def is_complete(self, tokens: Sequence[int]) -> bool:
        checked = self._global(tokens)
        return checked is not None and bool(self._grammar.is_complete(checked))


class ActionCodec2Tokenizer(ChunkTokenizer):
    name = "actioncodec2"
    is_exact = False


    cache_batch_size = 4096

    def __init__(
        self,
        processor,
        *,
        time_horizon: int,
        fps: float,
        action_space: str = "single_eef_delta",
        action_low=None,
        action_high=None,
        target_hz: float = 15.0,
        configured_action_dim: int | None = None,
        rotation_representation: str = "rotation_vector",
        rotation_multiplication: str = "right",
        motion_scale=1.0,
        gripper_threshold: float = 0.8,
        gripper_open_is_high: bool = True,
        interpolation: str = "cubic",
        artifact=None,
        variant=None,
        revision=None,
        previous_action_resampling: str = "strict",
        resampling_alignment: str = "legacy",
    ):
        self.fps = float(fps)
        self.target_hz = float(target_hz)
        self.source_action_space = str(action_space).lower()
        self.action_space = {
            "delta_eef": "single_eef_delta",
            "delta_joint": "single_joint7_delta",
        }.get(str(action_space).lower(), str(action_space).lower())
        if self.action_space not in {"single_eef_delta", "single_joint7_delta"}:
            raise ValueError("ActionCodec2 supports single_eef_delta or single_joint7_delta")
        self.previous_action_resampling = str(previous_action_resampling).lower()
        self.resampling_alignment = str(resampling_alignment).lower()
        if self.previous_action_resampling not in {"strict", "constant_velocity"}:
            raise ValueError("previous_action_resampling must be strict or constant_velocity")
        if self.resampling_alignment not in {"legacy", "interval_end"}:
            raise ValueError("resampling_alignment must be legacy or interval_end")
        if rotation_representation not in {"rotation_vector", "euler_xyz"}:
            raise ValueError("rotation_representation must be rotation_vector or euler_xyz")
        if rotation_multiplication not in {"left", "right"}:
            raise ValueError("rotation_multiplication must be left or right")
        self.rotation_representation = rotation_representation
        self.rotation_multiplication = rotation_multiplication
        motion_width = 6 if self.action_space == "single_eef_delta" else 7
        self.motion_scale = np.broadcast_to(
            np.asarray(motion_scale, np.float32), (motion_width,)
        ).copy()
        if not np.all(np.isfinite(self.motion_scale)) or np.any(self.motion_scale <= 0):
            raise ValueError("motion_scale must contain positive finite values")
        self.gripper_threshold = float(gripper_threshold)
        self.gripper_open_is_high = bool(gripper_open_is_high)
        self.interpolation = str(interpolation).lower()
        self.artifact = None if artifact is None else str(artifact)
        origin = getattr(processor, "_catalog_origin", {})
        self.variant = origin.get("name", variant)
        self.revision = origin.get("revision", revision)
        self.variant_fingerprint = origin.get("fingerprint")
        if self.action_space == "single_eef_delta":
            component = {
                "space": "eef",
                "position_dims": [0, 1, 2],
                "rotation_dims": [3, 4, 5],
                "rotation_format": "rotation_vector",
                "gripper_dim": 6,
            }
            action_dim = 7
        else:
            component = {"space": "joint", "joint_dims": list(range(7)), "gripper_dim": 7}
            action_dim = 8
        if configured_action_dim is not None and int(configured_action_dim) != action_dim:
            raise ValueError("action_dim does not match ActionCodec2 action space")
        component.update(
            gripper_threshold=self.gripper_threshold, gripper_open_is_high=self.gripper_open_is_high
        )
        preset = {
            "name": "opensource_actioncodec2_profile",
            "input_dim": action_dim,
            "input": "delta",
            "source_hz": self.fps,
            "target_hz": self.target_hz,
            "interpolation": interpolation,
            "components": {"arm": component},
        }
        self.preset_name = processor.register_action_space(preset).name
        self.processor = processor
        self.preset = processor.preset(self.preset_name)
        self.time_horizon = int(time_horizon)
        self.action_dim = int(self.preset.input_dim)
        components = tuple(self.preset.components)
        if len(components) != 1 or components[0].space not in {"joint", "eef"}:
            raise ValueError("ActionCodec2 requires one joint or EEF profile")
        self.profile_name = components[0].space
        self.component = components[0]
        profile = self.processor.profiles[self.profile_name]
        self.profile = profile
        self.profile_offset = int(profile.spec.token_offset)
        package = type(processor).__module__.rsplit(".routing", 1)[0]
        self._presets_module = importlib.import_module(f"{package}.presets")
        self._trajectory_module = importlib.import_module(f"{package}.trajectory")
        self._semantic_module = importlib.import_module(f"{package}.semantic")
        self._physical_module = importlib.import_module(f"{package}.physical_codec")
        self.vocab_size = int(profile.processor.vocab_size)


        self._native = _native_kernels()
        serializer = profile.processor.serializer
        self._setbpe = (
            self._native.SetBPE(
                list(serializer.bins),
                serializer.lookahead,
                [tuple(rule) for rule in serializer.spatial_merges],
                [tuple(rule) for rule in serializer.temporal_merges],
            )
            if self._native is not None and getattr(serializer, "scheme", None) == "set_bpe"
            else None
        )


        self.delta_native = any(
            c.input_order == 1 for c in profile.processor.config.physical.components.values()
        )
        self.requires_previous_action = bool(profile.processor.requires_previous_first_order)
        self.context_steps = int(self.requires_previous_action)
        self.requires_explicit_boundary = self.requires_previous_action
        if (
            self.requires_previous_action
            and self.fps != self.target_hz
            and self.previous_action_resampling == "strict"
        ):
            raise ValueError(
                "second-order ActionCodec2 needs matching rates or constant_velocity boundary resampling"
            )
        self.action_low = None if action_low is None else np.asarray(action_low, np.float32)
        self.action_high = None if action_high is None else np.asarray(action_high, np.float32)
        if (self.action_low is None or self.action_high is None) and (
            self.delta_native and self.requires_previous_action
        ):
            raise ValueError(
                "a stateful delta-native profile needs the dataset's action bounds for its "
                "boundary input; set tokenizer.action_stats or tokenizer.action_clip"
            )
        if self.action_low is None or self.action_high is None:


            self.action_low, self.action_high = self._profile_bounds()
        if self.action_low is not None and self.action_low.shape != (self.action_dim,):
            raise ValueError("action_low must match action_dim")
        if self.action_high is not None and self.action_high.shape != (self.action_dim,):
            raise ValueError("action_high must match action_dim")
        if self.delta_native:
            self._check_delta_native()
        internal_horizon = _resampled_horizon(self.time_horizon, self.fps, self.target_hz)
        self.internal_horizon = internal_horizon
        if self.resampling_alignment == "interval_end":
            if not self.requires_previous_action:
                raise ValueError("interval_end requires a second-order ActionCodec2 artifact")
            if not np.isclose(
                self.time_horizon / self.fps, internal_horizon / self.target_hz, rtol=0, atol=1e-9
            ):
                raise ValueError("interval_end requires an integral codec duration")
        self._source_preset = replace(self.preset, target_hz=self.fps)
        self._tokens_per_chunk = internal_horizon * sum(len(c.active_dims) for c in components)
        self._grammar = ActionCodec2Grammar(
            self.processor.grammar(preset=self.preset_name, horizon=internal_horizon),
            self.profile_offset,
            self.vocab_size,
        )

    def _check_delta_native(self):
        """A delta-native profile is used as fitted: no rescaling, resampling or re-framing."""
        if self.fps != self.target_hz:
            raise ValueError("this profile consumes the dataset's own rate; set target_hz to fps")
        if not np.all(self.motion_scale == 1.0):
            raise ValueError("this profile consumes actions as stored; motion_scale must be 1")
        if self.previous_action_resampling != "strict" or self.resampling_alignment != "legacy":
            raise ValueError("this profile has no rate conversion to resample a boundary for")
        for component in self.profile.processor.config.physical.components.values():
            if component.space != "so3":
                continue
            if (
                component.representation != self.rotation_representation
                or component.multiplication != self.rotation_multiplication
            ):
                raise ValueError(
                    "the dataset's rotation convention must match the profile's: set "
                    f"rotation_representation={component.representation!r} and "
                    f"rotation_multiplication={component.multiplication!r}"
                )

    def _profile_bounds(self):
        low = np.zeros(self.action_dim, dtype=np.float32)
        high = np.ones(self.action_dim, dtype=np.float32)
        profile_config = self.processor.profiles[self.profile_name].processor.config
        active = tuple(self.preset.components[0].active_dims)
        for physical in profile_config.physical.components.values():
            if physical.primitive_order == 0 or "order_1" not in physical.constraints:
                continue
            bound = physical.constraints["order_1"]
            minimum = np.broadcast_to(np.asarray(bound.minimum, np.float32), len(physical.dims))
            maximum = np.broadcast_to(np.asarray(bound.maximum, np.float32), len(physical.dims))
            for index, dimension in enumerate(physical.dims):
                if dimension in active:
                    target = active.index(dimension)
                    low[target], high[target] = (
                        minimum[index] / self.motion_scale[target],
                        maximum[index] / self.motion_scale[target],
                    )
        return low, high

    @classmethod
    def from_pretrained(
        cls, path, *, time_horizon, fps=20.0, action_space="single_eef_delta",
        variant=None, revision=None, variant_fingerprint=None, **kwargs
    ):
        return cls(
            _load_processor(path, variant=variant, revision=revision,
                            variant_fingerprint=variant_fingerprint),
            time_horizon=time_horizon,
            fps=fps,
            action_space=action_space,
            artifact=path,
            variant=variant,
            revision=revision,
            **kwargs,
        )

    @property
    def max_tokens_per_chunk(self):
        return self._tokens_per_chunk

    @property
    def grammar(self):
        return self._grammar

    def _canonical_input(self, values):
        result = np.asarray(values, np.float32).copy()
        squeeze = result.ndim == 2
        if squeeze:
            result = result[None]
        result[..., :-1] *= self.motion_scale
        if self.action_space == "single_eef_delta":
            if self.rotation_representation == "euler_xyz":
                flat = result[..., 3:6].reshape(-1, 3)
                result[..., 3:6] = (
                    Rotation.from_euler("xyz", flat).as_rotvec().reshape(result[..., 3:6].shape)
                )
            if self.rotation_multiplication == "left":
                result[..., 3:6] = self._left_to_right(result[..., 3:6])
        return result[0] if squeeze else result

    def _external_output(self, values):
        result = np.asarray(values, np.float32).copy()
        if self.action_space == "single_eef_delta":
            if self.rotation_multiplication == "left":
                result[..., 3:6] = self._right_to_left(result[..., 3:6])
            if self.rotation_representation == "euler_xyz":
                flat = result[..., 3:6].reshape(-1, 3)
                result[..., 3:6] = (
                    Rotation.from_rotvec(flat).as_euler("xyz").reshape(result[..., 3:6].shape)
                )
        result[..., :-1] /= self.motion_scale
        return result

    def _left_to_right(self, increments):
        """World-frame (left-multiplied) rotation increments as body-frame ones."""
        if self._native is not None:
            return self._native.rotvec_left_to_right_batch(increments)
        batch, steps = increments.shape[:2]
        current, output = Rotation.identity(batch), np.empty_like(increments)
        for step in range(steps):
            world = Rotation.from_rotvec(increments[:, step])
            following = world * current
            output[:, step] = (current.inv() * following).as_rotvec()
            current = following
        return output

    def _right_to_left(self, increments):
        """Body-frame (right-multiplied) rotation increments as world-frame ones."""
        if self._native is not None:
            return self._native.rotvec_right_to_left_batch(increments)
        batch, steps = increments.shape[:2]
        current, output = Rotation.identity(batch), np.empty_like(increments)
        for step in range(steps):
            body = Rotation.from_rotvec(increments[:, step])
            following = current * body
            output[:, step] = (following * current.inv()).as_rotvec()
            current = following
        return output

    def _boundary(self, previous_actions, count):
        if self.delta_native and previous_actions is not None and self.requires_previous_action:
            values = np.asarray(previous_actions, np.float32)
            if values.shape != (count, self.action_dim) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"previous_actions must be finite with shape [{count}, {self.action_dim}]"
                )
            return values[:, list(self.component.active_dims)]
        if previous_actions is None:
            if self.requires_previous_action:
                raise ValueError("ActionCodec2 second-order requires previous_actions")
            return None
        if not self.requires_previous_action:
            raise TypeError("first-order ActionCodec2 does not use previous_actions")
        values = np.asarray(previous_actions, np.float32)
        if values.shape != (count, self.action_dim) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"previous_actions must be finite with shape [{count}, {self.action_dim}]"
            )
        boundary = self._canonical_input(values[:, None])[:, 0]
        if self.previous_action_resampling == "constant_velocity":
            boundary[:, :-1] *= self.fps / self.target_hz
        return boundary[:, list(self.component.active_dims)]

    def _motion_width(self):
        return len(self.component.joint_dims or ()) if self.component.space == "joint" else 6

    def _native_canonicalize(self, values, *, source_hz, target_hz):
        """The runtime preset's ``canonicalize`` for this delta preset, over [B, T, D]."""
        component, native = self.component, self._native
        if component.space == "eef":
            pose = native.integrate_eef_deltas_batch(
                np.concatenate(
                    (
                        values[..., list(component.position_dims)],
                        values[..., list(component.rotation_dims)],
                    ),
                    axis=-1,
                )
            )
            pose = native.resample_uniform_batch(
                pose, source_hz, target_hz, "pose_xyzw", -1, self.interpolation
            )
            motion = native.eef_pose_absolute_batch(pose)
        else:
            source = np.cumsum(values[..., list(component.joint_dims)], axis=1, dtype=np.float32)
            motion = native.resample_uniform_batch(
                source, source_hz, target_hz, "linear", -1, self.interpolation
            )
        if component.gripper_dim is None:
            return motion
        gripper = native.resample_uniform_batch(
            values[..., component.gripper_dim : component.gripper_dim + 1],
            source_hz, target_hz, "hold", -1, self.interpolation,
        )
        threshold = component.gripper_threshold
        opened = gripper >= threshold if component.gripper_open_is_high else gripper <= threshold
        return np.concatenate((motion, opened.astype(np.float32)), axis=-1)

    def _native_resample_canonical(self, values, *, source_hz, target_hz, length):
        """The runtime's ``resample_canonical_trajectory`` over [B, T, D_active]."""
        component, native = self.component, self._native
        width = self._motion_width()
        motion = np.ascontiguousarray(values[..., :width])
        if component.space == "eef":
            pose = native.resample_uniform_batch(
                native.eef_pose_from_rotvec_batch(motion),
                source_hz, target_hz, "pose_xyzw", length, self.interpolation,
            )
            sampled = native.eef_pose_absolute_batch(pose)
        else:
            sampled = native.resample_uniform_batch(
                motion, source_hz, target_hz, "linear", length, self.interpolation
            )
        if component.gripper_dim is None:
            return sampled
        gripper = native.resample_uniform_batch(
            values[..., width : width + 1], source_hz, target_hz, "hold", length, self.interpolation
        )
        return np.concatenate((sampled, gripper), axis=-1)

    def _native_resample_intervals(self, values, *, source_hz, target_hz, length):
        """``_resample_action_intervals`` over [B, T, D_active]."""
        anchored = np.concatenate((np.zeros_like(values[:, :1]), values), axis=1)
        sampled = self._native_resample_canonical(
            anchored, source_hz=source_hz, target_hz=target_hz, length=length + 1
        )[:, 1:].copy()
        if self.component.gripper_dim is not None:
            sampled[..., -1:] = self._native.resample_uniform_batch(
                values[..., -1:], source_hz, target_hz, "hold", length, self.interpolation
            )
        return sampled

    def _native_externalize(self, values):
        """The runtime's ``externalize_canonical_trajectory`` without a state anchor."""
        component, native = self.component, self._native
        width = self._motion_width()
        motion = np.ascontiguousarray(values[..., :width])
        if component.space == "joint":
            previous = np.concatenate((np.zeros_like(motion[:, :1]), motion[:, :-1]), axis=1)
            output = motion - previous
        else:
            output = native.eef_pose_deltas_batch(native.eef_pose_from_rotvec_batch(motion))
        if component.gripper_dim is None:
            return np.asarray(output, dtype=np.float32)
        gripper = values[..., width : width + 1]
        if not component.gripper_open_is_high:
            gripper = 1.0 - gripper
        return np.asarray(np.concatenate((output, gripper), axis=-1), dtype=np.float32)

    def _native_decode_cells(self, rows, boundary):
        """The runtime processor's ``decode_batch`` with Set-BPE decoding in C++."""
        processor = self.profile.processor
        dims = tuple(self.component.active_dims)
        decoder = getattr(self._setbpe, "decode_checked_batch", self._setbpe.decode_batch)
        cells = decoder(rows, processor.action_dim, sum(1 << d for d in dims))
        previous = processor._dense_previous_first_order(boundary, dims, len(rows))
        decoded = processor.physical.decode_batch(
            processor._fill_inactive_cells(cells), previous_first_order=previous
        )
        return np.take(decoded, dims, axis=2)

    def _resample_action_intervals(self, trajectory, *, source_hz, target_hz, length):
        values = trajectory.values
        anchored = np.concatenate((np.zeros_like(values[:1]), values), axis=0)
        sampled = (
            self._presets_module.resample_canonical_trajectory(
                self._trajectory_module.ActiveTrajectory(anchored, trajectory.active_dims),
                self.component,
                source_hz=source_hz,
                target_hz=target_hz,
                target_length=length + 1,
                method=self.interpolation,
            )
            .values[1:]
            .copy()
        )
        if self.component.gripper_dim is not None:
            sampled[:, -1:] = self._semantic_module.resample_uniform(
                values[:, -1:],
                source_hz=source_hz,
                target_hz=target_hz,
                target_length=length,
                kind="hold",
                method=self.interpolation,
            )
        return self._trajectory_module.ActiveTrajectory(sampled, trajectory.active_dims)

    def canonical_targets(self, chunks):
        if self.delta_native:
            return np.array(self._check_chunks(chunks), np.float32)
        values = self._canonical_input(self._check_chunks(chunks))
        if self._native is not None:
            if self.resampling_alignment == "interval_end":
                source = self._native_canonicalize(values, source_hz=self.fps, target_hz=self.fps)
                active = self._native_resample_intervals(
                    source, source_hz=self.fps, target_hz=self.target_hz,
                    length=self.internal_horizon,
                )
            else:
                active = self._native_canonicalize(
                    values, source_hz=self.fps, target_hz=self.target_hz
                )
            dense = np.zeros(
                (len(active), self.internal_horizon, self.profile.processor.action_dim),
                dtype=np.float32,
            )
            dense[:, :, self.component.active_dims] = active
            return dense
        if self.resampling_alignment == "interval_end":
            source = self._source_preset.canonicalize(values)[self.component.name]
            trajectories = [
                self._resample_action_intervals(
                    row,
                    source_hz=self.fps,
                    target_hz=self.target_hz,
                    length=self.internal_horizon,
                )
                for row in source
            ]
        else:
            trajectories = self.preset.canonicalize(values, source_hz=self.fps)[self.component.name]
        dense = np.zeros(
            (len(trajectories), self.internal_horizon, self.profile.processor.action_dim),
            dtype=np.float32,
        )
        dense[:, :, self.component.active_dims] = np.stack(
            [trajectory.values for trajectory in trajectories]
        )
        return dense

    def _quantize_canonical(self, dense, previous_actions):
        boundary = self._boundary(previous_actions, len(dense))
        dense_boundary = None
        if boundary is not None:
            dense_boundary = np.zeros((len(dense), self.profile.processor.action_dim), np.float32)
            dense_boundary[:, self.component.active_dims] = boundary
        cells = self.profile.processor.quantize_batch(dense, previous_first_order=dense_boundary)
        return np.take(cells, self.component.active_dims, axis=2)

    def tokens_from_cells(self, cells):
        return self._streams_from_cells(np.asarray(cells)[None])[0]

    def _streams_from_cells(self, cells):
        """Token streams of compact [B, T, D_active] cell trajectories."""
        if self._setbpe is None:
            return [
                [
                    int(token)
                    for token in self.profile.processor.tokens_from_cells(
                        row, active_dims=self.component.active_dims
                    )
                ]
                for row in cells
            ]
        dense = np.full(
            (len(cells), cells.shape[1], self.profile.processor.action_dim), -1, dtype=np.int64
        )
        dense[:, :, list(self.component.active_dims)] = cells
        return self._setbpe.encode_batch(dense)

    def encode_batch(self, chunks, previous_actions=None):
        cells = self._quantize_canonical(self.canonical_targets(chunks), previous_actions)
        return self._streams_from_cells(cells)

    def decode_batch(self, streams, previous_actions=None):
        rows = list(streams)
        if self._setbpe is None or not hasattr(self._setbpe, "decode_checked_batch"):
            for index, stream in enumerate(rows):
                checked = _token_ids(stream, self.vocab_size)
                if checked is None:
                    raise ValueError("tokens must be integers within the profile vocabulary")
                rows[index] = checked
        boundary = self._boundary(previous_actions, len(rows))
        if self._setbpe is not None:
            decoded = self._native_decode_cells(rows, boundary)
        else:
            decoded = self.profile.processor.decode_batch(
                rows,
                active_dims=self.component.active_dims,
                previous_first_order=boundary,
            )
        if decoded.shape[1] != self.internal_horizon:
            raise ValueError("token stream does not match the configured chunk horizon")
        if self.delta_native:
            return np.asarray(decoded, np.float32)
        if self._native is not None:
            values = np.asarray(decoded, np.float32)
            if self.resampling_alignment == "interval_end":
                restored = self._native_resample_intervals(
                    values, source_hz=self.target_hz, target_hz=self.fps, length=self.time_horizon
                )
            else:
                restored = self._native_resample_canonical(
                    values, source_hz=self.target_hz, target_hz=self.fps, length=self.time_horizon
                )
            return self._external_output(self._native_externalize(restored))
        output = []
        for row in decoded:
            trajectory = self._trajectory_module.ActiveTrajectory(row, self.component.active_dims)
            if self.resampling_alignment == "interval_end":
                restored = self._resample_action_intervals(
                    trajectory,
                    source_hz=self.target_hz,
                    target_hz=self.fps,
                    length=self.time_horizon,
                )
            else:
                restored = self._presets_module.resample_canonical_trajectory(
                    trajectory,
                    self.component,
                    source_hz=self.target_hz,
                    target_hz=self.fps,
                    target_length=self.time_horizon,
                    method=self.interpolation,
                )
            output.append(
                self._presets_module.externalize_canonical_trajectory(
                    restored, self.component, current_state=None
                )
            )
        return self._external_output(np.stack(output))

    def validate_boundary_augmentation(self):
        if self.delta_native:
            raise ValueError("boundary augmentation is not implemented for delta-native profiles")
        if not self.requires_previous_action or self.resampling_alignment != "interval_end":
            raise ValueError("boundary augmentation requires a second-order interval_end profile")
        if any(
            c.input_order != 0 for c in self.profile.processor.config.physical.components.values()
        ):
            raise ValueError("boundary augmentation requires absolute canonical targets")

    def admissible_previous_actions(self, first_targets, previous_actions):
        """Which boundaries keep the first target inside the codec's order-2 limits."""
        try:
            return self._admissible_previous_actions(first_targets, previous_actions)
        except AttributeError as error:
            raise RuntimeError(
                "this ActionCodec2 artifact does not expose the internals boundary "
                "augmentation reads; use the revision pinned in the recipes"
            ) from error

    def _admissible_previous_actions(self, first_targets, previous_actions):
        boundary = self._boundary(previous_actions, len(previous_actions))
        dense_boundary = np.zeros_like(first_targets)
        dense_boundary[:, self.component.active_dims] = boundary
        valid = np.ones(len(previous_actions), dtype=bool)
        for component, runtime in self.profile.processor.physical._runtimes:
            if component.primitive_order != 2:
                continue
            dims = list(component.dims)
            scale = self._physical_module._scale(component)
            first = np.asarray(first_targets[:, dims], np.float64) * scale
            previous = np.asarray(dense_boundary[:, dims], np.float64) * scale
            if component.space == "so3":
                change = runtime._residual(
                    runtime._from_representation(first), runtime._from_representation(previous)
                )
                second = runtime._to_representation(change)
            else:
                second = first - previous
            bounds = component.constraints["order_2"]
            valid &= (
                (second >= np.asarray(bounds.minimum)) & (second <= np.asarray(bounds.maximum))
            ).all(axis=1)
        return valid

    def encode_augmented_batch(self, chunks, previous, steps, augmentation, rng):
        self.validate_boundary_augmentation()
        targets = self.canonical_targets(chunks)
        augmented, stats = augmentation.sample_validated(
            previous,
            steps,
            rng,
            validate=lambda indices, candidates: self.admissible_previous_actions(
                targets[indices, 0], candidates
            ),
            rotation_representation=self.rotation_representation,
            rotation_scale=self.motion_scale[3:6],
            rotation_multiplication=self.rotation_multiplication,
            action_space=("delta_eef" if self.profile_name == "eef" else "delta_joint"),
        )
        stats["clean_outside_order2"] = int(
            (~self.admissible_previous_actions(targets[:, 0], previous)).sum()
        )
        cells = self._quantize_canonical(targets, augmented)
        return self._streams_from_cells(cells), augmented, stats

    def encode(self, chunk, previous_action=None):
        previous = None if previous_action is None else np.asarray(previous_action)[None]
        return self.encode_batch(self._check_chunk(chunk)[None], previous)[0]

    def decode(self, tokens, previous_action=None):
        previous = None if previous_action is None else np.asarray(previous_action)[None]
        return self.decode_batch([tokens], previous)[0]

    def rollout_previous_actions(self, actions, *, execution_stride=1):
        values = np.asarray(actions, np.float32)
        previous, visited = np.zeros_like(values), np.zeros(len(values), dtype=bool)
        if not self.requires_previous_action:
            visited[:] = True
            return previous, visited
        if not len(values):
            return previous, visited
        stride, boundary = max(1, int(execution_stride)), values[0].copy()
        visited[0] = True
        for start in range(1, len(values), stride):
            visited[start], previous[start] = True, boundary
            chunk = self.chunk_at(values, start)
            stream = self.encode_batch(chunk[None], boundary[None])[0]
            decoded = self.decode_batch([stream], boundary[None])[0]
            boundary = decoded[min(stride, len(decoded)) - 1]
        return previous, visited

    def to_json(self):
        settings = {
            "name": self.name,
            "artifact": self.artifact,
            "time_horizon": self.time_horizon,
            "fps": self.fps,
            "target_hz": self.target_hz,
            "action_space": self.source_action_space,
            "action_dim": self.action_dim,
            "interpolation": self.interpolation,
            "rotation_representation": self.rotation_representation,
            "rotation_multiplication": self.rotation_multiplication,
            "motion_scale": self.motion_scale.tolist(),
            "gripper_threshold": self.gripper_threshold,
            "gripper_open_is_high": self.gripper_open_is_high,
        }
        if self.previous_action_resampling != "strict":
            settings["previous_action_resampling"] = self.previous_action_resampling
        if self.resampling_alignment != "legacy":
            settings["resampling_alignment"] = self.resampling_alignment
        if self.variant is not None:
            settings["variant"] = self.variant
        if self.variant_fingerprint:
            settings["variant_fingerprint"] = self.variant_fingerprint
        if self.revision:
            settings["revision"] = self.revision
        return settings
