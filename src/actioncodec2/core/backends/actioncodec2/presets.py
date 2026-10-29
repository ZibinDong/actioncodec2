"""User-facing action layouts compiled into shared joint/EEF trajectories."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ._validation import optional_finite_array, validate_dimension_partition
from .batch_semantics import (
    canonicalize_batch,
    canonicalize_component_batch,
    externalize_canonical_batch,
    resample_canonical_batch,
)
from .configuration import (
    ConfigError,
    _integer,
    _mapping,
    _number,
    _reject_unknown,
    _required,
    _UniqueKeyLoader,
)
from .semantic import wrap_to_pi as wrap_to_pi
from .trajectory import ActiveTrajectory

_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SPACES = {"eef", "joint"}
_INPUTS = {"absolute", "delta"}
_ROTATIONS = {"rotation_vector", "euler_xyz", "quaternion_xyzw"}
_JOINT_WRAPS = {"pi", "none"}
_INTERPOLATIONS = {"linear", "pchip", "cubic"}
_GRIPPER_DIM = {"eef": 6, "joint": 7}


def canonical_preset_name(name: str) -> str:
    """Preserve the historical six-coordinate alias without implying a gripper slot."""
    return "single_joint6_abs_no_gripper" if name == "single_joint6_abs_continuous_gripper" else name


def _format_dims(dims: Sequence[int]) -> str:
    values = tuple(dims)
    if len(values) == 1:
        return str(values[0])
    if values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}..{values[-1]}"
    return ",".join(map(str, values))


def _dims(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(f"{path} must be a list of dimensions")
    output = tuple(_integer(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not output:
        raise ConfigError(f"{path} must not be empty")
    if len(set(output)) != len(output):
        raise ConfigError(f"{path} contains a repeated dimension")
    return output


@dataclass(frozen=True)
class ComponentSpec:
    """One independently encoded arm/body component in an input tensor."""

    name: str
    space: str
    joint_dims: tuple[int, ...] | None = None
    joint_wrap: str | None = None
    position_dims: tuple[int, ...] | None = None
    rotation_dims: tuple[int, ...] | None = None
    rotation_format: str = "rotation_vector"
    gripper_dim: int | None = None
    gripper_closed_value: float = 0.0
    gripper_open_value: float = 1.0
    gripper_threshold: float | None = None
    gripper_open_is_high: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.gripper_open_is_high, bool):
            raise ConfigError("gripper_open_is_high must be boolean")
        if self.gripper_threshold is not None:
            _number(self.gripper_threshold, "gripper_threshold")
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ConfigError(f"component name must match {_NAME.pattern!r}")
        if self.space not in _SPACES:
            raise ConfigError(f"component {self.name!r}.space must be joint or eef")
        if self.rotation_format not in _ROTATIONS:
            raise ConfigError(
                f"component {self.name!r}.rotation_format must be one of {sorted(_ROTATIONS)}"
            )
        for label, dims in (
            ("joint_dims", self.joint_dims),
            ("position_dims", self.position_dims),
            ("rotation_dims", self.rotation_dims),
        ):
            if dims is not None:
                _dims(dims, f"component {self.name}.{label}")
        self._validate_motion_dims()
        if self.space == "joint":
            if self.joint_wrap is None:
                object.__setattr__(self, "joint_wrap", "pi")
            if self.joint_wrap not in _JOINT_WRAPS:
                raise ConfigError(
                    f"component {self.name!r}.joint_wrap must be one of {sorted(_JOINT_WRAPS)}"
                )
        elif self.joint_wrap is not None:
            raise ConfigError(
                f"component {self.name!r}.joint_wrap applies only to joint components"
            )
        raw_dims = self.source_dims
        if self.gripper_dim is not None:
            _integer(self.gripper_dim, f"component {self.name}.gripper_dim")
            if self.gripper_dim in raw_dims:
                raise ConfigError(f"component {self.name!r}.gripper_dim overlaps motion dims")
            for anchor in (self.gripper_closed_value, self.gripper_open_value):
                _number(anchor, f"component {self.name}.gripper anchor")
            if self.gripper_closed_value == self.gripper_open_value:
                raise ConfigError(
                    f"component {self.name!r}.gripper_closed_value and gripper_open_value "
                    "must differ"
                )

    def _validate_motion_dims(self) -> None:
        if self.space == "joint":
            if self.joint_dims is None or not 1 <= len(self.joint_dims) <= 7:
                raise ConfigError(f"joint component {self.name!r} requires 1..7 joint_dims")
            if self.position_dims is not None or self.rotation_dims is not None:
                raise ConfigError(f"joint component {self.name!r} cannot define EEF dims")
            return
        if self.joint_dims is not None:
            raise ConfigError(f"EEF component {self.name!r} cannot define joint_dims")
        if self.position_dims is None or len(self.position_dims) != 3:
            raise ConfigError(f"EEF component {self.name!r} requires three position_dims")
        expected = 4 if self.rotation_format == "quaternion_xyzw" else 3
        if self.rotation_dims is None or len(self.rotation_dims) != expected:
            raise ConfigError(
                f"EEF component {self.name!r} requires {expected} rotation_dims for "
                f"{self.rotation_format}"
            )
        if set(self.position_dims).intersection(self.rotation_dims):
            raise ConfigError(f"EEF component {self.name!r} reuses a pose dimension")

    @property
    def source_dims(self) -> tuple[int, ...]:
        """All source motion columns in canonical component order."""
        if self.space == "joint":
            assert self.joint_dims is not None
            return self.joint_dims
        assert self.position_dims is not None and self.rotation_dims is not None
        return (*self.position_dims, *self.rotation_dims)

    @property
    def all_source_dims(self) -> tuple[int, ...]:
        """Motion plus optional binary-gripper source columns."""
        return (
            self.source_dims if self.gripper_dim is None else (*self.source_dims, self.gripper_dim)
        )

    @property
    def active_dims(self) -> tuple[int, ...]:
        """Shared-profile slots emitted by this component."""
        motion_width = len(self.joint_dims or ()) if self.space == "joint" else 6
        dims = tuple(range(motion_width))
        return dims if self.gripper_dim is None else (*dims, _GRIPPER_DIM[self.space])

    @classmethod
    def from_dict(cls, name: str, payload: object, *, path: str) -> ComponentSpec:
        values = _mapping(payload, path)
        allowed = {
            "space",
            "joint_dims",
            "joint_wrap",
            "position_dims",
            "rotation_dims",
            "rotation_format",
            "gripper_dim",
            "gripper_closed_value",
            "gripper_open_value",
            "gripper_threshold",
            "gripper_open_is_high",
        }
        _reject_unknown(values, allowed, path)
        gripper_dim = values.get("gripper_dim")
        if gripper_dim is not None:
            gripper_dim = _integer(gripper_dim, f"{path}.gripper_dim")
        return cls(
            name=name,
            space=str(_required(values, "space", path)),
            joint_dims=None
            if "joint_dims" not in values
            else _dims(values["joint_dims"], f"{path}.joint_dims"),
            joint_wrap=None if values.get("joint_wrap") is None else str(values["joint_wrap"]),
            position_dims=None
            if "position_dims" not in values
            else _dims(values["position_dims"], f"{path}.position_dims"),
            rotation_dims=None
            if "rotation_dims" not in values
            else _dims(values["rotation_dims"], f"{path}.rotation_dims"),
            rotation_format=str(values.get("rotation_format", "rotation_vector")),
            gripper_dim=gripper_dim,
            gripper_threshold=(values.get("gripper_threshold") if values.get("gripper_threshold") is not None
                               else (0.8 if values.get("gripper_open_is_high", True) else 0.2)
                               if "gripper_open_is_high" in values or "gripper_threshold" in values else None),
            gripper_open_is_high=values.get("gripper_open_is_high", True),
            gripper_closed_value=_number(
                values.get("gripper_closed_value", 0.0), f"{path}.gripper_closed_value"
            ),
            gripper_open_value=_number(
                values.get("gripper_open_value", 1.0), f"{path}.gripper_open_value"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"space": self.space}
        if self.gripper_dim is not None and self.gripper_threshold is not None:
            payload.update(gripper_threshold=self.gripper_threshold,
                           gripper_open_is_high=self.gripper_open_is_high)
        if self.space == "joint":
            payload["joint_dims"] = list(self.joint_dims or ())
            payload["joint_wrap"] = self.joint_wrap
        else:
            payload.update(
                {
                    "position_dims": list(self.position_dims or ()),
                    "rotation_dims": list(self.rotation_dims or ()),
                    "rotation_format": self.rotation_format,
                }
            )
        if self.gripper_dim is not None:
            payload.update(
                {
                    "gripper_dim": self.gripper_dim,
                    "gripper_closed_value": self.gripper_closed_value,
                    "gripper_open_value": self.gripper_open_value,
                }
            )
        return payload


@dataclass(frozen=True)
class PresetConfig:
    """A dataset- and embodiment-independent physical ``[B,T,D]`` contract."""

    name: str
    input_dim: int
    input: str
    components: tuple[ComponentSpec, ...]
    source_hz: float = 15.0
    target_hz: float = 15.0
    interpolation: str = "cubic"
    token_order: str = "time_major"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonical_preset_name(self.name))
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ConfigError(f"preset.name must match {_NAME.pattern!r}")
        _integer(self.input_dim, "preset.input_dim", positive=True)
        if self.input not in _INPUTS:
            raise ConfigError("preset.input must be absolute or delta")
        if not self.components:
            raise ConfigError("preset.components must not be empty")
        if not all(isinstance(component, ComponentSpec) for component in self.components):
            raise ConfigError("preset.components must contain ComponentSpec values")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ConfigError("preset.components contains a duplicate name")
        self._validate_source_partition()
        for rate, path in ((self.source_hz, "source_hz"), (self.target_hz, "target_hz")):
            checked = _number(rate, f"preset.{path}")
            if checked <= 0.0:
                raise ConfigError(f"preset.{path} must be positive and finite")
        if self.interpolation not in _INTERPOLATIONS:
            raise ConfigError(f"preset.interpolation must be one of {sorted(_INTERPOLATIONS)}")
        if self.token_order not in {"time_major", "component_major"}:
            raise ConfigError("preset.token_order must be time_major or component_major")
        if self.input == "delta" and any(
            component.space == "eef" and component.rotation_format != "rotation_vector"
            for component in self.components
        ):
            raise ConfigError("EEF delta components require rotation_format=rotation_vector")

    def _validate_source_partition(self) -> None:
        validate_dimension_partition(
            ((component.name, component.all_source_dims) for component in self.components),
            self.input_dim,
            label="input dim",
            error_type=ConfigError,
        )

    @classmethod
    def from_dict(cls, payload: object, *, path: str = "preset") -> PresetConfig:
        values = _mapping(payload, path)
        allowed = {
            "name",
            "input_dim",
            "input",
            "source_hz",
            "target_hz",
            "interpolation",
            "components",
            "token_order",
        }
        _reject_unknown(values, allowed, path)
        raw_components = _mapping(_required(values, "components", path), f"{path}.components")
        components = tuple(
            ComponentSpec.from_dict(name, raw, path=f"{path}.components.{name}")
            for name, raw in raw_components.items()
        )
        return cls(
            name=str(_required(values, "name", path)),
            input_dim=_integer(
                _required(values, "input_dim", path), f"{path}.input_dim", positive=True
            ),
            input=str(_required(values, "input", path)),
            components=components,
            source_hz=_number(values.get("source_hz", 15.0), f"{path}.source_hz"),
            target_hz=_number(values.get("target_hz", 15.0), f"{path}.target_hz"),
            interpolation=str(values.get("interpolation", "cubic")),
            token_order=str(values.get("token_order", "time_major")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PresetConfig:
        """Load one strict user preset without executing arbitrary code."""
        source = Path(path)
        try:
            payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except yaml.YAMLError as error:
            raise ConfigError(f"cannot parse {source}: {error}") from error
        return cls.from_dict(payload, path=str(source))

    def save_yaml(self, path: str | Path) -> Path:
        """Save the complete reproducible preset and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return target

    def to_dict(self) -> dict[str, object]:
        payload = {
            "name": self.name,
            "input_dim": self.input_dim,
            "input": self.input,
            "source_hz": self.source_hz,
            "target_hz": self.target_hz,
            "interpolation": self.interpolation,
            "token_order": self.token_order,
            "components": {component.name: component.to_dict() for component in self.components},
        }
        return payload

    def summary(self) -> str:
        """Return a compact action-space label for discovery UIs."""
        spaces = "+".join(dict.fromkeys(component.space for component in self.components))
        return (
            f"{self.name} (D={self.input_dim}, {self.input}, {spaces}, {self.source_hz:g} Hz input)"
        )

    def describe(self) -> str:
        """Describe every source tensor column and its physical interpretation."""
        motion = "position" if self.input == "absolute" else "per-step delta"
        lines = [self.summary()]
        for component in self.components:
            if component.space == "joint":
                assert component.joint_dims is not None
                lines.append(
                    f"  dims {_format_dims(component.joint_dims)}: {component.name} "
                    f"generalized-joint {motion} [rad or registered canonical unit]"
                )
                if self.input == "absolute":
                    lines.append(
                        "    angles in (-pi, pi]; decode returns that interval"
                        if component.joint_wrap == "pi"
                        else "    continuous angles, may exceed one turn; decode never wraps"
                    )
            else:
                assert component.position_dims is not None
                assert component.rotation_dims is not None
                lines.append(
                    f"  dims {_format_dims(component.position_dims)}: {component.name} "
                    f"EEF translation {motion} (x, y, z in listed order) [m]"
                )
                rotation_motion = "orientation" if self.input == "absolute" else "increment"
                rotation_unit = (
                    "unit xyzw quaternion"
                    if component.rotation_format == "quaternion_xyzw"
                    else "rad"
                )
                lines.append(
                    f"  dims {_format_dims(component.rotation_dims)}: {component.name} EEF "
                    f"{rotation_motion} [{component.rotation_format}, {rotation_unit}]"
                )
                if self.input == "delta":
                    lines.append(
                        "    rotation_vector = axis * angle [rad], right-multiplicative "
                        "body-frame increment: R_next = R_previous @ Exp(rotvec)"
                    )
                elif component.rotation_format == "rotation_vector":
                    lines.append("    rotation_vector = axis * angle [rad], not Euler angles")
                elif component.rotation_format == "euler_xyz":
                    lines.append("    euler_xyz = extrinsic x, y, z Euler angles [rad]")
            if component.gripper_dim is not None:
                if component.gripper_threshold is not None:
                    comparison = ">=" if component.gripper_open_is_high else "<="
                    semantics = f"binary; open when raw {comparison} {component.gripper_threshold:g}"
                else:
                    semantics = (f"raw {component.gripper_closed_value:g} = closed, "
                                 f"raw {component.gripper_open_value:g} = open; "
                                 "levels are determined by the loaded profile")
                lines.append(f"  dim {component.gripper_dim}: {component.name} gripper command ({semantics})")
            else:
                lines.append(
                    f"  {component.name}: no gripper channel; all listed columns are motion"
                )
        if self.input == "absolute":
            lines.append(
                "  encode/decode require the same current_state (B,D), immediately before row 0"
            )
        else:
            lines.append(
                "  encode/decode: omit current_state; deltas are per-step increments, not velocities"
            )
        lines.append(
            f"  resampling: {self.source_hz:g} Hz default -> {self.target_hz:g} Hz codec "
            f"clock ({self.interpolation})"
        )
        return "\n".join(lines)

    def canonicalize(
        self,
        values: np.ndarray,
        *,
        source_hz: float | None = None,
        current_state: np.ndarray | None = None,
    ) -> dict[str, list[ActiveTrajectory]]:
        """Convert a batch into local paths for the two shared profiles."""
        batches = self._canonicalize_batch(values, source_hz=source_hz, current_state=current_state)
        return {
            component.name: [
                ActiveTrajectory(row, component.active_dims) for row in batches[component.name]
            ]
            for component in self.components
        }

    def _canonicalize_batch(
        self,
        values: np.ndarray,
        *,
        source_hz: float | None = None,
        current_state: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 3 or len(array) < 1 or array.shape[1] < 1:
            raise ValueError("preset input must have non-empty shape [B, T, D]")
        if array.shape[2] != self.input_dim:
            raise ValueError(
                f"preset {self.name!r} expects input_dim={self.input_dim}, got {array.shape[2]}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("preset input contains a non-finite value")
        state_array = optional_finite_array(current_state, shape=(len(array), self.input_dim))
        source_hz = self.source_hz if source_hz is None else source_hz
        return canonicalize_batch(self, array, source_hz=source_hz, current_state=state_array)

    def canonicalize_episode(
        self,
        values: np.ndarray,
        *,
        source_hz: float | None = None,
        current_state: np.ndarray | None = None,
    ) -> dict[str, ActiveTrajectory]:
        """Canonicalize one variable-length episode and optional ``(D,)`` anchor."""
        episode = np.asarray(values, dtype=np.float32)
        if episode.ndim != 2 or len(episode) < 1 or episode.shape[1] != self.input_dim:
            raise ValueError(
                f"preset {self.name!r} expects a non-empty [T, {self.input_dim}] episode"
            )
        state = optional_finite_array(current_state, shape=(self.input_dim,))
        canonical = self._canonicalize_batch(
            episode[None],
            source_hz=source_hz,
            current_state=None if state is None else state[None],
        )
        return {
            component.name: ActiveTrajectory(canonical[component.name][0], component.active_dims)
            for component in self.components
        }

    def profile_trajectories(
        self, values: np.ndarray, *, current_state: np.ndarray | None = None
    ) -> dict[str, list[ActiveTrajectory]]:
        """Canonicalize one episode and group its components by joint/EEF profile."""
        canonical = self.canonicalize_episode(values, current_state=current_state)
        output: dict[str, list[ActiveTrajectory]] = {}
        for component in self.components:
            output.setdefault(component.space, []).append(canonical[component.name])
        return output

    def _canonicalize_component(
        self,
        episode: np.ndarray,
        component: ComponentSpec,
        *,
        source_hz: float,
        current_state: np.ndarray | None,
    ) -> ActiveTrajectory:
        values = canonicalize_component_batch(
            self,
            np.asarray(episode, dtype=np.float32)[None],
            component,
            source_hz=source_hz,
            current_state=None
            if current_state is None
            else np.asarray(current_state, dtype=np.float32)[None],
        )
        return ActiveTrajectory(values[0], component.active_dims)


def format_action_spaces(
    presets: Mapping[str, PresetConfig],
    *,
    selected: str | None = None,
    action_dim: int | None = None,
    second_order_boundaries: Mapping[str, Mapping[str, int]] | None = None,
    codec_name: str = "ActionCodec2",
    chunk_horizon_decode: bool = False,
) -> str:
    """Explain registered tensor layouts without inferring semantics from shape."""
    if action_dim is not None:
        action_dim = _integer(action_dim, "action_dim", positive=True)
    if not presets:
        return "No action spaces are registered."
    candidates = sorted(
        (
            preset
            for preset in presets.values()
            if action_dim is None or preset.input_dim == action_dim
        ),
        key=lambda preset: (preset.input_dim, preset.name),
    )
    lines = [
        f"{codec_name} — action layout guide",
        f"Selected action_space: {selected or 'none (choose a layout before encoding)'}",
        "",
        "Input: NumPy arrays or Torch tensors, (T,D) or (B,T,D).",
        "  T = timesteps, D = columns per action, B = batch size; column indices start at 0.",
        "  Any positive T is accepted. A batch needs equal T; encode unequal-length episodes separately.",
        "  decode returns (B,T,D), including B=1 for a single episode; values are quantized, not lossless.",
        "",
        "How to choose:",
        "  joint = robot joint coordinates; EEF = end-effector (tool) position and orientation.",
        "  absolute = target coordinates; delta = per-step motion increments, not velocities.",
        "  Match column order, units, rotation convention, gripper commands and controller fps.",
        "  Shape alone cannot identify physical meaning: layouts sharing D are only candidates.",
        "  Consult your dataset/controller documentation if these meanings are unknown.",
        "  Supply physical units, not normalized [-1,1] features; convert degrees to radians.",
        "  Non-radian joint coordinates need an explicit mapping to the registered canonical units.",
        "  Pass the actual sampling rate as fps=...; defaults and codec rates appear below.",
        (
            "  Rows are independent chunks: decode(rows, horizon=T, fps=...) needs the chunk T."
            if chunk_horizon_decode
            else "  Resampling can change decoded T; token coverage determines the returned length."
        ),
        "",
        f"Registered configurations: {len(candidates)} shown / {len(presets)} total"
        + ("" if action_dim is None else f" (D={action_dim} candidates, not automatic selection)"),
    ]
    if not candidates:
        widths = ", ".join(map(str, sorted({preset.input_dim for preset in presets.values()})))
        lines.extend(
            [
                f"  No registered action space matches D={action_dim}. Available D: {widths}.",
                "  Confirm your column meanings and register a custom layout; do not pad/truncate to force a match.",
            ]
        )
    for preset in candidates:
        name = preset.name
        marker = " [selected]" if name == selected else ""
        description = preset.describe().splitlines()
        lines.extend(["", f"- {description[0]}{marker}"])
        lines.extend(f"  {line.strip()}" for line in description[1:])
        if second_order_boundaries is not None:
            boundaries = [
                f"{component!r}: (B,{width})"
                for component, width in second_order_boundaries.get(name, {}).items()
            ]
            lines.append(
                "  previous_first_order: "
                + (
                    "required in encode/decode, {" + ", ".join(boundaries) + "}; "
                    f"reconstructed increments at {preset.target_hz:g} Hz, B=1 for a single episode"
                    if boundaries
                    else "not required by this artifact for this layout"
                )
            )
    lines.extend(
        [
            "",
            "Next steps:",
            f"  Load:   {codec_name}.from_pretrained(path, action_space='NAME')",
            "  Switch: codec = codec.for_action_space('NAME')",
            "  Filter: codec.print_action_spaces(action_dim=D)",
            "  Config: codec.action_spaces['NAME'].to_dict()  # exact settings for a listed layout",
            "  Custom: codec.register_action_space(preset)  # explicit column definitions, then switch",
            "  Custom layouts can reorder columns; each joint component supports 1..7 coordinates.",
            "  EEF absolute rotations support rotation_vector, euler_xyz or quaternion_xyzw (unit x,y,z,w).",
            "  EEF delta rotations require rotation_vector with the convention shown above.",
            "  For absolute layouts, current_state is (B,D), even when actions are (T,D).",
        ]
    )
    return "\n".join(lines)


def resample_canonical_trajectory(
    trajectory: ActiveTrajectory,
    component: ComponentSpec,
    *,
    source_hz: float,
    target_hz: float,
    target_length: int | None = None,
    method: str = "linear",
) -> ActiveTrajectory:
    """Rate-convert one decoded canonical component without changing slots."""
    sampled = resample_canonical_batch(
        np.asarray(trajectory.values, dtype=np.float32)[None],
        component,
        source_hz=source_hz,
        target_hz=target_hz,
        target_length=target_length,
        method=method,
    )
    return ActiveTrajectory(sampled[0], trajectory.active_dims)


def externalize_canonical_trajectory(
    trajectory: ActiveTrajectory,
    component: ComponentSpec,
    *,
    current_state: np.ndarray | None,
) -> np.ndarray:
    """Return deltas, or absolute values when an external anchor is supplied."""
    return externalize_canonical_batch(
        np.asarray(trajectory.values, dtype=np.float32)[None],
        component,
        current_state=None
        if current_state is None
        else np.asarray(current_state, dtype=np.float32)[None],
    )[0]
