"""Validated ``config.yaml`` model contract for ActionCodec2."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from ._validation import validate_dimension_partition

CONFIG_NAME = "config.yaml"

MODEL_TYPE = "actioncodec2_unified"
CONFIG_VERSION = 1
_SPACES = ("additive", "so3")
_SO3_REPRESENTATIONS = ("rotation_vector", "euler_xyz")
_SO3_MULTIPLICATIONS = ("left", "right")
_PRIMITIVE_SELECTIONS = ("nearest", "smooth", "momentum_lazy")
_ORDER_KEY = re.compile(r"order_(0|[1-9][0-9]*)\Z")

NumberOrVector = float | tuple[float, ...]


class ConfigError(ValueError):
    """The serialized model definition is incomplete or inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in output
        except TypeError as error:
            raise ConfigError("config.yaml contains an invalid mapping key") from error
        if duplicate:
            raise ConfigError(f"config.yaml contains duplicate field {key!r}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"every field in {path} must be a string")
    return value


def _reject_unknown(payload: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        rendered = ", ".join(unknown)
        raise ConfigError(f"{path} contains unknown field(s): {rendered}")


def _required(payload: Mapping[str, object], key: str, path: str) -> object:
    if key not in payload:
        raise ConfigError(f"{path} is missing required field {key!r}")
    return payload[key]


def _integer(value: object, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ConfigError(f"{path} must be {qualifier}")
    return int(value)


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    return result


def _number_or_vector(value: object, width: int, path: str) -> NumberOrVector:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))
        if len(result) != width:
            raise ConfigError(f"{path} must be a scalar or contain {width} values")
        return result
    return _number(value, path)


def _expanded(value: NumberOrVector, width: int) -> tuple[float, ...]:
    if isinstance(value, tuple):
        return value
    return (value,) * width


def _serialize_number_or_vector(value: NumberOrVector) -> float | list[float]:
    return list(value) if isinstance(value, tuple) else value


def _parse_dims(value: object, input_dim: int, path: str) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        payload = _mapping(value, path)
        _reject_unknown(payload, {"start", "stop"}, path)
        start = _integer(_required(payload, "start", path), f"{path}.start")
        stop = _integer(_required(payload, "stop", path), f"{path}.stop")
        if stop <= start:
            raise ConfigError(f"{path}.stop must be greater than {path}.start")
        dims = tuple(range(start, stop))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        dims = tuple(_integer(item, f"{path}[{index}]") for index, item in enumerate(value))
        if not dims:
            raise ConfigError(f"{path} must not be empty")
    else:
        raise ConfigError(f"{path} must be a list or a start/stop range")

    if len(set(dims)) != len(dims):
        raise ConfigError(f"{path} contains a repeated dimension")
    outside = [dimension for dimension in dims if dimension >= input_dim]
    if outside:
        raise ConfigError(f"{path} contains dimension {outside[0]}, outside input_dim={input_dim}")
    return dims


def _serialize_dims(dims: tuple[int, ...]) -> object:
    if len(dims) > 1 and dims == tuple(range(dims[0], dims[-1] + 1)):
        return {"start": dims[0], "stop": dims[-1] + 1}
    return list(dims)


@dataclass(frozen=True)
class ConstraintConfig:
    """Allowed inclusive interval for one differential order."""

    minimum: NumberOrVector
    maximum: NumberOrVector

    @classmethod
    def from_dict(cls, payload: object, *, width: int, path: str) -> ConstraintConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"min", "max"}, path)
        minimum = _number_or_vector(_required(values, "min", path), width, f"{path}.min")
        maximum = _number_or_vector(_required(values, "max", path), width, f"{path}.max")
        if any(
            lower > upper
            for lower, upper in zip(
                _expanded(minimum, width), _expanded(maximum, width), strict=True
            )
        ):
            raise ConfigError(f"{path}.min must not exceed {path}.max")
        return cls(minimum=minimum, maximum=maximum)

    def to_dict(self) -> dict[str, object]:
        return {
            "min": _serialize_number_or_vector(self.minimum),
            "max": _serialize_number_or_vector(self.maximum),
        }


@dataclass(frozen=True)
class InputTransformConfig:
    """Input-unit conversion and optional source-side clipping."""

    scale: NumberOrVector = 1.0
    clip: bool = False

    @classmethod
    def from_dict(cls, payload: object | None, *, width: int, path: str) -> InputTransformConfig:
        if payload is None:
            return cls()
        values = _mapping(payload, path)
        _reject_unknown(values, {"scale", "clip"}, path)
        scale = _number_or_vector(values.get("scale", 1.0), width, f"{path}.scale")
        if any(value <= 0.0 for value in _expanded(scale, width)):
            raise ConfigError(f"{path}.scale must be positive")
        clip = values.get("clip", False)
        if not isinstance(clip, bool):
            raise ConfigError(f"{path}.clip must be true or false")
        return cls(scale=scale, clip=clip)

    def to_dict(self) -> dict[str, object]:
        return {
            "scale": _serialize_number_or_vector(self.scale),
            "clip": self.clip,
        }


def _parse_component_geometry(
    values: Mapping[str, object], *, input_dim: int, path: str
) -> tuple[tuple[int, ...], str, int, int, str | None, str | None]:
    dims = _parse_dims(_required(values, "dims", path), input_dim, f"{path}.dims")
    space = _required(values, "space", path)
    if space not in _SPACES:
        raise ConfigError(f"{path}.space must be one of {_SPACES}")
    input_order = _integer(_required(values, "input_order", path), f"{path}.input_order")
    primitive_order = _integer(
        _required(values, "primitive_order", path), f"{path}.primitive_order"
    )
    representation = values.get("representation")
    multiplication = values.get("multiplication")
    if space == "so3":
        if len(dims) != 3:
            raise ConfigError(f"{path} with space='so3' must contain exactly 3 dims")
        if input_order not in {0, 1} or primitive_order not in {1, 2}:
            raise ConfigError(
                f"{path} with space='so3' supports input_order 0 or 1 and primitive_order 1 or 2"
            )
        if representation not in _SO3_REPRESENTATIONS:
            raise ConfigError(f"{path}.representation must be one of {_SO3_REPRESENTATIONS}")
        if multiplication not in _SO3_MULTIPLICATIONS:
            raise ConfigError(f"{path}.multiplication must be one of {_SO3_MULTIPLICATIONS}")
    else:
        if input_order not in {0, 1, 2} or primitive_order not in {0, 1, 2}:
            raise ConfigError(
                f"{path} with space='additive' supports input_order and primitive_order 0, 1, or 2"
            )
        if representation is not None or multiplication is not None:
            raise ConfigError(
                f"{path}.representation and multiplication are only valid for space='so3'"
            )
    return dims, str(space), input_order, primitive_order, representation, multiplication


def _parse_component_constraints(
    values: Mapping[str, object], *, width: int, path: str
) -> dict[str, ConstraintConfig]:
    raw = _mapping(_required(values, "constraints", path), f"{path}.constraints")
    if not raw:
        raise ConfigError(f"{path}.constraints must not be empty")
    constraints: dict[str, ConstraintConfig] = {}
    for key, constraint in raw.items():
        if _ORDER_KEY.fullmatch(key) is None:
            raise ConfigError(
                f"{path}.constraints field {key!r} must be named order_0, order_1, ..."
            )
        constraints[key] = ConstraintConfig.from_dict(
            constraint, width=width, path=f"{path}.constraints.{key}"
        )
    return constraints


@dataclass(frozen=True)
class ComponentConfig:
    """One orthogonal physical component of the action vector."""

    dims: tuple[int, ...]
    space: str
    input_order: int
    primitive_order: int
    input_transform: InputTransformConfig
    constraints: Mapping[str, ConstraintConfig]
    distortion_limit: NumberOrVector
    primitive_selection: str
    representation: str | None = None
    multiplication: str | None = None

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        input_dim: int,
        path: str,
    ) -> ComponentConfig:
        values = _mapping(payload, path)
        allowed = {
            "dims",
            "space",
            "input_order",
            "primitive_order",
            "input_transform",
            "constraints",
            "distortion_limit",
            "primitive_selection",
            "representation",
            "multiplication",
        }
        _reject_unknown(values, allowed, path)
        dims, space, input_order, primitive_order, representation, multiplication = (
            _parse_component_geometry(values, input_dim=input_dim, path=path)
        )
        width = len(dims)
        constraints = _parse_component_constraints(values, width=width, path=path)

        input_transform = InputTransformConfig.from_dict(
            values.get("input_transform"),
            width=width,
            path=f"{path}.input_transform",
        )
        if input_transform.clip and f"order_{input_order}" not in constraints:
            raise ConfigError(
                f"{path}.input_transform.clip requires constraints.order_{input_order}"
            )

        distortion_limit = _number_or_vector(
            _required(values, "distortion_limit", path),
            width,
            f"{path}.distortion_limit",
        )
        if any(value <= 0.0 for value in _expanded(distortion_limit, width)):
            raise ConfigError(f"{path}.distortion_limit must be positive")
        if space == "so3" and isinstance(distortion_limit, tuple):
            raise ConfigError(
                f"{path}.distortion_limit must be one scalar geodesic angle for SO(3)"
            )

        primitive_selection = _required(values, "primitive_selection", path)
        if primitive_selection not in _PRIMITIVE_SELECTIONS:
            raise ConfigError(
                f"{path}.primitive_selection must be one of {', '.join(_PRIMITIVE_SELECTIONS)}"
            )
        if primitive_selection == "momentum_lazy" and (space != "so3" or primitive_order != 1):
            raise ConfigError(
                f"{path}.primitive_selection='momentum_lazy' requires first-order SO(3) primitives"
            )
        if (
            primitive_selection == "smooth"
            and space == "additive"
            and primitive_order not in {1, 2}
        ):
            raise ConfigError(
                f"{path}.primitive_selection='smooth' requires first- or second-order additive primitives"
            )
        if primitive_order == 2 and "order_2" not in constraints:
            raise ConfigError(f"{path} second-order primitives require constraints.order_2")
        return cls(
            dims=dims,
            space=space,
            input_order=input_order,
            primitive_order=primitive_order,
            input_transform=input_transform,
            constraints=MappingProxyType(constraints),
            distortion_limit=distortion_limit,
            primitive_selection=str(primitive_selection),
            representation=None if representation is None else str(representation),
            multiplication=None if multiplication is None else str(multiplication),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dims": _serialize_dims(self.dims),
            "space": self.space,
            "input_order": self.input_order,
            "primitive_order": self.primitive_order,
            "input_transform": self.input_transform.to_dict(),
            "constraints": {
                key: constraint.to_dict() for key, constraint in self.constraints.items()
            },
            "distortion_limit": _serialize_number_or_vector(self.distortion_limit),
            "primitive_selection": self.primitive_selection,
        }
        if self.space == "so3":
            payload["representation"] = self.representation
            payload["multiplication"] = self.multiplication
        return payload


@dataclass(frozen=True)
class PhysicalConfig:
    """Complete, non-overlapping component partition of the input's last axis."""

    input_dim: int
    components: Mapping[str, ComponentConfig]

    @classmethod
    def from_dict(cls, payload: object, *, path: str = "physical") -> PhysicalConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"input_dim", "components"}, path)
        input_dim = _integer(
            _required(values, "input_dim", path), f"{path}.input_dim", positive=True
        )
        raw_components = _mapping(_required(values, "components", path), f"{path}.components")
        if not raw_components:
            raise ConfigError(f"{path}.components must not be empty")

        components: dict[str, ComponentConfig] = {}
        for name, component_payload in raw_components.items():
            if not name.strip():
                raise ConfigError(f"{path}.components contains an empty component name")
            component = ComponentConfig.from_dict(
                component_payload,
                input_dim=input_dim,
                path=f"{path}.components.{name}",
            )
            components[name] = component

        validate_dimension_partition(
            ((name, component.dims) for name, component in components.items()),
            input_dim,
            label="physical dimension",
            error_type=ConfigError,
        )
        return cls(input_dim=input_dim, components=MappingProxyType(components))

    def to_dict(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "components": {
                name: component.to_dict() for name, component in self.components.items()
            },
        }


TOKENIZATION_SCHEMES = ("set_bpe", "unified_set_bpe", "bpe", "sparse_set_bpe")


@dataclass(frozen=True)
class SetBPEConfig:
    """Runtime settings of the tokenization stage."""

    lookahead: int
    allow_cross_component_merge: bool = False
    scheme: str = "set_bpe"

    @classmethod
    def from_dict(cls, payload: object, *, path: str = "set_bpe") -> SetBPEConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"lookahead", "allow_cross_component_merge", "scheme"}, path)
        lookahead = _integer(
            _required(values, "lookahead", path), f"{path}.lookahead", positive=True
        )
        allow_cross = values.get("allow_cross_component_merge", False)
        if not isinstance(allow_cross, bool):
            raise ConfigError(f"{path}.allow_cross_component_merge must be true or false")
        scheme = values.get("scheme", "set_bpe")
        if scheme not in TOKENIZATION_SCHEMES:
            raise ConfigError(f"{path}.scheme must be one of {', '.join(TOKENIZATION_SCHEMES)}")
        return cls(lookahead=lookahead, allow_cross_component_merge=allow_cross, scheme=scheme)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "lookahead": self.lookahead,
            "allow_cross_component_merge": self.allow_cross_component_merge,
        }

        if self.scheme != "set_bpe":
            payload["scheme"] = self.scheme
        return payload


@dataclass(frozen=True)
class PrimitiveTrainingConfig:
    """Serialized config-v1 metadata retained until artifacts are versioned again."""

    zero_primitive_components: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        physical: PhysicalConfig,
        path: str = "train.primitives",
    ) -> PrimitiveTrainingConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"zero_primitive_components"}, path)
        raw_names = _required(values, "zero_primitive_components", path)
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes, bytearray)):
            raise ConfigError(f"{path}.zero_primitive_components must be a list")
        names = tuple(raw_names)
        if not all(isinstance(name, str) and name for name in names):
            raise ConfigError(
                f"{path}.zero_primitive_components must contain non-empty component names"
            )
        if len(set(names)) != len(names):
            raise ConfigError(f"{path}.zero_primitive_components contains a duplicate")
        unknown = sorted(set(names) - set(physical.components))
        if unknown:
            raise ConfigError(
                f"{path}.zero_primitive_components contains unknown component {unknown[0]!r}"
            )

        selected = set(names)
        for name, component in physical.components.items():
            if component.space == "additive":
                if component.primitive_order not in {0, 1, 2}:
                    raise ConfigError(
                        f"{path} cannot train additive primitive_order="
                        f"{component.primitive_order} for component {name!r}"
                    )
                domain_order = component.primitive_order
                if name in selected and component.primitive_order == 0:
                    raise ConfigError(
                        f"{path}.zero_primitive_components cannot contain absolute "
                        f"component {name!r}"
                    )
            else:
                domain_order = component.primitive_order
            if f"order_{domain_order}" not in component.constraints:
                raise ConfigError(
                    f"physical component {name!r} needs constraints.order_{domain_order} "
                    "to train its primitives"
                )
        return cls(zero_primitive_components=names)

    def to_dict(self) -> dict[str, object]:
        return {"zero_primitive_components": list(self.zero_primitive_components)}


@dataclass(frozen=True)
class BPETrainingConfig:
    """Spatial and final vocabulary budgets used for whole-trajectory BPE."""

    spatial_vocab_size: int
    vocab_size: int

    @classmethod
    def from_dict(cls, payload: object, *, path: str = "train.bpe") -> BPETrainingConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"spatial_vocab_size", "vocab_size"}, path)
        spatial_vocab_size = _integer(
            _required(values, "spatial_vocab_size", path),
            f"{path}.spatial_vocab_size",
            positive=True,
        )
        vocab_size = _integer(
            _required(values, "vocab_size", path), f"{path}.vocab_size", positive=True
        )
        if spatial_vocab_size > vocab_size:
            raise ConfigError(f"{path}.spatial_vocab_size must not exceed vocab_size")
        return cls(spatial_vocab_size=spatial_vocab_size, vocab_size=vocab_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "spatial_vocab_size": self.spatial_vocab_size,
            "vocab_size": self.vocab_size,
        }


@dataclass(frozen=True)
class QualityControlConfig:
    """Pretraining-only policy for extreme first-order discontinuities."""

    hard_order_1_scale: float
    min_segment_length: int

    @classmethod
    def from_dict(
        cls, payload: object, *, path: str = "train.quality_control"
    ) -> QualityControlConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"hard_order_1_scale", "min_segment_length"}, path)
        scale = _number(
            _required(values, "hard_order_1_scale", path),
            f"{path}.hard_order_1_scale",
        )
        if scale <= 1.0:
            raise ConfigError(f"{path}.hard_order_1_scale must be greater than 1")
        return cls(
            hard_order_1_scale=scale,
            min_segment_length=_integer(
                _required(values, "min_segment_length", path),
                f"{path}.min_segment_length",
                positive=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "hard_order_1_scale": self.hard_order_1_scale,
            "min_segment_length": self.min_segment_length,
        }


@dataclass(frozen=True)
class SparseTrainingConfig:
    """ActionCodec2Sparse training-corpus construction."""

    chunk_horizon: int
    static_run_cap: int

    @classmethod
    def from_dict(cls, payload: object, *, path: str = "train.sparse") -> SparseTrainingConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"chunk_horizon", "static_run_cap"}, path)
        return cls(
            chunk_horizon=_integer(
                _required(values, "chunk_horizon", path), f"{path}.chunk_horizon", positive=True
            ),
            static_run_cap=_integer(
                _required(values, "static_run_cap", path), f"{path}.static_run_cap", positive=True
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {"chunk_horizon": self.chunk_horizon, "static_run_cap": self.static_run_cap}


@dataclass(frozen=True)
class TrainConfig:
    """Non-derivable fit choices and BPE budgets."""

    primitives: PrimitiveTrainingConfig
    bpe: BPETrainingConfig
    quality_control: QualityControlConfig | None = None
    sparse: SparseTrainingConfig | None = None

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        physical: PhysicalConfig,
        path: str = "train",
    ) -> TrainConfig:
        values = _mapping(payload, path)
        _reject_unknown(values, {"primitives", "bpe", "quality_control", "sparse"}, path)
        raw_quality_control = values.get("quality_control")
        raw_sparse = values.get("sparse")
        return cls(
            primitives=PrimitiveTrainingConfig.from_dict(
                _required(values, "primitives", path), physical=physical
            ),
            bpe=BPETrainingConfig.from_dict(_required(values, "bpe", path)),
            quality_control=(
                None
                if raw_quality_control is None
                else QualityControlConfig.from_dict(raw_quality_control)
            ),
            sparse=None if raw_sparse is None else SparseTrainingConfig.from_dict(raw_sparse),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {"primitives": self.primitives.to_dict(), "bpe": self.bpe.to_dict()}
        if self.quality_control is not None:
            payload["quality_control"] = self.quality_control.to_dict()
        if self.sparse is not None:
            payload["sparse"] = self.sparse.to_dict()
        return payload


@dataclass(frozen=True)
class ActionCodec2Config:
    """Strict YAML configuration for an ActionCodec2 processor artifact."""

    physical: PhysicalConfig
    set_bpe: SetBPEConfig
    train: TrainConfig | None = None
    model_type: str = MODEL_TYPE
    config_version: int = CONFIG_VERSION

    @classmethod
    def from_dict(cls, payload: object) -> ActionCodec2Config:
        values = _mapping(payload, "config")
        _reject_unknown(
            values,
            {"model_type", "config_version", "physical", "set_bpe", "train"},
            "config",
        )
        model_type = _required(values, "model_type", "config")
        if model_type != MODEL_TYPE:
            raise ConfigError(f"config.model_type must be {MODEL_TYPE!r}")
        config_version = _integer(
            _required(values, "config_version", "config"), "config.config_version"
        )
        if config_version != CONFIG_VERSION:
            raise ConfigError(
                f"unsupported config_version {config_version}; expected {CONFIG_VERSION}"
            )
        physical = PhysicalConfig.from_dict(_required(values, "physical", "config"))
        raw_train = values.get("train")
        set_bpe = SetBPEConfig.from_dict(_required(values, "set_bpe", "config"))
        train = None if raw_train is None else TrainConfig.from_dict(raw_train, physical=physical)
        sparse = set_bpe.scheme == "sparse_set_bpe"
        if train is not None and (train.sparse is not None) != sparse:
            raise ConfigError(
                "train.sparse is required for, and only valid with, set_bpe.scheme=sparse_set_bpe"
            )
        return cls(
            physical=physical,
            set_bpe=set_bpe,
            train=train,
            model_type=str(model_type),
            config_version=config_version,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ActionCodec2Config:
        source = Path(path)
        try:
            payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except OSError:
            raise
        except yaml.YAMLError as error:
            raise ConfigError(f"cannot parse {source}: {error}") from error
        return cls.from_dict(payload)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> ActionCodec2Config:
        """Load the strict numerical configuration from ``config.yaml``."""
        return cls.from_yaml(Path(directory) / CONFIG_NAME)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_type": self.model_type,
            "config_version": self.config_version,
            "physical": self.physical.to_dict(),
            "set_bpe": self.set_bpe.to_dict(),
        }
        if self.train is not None:
            payload["train"] = self.train.to_dict()
        return payload

    def save_yaml(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(
                self.to_dict(),
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

    def save_pretrained(self, directory: str | Path) -> None:
        """Write ``config.yaml`` without embedding primitive or BPE arrays."""
        self.save_yaml(Path(directory) / CONFIG_NAME)
