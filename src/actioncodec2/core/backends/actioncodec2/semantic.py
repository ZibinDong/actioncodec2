"""Canonical physical preprocessing for routed ActionCodec2 profiles."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline

from ._so3 import (
    euler_xyz_to_quaternion as _euler_xyz_to_quaternion,
)
from ._so3 import (
    multiply_quaternions as _quaternion_multiply,
)
from ._so3 import (
    normalize_quaternion as _normalize_quaternion,
)
from ._so3 import (
    quaternion_to_rotvec as _quaternion_to_rotvec,
)
from ._so3 import (
    rotvec_to_quaternion as _rotvec_to_quaternion,
)
from .native_backend import native_kernel


def integrate_additive(values: np.ndarray, order: int) -> np.ndarray:
    """Integrate along the penultimate (time) axis from a zero boundary."""
    result = np.asarray(values, dtype=np.float32)
    kernel = native_kernel("additive_transform")
    if kernel is not None:
        shape = result.shape
        return kernel(result.reshape(-1, *shape[-2:]), order, True).reshape(shape)
    for _ in range(order):
        result = np.cumsum(result, axis=-2, dtype=np.float32)
    return result


def difference_additive(values: np.ndarray, order: int) -> np.ndarray:
    """Take zero-anchored finite differences along the time axis."""
    result = np.asarray(values, dtype=np.float32)
    kernel = native_kernel("additive_transform")
    if kernel is not None:
        shape = result.shape
        return kernel(result.reshape(-1, *shape[-2:]), order, False).reshape(shape)
    for _ in range(order):
        previous = np.concatenate((np.zeros_like(result[..., :1, :]), result[..., :-1, :]), axis=-2)
        result = result - previous
    return result


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Return angles in radians folded into ``(-pi, pi]``."""
    angles = np.asarray(values, dtype=np.float64)
    kernel = native_kernel("wrap_angles")
    if kernel is not None:
        return kernel(angles)
    return np.asarray(np.pi - np.mod(np.pi - angles, 2.0 * np.pi), dtype=np.float32)


_INTERPOLATION_METHODS = {"linear", "pchip", "cubic"}


def _validate_rate(value: float, name: str) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return rate


def _quaternion_slerp(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> np.ndarray:
    first = _normalize_quaternion(np.asarray(left, dtype=np.float64))
    second = _normalize_quaternion(np.asarray(right, dtype=np.float64))
    dot = np.sum(first * second, axis=1)
    flip = dot < 0.0
    second[flip] *= -1.0
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    output = np.empty_like(first)
    near = dot > 0.9995
    output[near] = first[near] + weight[near, None] * (second[near] - first[near])
    regular = ~near
    if np.any(regular):
        angle = np.arccos(dot[regular])
        denominator = np.sin(angle)
        w = weight[regular]
        output[regular] = (
            np.sin((1.0 - w) * angle)[:, None] / denominator[:, None] * first[regular]
            + np.sin(w * angle)[:, None] / denominator[:, None] * second[regular]
        )
    return _normalize_quaternion(output)


def resampled_horizon(length: int, *, source_hz: float, target_hz: float) -> int:
    """Return the target control-step count for a regular source horizon."""
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("a trajectory must contain at least one sample")
    source_rate = _validate_rate(source_hz, "source_hz")
    target_rate = _validate_rate(target_hz, "target_hz")
    scaled = int(math.floor(int(length) * target_rate / source_rate + 0.5))
    return max(1, scaled)


def _target_times(
    length: int,
    source_hz: float,
    target_hz: float,
    *,
    target_length: int | None,
) -> np.ndarray:
    count = (
        resampled_horizon(length, source_hz=source_hz, target_hz=target_hz)
        if target_length is None
        else target_length
    )
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("target_length must be a positive integer")
    return np.arange(int(count), dtype=np.float64) / target_hz


def _guarded_natural_cubic(
    source_time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    sampled = np.asarray(
        CubicSpline(source_time, values, axis=0, bc_type="natural")(target_time),
        dtype=np.float64,
    )
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)

    tolerance = 1e-9 + 0.05 * (upper - lower)
    fallback_axes = np.any(
        (sampled < lower[None] - tolerance[None]) | (sampled > upper[None] + tolerance[None]),
        axis=0,
    )
    if np.any(fallback_axes):
        sampled[:, fallback_axes] = PchipInterpolator(
            source_time, values[:, fallback_axes], axis=0
        )(target_time)
    return sampled


def resample_uniform(
    values: np.ndarray,
    *,
    source_hz: float,
    target_hz: float,
    kind: str = "linear",
    target_length: int | None = None,
    method: str = "linear",
) -> np.ndarray:
    """Resample one regular control-horizon trajectory onto a target clock."""
    source_rate = _validate_rate(source_hz, "source_hz")
    target_rate = _validate_rate(target_hz, "target_hz")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite [T, D] array")
    if kind not in {"linear", "hold", "pose_xyzw", "pose_euler_xyz"}:
        raise ValueError("unsupported resampling kind")
    if method not in _INTERPOLATION_METHODS:
        raise ValueError(f"method must be one of {sorted(_INTERPOLATION_METHODS)}")
    if kind == "pose_xyzw" and array.shape[1] != 7:
        raise ValueError("pose_xyzw values must have shape [T, 7]")
    if kind == "pose_euler_xyz" and array.shape[1] != 6:
        raise ValueError("pose_euler_xyz values must have shape [T, 6]")
    kernel = native_kernel("resample_uniform_batch")
    if kernel is not None:
        if target_length is not None and (
            isinstance(target_length, bool)
            or not isinstance(target_length, (int, np.integer))
            or target_length < 1
        ):
            raise ValueError("target_length must be a positive integer")
        if kind == "pose_euler_xyz":
            array = np.concatenate((array[:, :3], _euler_xyz_to_quaternion(array[:, 3:6])), axis=1)
            kind = "pose_xyzw"
        return kernel(
            array[None],
            source_rate,
            target_rate,
            kind,
            -1 if target_length is None else int(target_length),
            method,
        )[0]
    standard_length = resampled_horizon(len(array), source_hz=source_rate, target_hz=target_rate)
    if source_rate == target_rate and target_length in {None, standard_length}:
        if kind == "pose_euler_xyz":
            array = np.concatenate((array[:, :3], _euler_xyz_to_quaternion(array[:, 3:6])), axis=1)
        return np.asarray(array, dtype=np.float32)

    target_time = _target_times(len(array), source_rate, target_rate, target_length=target_length)
    source_time = np.arange(len(array), dtype=np.float64) / source_rate
    interpolation_time = np.minimum(target_time, source_time[-1])
    source_position = target_time * source_rate
    left = np.minimum(np.floor(source_position).astype(np.int64), len(array) - 1)
    right = np.minimum(left + 1, len(array) - 1)
    weight = source_position - left
    if kind == "hold":
        result = array[left]
    else:
        pose = kind in {"pose_xyzw", "pose_euler_xyz"}
        continuous = array[:, :3] if pose else array
        if len(array) == 1:
            sampled = np.repeat(continuous, len(target_time), axis=0)
        elif method == "pchip":
            sampled = PchipInterpolator(source_time, continuous, axis=0)(interpolation_time)
        elif method == "cubic":
            sampled = _guarded_natural_cubic(source_time, continuous, interpolation_time)
        else:
            sampled = array[left, : continuous.shape[1]] + weight[:, None] * (
                array[right, : continuous.shape[1]] - array[left, : continuous.shape[1]]
            )
        if not pose:
            result = sampled
        else:
            quaternion = (
                array[:, 3:7] if kind == "pose_xyzw" else _euler_xyz_to_quaternion(array[:, 3:6])
            )
            if len(array) == 1:
                rotation = np.repeat(quaternion, len(target_time), axis=0)
            elif method == "linear":
                rotation = _quaternion_slerp(quaternion[left], quaternion[right], weight)
            else:
                rotation = RotationSpline(source_time, Rotation.from_quat(quaternion))(
                    interpolation_time
                ).as_quat()
            result = np.concatenate((sampled, rotation), axis=1)
    return np.asarray(result, dtype=np.float32)


def eef_pose_absolute(poses: np.ndarray) -> np.ndarray:
    """Convert canonical ``xyz + xyzw`` state to ``xyz + rotation_vector``."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError("poses must be a finite [T, 7] array")
    kernel = native_kernel("eef_pose_absolute_batch")
    if kernel is not None and len(values):
        return kernel(values[None])[0]
    return np.asarray(
        np.concatenate((values[:, :3], _quaternion_to_rotvec(values[:, 3:7])), axis=1),
        dtype=np.float32,
    )


def eef_pose_components(poses: np.ndarray, representation: str) -> np.ndarray:
    """Convert canonical ``xyz + xyzw`` poses to one registered source layout."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError("poses must be a finite [T, 7] array")
    quaternion = _normalize_quaternion(values[:, 3:7])
    if representation == "rotation_vector":
        rotation = _quaternion_to_rotvec(quaternion)
    elif representation == "euler_xyz":
        rotation = Rotation.from_quat(quaternion).as_euler("xyz")
    elif representation == "quaternion_xyzw":
        rotation = quaternion
    else:
        raise ValueError("unsupported EEF rotation representation")
    return np.asarray(np.concatenate((values[:, :3], rotation), axis=1), dtype=np.float32)


def eef_pose_from_components(
    position: np.ndarray,
    rotation: np.ndarray,
    representation: str,
) -> np.ndarray:
    """Build canonical ``xyz + xyzw`` poses from an explicit source layout."""
    translation = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(rotation, dtype=np.float64)
    if translation.ndim != 2 or translation.shape[1] != 3 or len(translation) < 1:
        raise ValueError("position must have non-empty shape [T, 3]")
    if len(orientation) != len(translation) or not np.all(np.isfinite(translation)):
        raise ValueError("position and rotation must have matching finite trajectories")
    if representation == "rotation_vector":
        if orientation.ndim != 2 or orientation.shape[1] != 3:
            raise ValueError("rotation_vector must have shape [T, 3]")
        quaternion = _rotvec_to_quaternion(orientation)
    elif representation == "euler_xyz":
        if orientation.ndim != 2 or orientation.shape[1] != 3:
            raise ValueError("euler_xyz must have shape [T, 3]")
        quaternion = _euler_xyz_to_quaternion(orientation)
    elif representation == "quaternion_xyzw":
        if orientation.ndim != 2 or orientation.shape[1] != 4:
            raise ValueError("quaternion_xyzw must have shape [T, 4]")
        quaternion = _normalize_quaternion(orientation)
    else:
        raise ValueError("unsupported EEF rotation representation")
    if not np.all(np.isfinite(orientation)):
        raise ValueError("position and rotation must be finite")
    return np.asarray(np.concatenate((translation, quaternion), axis=1), dtype=np.float32)


def eef_pose_local_absolute(
    poses: np.ndarray, *, initial_pose: np.ndarray | None = None
) -> np.ndarray:
    """Express measured poses relative to an explicit or inferred initial pose."""
    values = np.asarray(poses, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 7
        or len(values) < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("poses must be a finite non-empty [T, 7] array")
    if initial_pose is None:
        initial = values[:1]
    else:
        initial = np.asarray(initial_pose, dtype=np.float64)
        if initial.shape == (7,):
            initial = initial[None]
        if initial.shape != (1, 7) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_pose must be a finite [7] pose")
    quaternion = _normalize_quaternion(values[:, 3:7])
    initial_quaternion = _normalize_quaternion(initial[:, 3:7])
    inverse_start = initial_quaternion.copy()
    inverse_start[:, :3] *= -1.0
    relative = _quaternion_multiply(np.broadcast_to(inverse_start, quaternion.shape), quaternion)
    return np.asarray(
        np.concatenate((values[:, :3] - initial[:, :3], _quaternion_to_rotvec(relative)), axis=1),
        dtype=np.float32,
    )


def eef_pose_deltas(poses: np.ndarray) -> np.ndarray:
    """Differentiate a local ``xyz + xyzw`` path from zero/identity."""
    values = np.asarray(poses, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 7
        or len(values) < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("poses must be a finite non-empty [T, 7] array")
    kernel = native_kernel("eef_pose_deltas_batch")
    if kernel is not None:
        return kernel(values[None])[0]
    quaternion = _normalize_quaternion(values[:, 3:7])
    previous_position = np.concatenate((np.zeros((1, 3)), values[:-1, :3]), axis=0)
    identity = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    previous_quaternion = np.concatenate((identity, quaternion[:-1]), axis=0)
    inverse_previous = previous_quaternion.copy()
    inverse_previous[:, :3] *= -1.0
    increment = _quaternion_multiply(inverse_previous, quaternion)
    return np.asarray(
        np.concatenate(
            (values[:, :3] - previous_position, _quaternion_to_rotvec(increment)), axis=1
        ),
        dtype=np.float32,
    )


def eef_pose_from_local_absolute(poses: np.ndarray, *, initial_pose: np.ndarray) -> np.ndarray:
    """Attach an external initial pose to a local ``xyz + xyzw`` path."""
    values = np.asarray(poses, dtype=np.float64)
    initial = np.asarray(initial_pose, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 7
        or len(values) < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("poses must be a finite non-empty [T, 7] array")
    if initial.shape == (7,):
        initial = initial[None]
    if initial.shape != (1, 7) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_pose must be a finite [7] pose")
    local_quaternion = _normalize_quaternion(values[:, 3:7])
    initial_quaternion = _normalize_quaternion(initial[:, 3:7])
    quaternion = _quaternion_multiply(
        np.broadcast_to(initial_quaternion, local_quaternion.shape), local_quaternion
    )
    return np.asarray(
        np.concatenate((values[:, :3] + initial[:, :3], quaternion), axis=1),
        dtype=np.float32,
    )


def integrate_eef_deltas(deltas: np.ndarray) -> np.ndarray:
    """Integrate ``[dx,dy,dz,body-rotvec]`` from a zero/identity pose."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("deltas must be a finite [T, 6] array")
    if len(values) < 1:
        raise ValueError("deltas must contain at least one step")
    kernel = native_kernel("integrate_eef_deltas_batch")
    if kernel is not None:
        return kernel(values[None])[0]
    translation = np.cumsum(values[:, :3], axis=0)
    increments = _rotvec_to_quaternion(values[:, 3:6])
    orientation = np.empty((len(values), 4), dtype=np.float64)
    current = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    for index, increment in enumerate(increments):
        current = _quaternion_multiply(current, increment[None])
        orientation[index] = current[0]
    return np.asarray(np.concatenate((translation, orientation), axis=1), dtype=np.float32)


def normalize_gripper_actions(
    values: np.ndarray,
    *,
    closed_value: float,
    open_value: float,
) -> np.ndarray:
    """Map one raw controller gripper channel to the canonical open fraction."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 1 or not np.all(np.isfinite(array)):
        raise ValueError("gripper action must be a finite [T, 1] array")
    if not (math.isfinite(closed_value) and math.isfinite(open_value)):
        raise ValueError("gripper anchors must be finite")
    if closed_value == open_value:
        raise ValueError("gripper closed/open anchors must differ")
    kernel = native_kernel("gripper_transform")
    if kernel is not None:
        return kernel(array, "normalize", closed_value, open_value)
    fraction = (array - closed_value) / (open_value - closed_value)
    return np.clip(fraction, 0.0, 1.0).astype(np.float32)


def discretize_gripper_actions(
    values: np.ndarray,
    *,
    mode: str,
    threshold: float = 0.8,
) -> np.ndarray:
    """Quantize the canonical open fraction for the codec."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 1 or not np.all(np.isfinite(array)):
        raise ValueError("gripper fraction must be a finite [T, 1] array")
    if mode == "binary":
        if not math.isfinite(threshold):
            raise ValueError("gripper threshold must be finite")
        kernel = native_kernel("gripper_transform")
        if kernel is not None:
            return kernel(array, mode, threshold, 0)
        return (array >= threshold).astype(np.float32)
    if mode == "eleven":
        kernel = native_kernel("gripper_transform")
        if kernel is not None:
            return kernel(array, mode, threshold, 0)
        return (np.rint(array * 10.0) / 10.0).astype(np.float32)
    if mode == "twenty":
        return (np.rint(array * 20.0) / 20.0).astype(np.float32)
    raise ValueError("gripper mode must be 'binary', 'eleven' or 'twenty'")
