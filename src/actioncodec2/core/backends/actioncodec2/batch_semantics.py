"""Canonical adapters shared by episode fitting and batched inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._so3 import (
    conjugate_quaternion,
    multiply_quaternions,
    normalize_quaternion,
    quaternion_to_rotvec,
)
from .native_backend import native_kernel
from .semantic import (
    difference_additive,
    eef_pose_absolute,
    eef_pose_components,
    eef_pose_deltas,
    eef_pose_from_components,
    integrate_additive,
    integrate_eef_deltas,
    normalize_gripper_actions,
    resample_uniform,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from .presets import ComponentSpec, PresetConfig


def _resample(values, source_hz, target_hz, method, *, kind="linear", target_length=None):
    source_hz, target_hz = float(source_hz), float(target_hz)
    if source_hz == target_hz and target_length in (None, values.shape[1]):
        return values
    kernel = native_kernel("resample_uniform_batch")
    if kernel is not None:
        return kernel(
            values,
            source_hz,
            target_hz,
            kind,
            -1 if target_length is None else target_length,
            method,
        )
    return np.stack(
        [
            resample_uniform(
                row,
                source_hz=source_hz,
                target_hz=target_hz,
                kind=kind,
                target_length=target_length,
                method=method,
            )
            for row in values
        ]
    )


def _pose_map(values, kernel_name, reference):
    kernel = native_kernel(kernel_name)
    return kernel(values) if kernel is not None else np.stack([reference(row) for row in values])


def _rotvec_poses(values):
    kernel = native_kernel("eef_pose_from_rotvec_batch")
    if kernel is not None:
        return kernel(values)
    flat = values.reshape(-1, 6)
    return eef_pose_from_components(flat[:, :3], flat[:, 3:], "rotation_vector").reshape(
        *values.shape[:2], 7
    )


def _poses(values: np.ndarray, component: ComponentSpec) -> np.ndarray:
    position = np.take(values, component.position_dims, axis=-1)
    rotation = np.take(values, component.rotation_dims, axis=-1)
    if component.rotation_format == "rotation_vector":
        return _rotvec_poses(np.concatenate((position, rotation), axis=-1))
    return eef_pose_from_components(
        position.reshape(-1, 3), rotation.reshape(-1, rotation.shape[-1]), component.rotation_format
    ).reshape(*values.shape[:2], 7)


def _local(poses, anchor):
    quaternion = normalize_quaternion(poses[..., 3:7])
    inverse = conjugate_quaternion(normalize_quaternion(anchor[..., 3:7]))
    relative = multiply_quaternions(inverse, quaternion)
    return np.concatenate(
        (poses[..., :3].astype(np.float64) - anchor[..., :3], quaternion_to_rotvec(relative)),
        axis=-1,
    ).astype(np.float32)


def _joint(preset, values, component, source_hz, current_state):
    source = np.take(values, component.joint_dims, axis=-1)
    if preset.input == "delta":
        source = integrate_additive(source, 1)
        anchor = np.zeros((len(source), 1, source.shape[-1]), np.float32)
    else:
        anchor = (
            source[:, :1]
            if current_state is None
            else np.take(current_state, component.joint_dims, axis=-1)[:, None]
        )
        joined = np.concatenate((anchor, source), axis=1)
        if component.joint_wrap == "pi":
            outside = np.abs(joined) > np.pi + 1e-5
            if outside.any():
                row = int(np.flatnonzero(outside.any(axis=(1, 2)))[0])
                column = component.joint_dims[int(np.argmax(outside[row].any(axis=0)))]
                raise ValueError(
                    f"action space {preset.name!r} declares joint_wrap='pi' for component "
                    f"{component.name!r}, so its absolute joint columns must lie in (-pi, pi]; "
                    f"source column {column} reaches {joined[row][outside[row]][0]:.6g} rad. Wrap the "
                    "inputs, or register a layout with joint_wrap='none' for coordinates that "
                    "are continuous past one turn, such as a multi-turn or prismatic axis."
                )
        batch, steps, width = joined.shape
        kernel = native_kernel("unwrap_angles")
        unwrapped = (
            (
                kernel(joined.transpose(1, 0, 2).reshape(steps, batch * width), True)
                .reshape(steps, batch, width)
                .transpose(1, 0, 2)
            )
            if kernel is not None
            else np.unwrap(joined, axis=1)
        )
        anchor, source = unwrapped[:, :1], unwrapped[:, 1:]
    return np.asarray(
        _resample(source, source_hz, preset.target_hz, preset.interpolation) - anchor,
        dtype=np.float32,
    )


def _eef(preset, values, component, source_hz, current_state):
    if preset.input == "delta":
        delta = np.concatenate(
            (
                np.take(values, component.position_dims, axis=-1),
                np.take(values, component.rotation_dims, axis=-1),
            ),
            axis=-1,
        )
        poses = _pose_map(delta, "integrate_eef_deltas_batch", integrate_eef_deltas)
        sampled = _resample(
            poses, source_hz, preset.target_hz, preset.interpolation, kind="pose_xyzw"
        )
        return _pose_map(sampled, "eef_pose_absolute_batch", eef_pose_absolute)
    poses = _poses(values, component)
    anchor = poses[:, :1] if current_state is None else _poses(current_state[:, None], component)
    sampled = _resample(poses, source_hz, preset.target_hz, preset.interpolation, kind="pose_xyzw")
    return _local(sampled, anchor)


def canonicalize_batch(preset: PresetConfig, values, *, source_hz, current_state):
    """Return component ``(B,T_codec,D_active)`` arrays from validated input."""
    return {
        component.name: canonicalize_component_batch(
            preset, values, component, source_hz=source_hz, current_state=current_state
        )
        for component in preset.components
    }


def canonicalize_component_batch(preset, values, component, *, source_hz, current_state):
    """Convert one component of validated ``(B,T,D)`` source actions."""
    motion = (_joint if component.space == "joint" else _eef)(
        preset, values, component, source_hz, current_state
    )
    if component.gripper_dim is None:
        return motion
    gripper = values[..., component.gripper_dim : component.gripper_dim + 1]
    if component.gripper_threshold is not None:
        gripper = ((gripper >= component.gripper_threshold) if component.gripper_open_is_high
                   else (gripper <= component.gripper_threshold)).astype(np.float32)
    else:
        gripper = normalize_gripper_actions(
            gripper.reshape(-1, 1),
            closed_value=component.gripper_closed_value,
            open_value=component.gripper_open_value,
        ).reshape(*values.shape[:2], 1)
    gripper = _resample(gripper, source_hz, preset.target_hz, preset.interpolation, kind="hold")
    return np.concatenate((motion, gripper), axis=-1)


def _external_eef(motion, component, current_state):
    poses = _rotvec_poses(motion)
    if current_state is None:
        return _pose_map(poses, "eef_pose_deltas_batch", eef_pose_deltas)
    anchor = _poses(current_state[:, None], component)
    quaternion = multiply_quaternions(
        normalize_quaternion(anchor[..., 3:7]), normalize_quaternion(poses[..., 3:7])
    )
    attached = np.concatenate(
        (poses[..., :3].astype(np.float64) + anchor[..., :3], quaternion), axis=-1
    ).astype(np.float32)
    return eef_pose_components(attached.reshape(-1, 7), component.rotation_format).reshape(
        *attached.shape[:2], 3 + len(component.rotation_dims)
    )


def externalize_batch(
    values, component: ComponentSpec, *, source_hz, target_hz, target_length, method, current_state
):
    """Resample decoded paths and return ``(B,T_output,D_component)`` arrays."""
    sampled = resample_canonical_batch(
        values,
        component,
        source_hz=source_hz,
        target_hz=target_hz,
        target_length=target_length,
        method=method,
    )
    return externalize_canonical_batch(sampled, component, current_state=current_state)


def resample_canonical_batch(
    values, component: ComponentSpec, *, source_hz, target_hz, target_length, method
):
    """Rate-convert local ``(B,T,D_active)`` paths, holding gripper commands."""
    motion_width = len(component.joint_dims or ()) if component.space == "joint" else 6
    motion = values[..., :motion_width]
    if component.space == "eef":
        poses = _rotvec_poses(motion)
        poses = _resample(
            poses, source_hz, target_hz, method, kind="pose_xyzw", target_length=target_length
        )
        motion = _pose_map(poses, "eef_pose_absolute_batch", eef_pose_absolute)
    else:
        motion = _resample(motion, source_hz, target_hz, method, target_length=target_length)
    if component.gripper_dim is None:
        return np.asarray(motion, dtype=np.float32)
    gripper = _resample(
        values[..., motion_width : motion_width + 1],
        source_hz,
        target_hz,
        method,
        kind="hold",
        target_length=target_length,
    )
    return np.asarray(np.concatenate((motion, gripper), axis=-1), dtype=np.float32)


def externalize_canonical_batch(values, component: ComponentSpec, *, current_state):
    """Restore source coordinates or deltas from ``(B,T,D_active)`` local paths."""
    motion_width = len(component.joint_dims or ()) if component.space == "joint" else 6
    motion = values[..., :motion_width]
    if component.space == "eef":
        motion = _external_eef(motion, component, current_state)
    elif current_state is None:
        motion = difference_additive(motion, 1)
    else:
        motion = motion + np.take(current_state, component.joint_dims, axis=-1)[:, None]
        if component.joint_wrap == "pi":
            motion = wrap_to_pi(motion)
    if component.gripper_dim is None:
        return np.asarray(motion, dtype=np.float32)
    gripper = values[..., motion_width : motion_width + 1]
    if component.gripper_threshold is not None:
        gripper = gripper if component.gripper_open_is_high else 1.0 - gripper
        return np.asarray(np.concatenate((motion, gripper), axis=-1), dtype=np.float32)
    gripper = component.gripper_closed_value + gripper * (
        component.gripper_open_value - component.gripper_closed_value
    )
    return np.asarray(np.concatenate((motion, gripper), axis=-1), dtype=np.float32)
