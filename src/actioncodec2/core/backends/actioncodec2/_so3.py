"""Shared NumPy quaternion operations using the xyzw convention."""

from __future__ import annotations

import numpy as np

from .native_backend import native_kernel


def _mapped(values, operation, width):
    kernel = native_kernel("quaternion_map")
    if kernel is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    return kernel(array.reshape(-1, array.shape[-1]), operation).reshape(*array.shape[:-1], width)


def normalize_quaternion(values: np.ndarray) -> np.ndarray:
    """Normalize quaternion rows and reject an undefined zero rotation."""
    result = _mapped(values, "normalize", 4)
    if result is not None:
        return result
    quaternions = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("a pose contains a zero quaternion")
    return quaternions / norms


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of broadcastable xyzw arrays."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    kernel = native_kernel("quaternion_multiply")
    if kernel is not None:
        first, second = np.broadcast_arrays(first, second)
        return kernel(first.reshape(-1, 4), second.reshape(-1, 4)).reshape(first.shape)
    first_vector, first_scalar = first[..., :3], first[..., 3:4]
    second_vector, second_scalar = second[..., :3], second[..., 3:4]
    return normalize_quaternion(
        np.concatenate(
            (
                first_scalar * second_vector
                + second_scalar * first_vector
                + np.cross(first_vector, second_vector),
                first_scalar * second_scalar
                - np.sum(first_vector * second_vector, axis=-1, keepdims=True),
            ),
            axis=-1,
        )
    )


def conjugate_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of an xyzw quaternion array."""
    native = _mapped(quaternion, "conjugate", 4)
    if native is not None:
        return native
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., :3] *= -1.0
    return result


def rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to xyzw quaternions."""
    result = _mapped(rotvec, "from_rotvec", 4)
    if result is not None:
        return result
    values = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(values, axis=-1)
    scale = np.full_like(angle, 0.5)
    regular = angle >= 1e-12
    scale[regular] = np.sin(0.5 * angle[regular]) / angle[regular]
    return normalize_quaternion(
        np.concatenate((values * scale[..., None], np.cos(0.5 * angle)[..., None]), axis=-1)
    )


def quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to shortest-path rotation vectors."""
    result = _mapped(quaternion, "to_rotvec", 3)
    if result is not None:
        return result
    values = normalize_quaternion(quaternion).copy()
    values[values[..., 3] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(values[..., :3], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(values[..., 3], 0.0, 1.0))
    scale = np.full_like(angle, 2.0)
    regular = vector_norm >= 1e-12
    scale[regular] = angle[regular] / vector_norm[regular]
    return values[..., :3] * scale[..., None]


def euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert ``(N,3)`` XYZ Euler angles to xyzw quaternions."""
    values = np.asarray(euler, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("XYZ Euler angles must have shape [N, 3]")
    result = _mapped(values, "from_euler", 4)
    if result is not None:
        return result
    half = 0.5 * values
    sin_roll, sin_pitch, sin_yaw = np.sin(half).T
    cos_roll, cos_pitch, cos_yaw = np.cos(half).T
    result = np.empty((len(values), 4), dtype=np.float64)
    result[:, 0] = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    result[:, 1] = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    result[:, 2] = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    result[:, 3] = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    return normalize_quaternion(result)


def quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert ``(N,4)`` xyzw quaternions to XYZ Euler angles."""
    values = normalize_quaternion(quaternion)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("xyzw quaternions must have shape [N, 4]")
    result = _mapped(values, "to_euler", 3)
    if result is not None:
        return result
    x, y, z, w = values.T
    return np.stack(
        (
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        axis=1,
    )
