"""Rotation helpers for deriving delta actions from absolute poses."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["relative_rotation_vectors_between"]


def relative_rotation_vectors_between(
    previous_euler: np.ndarray, target_euler: np.ndarray
) -> np.ndarray:
    """Body-frame rotation vector taking each XYZ-Euler ``previous`` to ``target``."""
    previous = np.asarray(previous_euler, dtype=np.float64)
    target = np.asarray(target_euler, dtype=np.float64)
    if previous.shape != target.shape or previous.shape[-1:] != (3,):
        raise ValueError("previous and target Euler arrays must have the same [..., 3] shape")
    shape = previous.shape
    start = Rotation.from_euler("xyz", previous.reshape(-1, 3))
    end = Rotation.from_euler("xyz", target.reshape(-1, 3))
    return (start.inv() * end).as_rotvec().reshape(shape)
