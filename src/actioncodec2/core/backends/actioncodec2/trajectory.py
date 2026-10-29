"""Variable-width trajectory contract shared by statistics and fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._validation import active_dimensions
from .configuration import ActionCodec2Config

__all__ = ["ActiveTrajectory", "TrajectoryLike", "prepare_trajectory"]


@dataclass(frozen=True)
class ActiveTrajectory:
    """One complete trajectory over an explicit subset of codebook dimensions."""

    values: np.ndarray
    active_dims: tuple[int, ...]
    previous_first_order: np.ndarray | None = None

    def __init__(
        self,
        values: np.ndarray,
        active_dims: Sequence[int],
        *,
        previous_first_order: np.ndarray | None = None,
    ) -> None:
        object.__setattr__(self, "values", np.asarray(values))
        object.__setattr__(self, "active_dims", tuple(active_dims))
        previous = (
            None
            if previous_first_order is None
            else np.asarray(previous_first_order, dtype=np.float32)
        )
        if previous is not None and (
            previous.shape != (len(self.active_dims),) or not np.all(np.isfinite(previous))
        ):
            raise ValueError("previous_first_order must be finite with shape [D_active]")
        object.__setattr__(self, "previous_first_order", previous)

    @classmethod
    def from_padded(
        cls,
        values: np.ndarray,
        *,
        active_dims: Sequence[int],
        previous_first_order: np.ndarray | None = None,
    ) -> ActiveTrajectory:
        """Drop engineering-only dense padding before the semantic boundary."""
        array = np.asarray(values)
        dims = tuple(int(value) for value in active_dims)
        if array.ndim != 2 or not dims or min(dims) < 0 or max(dims) >= array.shape[1]:
            raise ValueError("padded values and active_dims do not define a valid trajectory")
        previous = None if previous_first_order is None else np.take(previous_first_order, dims)
        return cls(np.take(array, dims, axis=1), dims, previous_first_order=previous)


def trajectory_boundary(
    trajectory: TrajectoryLike, config: ActionCodec2Config, active_dims: tuple[int, ...]
) -> np.ndarray | None:
    """Expand explicit compact state to ``[D]`` for fitting and validation."""
    previous = getattr(trajectory, "previous_first_order", None)
    active = set(active_dims)
    required = any(
        c.primitive_order == 2 and active.intersection(c.dims)
        for c in config.physical.components.values()
    )
    if previous is None and required:
        raise ValueError("second-order trajectories require explicit previous_first_order")
    if previous is None and not any(
        c.primitive_order == 2 for c in config.physical.components.values()
    ):
        return None
    dense = np.zeros(config.physical.input_dim, dtype=np.float32)
    if previous is not None:
        dense[list(active_dims)] = previous
    return dense


TrajectoryLike = np.ndarray | ActiveTrajectory


def prepare_trajectory(
    trajectory: TrajectoryLike,
    *,
    dimension: int,
    index: int,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Validate a trajectory and return compact values plus global dimension ids."""
    if isinstance(trajectory, ActiveTrajectory):
        raw = trajectory.values
        active_dims = trajectory.active_dims
    else:
        raw = trajectory
        array = np.asarray(raw)
        active_dims = tuple(range(array.shape[1])) if array.ndim == 2 else ()

    values = np.asarray(raw, dtype=dtype)
    if values.ndim != 2 or len(values) < 1 or values.shape[1] < 1:
        raise ValueError(
            f"trajectory {index} must have non-empty shape [T, D_active], got {values.shape}"
        )
    if len(active_dims) != values.shape[1]:
        raise ValueError(
            f"trajectory {index} active_dims has {len(active_dims)} entries for "
            f"width {values.shape[1]}"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in active_dims):
        raise ValueError(f"trajectory {index} active_dims must contain integers")
    active_dims = active_dimensions(
        active_dims, dimension, subject=f"trajectory {index} active_dims"
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"trajectory {index} contains a non-finite value")
    return np.ascontiguousarray(values), active_dims
