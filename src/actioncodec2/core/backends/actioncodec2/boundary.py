"""Explicit measured-context preparation for stateful trajectory fitting."""

from __future__ import annotations

import numpy as np

from .configuration import ActionCodec2Config
from .physical_codec import _scale, _SO3Runtime
from .trajectory import ActiveTrajectory, TrajectoryLike, prepare_trajectory, trajectory_boundary


def source_differences(
    trajectory: TrajectoryLike, config: ActionCodec2Config
) -> dict[str, tuple[tuple[int, ...], np.ndarray, np.ndarray]]:
    """Measure physical first/second differences with the declared group law."""
    values, dims = prepare_trajectory(trajectory, dimension=config.physical.input_dim, index=0)
    boundary = trajectory_boundary(trajectory, config, dims)
    output = {}
    for name, component in config.physical.components.items():
        present = tuple(d for d in component.dims if d in dims)
        if not present or component.primitive_order == 0:
            continue
        local = [dims.index(d) for d in present]
        indices = [component.dims.index(d) for d in present]
        scale = np.take(_scale(component), indices)
        raw = values[:, local].astype(np.float64) * scale
        previous = None if boundary is None else boundary[list(present)].astype(np.float64) * scale
        if component.space == "additive":
            if component.input_order == 0:
                first = np.diff(raw, axis=0, prepend=np.zeros((1, len(present))))
            elif component.input_order == 1:
                first = raw
            else:
                if previous is None:
                    raise ValueError("order-2 input statistics require previous_first_order")
                first = np.cumsum(raw, axis=0) + previous
            second = (
                np.diff(first, axis=0)
                if previous is None
                else np.diff(first, axis=0, prepend=previous[None])
            )
        else:
            if len(present) != 3:
                raise ValueError("differential statistics require complete SO(3) components")
            runtime = _SO3Runtime(component, (np.zeros(1),) * 3)
            quaternion = runtime._from_representation(raw)
            if component.input_order == 0:
                anchors = np.concatenate((np.array([[0.0, 0.0, 0.0, 1.0]]), quaternion[:-1]))
                increments = runtime._residual(quaternion, anchors)
            else:
                increments = quaternion
            first = runtime._to_representation(increments)
            if previous is None:
                changes = runtime._residual(increments[1:], increments[:-1])
            else:
                anchors = np.concatenate(
                    (runtime._from_representation(previous[None]), increments[:-1])
                )
                changes = runtime._residual(increments, anchors)
            second = runtime._to_representation(changes)
        output[name] = present, first, second
    return output


def bootstrap_second_order_trajectory(
    trajectory: ActiveTrajectory, config: ActionCodec2Config
) -> ActiveTrajectory:
    """Use two observed positions as context and encode the remaining suffix."""
    values, dims = prepare_trajectory(trajectory, dimension=config.physical.input_dim, index=0)
    if len(values) < 3:
        raise ValueError(
            "second-order bootstrap requires two context frames and at least one target"
        )
    if trajectory.previous_first_order is not None:
        raise ValueError("trajectory already has previous_first_order; do not bootstrap it again")
    result = values[2:].copy()
    previous = np.zeros(len(dims), dtype=np.float32)
    for component in config.physical.components.values():
        local = [dims.index(d) for d in component.dims if d in dims]
        if not local:
            continue
        if component.input_order != 0:
            raise ValueError("measured-context bootstrap requires input_order=0")
        if component.primitive_order == 0:
            continue
        if component.space == "additive":
            result[:, local] -= values[1, local]
            previous[local] = values[1, local] - values[0, local]
        else:
            if len(local) != 3:
                raise ValueError("bootstrap requires complete SO(3) components")
            runtime = _SO3Runtime(component, (np.zeros(1),) * 3)
            quaternion = runtime._from_representation(values[:, local] * _scale(component))
            increment = runtime._residual(quaternion[1:2], quaternion[:1])

            from ._so3 import conjugate_quaternion, multiply_quaternions

            anchor = conjugate_quaternion(quaternion[1:2])
            local_path = (
                multiply_quaternions(quaternion[2:], anchor)
                if runtime.left
                else multiply_quaternions(anchor, quaternion[2:])
            )
            result[:, local] = runtime._to_representation(local_path) / _scale(component)
            previous[local] = runtime._to_representation(increment)[0] / _scale(component)
    return ActiveTrajectory(result, dims, previous_first_order=previous)
