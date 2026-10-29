"""Configuration-driven physical quantization."""

from __future__ import annotations

import itertools
import os
from typing import Protocol

import numpy as np

from ._so3 import (
    conjugate_quaternion as _conjugate,
)
from ._so3 import (
    euler_xyz_to_quaternion as _euler_xyz_to_quaternion,
)
from ._so3 import (
    multiply_quaternions as _multiply,
)
from ._so3 import (
    quaternion_to_euler_xyz as _quaternion_to_euler_xyz,
)
from ._so3 import (
    quaternion_to_rotvec as _quaternion_to_rotvec,
)
from ._so3 import (
    rotvec_to_quaternion as _rotvec_to_quaternion,
)
from ._validation import broadcast_vector
from .configuration import ActionCodec2Config, ComponentConfig, ConfigError
from .native_backend import native_kernel, require_native, resolve_backend
from .primitives import PrimitiveStore
from .semantic import difference_additive as _difference_additive
from .semantic import integrate_additive as _integrate_additive

_FLOAT32_SLACK = np.float32(8 * np.finfo(np.float32).eps)
_IDENTITY = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)

_SELECTION_CODES = {"nearest": 0, "smooth": 1, "momentum_lazy": 2}


def _constraint(component: ComponentConfig, order: int) -> tuple[np.ndarray, np.ndarray] | None:
    value = component.constraints.get(f"order_{order}")
    if value is None:
        return None
    return (
        np.asarray(value.minimum, dtype=np.float32),
        np.asarray(value.maximum, dtype=np.float32),
    )


def _scale(component: ComponentConfig) -> np.ndarray:
    return broadcast_vector(
        component.input_transform.scale,
        len(component.dims),
        dtype=np.float32,
        name="input_transform.scale",
    )


def _transform(component: ComponentConfig, actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32) * _scale(component)
    if component.input_transform.clip:
        bounds = _constraint(component, component.input_order)
        if bounds is None:
            raise ConfigError(f"clipping requires constraints.order_{component.input_order}")
        values = np.clip(values, bounds[0], bounds[1])
    return np.asarray(values, dtype=np.float32)


class _ComponentRuntime(Protocol):
    def encode_batch(
        self, actions: np.ndarray, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray: ...

    def decode_batch(
        self, cells: np.ndarray, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray: ...


class _AdditiveRuntime:
    def __init__(self, component: ComponentConfig, primitives: tuple[np.ndarray, ...]) -> None:
        self.config = component
        self.primitives = tuple(np.asarray(value, dtype=np.float32) for value in primitives)
        self.width = len(component.dims)
        self.order = component.primitive_order
        self._second_order_axes = (
            tuple(axis.astype(np.float64) for axis in self.primitives) if self.order == 2 else ()
        )
        self._second_order_sorted = tuple(
            bool(np.all(axis[1:] > axis[:-1])) for axis in self._second_order_axes
        )
        self._second_order_zero_bounds = tuple(
            (int(np.searchsorted(axis, 0)), int(np.searchsorted(axis, 0, side="right")) - 1)
            if ordered
            else (0, 0)
            for axis, ordered in zip(
                self._second_order_axes, self._second_order_sorted, strict=True
            )
        )
        raw_position_bounds = _constraint(component, 0)
        self.position_bounds = (
            None
            if raw_position_bounds is None
            else (
                np.broadcast_to(raw_position_bounds[0], (self.width,)).copy(),
                np.broadcast_to(raw_position_bounds[1], (self.width,)).copy(),
            )
        )
        self.limit = broadcast_vector(
            component.distortion_limit,
            self.width,
            dtype=np.float32,
            name="distortion_limit",
        )
        table_width = max(len(values) for values in self.primitives)
        table = np.full((self.width, table_width), np.inf, dtype=np.float32)
        for coordinate, values in enumerate(self.primitives):
            table[coordinate, : len(values)] = values
        self._native_table = np.ascontiguousarray(table)
        self._native_bins = np.asarray([len(values) for values in self.primitives], dtype=np.int32)
        self._native_binary = np.zeros(self.width, dtype=np.uint8)

    def _pick_batch(self, request: np.ndarray, direction: np.ndarray) -> np.ndarray:
        flat_request = np.asarray(request, dtype=np.float32).reshape(-1, self.width)
        flat_direction = np.asarray(direction, dtype=np.float32).reshape(-1, self.width)
        kernel = native_kernel("choose_cells")
        if kernel is not None:
            return kernel(
                self._native_table,
                self._native_bins,
                flat_request,
                flat_direction,
                self.limit,
                self.config.primitive_selection == "smooth",
            ).reshape(request.shape)
        choice = np.empty(flat_request.shape, dtype=np.int64)
        for coordinate, primitives in enumerate(self.primitives):
            distance = np.abs(flat_request[:, coordinate, None] - primitives[None, :])
            nearest = np.argmin(distance, axis=1)
            if self.config.primitive_selection == "nearest":
                choice[:, coordinate] = nearest
                continue
            feasible = distance <= self.limit[coordinate] + _FLOAT32_SLACK
            preferred = feasible & (
                np.sign(primitives)[None, :] * flat_direction[:, coordinate, None] >= 0.0
            )
            candidates = np.where(np.any(preferred, axis=1, keepdims=True), preferred, feasible)
            smooth = np.argmin(np.where(candidates, distance, np.inf), axis=1)
            choice[:, coordinate] = np.where(np.any(feasible, axis=1), smooth, nearest)
        return choice.reshape(request.shape)

    def _advance_batch(self, state: np.ndarray, primitive: np.ndarray) -> None:
        higher = primitive
        for level in range(self.order - 1, -1, -1):
            state[:, level] += higher
            higher = state[:, level]
        if self.order == 1 and self.position_bounds is not None:
            state[:, 0] = np.clip(state[:, 0], self.position_bounds[0], self.position_bounds[1])

    def encode_batch(
        self, actions: np.ndarray, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Quantize ``[B,T,d]`` actions, vectorizing all trajectories per step."""
        if self.order == 2:
            return self._encode_second_order(actions, previous_first_order)
        transformed = _transform(self.config, actions)
        target = _integrate_additive(transformed, self.config.input_order)
        batch, steps, _ = target.shape
        if self.order == 0:
            return self._pick_batch(target, np.zeros_like(target))
        if resolve_backend("auto") == "native":
            lower, upper = (
                self.position_bounds
                if self.position_bounds is not None
                else (
                    np.full(self.width, -np.inf, dtype=np.float32),
                    np.full(self.width, np.inf, dtype=np.float32),
                )
            )
            starts = np.zeros((batch, self.width), dtype=np.float32)
            cells = require_native().physical_encode_batch(
                self._native_table,
                self._native_bins,
                self._native_binary,
                np.ascontiguousarray(lower, dtype=np.float32),
                np.ascontiguousarray(upper, dtype=np.float32),
                np.ascontiguousarray(target, dtype=np.float32),
                starts,
                starts,
                self.order,
                min(batch, os.cpu_count() or 1),
                np.ascontiguousarray(self.limit, dtype=np.float32),
                self.config.primitive_selection == "smooth",
            )
            return np.asarray(cells, dtype=np.int64)

        cells = np.empty(target.shape, dtype=np.int64)
        state = np.zeros((batch, self.order, self.width), dtype=np.float32)
        direction = np.zeros((batch, self.width), dtype=np.float32)
        zero = np.zeros((batch, self.width), dtype=np.float32)
        for step in range(steps):
            baseline = state.copy()
            self._advance_batch(baseline, zero)
            request = target[:, step] - baseline[:, 0]
            choice = self._pick_batch(request, direction)
            primitive = np.stack(
                [self.primitives[index][choice[:, index]] for index in range(self.width)],
                axis=1,
            )
            self._advance_batch(state, primitive)
            moved = primitive != 0.0
            direction = np.where(moved, np.sign(primitive), direction)
            cells[:, step] = choice
        return cells

    def decode_batch(
        self, cells: np.ndarray, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Decode ``[B,T,d]`` cells to actions in the component's input units."""
        if self.order == 2:
            return self._decode_second_order(cells, previous_first_order)
        indices = np.asarray(cells, dtype=np.int64)
        kernel = native_kernel("additive_decode")
        if kernel is not None:
            lower, upper = (
                self.position_bounds
                if self.position_bounds is not None
                else (
                    np.full(self.width, -np.inf, dtype=np.float32),
                    np.full(self.width, np.inf, dtype=np.float32),
                )
            )
            return kernel(
                self._native_table,
                indices,
                _scale(self.config),
                lower,
                upper,
                self.config.input_order,
                self.order,
            )
        if self.order == 0:
            positions = np.stack(
                [self.primitives[index][indices[:, :, index]] for index in range(self.width)],
                axis=2,
            )
        elif self.position_bounds is not None and resolve_backend("auto") == "native":
            batch = len(indices)
            starts = np.zeros((batch, self.width), dtype=np.float32)
            positions = require_native().physical_decode_batch(
                self._native_table,
                self._native_bins,
                self._native_binary,
                np.ascontiguousarray(self.position_bounds[0], dtype=np.float32),
                np.ascontiguousarray(self.position_bounds[1], dtype=np.float32),
                np.ascontiguousarray(indices, dtype=np.int32),
                starts,
                starts,
                self.order,
                min(batch, os.cpu_count() or 1),
            )
        else:
            picked = np.stack(
                [self.primitives[index][indices[:, :, index]] for index in range(self.width)],
                axis=2,
            )
            batch, steps, _ = indices.shape
            positions = np.empty(indices.shape, dtype=np.float32)
            state = np.zeros((batch, self.order, self.width), dtype=np.float32)
            for step in range(steps):
                self._advance_batch(state, picked[:, step])
                positions[:, step] = state[:, 0]
        actions = _difference_additive(positions, self.config.input_order)
        return np.asarray(actions / _scale(self.config), dtype=np.float32)

    def _encode_second_order(
        self, actions: np.ndarray, previous_first_order: np.ndarray
    ) -> np.ndarray:
        transformed = _transform(self.config, actions).astype(np.float64)
        velocity = previous_first_order.astype(np.float64) * _scale(self.config)
        target = transformed
        if self.config.input_order == 2:
            target = np.cumsum(target, axis=1) + velocity[:, None]
        if self.config.input_order >= 1:
            target = np.cumsum(target, axis=1)
        if resolve_backend("auto") == "native" and hasattr(
            require_native(), "additive_second_order_encode_batch"
        ):
            return np.asarray(
                require_native().additive_second_order_encode_batch(
                    self._native_table,
                    self._native_bins,
                    np.ascontiguousarray(target),
                    np.ascontiguousarray(velocity),
                    self.limit,
                    self.config.primitive_selection == "smooth",
                    min(
                        len(actions),
                        max(1, int(os.environ.get("ACTIONCODEC2_PHYSICAL_THREADS", "16"))),
                    ),
                ),
                dtype=np.int64,
            )
        position = np.zeros_like(velocity)
        direction = np.zeros_like(velocity)
        cells = np.empty(actions.shape, dtype=np.int64)
        for step in range(actions.shape[1]):
            request = target[:, step] - (position + velocity)

            choice = self._pick_second_order(request, direction)
            primitive = np.stack(
                [axis[choice[:, j]] for j, axis in enumerate(self.primitives)], axis=1
            )
            velocity += primitive
            position += velocity
            direction = np.where(primitive != 0, np.sign(primitive), direction)
            cells[:, step] = choice
        return cells

    def _pick_second_order(self, request: np.ndarray, direction: np.ndarray) -> np.ndarray:
        choice = np.empty(request.shape, dtype=np.int64)
        for j, axis in enumerate(self._second_order_axes):
            if self._second_order_sorted[j] and (
                self.config.primitive_selection == "smooth" or len(axis) > 32
            ):
                right = np.searchsorted(axis, request[:, j])
                left = np.clip(right - 1, 0, len(axis) - 1)
                right = np.minimum(right, len(axis) - 1)
                nearest = np.where(
                    np.abs(request[:, j] - axis[left]) <= np.abs(request[:, j] - axis[right]),
                    left,
                    right,
                )

                tied = (nearest > 0) & (
                    np.abs(request[:, j] - axis[np.maximum(nearest - 1, 0)])
                    == np.abs(request[:, j] - axis[nearest])
                )
                if np.any(tied):
                    nearest[tied] = np.argmin(np.abs(request[tied, j, None] - axis), axis=1)
                if self.config.primitive_selection == "smooth":
                    nonnegative, nonpositive = self._second_order_zero_bounds[j]
                    preferred = np.where(
                        direction[:, j] > 0,
                        np.maximum(nearest, min(nonnegative, len(axis) - 1)),
                        np.where(
                            direction[:, j] < 0,
                            np.minimum(nearest, max(nonpositive, 0)),
                            nearest,
                        ),
                    )
                    admissible = (np.sign(axis[preferred]) * direction[:, j] >= 0) & (
                        np.abs(request[:, j] - axis[preferred]) <= self.limit[j]
                    )
                    nearest = np.where(admissible, preferred, nearest)
                choice[:, j] = nearest
                continue

            distance = np.abs(request[:, j, None] - axis)
            nearest = np.argmin(distance, axis=1)
            if self.config.primitive_selection == "nearest":
                choice[:, j] = nearest
                continue
            feasible = distance <= self.limit[j]
            preferred = feasible & (np.sign(axis)[None] * direction[:, j, None] >= 0)
            candidates = np.where(preferred.any(axis=1, keepdims=True), preferred, feasible)
            smooth = np.argmin(np.where(candidates, distance, np.inf), axis=1)
            choice[:, j] = np.where(feasible.any(axis=1), smooth, nearest)
        return choice

    def _decode_second_order(
        self, cells: np.ndarray, previous_first_order: np.ndarray
    ) -> np.ndarray:
        kernel = native_kernel("additive_reconstruct")
        if kernel is not None:
            return kernel(
                self._native_table,
                cells,
                previous_first_order,
                _scale(self.config),
                self.config.input_order,
                self.order,
            )
        picked = np.stack(
            [axis[cells[:, :, j]] for j, axis in enumerate(self.primitives)], axis=2
        ).astype(np.float64)
        if self.config.input_order == 2:
            result = picked
        else:
            result = np.cumsum(picked, axis=1) + previous_first_order[:, None] * _scale(self.config)
            if self.config.input_order == 0:
                result = np.cumsum(result, axis=1)
        return np.asarray(result / _scale(self.config), dtype=np.float32)


class _SO3Runtime:
    def __init__(self, component: ComponentConfig, primitives: tuple[np.ndarray, ...]) -> None:
        self.config = component
        self.primitives = tuple(np.asarray(value, dtype=np.float64) for value in primitives)
        self.left = component.multiplication == "left"
        self.limit = float(component.distortion_limit)
        if component.primitive_order == 2:
            bounds = component.constraints["order_2"]
            corner = np.maximum(np.abs(bounds.minimum), np.abs(bounds.maximum))
            corner = np.broadcast_to(corner, (3,))
            radius = (
                np.linalg.norm(corner)
                if component.representation == "rotation_vector"
                else np.sum(corner)
            )
            self.second_order_radius = min(np.pi, float(radius) + 3 * self.limit)
        table_width = max(len(values) for values in self.primitives)
        table = np.full((3, table_width), np.inf, dtype=np.float64)
        for coordinate, values in enumerate(self.primitives):
            table[coordinate, : len(values)] = values
        self._native_table = table
        self._native_bins = np.asarray([len(values) for values in self.primitives], dtype=np.int32)
        self.lazy = component.primitive_selection == "momentum_lazy"
        self._zero = np.asarray([int(np.argmin(np.abs(values))) for values in self.primitives])
        if self.lazy and any(
            values[index] != 0.0 for values, index in zip(self.primitives, self._zero, strict=True)
        ):
            raise ValueError("momentum_lazy selection requires an exact zero primitive per axis")

    def _from_representation(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape[:-1]
        flat = np.asarray(values).reshape(-1, 3)
        converted = (
            _rotvec_to_quaternion(flat)
            if self.config.representation == "rotation_vector"
            else _euler_xyz_to_quaternion(flat)
        )
        return converted.reshape(*shape, 4)

    def _to_representation(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape[:-1]
        flat = np.asarray(values).reshape(-1, 4)
        converted = (
            _quaternion_to_rotvec(flat)
            if self.config.representation == "rotation_vector"
            else _quaternion_to_euler_xyz(flat)
        )
        return converted.reshape(*shape, 3)

    def _residual(self, target: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
        return (
            _multiply(target, _conjugate(reconstruction))
            if self.left
            else _multiply(_conjugate(reconstruction), target)
        )

    def _apply(self, state: np.ndarray, increment: np.ndarray) -> np.ndarray:
        return _multiply(increment, state) if self.left else _multiply(state, increment)

    def _nearest_batch(self, request: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.argmin(np.abs(request[:, index, None] - values[None, :]), axis=1)
                for index, values in enumerate(self.primitives)
            ],
            axis=1,
        ).astype(np.int64, copy=False)

    def _smooth(
        self,
        request: np.ndarray,
        reconstruction: np.ndarray,
        target: np.ndarray,
        direction: np.ndarray,
        nearest: np.ndarray,
        previous: np.ndarray | None = None,
    ) -> np.ndarray:
        axes = []
        for coordinate, values in enumerate(self.primitives):
            candidates = np.flatnonzero(
                np.abs(values - request[coordinate]) <= 2.0 * self.limit + 1e-12
            )
            if nearest[coordinate] not in candidates:
                candidates = np.append(candidates, nearest[coordinate])
            axes.append(candidates)

        best_any: tuple | None = None
        best_preferred: tuple | None = None
        repeat = previous is not None and tuple(int(v) for v in previous)
        for raw_choice in itertools.product(*axes):
            choice = tuple(int(value) for value in raw_choice)
            rotvec = np.asarray([self.primitives[index][choice[index]] for index in range(3)])
            rebuilt = self._apply(reconstruction, _rotvec_to_quaternion(rotvec))
            error = float(np.linalg.norm(_quaternion_to_rotvec(self._residual(target, rebuilt))))
            if error > self.limit + 1e-12:
                continue
            if self.lazy and choice == repeat:
                return np.asarray(choice, dtype=np.int64)
            offset = np.abs(np.asarray(choice) - self._zero)
            candidate = (
                (int(np.count_nonzero(offset)), int(offset.sum()), error, choice)
                if self.lazy
                else (error, choice)
            )
            if best_any is None or candidate < best_any:
                best_any = candidate

            if np.dot(rotvec, direction) >= -1e-15 and (
                best_preferred is None or candidate < best_preferred
            ):
                best_preferred = candidate
        selected = best_preferred or best_any
        return nearest if selected is None else np.asarray(selected[-1], dtype=np.int64)

    def encode_batch(
        self, actions: np.ndarray, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Quantize ``[B,T,3]`` rotations with a vectorized batch state."""
        if self.config.primitive_order == 2:
            return self._encode_second_order(actions, previous_first_order)
        transformed = _transform(self.config, actions).astype(np.float64)
        if resolve_backend("auto") == "native":
            cells = require_native().so3_encode_batch(
                self._native_table,
                self._native_bins,
                np.ascontiguousarray(transformed, dtype=np.float32),
                self.config.input_order,
                self.config.representation == "rotation_vector",
                self.left,
                self.limit,
                _SELECTION_CODES[self.config.primitive_selection],
                min(len(transformed), os.cpu_count() or 1),
            )
            return np.asarray(cells, dtype=np.int64)
        converted = self._from_representation(transformed)
        batch, steps = transformed.shape[:2]
        true = np.broadcast_to(_IDENTITY, (batch, 4)).copy()
        reconstruction = true.copy()
        direction = np.zeros((batch, 3), dtype=np.float64)
        cells = np.empty((batch, steps, 3), dtype=np.int64)
        previous = np.broadcast_to(self._zero, (batch, 3))
        for step in range(steps):
            value = converted[:, step]
            true = value if self.config.input_order == 0 else self._apply(true, value)
            residual = _quaternion_to_rotvec(self._residual(true, reconstruction))
            choice = self._nearest_batch(residual)
            if self.config.primitive_selection != "nearest":
                choice = np.stack(
                    [
                        self._smooth(
                            residual[index],
                            reconstruction[index],
                            true[index],
                            direction[index],
                            choice[index],
                            previous[index] if self.lazy else None,
                        )
                        for index in range(batch)
                    ]
                )
            previous = choice
            primitive = np.stack(
                [self.primitives[index][choice[:, index]] for index in range(3)], axis=1
            )
            reconstruction = self._apply(reconstruction, _rotvec_to_quaternion(primitive))
            moved = np.linalg.norm(primitive, axis=1) > 0.0
            direction[moved] = primitive[moved]
            cells[:, step] = choice
        return cells

    def decode_batch(
        self, cells: np.ndarray, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Decode ``[B,T,3]`` cells in the configured boundary representation."""
        kernel = native_kernel("so3_reconstruct")
        if kernel is not None:
            previous = previous_first_order
            if previous is None:
                previous = np.zeros((len(cells), 3), dtype=np.float32)
            return kernel(
                self._native_table,
                cells,
                previous,
                _scale(self.config),
                self.config.input_order,
                self.config.primitive_order,
                self.left,
                self.config.representation == "rotation_vector",
                getattr(self, "second_order_radius", np.inf),
            )
        if self.config.primitive_order == 2:
            return self._decode_second_order(cells, previous_first_order)
        indices = np.asarray(cells, dtype=np.int64)
        rotvec = np.stack(
            [self.primitives[index][indices[:, :, index]] for index in range(3)], axis=2
        )
        if self.config.input_order == 1:
            represented = (
                rotvec
                if self.config.representation == "rotation_vector"
                else self._to_representation(_rotvec_to_quaternion(rotvec.astype(np.float32)))
            )
            actions = np.asarray(represented / _scale(self.config), dtype=np.float32)

            return _difference_additive(np.cumsum(actions, axis=1, dtype=np.float32), 1)

        batch, steps = indices.shape[:2]
        reconstruction = np.broadcast_to(_IDENTITY, (batch, 4)).copy()
        quaternion = np.empty((batch, steps, 4), dtype=np.float64)
        primitive_quaternion = _rotvec_to_quaternion(rotvec)
        for step in range(steps):
            reconstruction = self._apply(reconstruction, primitive_quaternion[:, step])
            quaternion[:, step] = reconstruction
        represented = self._to_representation(quaternion)
        return np.asarray(represented / _scale(self.config), dtype=np.float32)

    def _encode_second_order(
        self, actions: np.ndarray, previous_first_order: np.ndarray
    ) -> np.ndarray:
        converted = self._from_representation(_transform(self.config, actions).astype(np.float64))
        increment = self._from_representation(
            previous_first_order.astype(np.float64) * _scale(self.config)
        )
        if resolve_backend("auto") == "native" and hasattr(
            require_native(), "so3_second_order_encode_batch"
        ):
            return np.asarray(
                require_native().so3_second_order_encode_batch(
                    self._native_table,
                    self._native_bins,
                    np.ascontiguousarray(converted),
                    np.ascontiguousarray(increment),
                    self.config.input_order,
                    self.left,
                    self.limit,
                    self.second_order_radius,
                    self.config.primitive_selection == "smooth",
                    min(
                        len(actions),
                        max(1, int(os.environ.get("ACTIONCODEC2_PHYSICAL_THREADS", "16"))),
                    ),
                ),
                dtype=np.int64,
            )
        batch, steps = actions.shape[:2]
        true = np.broadcast_to(_IDENTITY, (batch, 4)).copy()
        reconstruction = true.copy()
        direction = np.zeros((batch, 3), dtype=np.float64)
        cells = np.empty((batch, steps, 3), dtype=np.int64)
        for step in range(steps):
            true = (
                converted[:, step]
                if self.config.input_order == 0
                else self._apply(true, converted[:, step])
            )
            desired = self._residual(true, reconstruction)
            correction = self._residual(desired, increment)
            request = _quaternion_to_rotvec(correction)
            choice = self._nearest_batch(request)
            if self.config.primitive_selection == "smooth":
                choice = self._smooth_second_order_batch(request, correction, direction, choice)
            primitive = np.stack(
                [axis[choice[:, j]] for j, axis in enumerate(self.primitives)], axis=1
            )
            primitive = self._second_order_primitive(primitive)
            increment = self._apply(increment, _rotvec_to_quaternion(primitive))
            reconstruction = self._apply(reconstruction, increment)
            moved = np.linalg.norm(primitive, axis=1) > 0
            direction[moved] = primitive[moved]
            cells[:, step] = choice
        return cells

    def _second_order_primitive(self, rotvec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        return rotvec * np.minimum(
            1.0, self.second_order_radius / np.maximum(norm, np.finfo(float).tiny)
        )

    def _smooth_second_order_batch(
        self,
        request: np.ndarray,
        correction: np.ndarray,
        direction: np.ndarray,
        nearest: np.ndarray,
    ) -> np.ndarray:
        if len(request) > 64:
            return np.concatenate(
                [
                    self._smooth_second_order_batch(
                        request[i : i + 64],
                        correction[i : i + 64],
                        direction[i : i + 64],
                        nearest[i : i + 64],
                    )
                    for i in range(0, len(request), 64)
                ]
            )
        axes, counts = [], []
        for j, axis in enumerate(self.primitives):
            mask = np.abs(axis[None] - request[:, j, None]) <= 2 * self.limit + 1e-12
            mask[np.arange(len(request)), nearest[:, j]] = True
            count = mask.sum(axis=1)
            rows, columns = np.nonzero(mask)
            slots = np.arange(len(rows)) - np.repeat(np.cumsum(count) - count, count)
            candidates = np.zeros((len(request), int(count.max())), dtype=np.int64)
            candidates[rows, slots] = columns
            axes.append(candidates)
            counts.append(count)
        counts = np.stack(counts, axis=1)
        widths = counts.max(axis=0)
        total = int(np.prod(widths))
        best = nearest.copy()
        best_error = np.full(len(request), np.inf)
        best_preferred = np.zeros(len(request), dtype=bool)
        for offset in range(0, total, 4096):
            lattice = np.stack(
                np.unravel_index(np.arange(offset, min(total, offset + 4096)), tuple(widths)),
                axis=1,
            )
            valid = np.all(lattice[None] < counts[:, None], axis=2)
            choices = np.stack(
                [
                    axis[
                        np.arange(len(request))[:, None],
                        np.minimum(lattice[None, :, j], counts[:, j, None] - 1),
                    ]
                    for j, axis in enumerate(axes)
                ],
                axis=2,
            )
            rotvec = np.stack(
                [axis[choices[:, :, j]] for j, axis in enumerate(self.primitives)], axis=2
            )
            rotvec = self._second_order_primitive(rotvec)
            primitive = _rotvec_to_quaternion(rotvec)
            residual = _multiply(_conjugate(primitive), correction[:, None])
            error = np.linalg.norm(_quaternion_to_rotvec(residual), axis=2)
            valid &= error <= self.limit + 1e-12
            preferred = valid & (np.sum(rotvec * direction[:, None], axis=2) >= -1e-15)
            has_preferred = preferred.any(axis=1)
            eligible = np.where(has_preferred[:, None], preferred, valid)
            scores = np.where(eligible, error, np.inf)
            index = np.argmin(scores, axis=1)
            score = scores[np.arange(len(request)), index]
            improve = np.isfinite(score) & (
                (has_preferred & ~best_preferred)
                | ((has_preferred == best_preferred) & (score < best_error))
            )
            best[improve] = choices[np.arange(len(request)), index][improve]
            best_error[improve] = score[improve]
            best_preferred[improve] = has_preferred[improve]
        return best

    def _decode_second_order(
        self, cells: np.ndarray, previous_first_order: np.ndarray
    ) -> np.ndarray:
        rotvec = np.stack([axis[cells[:, :, j]] for j, axis in enumerate(self.primitives)], axis=2)
        rotvec = self._second_order_primitive(rotvec)
        changes = _rotvec_to_quaternion(rotvec)
        increment = self._from_representation(
            previous_first_order.astype(np.float64) * _scale(self.config)
        )
        reconstruction = np.broadcast_to(_IDENTITY, increment.shape).copy()
        output = np.empty((*cells.shape[:2], 4), dtype=np.float64)
        for step in range(cells.shape[1]):
            increment = self._apply(increment, changes[:, step])
            reconstruction = self._apply(reconstruction, increment)
            output[:, step] = increment if self.config.input_order == 1 else reconstruction
        return np.asarray(self._to_representation(output) / _scale(self.config), dtype=np.float32)


class PhysicalCodec:
    """Compose independent physical components in global action-dimension order."""

    def __init__(self, config: ActionCodec2Config, primitives: PrimitiveStore) -> None:
        if primitives.dimension != config.physical.input_dim:
            raise ConfigError(
                f"primitive dimension {primitives.dimension} does not match input_dim "
                f"{config.physical.input_dim}"
            )
        self.config = config
        self.primitives = primitives
        runtimes: list[tuple[ComponentConfig, _ComponentRuntime]] = []
        for component in config.physical.components.values():
            component_primitives = tuple(primitives.values[index] for index in component.dims)
            runtime: _ComponentRuntime = (
                _AdditiveRuntime(component, component_primitives)
                if component.space == "additive"
                else _SO3Runtime(component, component_primitives)
            )
            runtimes.append((component, runtime))
        self._runtimes = tuple(runtimes)
        self.bins = primitives.bins
        self.requires_previous_first_order = any(c.primitive_order == 2 for c, _ in runtimes)

    def _check_previous_first_order(
        self, value: np.ndarray | None, batch: int
    ) -> np.ndarray | None:
        if value is None:
            if self.requires_previous_first_order:
                raise ValueError(
                    "second-order tokenization requires previous_first_order; no zero initialization is inferred"
                )
            return None
        array = np.asarray(value, dtype=np.float32)
        shape = (batch, self.config.physical.input_dim)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"previous_first_order must be finite with shape {shape}")
        return array

    def _check_actions(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        expected = self.config.physical.input_dim
        if values.ndim != 3 or values.shape[2] != expected or values.shape[1] < 1:
            raise ValueError(f"actions must have non-empty shape [B, T, {expected}]")
        if not np.all(np.isfinite(values)):
            raise ValueError("actions must contain only finite values")
        return values

    def final_first_order(
        self, cells: np.ndarray, *, previous_first_order: np.ndarray
    ) -> np.ndarray:
        """Return ``[B,D]`` state after these cells, without position differencing."""
        values = self._check_cells(cells)
        previous = self._check_previous_first_order(previous_first_order, len(values))
        result = np.zeros((len(values), self.config.physical.input_dim), dtype=np.float32)
        for component, runtime in self._runtimes:
            if component.primitive_order != 2:
                continue
            local = np.take(values, component.dims, axis=2)
            boundary = np.take(previous, component.dims, axis=1)
            picked = np.stack(
                [axis[local[:, :, j]] for j, axis in enumerate(runtime.primitives)], axis=2
            )
            if component.space == "additive":
                final = boundary.astype(np.float64) + np.sum(
                    picked.astype(np.float64), axis=1
                ) / _scale(component)
            else:
                increment = runtime._from_representation(
                    boundary.astype(np.float64) * _scale(component)
                )
                changes = _rotvec_to_quaternion(runtime._second_order_primitive(picked))
                for step in range(local.shape[1]):
                    increment = runtime._apply(increment, changes[:, step])
                final = runtime._to_representation(increment) / _scale(component)
            result[:, component.dims] = final
        return result

    def _check_cells(self, cells: np.ndarray) -> np.ndarray:
        values = np.asarray(cells, dtype=np.int64)
        expected = self.config.physical.input_dim
        if values.ndim != 3 or values.shape[2] != expected or values.shape[1] < 1:
            raise ValueError(f"cells must have non-empty shape [B, T, {expected}]")
        counts = np.asarray(self.bins, dtype=np.int64)
        if values.size and (np.any(values < 0) or np.any(values >= counts[None, None, :])):
            raise ValueError("a cell index is outside its dimension's primitive set")
        return values

    def encode_batch(
        self, actions: np.ndarray, *, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Quantize ``[B,T,D]`` with a required ``[B,D]`` boundary for order 2."""
        values = self._check_actions(actions)
        previous = self._check_previous_first_order(previous_first_order, len(values))
        cells = np.empty(values.shape, dtype=np.int64)
        for component, runtime in self._runtimes:
            component_cells = runtime.encode_batch(
                np.take(values, component.dims, axis=2),
                None if previous is None else np.take(previous, component.dims, axis=1),
            )
            for local, global_dimension in enumerate(component.dims):
                cells[:, :, global_dimension] = component_cells[:, :, local]
        return cells

    def decode_batch(
        self, cells: np.ndarray, *, previous_first_order: np.ndarray | None = None
    ) -> np.ndarray:
        """Decode ``[B,T,D]`` using the same ``[B,D]`` boundary as encoding."""
        values = self._check_cells(cells)
        previous = self._check_previous_first_order(previous_first_order, len(values))
        actions = np.empty(values.shape, dtype=np.float32)
        for component, runtime in self._runtimes:
            component_actions = runtime.decode_batch(
                np.take(values, component.dims, axis=2),
                None if previous is None else np.take(previous, component.dims, axis=1),
            )
            for local, global_dimension in enumerate(component.dims):
                actions[:, :, global_dimension] = component_actions[:, :, local]
        return actions
