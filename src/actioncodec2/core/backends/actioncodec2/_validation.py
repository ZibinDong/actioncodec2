"""Shared validation primitives for ActionCodec2 public boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

import numpy as np

ErrorT = TypeVar("ErrorT", bound=Exception)


def active_dimensions(
    active_dims: Sequence[int] | None,
    dimension: int,
    *,
    subject: str = "active_dims",
) -> tuple[int, ...]:
    """Normalize and validate a non-empty increasing subset of dimensions."""
    dims = (
        tuple(range(dimension)) if active_dims is None else tuple(int(item) for item in active_dims)
    )
    if not dims:
        raise ValueError(f"{subject} must not be empty")
    if any(left >= right for left, right in zip(dims, dims[1:], strict=False)):
        raise ValueError(f"{subject} must be strictly increasing")
    if dims[0] < 0 or dims[-1] >= dimension:
        raise ValueError(f"{subject} contains a dimension outside the vocabulary")
    return dims


def optional_finite_array(
    value: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    name: str = "current_state",
) -> np.ndarray | None:
    """Convert an optional float32 array and enforce its exact shape/finiteness."""
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        rendered = ", ".join(str(item) for item in shape)
        raise ValueError(f"{name} must have shape [{rendered}]")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def broadcast_vector(
    value: object,
    width: int,
    *,
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    """Broadcast a scalar or width-sized value to one numerical vector."""
    array = np.asarray(value, dtype=dtype)
    try:
        return np.broadcast_to(array, (width,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or contain {width} values") from error


def validate_dimension_partition(
    groups: Iterable[tuple[str, Sequence[int]]],
    dimension: int,
    *,
    label: str,
    error_type: type[ErrorT] = ValueError,
) -> None:
    """Validate that named dimension groups form one exact non-overlapping partition."""
    owners: dict[int, str] = {}
    for name, dims in groups:
        for index in dims:
            if index < 0 or index >= dimension:
                raise error_type(f"{label} {index} used by {name!r} is outside input_dim")
            if index in owners:
                raise error_type(f"{label} {index} belongs to both {owners[index]!r} and {name!r}")
            owners[index] = name
    missing = sorted(set(range(dimension)) - set(owners))
    if missing:
        raise error_type(f"{label}(s) {missing} are not assigned")
