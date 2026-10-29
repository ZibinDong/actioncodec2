"""Numerical primitive parameters stored in ``primitives.npz``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .configuration import ConfigError

PRIMITIVES_NAME = "primitives.npz"


def _key(dimension: int) -> str:
    return f"dimension_{dimension:04d}"


class PrimitiveStore:
    """Immutable numerical primitive axes in global action-dimension order."""

    def __init__(self, values: Sequence[np.ndarray | Sequence[float]]) -> None:
        if not values:
            raise ConfigError("primitives must contain at least one dimension")
        arrays = []
        for dimension, raw in enumerate(values):
            array = np.asarray(raw)
            if array.dtype.hasobject:
                raise ConfigError(f"primitive dimension {dimension} must not use object dtype")
            if array.ndim != 1 or not len(array):
                raise ConfigError(f"primitive dimension {dimension} must be a non-empty vector")
            real_numeric = np.issubdtype(array.dtype, np.integer) or np.issubdtype(
                array.dtype, np.floating
            )
            if not real_numeric or not np.all(np.isfinite(array)):
                raise ConfigError(f"primitive dimension {dimension} must contain finite numbers")
            arrays.append(np.array(array, copy=True))
        self.values = tuple(arrays)
        self.bins = tuple(len(array) for array in arrays)

    @property
    def dimension(self) -> int:
        return len(self.values)

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, input_dim: int) -> PrimitiveStore:
        """Load exactly one safe numeric vector per expected global dimension."""
        path = Path(directory) / PRIMITIVES_NAME
        with np.load(path, allow_pickle=False) as payload:
            expected = {_key(index) for index in range(input_dim)}
            actual = set(payload.files)
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            if missing:
                raise ConfigError(f"{PRIMITIVES_NAME} is missing array(s): {', '.join(missing)}")
            if unknown:
                raise ConfigError(
                    f"{PRIMITIVES_NAME} contains unknown array(s): {', '.join(unknown)}"
                )
            values = [payload[_key(index)] for index in range(input_dim)]
        return cls(values)

    def save_pretrained(self, directory: str | Path) -> None:
        """Write ``primitives.npz`` without pickled objects or path metadata."""
        path = Path(directory) / PRIMITIVES_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{_key(index): value for index, value in enumerate(self.values)})
