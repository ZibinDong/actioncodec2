"""Dense primitive cells to zero-free sparse frames, and sparse training corpora."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..configuration import ActionCodec2Config, ConfigError, _integer
from ..primitives import PrimitiveStore


@dataclass(frozen=True)
class SparseLayout:
    """Maps dense physical cells to zero-free sparse atoms."""

    bins: tuple[int, ...]
    zero: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not self.bins or len(self.bins) != len(self.zero):
            raise ValueError("bins and zero must have one entry per slot")
        if len(self.bins) > 15:
            raise ValueError("sparse frames support at most 15 dense slots")
        for count, zero in zip(self.bins, self.zero, strict=True):
            _integer(count, "bin count", positive=True)
            if zero is not None:
                _integer(zero, "zero index")
            if count < 1 or (zero is not None and not 0 <= zero < count):
                raise ValueError("every slot needs a bin and an in-range zero index")
            if zero is not None and count < 2:
                raise ValueError("a motion slot needs at least one nonzero primitive")

    @classmethod
    def from_config(cls, config: ActionCodec2Config, primitives: PrimitiveStore) -> SparseLayout:
        """Motion slots (order 1 or 2) keep their exact zero primitive; order 0 is change-coded."""
        zero: list[int | None] = [None] * len(primitives.bins)
        for component in config.physical.components.values():
            if component.primitive_order == 0:
                continue
            for dimension in component.dims:
                axis = np.asarray(primitives.values[dimension])
                matches = np.flatnonzero(axis == 0)
                if not len(matches):
                    raise ConfigError(f"dimension {dimension} has no exact zero primitive")
                zero[dimension] = int(matches[0])
        return cls(tuple(int(b) for b in primitives.bins), tuple(zero))

    @property
    def dims(self) -> int:
        return len(self.bins)

    @property
    def e_dim(self) -> int:
        """Sparse slot index of the end-of-frame atom."""
        return len(self.bins)

    @property
    def sparse_bins(self) -> tuple[int, ...]:
        """Motion slots drop their zero primitive; ``E`` has exactly one value."""
        return tuple(b - (z is not None) for b, z in zip(self.bins, self.zero, strict=True)) + (1,)

    @property
    def native_zero(self) -> list[int]:
        return [-1 if z is None else int(z) for z in self.zero]

    def sparsify(self, cells: np.ndarray) -> np.ndarray:
        """Dense chunk ``(T, D)`` to sparse frames ``(T, D + 1)``."""
        cells = np.asarray(cells, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != self.dims or not len(cells):
            raise ValueError(f"cells must have shape (T, {self.dims}) with T >= 1")
        if np.any(cells >= np.asarray(self.bins)):
            raise ValueError("cell outside the primitive range")
        out = np.full((len(cells), self.dims + 1), -1, dtype=np.int64)
        for d, z in enumerate(self.zero):
            column = cells[:, d]
            if np.any((column >= 0) != (column[0] >= 0)):
                raise ValueError("a slot must be active for the whole chunk or not at all")
            if column[0] < 0:
                continue
            if z is None:
                changed = np.concatenate(([True], column[1:] != column[:-1]))
                out[changed, d] = column[changed]
            else:
                moving = column != z
                out[moving, d] = column[moving] - (column[moving] > z)
        content = np.flatnonzero(np.any(out[:, : self.dims] >= 0, axis=1))
        if len(content):
            out[: content[-1], self.dims] = 0
        return out

    def densify(self, sparse: np.ndarray, active: Sequence[bool]) -> np.ndarray:
        """Inverse of :meth:`sparsify` given which dense slots are active."""
        sparse = np.asarray(sparse, dtype=np.int64)
        out = np.full((len(sparse), self.dims), -1, dtype=np.int64)
        for d, z in enumerate(self.zero):
            if not active[d]:
                continue
            column = sparse[:, d]
            if z is None:
                if column[0] < 0:
                    raise ValueError("an absolute slot must be stated on frame 0")
                index = np.maximum.accumulate(np.where(column >= 0, np.arange(len(column)), 0))
                out[:, d] = column[index]
            else:
                out[:, d] = np.where(column < 0, z, column + (column >= z))
        return out


def sparsify_chunks(
    layout: SparseLayout, chunks: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :meth:`SparseLayout.sparsify` over many dense chunks ``[(T_i, D)]``."""
    lengths = np.asarray([len(c) for c in chunks], dtype=np.int64)
    if not len(lengths) or np.any(lengths < 1):
        raise ValueError("chunks must be non-empty")
    starts = np.concatenate(([0], np.cumsum(lengths)))
    arrays = [np.asarray(c) for c in chunks]
    for values in arrays:
        if values.ndim != 2 or values.shape[1] != layout.dims or values.dtype.kind not in "iu":
            raise ValueError("training cells must be integer matrices matching the sparse layout")
        if np.any(values < -1) or np.any(values >= np.asarray(layout.bins)):
            raise ValueError("cell outside the primitive range")
        if np.any((values >= 0) != (values[0] >= 0)):
            raise ValueError("a slot must be active for the whole chunk or not at all")
    dtype = np.int16 if max(layout.bins) <= np.iinfo(np.int16).max else np.int32
    cells = np.concatenate(arrays, dtype=dtype)
    if getattr(layout, "dense", False):
        out = np.full((len(cells), layout.dims + 1), -1, dtype=dtype)
        out[:, : layout.dims] = cells
        out[:, layout.dims] = 0
        out[starts[1:] - 1, layout.dims] = -1
        return out, starts
    chunk = np.repeat(np.arange(len(lengths)), lengths)
    first = np.zeros(len(cells), dtype=bool)
    first[starts[:-1]] = True
    out = np.full((len(cells), layout.dims + 1), -1, dtype=dtype)
    for d, z in enumerate(layout.zero):
        column = cells[:, d]
        active = column >= 0
        if z is None:
            changed = active & (first | (column != np.roll(column, 1)))
            out[changed, d] = column[changed]
        else:
            moving = active & (column != z)
            out[moving, d] = column[moving] - (column[moving] > z)
    local = np.arange(len(cells)) - starts[chunk]
    content = np.any(out[:, : layout.dims] >= 0, axis=1)
    last = np.maximum.reduceat(np.where(content, local, -1), starts[:-1])
    out[local < last[chunk], layout.dims] = 0
    return out, starts


def static_frames(cells: np.ndarray, layout: SparseLayout) -> np.ndarray:
    """Frames whose motion cells are all zero and whose absolute slots repeat the last frame."""
    moving = np.zeros(len(cells), dtype=bool)
    for d, zero in enumerate(layout.zero):
        column = cells[:, d]
        if column[0] < 0:
            continue
        if zero is None:
            moving[1:] |= column[1:] != column[:-1]
        else:
            moving |= column != zero
    return ~moving


def compress_static(cells: np.ndarray, layout: SparseLayout, run_cap: int) -> np.ndarray:
    """Drop leading/trailing static frames and cap interior static runs at ``run_cap``."""
    static = static_frames(cells, layout)
    if static.all():
        return cells[:1]
    keep = np.zeros(len(cells), dtype=bool)
    first, last = int(np.argmax(~static)), len(cells) - int(np.argmax(~static[::-1]))
    keep[first:last] = True
    index = np.arange(len(cells))
    run_start = np.maximum.accumulate(np.where(~static, index + 1, 0))
    keep &= ~static | (index - run_start < run_cap)
    return cells[keep]


def training_chunks(
    trajectories: Sequence[np.ndarray],
    layout: SparseLayout,
    *,
    horizon: int,
    static_run_cap: int,
    dtype: np.dtype | type = np.int64,
) -> list[np.ndarray]:
    """Static-compressed cell trajectories cut into aligned ``horizon`` chunks."""
    out = []
    for raw in trajectories:
        cells = compress_static(np.asarray(raw), layout, static_run_cap)
        for start in range(0, len(cells), horizon):
            piece = cells[start : start + horizon]
            if len(piece) >= 2 or start == 0:
                out.append(np.asarray(piece, dtype=dtype))
    return out
