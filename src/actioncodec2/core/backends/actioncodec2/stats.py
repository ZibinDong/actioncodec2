"""Streaming corpus statistics for first- and second-order primitive design."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .boundary import source_differences
from .configuration import ActionCodec2Config
from .trajectory import TrajectoryLike, prepare_trajectory

__all__ = [
    "DEFAULT_QUANTILES",
    "AxisStatistics",
    "CorpusStatistics",
    "analyze_trajectories",
]

DEFAULT_QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


class _StreamingAxis:
    """Exact moments and bounded-memory deterministic quantile samples."""

    def __init__(self, sample_size: int, rng: np.random.Generator) -> None:
        self.sample_size = sample_size
        self.rng = rng
        self.count = 0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.mean = 0.0
        self.m2 = 0.0
        self.sample = np.empty(0, dtype=np.float32)
        self.priority = np.empty(0, dtype=np.float64)

    def update(self, raw: np.ndarray) -> None:
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not len(values):
            return
        batch_count = len(values)
        batch_mean = float(values.mean(dtype=np.float64))
        batch_m2 = float(np.square(values.astype(np.float64) - batch_mean).sum())
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

        priority = self.rng.random(batch_count)
        candidates = np.concatenate((self.sample, values))
        candidate_priority = np.concatenate((self.priority, priority))
        if len(candidates) > self.sample_size:
            keep = np.argpartition(candidate_priority, self.sample_size - 1)[: self.sample_size]
            candidates = candidates[keep]
            candidate_priority = candidate_priority[keep]
        self.sample = candidates
        self.priority = candidate_priority

    def finish(self, quantiles: Sequence[float]) -> AxisStatistics:
        if self.count == 0:
            return AxisStatistics(0, None, None, None, None, {}, {})
        signed = np.quantile(self.sample, quantiles)
        absolute = np.quantile(np.abs(self.sample), quantiles)
        return AxisStatistics(
            samples=self.count,
            minimum=self.minimum,
            maximum=self.maximum,
            mean=self.mean,
            std=float(np.sqrt(self.m2 / self.count)),
            quantiles={f"{q:g}": float(value) for q, value in zip(quantiles, signed, strict=True)},
            absolute_quantiles={
                f"{q:g}": float(value) for q, value in zip(quantiles, absolute, strict=True)
            },
        )


@dataclass(frozen=True)
class AxisStatistics:
    """Statistics for one global codebook coordinate at one signal order."""

    samples: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    quantiles: dict[str, float]
    absolute_quantiles: dict[str, float]


@dataclass(frozen=True)
class CorpusStatistics:
    """Per-axis input, first-difference and optional second-difference measurements."""

    dimension: int
    episodes: int
    frames: int
    active_values: int
    binary_dims: list[int]
    trajectory_arities: dict[str, int]
    origin_max_abs: list[float | None]
    order_0: list[AxisStatistics]
    order_1: list[AxisStatistics]
    quantiles: list[float]
    quantile_sample_size: int
    order_2: list[AxisStatistics] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema"] = "actioncodec2_first_order_statistics/1"
        if self.order_2 is None:
            payload.pop("order_2")
        else:
            payload["schema"] = "actioncodec2_differential_statistics/2"
        return payload

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _validate_options(
    dimension: int,
    binary_dims: Sequence[int],
    quantiles: Sequence[float],
    sample_size: int,
) -> tuple[set[int], tuple[float, ...]]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    binary = {int(value) for value in binary_dims}
    if any(value < 0 or value >= dimension for value in binary):
        raise ValueError("binary_dims contains a dimension outside the codebook")
    checked_quantiles = tuple(float(value) for value in quantiles)
    if not checked_quantiles or any(not 0.0 <= value <= 1.0 for value in checked_quantiles):
        raise ValueError("quantiles must contain values in [0, 1]")
    if sample_size < 1:
        raise ValueError("quantile_sample_size must be positive")
    return binary, checked_quantiles


def analyze_trajectories(
    trajectories: Iterable[TrajectoryLike],
    *,
    dimension: int,
    binary_dims: Sequence[int] = (),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    quantile_sample_size: int = 1_000_000,
    seed: int = 0,
    config: ActionCodec2Config | None = None,
) -> CorpusStatistics:
    'Measure variable-width canonical absolute trajectories in one pass.'
    binary, checked_quantiles = _validate_options(
        dimension, binary_dims, quantiles, quantile_sample_size
    )
    rng = np.random.default_rng(seed)
    order_0 = [_StreamingAxis(quantile_sample_size, rng) for _ in range(dimension)]
    order_1 = [_StreamingAxis(quantile_sample_size, rng) for _ in range(dimension)]
    order_2 = (
        None
        if config is None
        else [_StreamingAxis(quantile_sample_size, rng) for _ in range(dimension)]
    )
    if config is not None and config.physical.input_dim != dimension:
        raise ValueError("statistics dimension must match config.physical.input_dim")
    origin = np.full(dimension, -np.inf, dtype=np.float64)
    arities: dict[int, int] = {}
    episodes = frames = active_values = 0

    for index, raw in enumerate(trajectories):
        values, active_dims = prepare_trajectory(
            raw, dimension=dimension, index=index, dtype=np.float32
        )
        episodes += 1
        frames += len(values)
        active_values += values.size
        arities[len(active_dims)] = arities.get(len(active_dims), 0) + 1
        for local, global_dim in enumerate(active_dims):
            axis = values[:, local]
            order_0[global_dim].update(axis)
            origin[global_dim] = max(origin[global_dim], abs(float(axis[0])))
            if global_dim not in binary and config is None:
                delta = np.diff(axis, prepend=np.float32(0.0))
                order_1[global_dim].update(delta)
        if config is not None:
            for dims, first, second in source_differences(raw, config).values():
                for local, global_dim in enumerate(dims):
                    if global_dim not in binary:
                        order_1[global_dim].update(first[:, local])
                        order_2[global_dim].update(second[:, local])

    if episodes == 0:
        raise ValueError("statistics require at least one complete trajectory")
    return CorpusStatistics(
        dimension=dimension,
        episodes=episodes,
        frames=frames,
        active_values=active_values,
        binary_dims=sorted(binary),
        trajectory_arities={str(key): value for key, value in sorted(arities.items())},
        origin_max_abs=[None if value == -np.inf else float(value) for value in origin],
        order_0=[axis.finish(checked_quantiles) for axis in order_0],
        order_1=[axis.finish(checked_quantiles) for axis in order_1],
        quantiles=list(checked_quantiles),
        quantile_sample_size=quantile_sample_size,
        order_2=None if order_2 is None else [axis.finish(checked_quantiles) for axis in order_2],
    )
