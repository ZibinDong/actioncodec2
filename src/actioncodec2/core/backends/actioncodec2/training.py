'Whole-trajectory primitive construction and BPE fitting.'

from __future__ import annotations

import math
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from ._validation import broadcast_vector
from .bpe import BPESerializer, SetBPESerializer
from .bpe_training import train_bpe
from .configuration import ActionCodec2Config, ConfigError
from .physical_codec import PhysicalCodec
from .primitives import PrimitiveStore
from .processor import ActionCodec2Processor
from .setbpe.spatial import aggregate_records
from .setbpe.spatial_training import train_spatial
from .setbpe.temporal_training import train_temporal
from .trajectory import TrajectoryLike, prepare_trajectory, trajectory_boundary


def _zero_anchored_grid(lower: np.float32, upper: np.float32, spacing: np.float32) -> np.ndarray:
    """Build a float32 endpoint-preserving grid with an exact zero."""
    below = int(np.ceil(-lower / spacing))
    above = int(np.ceil(upper / spacing))
    negative = np.linspace(lower, np.float32(0.0), below + 1, dtype=np.float32)[:-1]
    positive = np.linspace(np.float32(0.0), upper, above + 1, dtype=np.float32)
    return np.concatenate((negative, positive))


def _additive_axis(
    lower: np.float32,
    upper: np.float32,
    distortion_limit: np.float32,
    *,
    zero_primitive: bool,
) -> np.ndarray:
    """Construct an endpoint-preserving additive cover at the distortion limit."""
    width = np.float32(upper - lower)
    if width == 0.0:
        return np.asarray([lower], dtype=np.float32)
    spacing = np.float32(2.0) * distortion_limit
    if zero_primitive and lower < 0.0 < upper:
        return _zero_anchored_grid(lower, upper, spacing)
    count = 1 + int(np.ceil(width / spacing))
    return np.linspace(lower, upper, count, dtype=np.float32)


def build_primitives(config: ActionCodec2Config) -> PrimitiveStore:
    'Derive primitive arrays and force an exact zero on every axis.'
    if config.train is None:
        raise ConfigError("config.train is required to build primitives")
    output: list[np.ndarray | None] = [None] * config.physical.input_dim
    for component in config.physical.components.values():
        width = len(component.dims)
        if component.space == "additive":
            domain = component.constraints[f"order_{component.primitive_order}"]
            lower = broadcast_vector(
                domain.minimum, width, dtype=np.dtype(np.float32), name="minimum"
            )
            upper = broadcast_vector(
                domain.maximum, width, dtype=np.dtype(np.float32), name="maximum"
            )
            epsilon = broadcast_vector(
                component.distortion_limit,
                width,
                dtype=np.dtype(np.float32),
                name="distortion_limit",
            )
            if component.primitive_order == 2:
                # Relaxed decoder: mu=(epsilon, 2*epsilon, 2*epsilon).
                lower = lower - 2 * epsilon
                upper = upper + 2 * epsilon
            axes = [
                _additive_axis(
                    lower[index],
                    upper[index],
                    epsilon[index],
                    zero_primitive=True,
                )
                for index in range(width)
            ]
        else:
            domain = component.constraints[f"order_{component.primitive_order}"]
            radius_dtype = (
                np.float64 if component.representation == "rotation_vector" else np.float32
            )
            lower = broadcast_vector(
                domain.minimum, width, dtype=np.dtype(radius_dtype), name="minimum"
            )
            upper = broadcast_vector(
                domain.maximum, width, dtype=np.dtype(radius_dtype), name="maximum"
            )
            corner = np.maximum(np.abs(lower), np.abs(upper))
            source_radius = (
                float(np.linalg.norm(corner))
                if component.representation == "rotation_vector"
                else float(np.sum(corner, dtype=np.float32))
            )
            source_radius = min(math.pi, source_radius)
            epsilon = float(component.distortion_limit)
            allowance = (3 if component.primitive_order == 2 else 1) * epsilon
            cover_radius = float(min(math.pi, source_radius + allowance))
            maximum_spacing = 2.0 * epsilon / math.sqrt(3.0)
            count = int(math.ceil(2.0 * cover_radius / maximum_spacing)) + 1
            if count % 2 == 0:
                count += 1
            axis = np.linspace(-cover_radius, cover_radius, count, dtype=np.float64)
            if count % 2 == 1:
                axis[count // 2] = 0.0
            axes = [axis.copy() for _ in range(width)]
        for dimension, axis in zip(component.dims, axes, strict=True):
            output[dimension] = axis
    if any(axis is None for axis in output):
        raise RuntimeError("internal error: primitive construction left an unassigned dimension")
    store = PrimitiveStore([axis for axis in output if axis is not None])
    missing_zero = [
        dimension
        for dimension, axis in enumerate(store.values)
        if not np.any(np.asarray(axis) == 0)
    ]
    if missing_zero:
        raise ConfigError(
            "every fitted primitive axis must contain exact zero; missing dimension(s) "
            f"{missing_zero}. Ensure each configured domain contains zero."
        )
    return store


def _validate_component_support(
    active_dims: tuple[int, ...], config: ActionCodec2Config, *, index: int
) -> None:
    active = set(active_dims)
    for name, component in config.physical.components.items():
        present = active.intersection(component.dims)
        if component.space == "so3" and present and present != set(component.dims):
            raise ValueError(f"trajectory {index} activates only part of SO(3) component {name!r}")


def _length_buckets(
    pending: Sequence[tuple[np.ndarray, tuple[int, ...], np.ndarray | None]],
    *,
    maximum_ratio: float = 1.25,
) -> list[list[int]]:
    """Group nearby lengths so ragged episodes still use vectorized quantization."""
    ordered = sorted(range(len(pending)), key=lambda index: len(pending[index][0]))
    buckets: list[list[int]] = []
    current: list[int] = []
    shortest = 0
    for index in ordered:
        length = len(pending[index][0])
        if current and length > shortest * maximum_ratio:
            buckets.append(current)
            current = []
        if not current:
            shortest = length
        current.append(index)
    if current:
        buckets.append(current)
    return buckets


class _CellCorpus(Sequence[np.ndarray]):
    """Contiguous cell chunks plus compact episode offsets."""

    _actioncodec2_validated = True

    def __init__(self, chunks: list[np.ndarray], offsets: list[np.ndarray]) -> None:
        self.chunks: list[np.ndarray | None] = list(chunks)
        self.offsets = tuple(offsets)
        counts = np.asarray([len(value) - 1 for value in offsets], dtype=np.int64)
        self._episode_ends = np.cumsum(counts)

    def __len__(self) -> int:
        return int(self._episode_ends[-1]) if len(self._episode_ends) else 0

    def __getitem__(self, index: int) -> np.ndarray:
        if not isinstance(index, int):
            raise TypeError("cell corpus indices must be integers")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chunk = bisect_right(self._episode_ends, index)
        previous = 0 if chunk == 0 else int(self._episode_ends[chunk - 1])
        local = index - previous
        start, stop = self.offsets[chunk][local : local + 2]
        values = self.chunks[chunk]
        if values is None:
            raise RuntimeError("cell corpus chunk has already been consumed")
        return values[int(start) : int(stop)]

    def __iter__(self):
        for chunk, offsets in zip(self.chunks, self.offsets, strict=True):
            if chunk is None:
                raise RuntimeError("cell corpus chunk has already been consumed")
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
                yield chunk[int(start) : int(stop)]

    def release_chunk(self, index: int) -> None:
        """Release a replayed chunk while retaining compact episode metadata."""
        self.chunks[index] = None


def _quantize_pending_batch(
    pending: Sequence[tuple[np.ndarray, tuple[int, ...], np.ndarray | None]],
    physical: PhysicalCodec,
    cell_dtype: type[np.signedinteger],
) -> list[np.ndarray]:
    """Quantize one bounded ragged batch and restore its original order."""
    result: list[np.ndarray | None] = [None] * len(pending)
    dimension = physical.config.physical.input_dim
    for indices in _length_buckets(pending):
        length = max(len(pending[index][0]) for index in indices)
        batch = np.zeros((len(indices), length, dimension), dtype=np.float32)
        active_by_item: list[tuple[int, ...]] = []
        for local, index in enumerate(indices):
            trajectory, active_dims, _ = pending[index]
            item = batch[local, : len(trajectory)]
            item[:, active_dims] = trajectory
            active_by_item.append(active_dims)
        boundaries = [pending[index][2] for index in indices]
        previous = (
            None
            if all(value is None for value in boundaries)
            else np.stack(
                [
                    np.zeros(dimension, dtype=np.float32) if value is None else value
                    for value in boundaries
                ]
            )
        )
        encoded = physical.encode_batch(batch, previous_first_order=previous).astype(
            cell_dtype, copy=False
        )
        for local, index in enumerate(indices):
            item_length = len(pending[index][0])
            dense = np.full((item_length, dimension), -1, dtype=cell_dtype)
            dense[:, active_by_item[local]] = encoded[local][:item_length, active_by_item[local]]
            result[index] = dense
    if any(value is None for value in result):
        raise RuntimeError("internal error: trajectory batch was not fully quantized")
    return [value for value in result if value is not None]


def _quantize_trajectories(
    trajectories: Iterable[TrajectoryLike],
    physical: PhysicalCodec,
    *,
    batch_size: int,
    max_cell_bytes: int | None,
) -> tuple[_CellCorpus, int, int, dict[int, int]]:
    'Quantize trajectories in length-homogeneous vectorized batches.'
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cell_chunks: list[np.ndarray] = []
    offset_chunks: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    cell_lengths: list[int] = []
    chunk_actions = 0
    action_count = 0
    active_value_count = 0
    arity_counts: dict[int, int] = defaultdict(int)
    source_index = 0
    retained_cell_bytes = 0
    cell_dtype = np.int16 if max(physical.bins) <= np.iinfo(np.int16).max else np.int32
    iterator = iter(trajectories)

    def flush_cells() -> None:
        nonlocal chunk_actions, retained_cell_bytes
        if not cell_parts:
            return
        chunk = np.concatenate(cell_parts, axis=0)
        projected = retained_cell_bytes + chunk.nbytes
        if max_cell_bytes is not None and projected > max_cell_bytes:
            raise MemoryError(
                "quantized ActionCodec2 corpus would retain at least "
                f"{projected / 2**30:.2f} GiB, exceeding the configured "
                f"{max_cell_bytes / 2**30:.2f} GiB limit; use stratified fitting, "
                "lower coverage, or explicitly raise --max-cell-memory-gib"
            )
        cell_chunks.append(chunk)
        retained_cell_bytes = projected
        offset_chunks.append(np.concatenate(([0], np.cumsum(cell_lengths, dtype=np.int64))))
        cell_parts.clear()
        cell_lengths.clear()
        chunk_actions = 0

    while True:
        pending: list[tuple[np.ndarray, tuple[int, ...], np.ndarray | None]] = []
        for _ in range(batch_size):
            try:
                raw = next(iterator)
            except StopIteration:
                break
            trajectory, active_dims = prepare_trajectory(
                raw,
                index=source_index,
                dimension=physical.config.physical.input_dim,
                dtype=np.float32,
            )
            _validate_component_support(active_dims, physical.config, index=source_index)
            source_index += 1
            action_count += len(trajectory)
            active_value_count += len(trajectory) * len(active_dims)
            arity_counts[len(active_dims)] += 1
            pending.append(
                (trajectory, active_dims, trajectory_boundary(raw, physical.config, active_dims))
            )
        if not pending:
            break

        for value in _quantize_pending_batch(pending, physical, cell_dtype):
            cell_parts.append(value)
            cell_lengths.append(len(value))
            chunk_actions += len(value)
        if chunk_actions >= 1_000_000:
            flush_cells()
    flush_cells()
    cells = _CellCorpus(cell_chunks, offset_chunks)
    if not cells:
        raise ValueError("training requires at least one complete trajectory")
    return cells, action_count, active_value_count, dict(sorted(arity_counts.items()))


def _aggregate_trajectory_records(
    cells: Sequence[np.ndarray], bins: Sequence[int], action_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate records with memory bounded by one row chunk plus unique rows."""
    del action_count  # retained in the signature for artifact-report compatibility
    dimension = len(tuple(bins))
    widths = tuple(int(count).bit_length() for count in bins)
    if sum(widths) <= 64:
        return _aggregate_packed_records(cells, bins, widths)
    counts_by_row: dict[bytes, int] = defaultdict(int)
    pending: list[np.ndarray] = []
    pending_rows = 0
    chunk_rows = 1_000_000

    def flush() -> None:
        nonlocal pending_rows
        if not pending:
            return
        block = (
            np.asarray(pending[0], dtype=np.int32)
            if len(pending) == 1
            else np.concatenate(pending, axis=0).astype(np.int32, copy=False)
        )
        unique, counts = aggregate_records(block, bins)
        for row, count in zip(unique.astype(np.int32, copy=False), counts, strict=True):
            counts_by_row[row.tobytes()] += int(count)
        pending.clear()
        pending_rows = 0

    for trajectory in cells:
        pending.append(np.asarray(trajectory, dtype=np.int32))
        pending_rows += len(trajectory)
        if pending_rows >= chunk_rows:
            flush()
    flush()
    keys = sorted(counts_by_row)
    records = np.frombuffer(b"".join(keys), dtype=np.int32).reshape(-1, dimension)
    weights = np.fromiter((counts_by_row[key] for key in keys), dtype=np.int64, count=len(keys))
    return records, weights


def _aggregate_packed_records(
    cells: Sequence[np.ndarray], bins: Sequence[int], widths: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate short ragged cell rows using one exact uint64 key per row."""
    counts_by_key: dict[int, int] = defaultdict(int)
    pending: list[np.ndarray] = []
    pending_rows = 0
    chunk_rows = 1_000_000

    def flush() -> None:
        nonlocal pending_rows
        if not pending:
            return
        block = pending[0] if len(pending) == 1 else np.concatenate(pending, axis=0)
        keys = np.zeros(len(block), dtype=np.uint64)
        shift = 0
        for dimension, width in enumerate(widths):
            encoded = np.asarray(block[:, dimension], dtype=np.int64) + 1
            keys |= encoded.astype(np.uint64) << np.uint64(shift)
            shift += width
        unique, counts = np.unique(keys, return_counts=True)
        for key, count in zip(unique, counts, strict=True):
            counts_by_key[int(key)] += int(count)
        pending.clear()
        pending_rows = 0

    for trajectory in cells:
        pending.append(trajectory)
        pending_rows += len(trajectory)
        if pending_rows >= chunk_rows:
            flush()
    flush()

    keys = np.fromiter(sorted(counts_by_key), dtype=np.uint64, count=len(counts_by_key))
    records = np.empty((len(keys), len(tuple(bins))), dtype=np.int32)
    shift = 0
    for dimension, width in enumerate(widths):
        mask = np.uint64((1 << width) - 1)
        records[:, dimension] = ((keys >> np.uint64(shift)) & mask).astype(np.int32) - 1
        shift += width
    weights = np.fromiter(
        (counts_by_key[int(key)] for key in keys), dtype=np.int64, count=len(keys)
    )
    return records, weights


def _fit_report(
    config: ActionCodec2Config,
    primitives: PrimitiveStore,
    codec: ActionCodec2Processor,
    *,
    trajectory_count: int,
    actions: int,
    active_values: int,
    arity_counts: dict[int, int],
    cell_dtype: str,
    unique_records: int,
    spatial_report: dict[str, object],
    temporal_report: dict[str, object],
    timestamps: tuple[float, float, float, float, float, float],
) -> dict[str, object]:
    started, primitive_done, quantized, aggregated, spatial_done, finished = timestamps
    scheme = config.set_bpe.scheme
    stages: dict[str, object] = (
        {"bpe": temporal_report}
        if scheme == "bpe"
        else {"spatial": spatial_report, "temporal": temporal_report}
    )
    return {
        "trajectories": trajectory_count,
        "actions": actions,
        "active_values": active_values,
        "trajectory_arities": {str(key): value for key, value in arity_counts.items()},
        "bpe_boundary": "complete_component_trajectory",
        "bpe_training_domain": "quantized_primitive_cells",
        "physical_boundary": (
            "explicit_previous_first_order"
            if codec.requires_previous_first_order
            else "zero_position_identity_orientation"
        ),
        "component_primitive_orders": {
            name: component.primitive_order
            for name, component in config.physical.components.items()
        },
        "allow_cross_component_merge": config.set_bpe.allow_cross_component_merge,
        "lookahead": config.set_bpe.lookahead,
        "physical_bins": list(primitives.bins),
        "cell_dtype": cell_dtype,
        "unique_spatial_records": unique_records,
        "primitive_atoms": sum(primitives.bins),
        "tokenization_scheme": scheme,
        **stages,
        "vocab_size": codec.vocab_size,
        "tokens_per_action": float(temporal_report["tokens_per_action"]),
        "seconds": {
            "primitives": primitive_done - started,
            "quantize": quantized - primitive_done,
            "aggregate": aggregated - quantized,
            "spatial_train": spatial_done - aggregated,
            "spatial": spatial_done - quantized,
            "temporal": finished - spatial_done,
            "total": finished - started,
        },
    }


def _fit_set_bpe(
    cells: Sequence[np.ndarray],
    config: ActionCodec2Config,
    primitives: PrimitiveStore,
    actions: int,
    *,
    backend: str,
    threads: int,
    progress: bool,
) -> tuple[SetBPESerializer, dict[str, object], dict[str, object], int, float, float]:
    records, weights = _aggregate_trajectory_records(cells, primitives.bins, actions)
    aggregated = time.perf_counter()
    bpe = config.train.bpe if config.train is not None else None
    assert bpe is not None
    spatial, spatial_report = train_spatial(
        records,
        weights,
        primitives.bins,
        vocab_size=bpe.spatial_vocab_size,
        backend=backend,
        threads=threads,
        progress=progress,
    )
    spatial_done = time.perf_counter()
    vocabulary, temporal_report = train_temporal(
        cells,
        spatial,
        vocab_size=bpe.vocab_size,
        lookahead=config.set_bpe.lookahead,
        backend=backend,
        threads=threads,
        progress=progress,
    )
    serializer = SetBPESerializer(
        bins=primitives.bins,
        lookahead=config.set_bpe.lookahead,
        spatial_merges=spatial.merges,
        temporal_merges=vocabulary.merges,
    )
    return serializer, spatial_report, temporal_report, len(records), aggregated, spatial_done


def _fit_bpe(
    cells: Sequence[np.ndarray],
    config: ActionCodec2Config,
    primitives: PrimitiveStore,
    *,
    backend: str,
    threads: int,
    progress: bool,
) -> tuple[BPESerializer, dict[str, object], dict[str, object], int, float, float]:
    'Fit the FAST-style baseline on the same quantized cells.'
    started = time.perf_counter()
    bpe = config.train.bpe if config.train is not None else None
    assert bpe is not None
    vocabulary, report = train_bpe(
        cells,
        primitives.bins,
        vocab_size=bpe.vocab_size,
        max_token_cells=config.set_bpe.lookahead * config.physical.input_dim,
        backend=backend,
        threads=threads,
        progress=progress,
    )
    serializer = BPESerializer(
        bins=primitives.bins,
        lookahead=config.set_bpe.lookahead,
        merges=vocabulary.merges,
    )
    return serializer, {}, report, 0, started, started


def _validate_fit_options(config: ActionCodec2Config, max_cell_bytes: int | None) -> None:
    if config.train is None:
        raise ConfigError("config.train is required for fitting")
    if config.set_bpe.scheme == "sparse_set_bpe":
        raise ConfigError("sparse_set_bpe profiles are fitted by ActionCodec2Sparse")
    if max_cell_bytes is not None and (
        isinstance(max_cell_bytes, bool)
        or not isinstance(max_cell_bytes, int)
        or max_cell_bytes < 1
    ):
        raise ValueError("max_cell_bytes must be a positive integer or None")
    if config.set_bpe.allow_cross_component_merge:
        raise ConfigError(
            "allow_cross_component_merge=true requires the future multi-component "
            "corpus representation; v1 fitting supports the default isolated boundary only"
        )
    if config.set_bpe.scheme == "bpe" and config.train.bpe.spatial_vocab_size != (
        config.train.bpe.vocab_size
    ):
        raise ConfigError(
            "set_bpe.scheme=bpe has a single merge stage; set "
            "train.bpe.spatial_vocab_size equal to train.bpe.vocab_size to make the "
            "unused split explicit"
        )


class ProfileTrainer:
    'Internal whole-trajectory fitter for one physical profile.'

    def __init__(self, config: ActionCodec2Config) -> None:
        self.config = config

    def fit(
        self,
        trajectories: Iterable[TrajectoryLike],
        *,
        backend: str = "auto",
        threads: int = 0,
        batch_size: int = 256,
        progress: bool = False,
        max_cell_bytes: int | None = None,
    ) -> tuple[ActionCodec2Processor, dict[str, object]]:
        'Fit primitives and the configured tokenization scheme on whole trajectories.'
        config = self.config
        _validate_fit_options(config, max_cell_bytes)
        started = time.perf_counter()
        primitives = build_primitives(config)
        primitive_done = time.perf_counter()

        atoms = sum(primitives.bins)
        bpe = config.train.bpe
        if bpe.spatial_vocab_size < atoms:
            raise ConfigError(
                f"train.bpe.spatial_vocab_size must be at least the {atoms} primitive atoms"
            )

        physical = PhysicalCodec(config, primitives)
        cells, actions, active_values, arity_counts = _quantize_trajectories(
            trajectories,
            physical,
            batch_size=batch_size,
            max_cell_bytes=max_cell_bytes,
        )
        trajectory_count = len(cells)
        first_cell_dtype = str(cells[0].dtype)
        quantized = time.perf_counter()

        if config.set_bpe.scheme == "unified_set_bpe":
            from .unified_set_bpe import fit_unified_set_bpe

            serializer, unified_report = fit_unified_set_bpe(
                config, cells, threads=threads, backend=backend, progress=progress
            )
            temporal_report = {
                **unified_report["temporal"],
                "tokens_per_action": unified_report["temporal"]["final_instances"] / actions,
            }
            fitted = serializer, {}, temporal_report, 0, quantized, quantized
        elif config.set_bpe.scheme == "bpe":
            fitted = _fit_bpe(
                cells, config, primitives, backend=backend, threads=threads, progress=progress
            )
        else:
            fitted = _fit_set_bpe(
                cells,
                config,
                primitives,
                actions,
                backend=backend,
                threads=threads,
                progress=progress,
            )
        serializer, spatial_report, temporal_report, unique_records, aggregated, spatial_done = (
            fitted
        )
        codec = ActionCodec2Processor(config, primitives, serializer)
        finished = time.perf_counter()

        report = _fit_report(
            config,
            primitives,
            codec,
            trajectory_count=trajectory_count,
            actions=actions,
            active_values=active_values,
            arity_counts=arity_counts,
            cell_dtype=first_cell_dtype,
            unique_records=unique_records,
            spatial_report=spatial_report,
            temporal_report=temporal_report,
            timestamps=(
                started,
                primitive_done,
                quantized,
                aggregated,
                spatial_done,
                finished,
            ),
        )
        return codec, report
