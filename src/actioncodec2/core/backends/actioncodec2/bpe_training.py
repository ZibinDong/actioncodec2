'Plain-BPE fitting implementations.'

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from .bpe_vocabulary import BPEVocabulary
from .native_backend import require_native, resolve_backend

__all__ = ["train_bpe", "cell_chunks"]


def cell_chunks(
    episodes: Iterable[np.ndarray],
    *,
    max_chunk_rows: int = 1_000_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    'Yield ``(cells, episode_offsets)`` blocks for the native trainer.'
    chunks = getattr(episodes, "chunks", None)
    offsets = getattr(episodes, "offsets", None)
    if getattr(episodes, "_actioncodec2_validated", False) and chunks is not None:
        for index, (codes, bounds) in enumerate(zip(chunks, offsets, strict=True)):
            if codes is None:
                raise RuntimeError("cell corpus chunk has already been consumed")
            yield np.ascontiguousarray(codes), np.ascontiguousarray(bounds, dtype=np.int64)
            episodes.release_chunk(index)
        return
    pending: list[np.ndarray] = []
    lengths: list[int] = []
    rows = 0

    def flush() -> tuple[np.ndarray, np.ndarray]:
        block = np.concatenate(pending, axis=0)
        bounds = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        pending.clear()
        lengths.clear()
        return np.ascontiguousarray(block), bounds

    for episode in episodes:
        values = np.asarray(episode)
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError("episodes must contain integer cell indices")
        if values.ndim != 2 or len(values) < 1:
            raise ValueError("every episode must have non-empty shape [T, D]")
        pending.append(values)
        lengths.append(len(values))
        rows += len(values)
        if rows >= max_chunk_rows:
            yield flush()
            rows = 0
    if pending:
        yield flush()


def _flatten(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]], bins: Sequence[int]
) -> tuple[list[list[int]], int]:
    """Serialize every episode row-major over its active support."""
    counts = np.asarray(tuple(int(value) for value in bins), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    streams: list[list[int]] = []
    actions = 0
    for codes, bounds in chunks:
        values = np.asarray(codes, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != len(counts):
            raise ValueError("cells must have one column per bin count")
        if np.any(values < -1) or np.any(values >= counts[None, :]):
            raise ValueError("cells must contain -1 or an in-range cell index")
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            episode = values[int(start) : int(stop)]
            if not len(episode):
                raise ValueError("every episode must be non-empty")
            support = episode >= 0
            if np.any(support != support[:1]):
                raise ValueError("active dimensions must remain constant within an episode")
            present = np.flatnonzero(support[0])
            if not len(present):
                raise ValueError("every control step needs at least one active coordinate")
            shifted = episode[:, present] + offsets[present][None, :]
            streams.append(shifted.reshape(-1).tolist())
            actions += len(episode)
    if not streams:
        raise ValueError("BPE training requires at least one complete trajectory")
    return streams, actions


def _train_bpe_python(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    bins: Sequence[int],
    *,
    vocab_size: int,
    max_token_cells: int,
    progress: bool = False,
) -> tuple[BPEVocabulary, dict[str, object]]:
    'Fit exact greedy BPE with sparse counters over observed ordered pairs.'
    atoms = int(sum(int(value) for value in bins))
    if vocab_size < atoms:
        raise ValueError(f"vocab_size {vocab_size} is below the {atoms} primitive atoms")
    if max_token_cells < 1:
        raise ValueError("max_token_cells must be positive")
    streams, actions = _flatten(chunks, bins)
    initial_length = sum(len(stream) for stream in streams)
    length = initial_length

    counts: dict[tuple[int, int], int] = defaultdict(int)
    carriers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index, stream in enumerate(streams):
        for pair in zip(stream[:-1], stream[1:], strict=True):
            counts[pair] += 1
            carriers[pair].add(index)
    heap = [(-value, pair) for pair, value in counts.items()]
    heapq.heapify(heap)

    token_cells = [1] * atoms
    merges: list[tuple[int, int]] = []
    oversized: set[tuple[int, int]] = set()
    observed_cells = 1
    while len(merges) + atoms < vocab_size and heap:
        negative, pair = heapq.heappop(heap)
        frequency = counts.get(pair, 0)
        if frequency <= 0 or frequency != -negative or pair in oversized:
            continue
        span = token_cells[pair[0]] + token_cells[pair[1]]
        if span > max_token_cells:
            oversized.add(pair)
            continue
        child = atoms + len(merges)
        token_cells.append(span)
        observed_cells = max(observed_cells, span)
        touched: set[tuple[int, int]] = set()
        for index in sorted(carriers.pop(pair, ())):
            stream = streams[index]
            output: list[int] = []
            position = 0
            while position < len(stream):
                if (
                    position + 1 < len(stream)
                    and stream[position] == pair[0]
                    and stream[position + 1] == pair[1]
                ):
                    if output:
                        counts[(output[-1], pair[0])] -= 1
                        touched.add((output[-1], pair[0]))
                        counts[(output[-1], child)] += 1
                        carriers[(output[-1], child)].add(index)
                        touched.add((output[-1], child))
                    if position + 2 < len(stream):
                        follower = stream[position + 2]
                        counts[(pair[1], follower)] -= 1
                        touched.add((pair[1], follower))
                        counts[(child, follower)] += 1
                        carriers[(child, follower)].add(index)
                        touched.add((child, follower))
                    counts[pair] -= 1
                    length -= 1
                    output.append(child)
                    position += 2
                else:
                    output.append(stream[position])
                    position += 1
            streams[index] = output
        merges.append(pair)
        for member in touched:
            value = counts.get(member, 0)
            if value > 0:
                heapq.heappush(heap, (-value, member))
        if progress and len(merges) % 256 == 0:
            print(f"  bpe merge {len(merges):>5}  tokens/action {length / actions:.4f}", flush=True)

    vocabulary = BPEVocabulary(tuple(int(value) for value in bins), tuple(merges))
    report: dict[str, object] = {
        "episodes": len(streams),
        "actions": actions,
        "atoms": atoms,
        "merges": len(merges),
        "vocab_size": vocabulary.vocab_size,
        "cells": initial_length,
        "tokens_per_action_initial": initial_length / actions,
        "tokens_per_action": length / actions,
        "compression": initial_length / length,
        "max_token_cells": int(max_token_cells),
        "max_token_cells_observed": observed_cells,
        "oversized_pairs_skipped": len(oversized),
        "trainer": "python_sparse_exact",
    }
    return vocabulary, report


def train_bpe(
    episodes: Iterable[np.ndarray],
    bins: Sequence[int],
    *,
    vocab_size: int,
    max_token_cells: int,
    backend: str = "auto",
    progress: bool = False,
    threads: int = 0,
) -> tuple[BPEVocabulary, dict[str, object]]:
    'Fit plain BPE on the row-major flattened primitive-cell stream.'
    if resolve_backend(backend) == "python":
        return _train_bpe_python(
            cell_chunks(episodes),
            bins,
            vocab_size=vocab_size,
            max_token_cells=max_token_cells,
            progress=progress,
        )
    merges, report = require_native().train_bpe(
        cell_chunks(episodes),
        [int(count) for count in bins],
        int(vocab_size),
        int(max_token_cells),
        int(threads),
    )
    vocabulary = BPEVocabulary(
        tuple(int(count) for count in bins),
        tuple((int(left), int(right)) for left, right in merges),
    )
    return vocabulary, dict(report)
