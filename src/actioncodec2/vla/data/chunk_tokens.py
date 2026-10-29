"""Pre-encoded action-token streams, addressed by dataset position."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

__all__ = ["ChunkTokenStore", "ChunkTokenError"]

LAYOUT_VERSION = 2
MANIFEST_NAME = "manifest.json"
TOKENS_NAME = "tokens.npy"
LENGTHS_NAME = "lengths.npy"
PREVIOUS_NAME = "previous.npy"
PARTIAL_MANIFEST_NAME = "partial-manifest.json"


class ChunkTokenError(RuntimeError):
    """A learned-token cache is missing, stale or incomplete."""


def _file_digest(path: Path) -> str:
    """SHA-256 of a file's complete contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_stamp(tokenizer) -> dict[str, object] | None:
    """The tokenizer artifact's content, so a cache follows the files and not where they live."""
    value = getattr(tokenizer, "artifact", None)
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        return {"missing": str(value)}
    if path.is_file():
        return {path.name: _file_digest(path)}
    return {
        str(file.relative_to(path)): _file_digest(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _dataset_stamp(source) -> dict[str, object]:
    import pyarrow.parquet as pq

    root = Path(source.root)
    info = root / "meta" / "info.json"
    actions = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        digest = hashlib.sha256()
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=65536, columns=list(source.action_keys), use_threads=False
        ):
            digest.update(batch.schema.serialize())
            digest.update(batch.serialize())
        actions[path.relative_to(root).as_posix()] = digest.hexdigest()
    return {
        "name": root.name,
        "info": _file_digest(info) if info.is_file() else None,
        "action_keys": list(source.action_keys),
        "actions": actions,
        "episodes": [
            [record.index, record.length, list(record.data_range),
             record.data_file.relative_to(root).as_posix(), record.data_file_start]
            for record in source.episodes
        ],
    }


def _identity(dataset) -> tuple[dict[str, object], str]:
    try:
        tokenizer = dict(dataset.tokenizer.to_json())
    except ValueError:
        tokenizer = {
            "name": str(dataset.tokenizer.name),
            "time_horizon": int(dataset.tokenizer.time_horizon),
            "vocab_size": int(dataset.tokenizer.vocab_size),
        }
    # Compute placement cannot affect deterministic token ids.  Omitting it
    # lets an offline GPU cache be reused by the CPU tokenizer configuration
    # used in the distributed VLA dataloaders.
    tokenizer.pop("device", None)
    # Files are identified by content below, never by the path they were found at.
    tokenizer.pop("artifact", None)
    for key, value in list(tokenizer.items()):
        if (key.endswith("_path") or key == "action_stats") and isinstance(value, str):
            file = Path(value)
            tokenizer[key] = _file_digest(file) if file.is_file() else value
    payload = {
        "layout_version": LAYOUT_VERSION,
        "tokenizer": tokenizer,
        "artifact": _artifact_stamp(dataset.tokenizer),
        "horizon": int(dataset.horizon),
        "sample_stride": int(dataset.sample_stride),
        "roots": [_dataset_stamp(source) for source in dataset.datasets],
        "samples": len(dataset),
        "episodes": [[int(root), int(episode)] for root, episode in dataset.entries],
    }
    if dataset.sample_offset:
        payload["sample_offset"] = int(dataset.sample_offset)
    if dataset.stateful:
        payload["previous_action"] = {
            "source": dataset.previous_action_source,
            "execution_stride": int(dataset.execution_stride),
        }
    action_offset = int(dataset.action_offset)
    if action_offset:
        payload["action_offset"] = action_offset
        payload["derived_action"] = {
            "space": dataset.action_space,
            "rotation_representation": dataset.action_rotation_representation,
            "gripper_threshold": dataset.gripper_threshold,
            "version": 1,
        }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return payload, digest


class ChunkTokenStore:
    """Memory-mapped final token streams addressed by dataset position."""

    #: Seconds between checks while another process builds the cache.
    POLL_SECONDS = 2.0

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads(
            (self.directory / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.fingerprint = str(self.manifest["fingerprint"])
        self.samples = int(self.manifest["samples"])
        self.max_tokens = int(self.manifest["max_tokens"])
        self.vocab_size = int(self.manifest["vocab_size"])
        self.stateful = bool(self.manifest.get("stateful", False))
        self._tokens: np.ndarray | None = None
        self._lengths: np.ndarray | None = None
        self._previous: np.ndarray | None = None

    def __getstate__(self) -> dict:
        # Re-open the memory maps in each DataLoader worker.
        state = dict(self.__dict__)
        state["_tokens"] = state["_lengths"] = state["_previous"] = None
        return state

    @property
    def tokens(self) -> np.ndarray:
        if self._tokens is None:
            self._tokens = np.load(self.directory / TOKENS_NAME, mmap_mode="r")
        return self._tokens

    @property
    def lengths(self) -> np.ndarray:
        if self._lengths is None:
            self._lengths = np.load(self.directory / LENGTHS_NAME, mmap_mode="r")
        return self._lengths

    @property
    def previous(self) -> np.ndarray | None:
        """``[samples, action_dim]`` boundary actions, or ``None`` if stateless."""
        if not self.stateful:
            return None
        if self._previous is None:
            self._previous = np.load(self.directory / PREVIOUS_NAME, mmap_mode="r")
        return self._previous

    def tokens_at(self, position: int) -> list[int]:
        length = int(self.lengths[position])
        return self.tokens[position, :length].astype(np.int64, copy=False).tolist()

    @classmethod
    def directory_for(cls, dataset, cache_root: str | Path) -> tuple[Path, str]:
        _, fingerprint = _identity(dataset)
        name = Path(dataset.datasets[0].root).name
        return Path(cache_root) / "tokens" / f"{name}-{fingerprint[:16]}", fingerprint

    @classmethod
    def build(
        cls,
        dataset,
        directory: str | Path,
        *,
        verbose: bool = True,
        heartbeat: Callable[[], None] | None = None,
    ):
        """Materialize the token stream of every chunk into ``directory``.

        An interrupted build leaves partial arrays behind; the next call with the
        same fingerprint resumes from the rows already written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity, fingerprint = _identity(dataset)
        samples = len(dataset)
        maximum = int(dataset.tokenizer.max_tokens_per_chunk)
        token_dtype = np.uint16 if dataset.tokenizer.vocab_size <= 65536 else np.uint32
        length_dtype = np.uint16 if maximum <= 65535 else np.uint32
        stateful = bool(dataset.stateful)
        action_dim = int(dataset.tokenizer.action_dim)
        token_partial = directory / f"{TOKENS_NAME}.partial"
        length_partial = directory / f"{LENGTHS_NAME}.partial"
        previous_partial = directory / f"{PREVIOUS_NAME}.partial"
        partial_manifest = directory / PARTIAL_MANIFEST_NAME
        partial_identity = {
            "fingerprint": fingerprint,
            "samples": samples,
            "max_tokens": maximum,
            "token_dtype": np.dtype(token_dtype).str,
            "length_dtype": np.dtype(length_dtype).str,
        }
        partials = [token_partial, length_partial] + ([previous_partial] if stateful else [])

        resume = all(path.is_file() for path in partials) and partial_manifest.is_file()
        if resume:
            try:
                resume = (
                    json.loads(partial_manifest.read_text(encoding="utf-8"))
                    == partial_identity
                )
                tokens = np.load(token_partial, mmap_mode="r+")
                lengths = np.load(length_partial, mmap_mode="r+")
                previous = (
                    np.load(previous_partial, mmap_mode="r+") if stateful else None
                )
                resume = resume and (
                    tokens.shape == (samples, maximum)
                    and lengths.shape == (samples,)
                    and tokens.dtype == token_dtype
                    and lengths.dtype == length_dtype
                    and np.all(lengths <= maximum)
                    and (previous is None or previous.shape == (samples, action_dim))
                )
            except (OSError, ValueError):
                resume = False
        if not resume:
            for path in partials:
                path.unlink(missing_ok=True)
            tokens = np.lib.format.open_memmap(
                token_partial, mode="w+", dtype=token_dtype, shape=(samples, maximum)
            )
            lengths = np.lib.format.open_memmap(
                length_partial, mode="w+", dtype=length_dtype, shape=(samples,)
            )
            tokens[:] = 0
            lengths[:] = 0
            previous = None
            if stateful:
                previous = np.lib.format.open_memmap(
                    previous_partial, mode="w+", dtype=np.float32, shape=(samples, action_dim)
                )
                previous[:] = 0
        partial_manifest.write_text(
            json.dumps(partial_identity, sort_keys=True), encoding="utf-8"
        )
        written = int(np.count_nonzero(lengths))
        resumed_written = written
        if verbose and written:
            print(
                f"[chunk-tokens] resuming {written}/{samples} cached chunks",
                flush=True,
            )
        cache_batch_size = max(
            1, int(getattr(dataset.tokenizer, "cache_batch_size", 1))
        )
        pending_chunks: list[np.ndarray] = []
        pending_boundaries: list[np.ndarray] = []
        pending_positions: list[np.ndarray] = []
        pending_count = 0

        def flush_pending(token_rows: np.ndarray, length_rows: np.ndarray) -> int:
            nonlocal pending_count
            if not pending_count:
                return 0
            if heartbeat is not None:
                heartbeat()
            chunks = np.concatenate(pending_chunks, axis=0)
            positions = np.concatenate(pending_positions)
            boundaries = np.concatenate(pending_boundaries, axis=0) if stateful else None
            streams = dataset.tokenizer.encode_batch(chunks, boundaries)
            if len(streams) != len(positions):
                raise ChunkTokenError(
                    f"tokenizer produced {len(streams)} streams for "
                    f"{len(positions)} sample positions"
                )
            for position, stream in zip(positions, streams, strict=True):
                row = np.asarray(stream, dtype=np.int64)
                if not 1 <= len(row) <= maximum:
                    raise ChunkTokenError("a token stream has an invalid length")
                if row.min() < 0 or row.max() >= dataset.tokenizer.vocab_size:
                    raise ChunkTokenError("a token stream contains an out-of-range id")
                token_rows[position, : len(row)] = row
                length_rows[position] = len(row)
            if stateful:
                previous[positions] = boundaries
            count = len(positions)
            pending_chunks.clear()
            pending_boundaries.clear()
            pending_positions.clear()
            pending_count = 0
            if heartbeat is not None:
                heartbeat()
            return count

        started = time.time()
        for entry, (dataset_index, episode_index) in enumerate(dataset.entries):
            if heartbeat is not None:
                heartbeat()
            begin, end = int(dataset.offsets[entry]), int(dataset.offsets[entry + 1])
            missing = np.flatnonzero(lengths[begin:end] == 0)
            if not len(missing):
                continue
            chunks, boundaries = dataset.episode_chunks(dataset_index, episode_index)
            if len(chunks) != end - begin:
                raise ChunkTokenError(
                    f"episode {episode_index} produced {len(chunks)} chunks for "
                    f"{end - begin} sample positions"
                )
            pending_chunks.append(chunks[missing])
            if stateful:
                pending_boundaries.append(boundaries[missing])
            pending_positions.append(begin + missing)
            pending_count += len(missing)
            if pending_count >= cache_batch_size:
                written += flush_pending(tokens, lengths)
            if verbose and entry and entry % 100 == 0:
                rate = (written - resumed_written) / max(time.time() - started, 1e-9)
                print(
                    f"[chunk-tokens] {entry}/{len(dataset.entries)} episodes "
                    f"({rate:.0f} chunks/s)",
                    flush=True,
                )
        written += flush_pending(tokens, lengths)
        tokens.flush()
        lengths.flush()
        if previous is not None:
            previous.flush()
        previous = None
        del tokens, lengths
        if written != samples:
            raise ChunkTokenError(f"wrote {written} of {samples} token streams")
        token_partial.replace(directory / TOKENS_NAME)
        length_partial.replace(directory / LENGTHS_NAME)
        if stateful:
            previous_partial.replace(directory / PREVIOUS_NAME)
        manifest = {
            "fingerprint": fingerprint,
            "samples": samples,
            "max_tokens": maximum,
            "vocab_size": int(dataset.tokenizer.vocab_size),
            "stateful": stateful,
            "identity": identity,
        }
        temporary = directory / f"{MANIFEST_NAME}.tmp"
        temporary.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        temporary.replace(directory / MANIFEST_NAME)
        partial_manifest.unlink(missing_ok=True)
        if verbose:
            print(
                f"[chunk-tokens] built {directory} in {time.time() - started:.1f}s",
                flush=True,
            )
        return cls(directory)

    @classmethod
    def _open_if_ready(cls, directory: Path, fingerprint: str):
        if not (directory / MANIFEST_NAME).exists():
            return None
        store = cls(directory)
        return store if store.fingerprint == fingerprint else None

    @staticmethod
    def _recover_stale_lock(lock: Path, stale_lock_timeout: float, verbose: bool) -> bool:
        """Remove ``lock`` if its owner stopped refreshing it; whether it was removed."""
        # Renaming is atomic, so of several waiters exactly one takes the lock apart.
        graveyard = lock.with_name(f"{lock.name}.stale-{os.getpid()}")
        try:
            os.rename(lock, graveyard)
        except FileNotFoundError:
            return False
        age = time.time() - graveyard.stat().st_mtime
        if age <= stale_lock_timeout:
            # Another waiter replaced the stale lock with a live one first.
            try:
                os.rename(graveyard, lock)
            except OSError:
                graveyard.unlink(missing_ok=True)
            return False
        graveyard.unlink(missing_ok=True)
        if verbose:
            print(f"[chunk-tokens] recovered stale lock {lock} ({age:.0f}s old)", flush=True)
        return True

    @classmethod
    def ensure(
        cls,
        dataset,
        cache_root: str | Path,
        *,
        timeout: float = 43200.0,
        stale_lock_timeout: float = 900.0,
        verbose: bool = True,
    ):
        """The cache for ``dataset``: found, built here, or built by another process."""
        if stale_lock_timeout <= 0:
            raise ValueError("stale_lock_timeout must be positive")
        directory, fingerprint = cls.directory_for(dataset, cache_root)
        directory.parent.mkdir(parents=True, exist_ok=True)
        lock = directory.with_name(directory.name + ".lock")
        deadline = time.time() + timeout
        announced = False
        while True:
            store = cls._open_if_ready(directory, fingerprint)
            if store is not None:
                return store
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - lock.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > stale_lock_timeout:
                    cls._recover_stale_lock(lock, stale_lock_timeout, verbose)
                    continue
                if verbose and not announced:
                    print(f"[chunk-tokens] waiting for {directory}", flush=True)
                    announced = True
                if time.time() >= deadline:
                    raise ChunkTokenError(f"timed out waiting for {directory}") from None
                time.sleep(cls.POLL_SECONDS)
                continue
            try:
                os.write(handle, f"{socket.gethostname()}:{os.getpid()}\n".encode())
                os.close(handle)
                # A builder may have finished between our check and taking the lock.
                store = cls._open_if_ready(directory, fingerprint)
                if store is not None:
                    return store
                last_heartbeat = 0.0

                def heartbeat() -> None:
                    nonlocal last_heartbeat
                    now = time.monotonic()
                    if now - last_heartbeat < 30.0:
                        return
                    try:
                        os.utime(lock, None)
                    except FileNotFoundError as error:
                        raise ChunkTokenError(
                            f"cache-build lock disappeared: {lock}"
                        ) from error
                    last_heartbeat = now

                heartbeat()
                return cls.build(
                    dataset,
                    directory,
                    verbose=verbose,
                    heartbeat=heartbeat,
                )
            finally:
                lock.unlink(missing_ok=True)
