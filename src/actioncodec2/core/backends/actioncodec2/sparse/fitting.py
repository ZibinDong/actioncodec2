"""Profile-level ActionCodec2Sparse fitting and default profile derivation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import numpy as np

from ..configuration import ActionCodec2Config, ConfigError
from ..physical_codec import PhysicalCodec
from ..primitives import PrimitiveStore
from ..training import _quantize_trajectories, build_primitives
from ..trajectory import TrajectoryLike
from .layout import SparseLayout, training_chunks
from .processor import ActionCodec2SparseProcessor
from .training import fit_sparse_vocabulary

DEFAULT_VOCAB_SIZE = 4096
DEFAULT_CHUNK_HORIZON = 32
DEFAULT_STATIC_RUN_CAP = 8


def sparse_profile_config(
    config: ActionCodec2Config,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    chunk_horizon: int = DEFAULT_CHUNK_HORIZON,
    static_run_cap: int = DEFAULT_STATIC_RUN_CAP,
) -> ActionCodec2Config:
    """Derive an ActionCodec2Sparse profile from a first- or second-order ActionCodec2 profile."""
    if config.train is None:
        raise ConfigError("a sparse profile is derived from a config with a train section")
    payload = config.to_dict()
    payload["set_bpe"]["scheme"] = "sparse_set_bpe"
    for component in payload["physical"]["components"].values():
        if component["space"] == "so3" and component["primitive_order"] == 1:
            component["primitive_selection"] = "momentum_lazy"
    layout = SparseLayout.from_config(config, build_primitives(config))
    payload["train"]["bpe"] = {
        "spatial_vocab_size": sum(layout.sparse_bins),
        "vocab_size": int(vocab_size),
    }
    payload["train"]["sparse"] = {
        "chunk_horizon": int(chunk_horizon),
        "static_run_cap": int(static_run_cap),
    }
    return ActionCodec2Config.from_dict(payload)


class SparseProfileTrainer:
    """Fit one ActionCodec2Sparse profile on complete canonical trajectories."""

    def __init__(self, config: ActionCodec2Config) -> None:
        if config.set_bpe.scheme != "sparse_set_bpe" or config.train is None:
            raise ConfigError("SparseProfileTrainer needs a sparse_set_bpe config with train")
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
    ) -> tuple[ActionCodec2SparseProcessor, dict]:
        """Quantize then fit; see :meth:`fit_cells` for the token-layer stage."""
        started = time.perf_counter()
        primitives = build_primitives(self.config)
        cells, actions, active_values, arities = _quantize_trajectories(
            trajectories,
            PhysicalCodec(self.config, primitives),
            batch_size=batch_size,
            max_cell_bytes=max_cell_bytes,
        )
        quantized = time.perf_counter()
        processor, report = self.fit_cells(
            cells, primitives=primitives, backend=backend, threads=threads, progress=progress
        )
        report.update(
            trajectories=len(cells),
            actions=actions,
            active_values=active_values,
            trajectory_arities={str(k): v for k, v in arities.items()},
        )
        report["seconds"]["quantize"] = quantized - started
        return processor, report

    def fit_cells(
        self,
        cells: Sequence[np.ndarray],
        *,
        primitives: PrimitiveStore | None = None,
        backend: str = "auto",
        threads: int = 0,
        progress: bool = False,
    ) -> tuple[ActionCodec2SparseProcessor, dict]:
        """Fit the token layer on quantized cell trajectories ``[(T_i, D)]`` (``-1`` inactive)."""
        config = self.config
        primitives = build_primitives(config) if primitives is None else primitives
        layout = SparseLayout.from_config(config, primitives)
        options = config.train.sparse
        chunks = training_chunks(
            cells, layout, horizon=options.chunk_horizon, static_run_cap=options.static_run_cap
        )
        vocabulary, report = fit_sparse_vocabulary(
            layout,
            chunks,
            spatial_vocab_size=config.train.bpe.spatial_vocab_size,
            vocab_size=config.train.bpe.vocab_size,
            lookahead=config.set_bpe.lookahead,
            backend=backend,
            threads=threads,
            progress=progress,
        )
        report["corpus"] = {
            "trajectories": len(cells),
            "source_frames": int(sum(len(c) for c in cells)),
            "training_frames": report["frames"],
            "chunk_horizon": options.chunk_horizon,
            "static_run_cap": options.static_run_cap,
        }
        report["tokenization_scheme"] = "sparse_set_bpe"
        return ActionCodec2SparseProcessor(config, primitives, vocabulary), report
