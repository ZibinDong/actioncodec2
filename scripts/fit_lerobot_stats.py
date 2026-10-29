"""Fit G0.5 proprio and action normalization statistics from LeRobot v3 datasets.

Writes ``proprio_stats.json`` (mean/std/q01/q99 of the state) and
``action_stats.json`` (q001/q999 of the action) into ``--output``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from actioncodec2.vla.data.splits import belongs_to_split


def _column(
    root: Path, name: str, *, root_index: int, validation_fraction: float
) -> np.ndarray:
    table = ds.dataset(root / "data", format="parquet")
    if name not in table.schema.names:
        raise ValueError(f"{root}: missing {name!r} column")
    columns = [name]
    if validation_fraction:
        if "episode_index" not in table.schema.names:
            raise ValueError(f"{root}: episode_index is required for a train/validation split")
        columns.append("episode_index")
    parts = []
    for batch in table.to_batches(columns=columns, batch_size=16384):
        column = batch.column(0)
        if validation_fraction:
            episodes = batch.column(1).to_numpy(zero_copy_only=False)
            unique, inverse = np.unique(episodes, return_inverse=True)
            selected = np.array([
                belongs_to_split(episode, root_index, validation_fraction, "train")
                for episode in unique
            ], dtype=bool)
            column = pc.filter(column, selected[inverse])
        rows = len(column)
        # Vector columns are lists of equal length; read them as one flat buffer
        # instead of materializing a Python list per row.
        if rows == 0:
            continue
        lengths = pc.list_value_length(column)
        width = lengths[0].as_py()
        if width is None or not pc.all(pc.equal(lengths, width)).as_py():
            raise ValueError(f"{root}: {name!r} must be a vector column of one width")
        values = column.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        values = values.reshape(rows, width)
        if not np.isfinite(values).all():
            raise ValueError(f"{root}: {name!r} must be a finite vector column")
        parts.append(values)
    if not parts:
        raise ValueError(f"{root}: no rows in {name!r}")
    return np.concatenate(parts)


def fit(
    roots: list[Path],
    output: Path,
    *,
    validation_fraction: float,
    gripper_threshold: float | None = None,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> None:
    validation_fraction = float(validation_fraction)
    belongs_to_split(0, 0, validation_fraction, "train")
    if not roots:
        raise ValueError("at least one dataset root is required")
    state = np.concatenate([
        _column(root, state_key, root_index=index, validation_fraction=validation_fraction)
        for index, root in enumerate(roots)
    ])
    action = np.concatenate([
        _column(root, action_key, root_index=index, validation_fraction=validation_fraction)
        for index, root in enumerate(roots)
    ])
    output.mkdir(parents=True, exist_ok=True)
    proprio = {
        "train_mean": state.mean(axis=0).tolist(),
        "train_std": state.std(axis=0).tolist(),
        "q01": np.quantile(state, 0.01, axis=0).tolist(),
        "q99": np.quantile(state, 0.99, axis=0).tolist(),
        "validation_fraction": validation_fraction,
    }
    actions = {
        "q001": np.quantile(action, 0.001, axis=0).tolist(),
        "q999": np.quantile(action, 0.999, axis=0).tolist(),
        "validation_fraction": validation_fraction,
    }
    if gripper_threshold is not None:
        actions["gripper_threshold"] = gripper_threshold
    for name, payload in (("proprio_stats.json", proprio), ("action_stats.json", actions)):
        (output / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="LeRobot dataset directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-fraction", type=float, required=True,
        help="match data.validation_fraction and pass roots in the same order; "
        "only training episodes contribute (0 explicitly uses every episode)",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        help="record the gripper threshold the statistics belong to; training then "
        "checks it against data.gripper_threshold",
    )
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    args = parser.parse_args()
    fit(
        args.roots,
        args.output,
        validation_fraction=args.validation_fraction,
        gripper_threshold=args.gripper_threshold,
        state_key=args.state_key,
        action_key=args.action_key,
    )


if __name__ == "__main__":
    main()
