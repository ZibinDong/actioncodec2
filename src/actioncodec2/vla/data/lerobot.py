"""LeRobot dataset adapter for G05's episode-wise action chunks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["EpisodeRecord", "LeRobotEpisodes"]


@dataclass(frozen=True)
class EpisodeRecord:
    index: int
    length: int
    task: str
    data_range: tuple[int, int]
    data_file: Path
    data_file_start: int
    video_files: dict[str, Path]
    video_start_seconds: dict[str, float]


class LeRobotEpisodes:
    """Episode index backed by LeRobot metadata and its video decoder."""

    def __init__(
        self,
        root: str | Path,
        *,
        video_keys: Sequence[str] | None = None,
        action_key: str = "action",
        state_key: str = "observation.state",
        action_keys: Sequence[str] | None = None,
        state_keys: Sequence[str] | None = None,
        table_cache: int = 8,
    ) -> None:
        from lerobot.datasets import LeRobotDatasetMetadata
        from lerobot.utils.import_utils import get_safe_default_video_backend

        self.root = Path(root)
        self.repo_id = f"local/{self.root.name}"
        self.meta = LeRobotDatasetMetadata(self.repo_id, root=self.root, token=False)
        self.action_keys = tuple(action_keys) if action_keys else (action_key,)
        self.state_keys = tuple(state_keys) if state_keys else (state_key,)
        self.action_key, self.state_key = self.action_keys[0], self.state_keys[0]
        self.camera_keys = tuple(self.meta.camera_keys)
        self.video_keys = tuple(video_keys) if video_keys else self.camera_keys
        missing = [key for key in self.video_keys if key not in self.camera_keys]
        if missing:
            raise KeyError(f"{self.root} has no camera {missing}; it declares {list(self.camera_keys)}")
        self.fps = float(self.meta.fps)
        self.video_backend = get_safe_default_video_backend()

        has_tasks = "tasks" in self.meta.episodes.column_names
        keys = ["episode_index", "length", "data/chunk_index",
                "data/file_index", "dataset_from_index", "dataset_to_index"]
        if has_tasks:
            keys.append("tasks")
        for key in self.video_keys:
            keys.extend((f"videos/{key}/chunk_index", f"videos/{key}/file_index",
                         f"videos/{key}/from_timestamp"))
        rows = self.meta.episodes.select_columns(keys).to_dict()
        file_of = list(zip(rows["data/chunk_index"], rows["data/file_index"], strict=True))
        file_start: dict[tuple[int, int], int] = {}
        for key, start in zip(file_of, rows["dataset_from_index"], strict=True):
            file_start[key] = min(file_start.get(key, int(start)), int(start))

        if not has_tasks:
            # Some v3 datasets record the instruction only per frame, in
            # ``task_index`` (names live in ``meta/tasks``).
            rows["tasks"] = self._first_frame_tasks(rows, file_of, file_start)

        self.episodes: list[EpisodeRecord] = []
        for pos, (chunk, file) in enumerate(file_of):
            tasks = rows["tasks"][pos]
            self.episodes.append(EpisodeRecord(
                index=int(rows["episode_index"][pos]),
                length=int(rows["length"][pos]),
                task=str(tasks[0]) if tasks else "",
                data_range=(int(rows["dataset_from_index"][pos]),
                            int(rows["dataset_to_index"][pos])),
                data_file=self.root / self.meta.data_path.format(chunk_index=chunk, file_index=file),
                data_file_start=file_start[(chunk, file)],
                video_files={key: self.root / self.meta.video_path.format(
                    video_key=key,
                    chunk_index=rows[f"videos/{key}/chunk_index"][pos],
                    file_index=rows[f"videos/{key}/file_index"][pos],
                ) for key in self.video_keys},
                video_start_seconds={key: float(rows[f"videos/{key}/from_timestamp"][pos])
                                     for key in self.video_keys},
            ))
        self.by_index = {record.index: record for record in self.episodes}
        self.table_cache = max(1, int(table_cache))
        self._tables: OrderedDict[Path, object] = OrderedDict()

    def _first_frame_tasks(
        self, rows: dict, file_of: list[tuple[int, int]], file_start: dict[tuple[int, int], int]
    ) -> list[list[str]]:
        import pyarrow.parquet as pq

        names = {int(index): str(task) for task, index in self.meta.tasks["task_index"].items()}
        indices: dict[tuple[int, int], np.ndarray] = {}
        tasks = []
        for (chunk, file), start in zip(file_of, rows["dataset_from_index"], strict=True):
            if (chunk, file) not in indices:
                path = self.root / self.meta.data_path.format(chunk_index=chunk, file_index=file)
                column = pq.read_table(path, columns=["task_index"]).column("task_index")
                indices[(chunk, file)] = column.to_numpy()
            tasks.append([names[int(indices[(chunk, file)][int(start) - file_start[(chunk, file)]])]])
        return tasks

    def __len__(self) -> int:
        return len(self.episodes)

    def _table(self, path: Path):
        import pyarrow.parquet as pq

        table = self._tables.get(path)
        if table is None:
            columns = list(dict.fromkeys((*self.action_keys, *self.state_keys)))
            table = pq.read_table(path, columns=columns)
            while len(self._tables) >= self.table_cache:
                self._tables.popitem(last=False)
            self._tables[path] = table
        self._tables.move_to_end(path)
        return table

    def frames(self, episode: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """Read only an episode's action and state columns, without video."""
        start, stop = episode.data_range
        local = start - episode.data_file_start
        take = stop - start
        table = self._table(episode.data_file)
        if local < 0 or local + take > table.num_rows:
            raise RuntimeError(f"episode {episode.index} has invalid data range {start}:{stop}")
        block = table.slice(local, take)

        def concatenate(keys: tuple[str, ...]) -> np.ndarray:
            parts = [np.stack(block.column(key).to_numpy(zero_copy_only=False))
                     .astype(np.float32).reshape(take, -1) for key in keys]
            return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

        return concatenate(self.action_keys), concatenate(self.state_keys)

    def video_frame(self, episode: EpisodeRecord, key: str, step: int, *, decoder_cache=None) -> np.ndarray:
        """Decode one selected camera through LeRobot's video backend."""
        from lerobot.datasets.video_utils import decode_video_frames, decode_video_frames_torchcodec

        timestamp = episode.video_start_seconds[key] + step / self.fps
        path = episode.video_files[key]
        if self.video_backend == "torchcodec":
            frames = decode_video_frames_torchcodec(
                path, [timestamp], 1e-4, decoder_cache=decoder_cache, return_uint8=True,
            )
        else:
            frames = decode_video_frames(
                path, [timestamp], 1e-4, backend=self.video_backend, return_uint8=True,
            )
        return frames[0].permute(1, 2, 0).numpy()
