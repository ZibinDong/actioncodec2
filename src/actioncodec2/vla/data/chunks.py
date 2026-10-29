"""Fixed-horizon LeRobot samples for G0.5 training and evaluation."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import initial_seed
from torch.utils.data import Dataset

from actioncodec2.vla.data.augmentation import PreviousActionAugmentation
from actioncodec2.vla.data.lerobot import EpisodeRecord, LeRobotEpisodes
from actioncodec2.vla.data.splits import belongs_to_split

__all__ = ["ActionChunkDataset", "TokenTable"]


class TokenTable(Sequence):
    """Variable-length token rows stored as two flat arrays.

    Flat NumPy storage stays shared between forked DataLoader workers, whereas
    lists of Python ints are copied page by page as their refcounts change.
    """

    __slots__ = ("_values", "_offsets")

    def __init__(self, rows: Sequence[Sequence[int]]) -> None:
        lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
        self._offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._offsets[1:])
        self._values = np.fromiter(
            (token for row in rows for token in row),
            dtype=np.int32,
            count=int(self._offsets[-1]),
        )

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, index: int) -> list[int]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._values[self._offsets[index] : self._offsets[index + 1]].tolist()


@dataclass
class EpisodeCache:
    """Everything about one episode that does not need a video decoder."""

    record: EpisodeRecord
    #: Source action column, ``[T, D]`` (a state trajectory when ``action_offset=1``).
    actions: np.ndarray
    states: np.ndarray
    #: Tokens of the episode's sampled chunks, encoded on first use when no
    #: token cache is attached.
    tokens: TokenTable | None = None
    #: Boundary action at every step, computed on first use.
    previous_actions: np.ndarray | None = None


class ActionChunkDataset(Dataset):
    """Action-token chunks with their observations, addressed by a flat index."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        processor,
        tokenizer,
        *,
        horizon: int,
        video_keys: Sequence[str] | None = None,
        action_keys: Sequence[str] | None = None,
        state_keys: Sequence[str] | None = None,
        action_offset: int = 0,
        action_space: str = "delta_eef",
        action_rotation_representation: str = "rotation_vector",
        gripper_threshold: float = 0.5,
        episode_filter: Sequence[tuple[int, int]] | None = None,
        stride: int = 1,
        max_episodes: int | None = None,
        require_nonempty_task: bool = True,
        execution_stride: int | None = None,
        previous_action_source: str = "reconstructed",
        previous_action_augmentation: dict | None = None,
        augmentation_seed: int = 0,
        lanes: int = 8,
        max_readers: int | None = None,
        max_cached_episodes: int | None = None,
        table_cache: int | None = None,
        validation_fraction: float = 0.0,
        split: str = "all",
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.horizon = int(horizon)
        self.lanes = max(1, int(lanes))
        self.datasets = [
            LeRobotEpisodes(
                root,
                video_keys=video_keys,
                action_keys=action_keys,
                state_keys=state_keys,
                table_cache=int(table_cache if table_cache is not None else self.lanes),
            )
            for root in roots
        ]
        self.action_offset = int(action_offset)
        if self.action_offset not in {0, 1}:
            raise ValueError("action_offset must be 0 or 1")
        self.action_space = str(action_space).lower()
        self.action_rotation_representation = str(action_rotation_representation).lower()
        self.gripper_threshold = float(gripper_threshold)
        self.video_keys = self.datasets[0].video_keys
        for root, dataset in zip(roots, self.datasets, strict=True):
            if dataset.video_keys != self.video_keys:
                raise ValueError(
                    f"{root} exposes cameras {list(dataset.video_keys)}, but the first root "
                    f"exposes {list(self.video_keys)}; set data.video_keys to the shared subset"
                )
        self.max_readers = int(
            max_readers if max_readers is not None else self.lanes * max(1, len(self.video_keys))
        )
        if self.max_readers <= 0:
            raise ValueError("max_readers must be positive")
        self.stateful = bool(tokenizer.requires_previous_action)
        # A stateful codec defaults to re-planning every step, which keeps every
        # timestep a valid chunk start.
        self.execution_stride = int(execution_stride or (1 if self.stateful else self.horizon))
        self.previous_action_source = str(previous_action_source).lower()
        if self.previous_action_source not in {"reconstructed", "ground_truth"}:
            raise ValueError("previous_action_source must be 'reconstructed' or 'ground_truth'")
        if self.stateful and self.action_offset and self.previous_action_source == "reconstructed":
            raise ValueError(
                "derived actions (action_offset=1) require previous_action_source: ground_truth"
            )
        self.boundary_augmentation = (
            None
            if previous_action_augmentation is None
            else PreviousActionAugmentation(**previous_action_augmentation)
        )
        self.augmentation_seed = int(augmentation_seed)
        self._augmentation_rng: np.random.Generator | None = None
        self._augmentation_pid: int | None = None
        if self.boundary_augmentation is not None:
            if not hasattr(tokenizer, "encode_augmented_batch"):
                raise ValueError(
                    f"{tokenizer.name} does not support boundary augmentation "
                    "(it has no encode_augmented_batch)"
                )
            if not self.stateful or self.previous_action_source != "ground_truth":
                raise ValueError("boundary augmentation needs a stateful tokenizer and ground_truth boundaries")
            if self.action_offset:
                raise ValueError("boundary augmentation requires action_offset=0")
            if not processor.requires_previous_action:
                raise ValueError("boundary augmentation requires the model to receive previous_action")
            validate = getattr(tokenizer, "validate_boundary_augmentation", None)
            if validate is not None:
                validate()

        allowed = set(episode_filter) if episode_filter is not None else None
        self.entries: list[tuple[int, int]] = []
        lengths: list[int] = []
        for dataset_index, dataset in enumerate(self.datasets):
            records = dataset.episodes[:max_episodes] if max_episodes is not None else dataset.episodes
            for record in records:
                key = (dataset_index, record.index)
                if not belongs_to_split(record.index, dataset_index, validation_fraction, split):
                    continue
                if allowed is not None and key not in allowed:
                    continue
                if require_nonempty_task and not record.task.strip():
                    continue
                if record.length < self.horizon + self.action_offset:
                    continue
                self.entries.append(key)
                lengths.append(record.length)
        if not self.entries:
            raise ValueError("no episodes matched the requested filter")

        # A reconstructed boundary exists only where a rollout re-plans, i.e. at
        # multiples of ``execution_stride``; the sample grid is intersected with
        # it.  A tokenizer may also need ``context_steps`` of recorded history
        # before its first reconstructed chunk.  The grid is arithmetic, so no
        # episode is read to size the dataset.
        reconstructed = self.stateful and self.previous_action_source == "reconstructed"
        self.sample_offset = int(getattr(tokenizer, "context_steps", 0)) if reconstructed else 0
        self.stride = max(1, int(stride))
        if reconstructed:
            self.sample_stride = math.lcm(self.stride, min(self.execution_stride, self.horizon))
        else:
            self.sample_stride = self.stride
        valid_starts = np.asarray(lengths, dtype=np.int64) - self.horizon - self.action_offset + 1
        counts = np.maximum(
            0, (valid_starts - self.sample_offset + self.sample_stride - 1) // self.sample_stride
        )
        self.offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self._entry_of = {key: entry for entry, key in enumerate(self.entries)}

        self.max_cached_episodes = int(
            max_cached_episodes if max_cached_episodes is not None else self.lanes
        )
        if self.max_cached_episodes <= 0:
            raise ValueError("max_cached_episodes must be positive")
        self.caches: OrderedDict[tuple[int, int], EpisodeCache] = OrderedDict()
        self.token_store = None
        self._video_decoder_cache = None

    # ---------------------------------------------------------------- indexing

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, position: int) -> tuple[int, int, int]:
        """``position -> (root_index, episode_index, step)``."""
        if not 0 <= position < len(self):
            raise IndexError(position)
        entry = int(np.searchsorted(self.offsets, position, side="right") - 1)
        step = self.sample_offset + int(position - self.offsets[entry]) * self.sample_stride
        dataset_index, episode_index = self.entries[entry]
        return dataset_index, episode_index, step

    def position_of(self, dataset_index: int, episode_index: int, step: int) -> int:
        """Flat index of one ``(episode, step)`` on the sample grid."""
        entry = self._entry_of[(dataset_index, episode_index)]
        offset = step - self.sample_offset
        if offset < 0 or offset % self.sample_stride:
            raise IndexError(f"step {step} is not on the sample grid (stride {self.sample_stride})")
        return int(self.offsets[entry]) + offset // self.sample_stride

    def entry_roots(self) -> list[int]:
        """Root index of every entry, in entry order (for the balanced sampler)."""
        return [dataset_index for dataset_index, _ in self.entries]

    def sampled_starts(self, episode_length: int) -> np.ndarray:
        """Chunk starts an episode of ``episode_length`` contributes, in index order."""
        valid = episode_length - self.horizon - self.action_offset + 1
        return np.arange(self.sample_offset, max(valid, 0), self.sample_stride, dtype=np.int64)

    # --------------------------------------------------------------- episodes

    def _episode(self, dataset_index: int, episode_index: int) -> EpisodeCache:
        key = (dataset_index, episode_index)
        cached = self.caches.get(key)
        if cached is not None:
            self.caches.move_to_end(key)
            return cached
        dataset = self.datasets[dataset_index]
        try:
            record = dataset.by_index[episode_index]
        except KeyError:
            raise KeyError(f"episode {episode_index} is not in dataset {dataset_index}") from None
        actions, states = dataset.frames(record)
        cached = EpisodeCache(record=record, actions=actions, states=states)
        while len(self.caches) >= self.max_cached_episodes:
            self.caches.popitem(last=False)
        self.caches[key] = cached
        return cached

    def chunks_at(self, actions: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """Target chunks ``[len(starts), horizon, D]`` in training action units."""
        indices = starts[:, None] + self.action_offset + np.arange(self.horizon, dtype=np.int64)
        targets = actions[indices]
        if not self.action_offset:
            return targets
        return self._derived_actions(actions[indices - 1], targets)

    def _derived_actions(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Deltas between consecutive source states; the gripper stays absolute."""
        derived = np.asarray(current - previous, dtype=np.float32)
        derived[..., -1] = (current[..., -1] >= self.gripper_threshold).astype(np.float32)
        if self.action_space == "delta_eef" and self.action_rotation_representation == "rotation_vector":
            from actioncodec2.vla.data.so3 import relative_rotation_vectors_between

            derived[..., 3:6] = relative_rotation_vectors_between(
                previous[..., 3:6], current[..., 3:6]
            )
        return derived

    def recorded_previous_actions(self, actions: np.ndarray) -> np.ndarray:
        """Recorded training action preceding every step; zero at the episode start."""
        previous = np.zeros((len(actions), actions.shape[1]), dtype=np.float32)
        if len(actions) <= 1:
            return previous
        if self.action_offset:
            previous[1:] = self._derived_actions(actions[:-1], actions[1:])
        else:
            previous[1:] = actions[:-1]
        return previous

    def previous_actions(self, dataset_index: int, episode_index: int) -> np.ndarray:
        """Boundary action a stateful tokenizer continues from, at every step."""
        cache = self._episode(dataset_index, episode_index)
        if cache.previous_actions is None:
            if self.stateful and self.previous_action_source == "reconstructed":
                cache.previous_actions, _ = self.tokenizer.rollout_previous_actions(
                    cache.actions, execution_stride=self.execution_stride
                )
            else:
                cache.previous_actions = self.recorded_previous_actions(cache.actions)
        return cache.previous_actions

    def episode_chunks(
        self, dataset_index: int, episode_index: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(chunks, boundaries)`` for the episode's sampled starts.

        ``boundaries`` is ``None`` for tokenizers that decode a chunk on its own.
        """
        actions = self._episode(dataset_index, episode_index).actions
        starts = self.sampled_starts(len(actions))
        boundaries = None
        if self.stateful:
            boundaries = np.asarray(
                self.previous_actions(dataset_index, episode_index)[starts], dtype=np.float32
            )
        return self.chunks_at(actions, starts), boundaries

    # ----------------------------------------------------------------- tokens

    def use_token_cache(self, store) -> None:
        """Serve tokens (and stateful boundaries) from a :class:`ChunkTokenStore`."""
        if store is not None and self.boundary_augmentation is not None:
            raise ValueError("a fixed token cache cannot serve online boundary augmentation")
        if store is not None and store.samples != len(self):
            raise ValueError(
                f"the token cache holds {store.samples} chunks but this dataset has "
                f"{len(self)} samples"
            )
        self.token_store = store
        self.caches.clear()

    def tokens_at(self, position: int) -> list[int]:
        """Action tokens of one sample."""
        if self.token_store is not None:
            return self.token_store.tokens_at(position)
        dataset_index, episode_index, step = self.locate(position)
        cache = self._episode(dataset_index, episode_index)
        if cache.tokens is None:
            chunks, boundaries = self.episode_chunks(dataset_index, episode_index)
            cache.tokens = TokenTable(self.tokenizer.encode_batch(chunks, boundaries))
        return cache.tokens[(step - self.sample_offset) // self.sample_stride]

    def previous_action_at(self, position: int) -> np.ndarray:
        """Boundary action the sample's tokens were encoded against."""
        if self.token_store is not None and self.token_store.previous is not None:
            return np.asarray(self.token_store.previous[position], dtype=np.float32)
        dataset_index, episode_index, step = self.locate(position)
        return self.previous_actions(dataset_index, episode_index)[step]

    # ---------------------------------------------------------------- samples

    def _frames(self, dataset_index: int, episode_index: int, step: int) -> list[np.ndarray]:
        record = self._episode(dataset_index, episode_index).record
        if not 0 <= step < record.length:
            raise IndexError(f"frame {step} is outside episode {episode_index}")
        source = self.datasets[dataset_index]
        if source.video_backend == "torchcodec" and self._video_decoder_cache is None:
            from lerobot.datasets.video_utils import VideoDecoderCache

            self._video_decoder_cache = VideoDecoderCache(max_size=self.max_readers)
        return [
            source.video_frame(record, key, step, decoder_cache=self._video_decoder_cache)
            for key in self.video_keys
        ]

    def _model_previous_action(self, position: int) -> np.ndarray | None:
        if not self.processor.requires_previous_action:
            return None
        if self.stateful:
            return self.previous_action_at(position)
        # The model asked for a previous action the tokenizer does not use.
        dataset_index, episode_index, step = self.locate(position)
        actions = self._episode(dataset_index, episode_index).actions
        return self.recorded_previous_actions(actions[: step + 1])[step]

    def _sample(
        self,
        position: int,
        tokens: Sequence[int] | None = None,
        previous_action: np.ndarray | None = None,
    ) -> dict[str, object]:
        dataset_index, episode_index, step = self.locate(position)
        cache = self._episode(dataset_index, episode_index)
        return self.processor.build_sample(
            self._frames(dataset_index, episode_index, step),
            cache.record.task,
            self.tokens_at(position) if tokens is None else tokens,
            state=cache.states[step],
            previous_action=(
                self._model_previous_action(position) if previous_action is None else previous_action
            ),
        )

    def __getitem__(self, position: int) -> dict[str, object]:
        if self.boundary_augmentation is not None:
            return self.__getitems__([position])[0]
        return self._sample(position)

    def __getitems__(self, positions: Sequence[int]) -> list[dict[str, object]]:
        """Fetch a batch; boundary augmentation re-encodes the whole batch at once."""
        if self.boundary_augmentation is None:
            return [self._sample(position) for position in positions]
        chunks, previous, steps = [], [], []
        for position in positions:
            dataset_index, episode_index, step = self.locate(position)
            actions = self._episode(dataset_index, episode_index).actions
            chunks.append(self.chunks_at(actions, np.asarray([step], dtype=np.int64))[0])
            previous.append(self.previous_actions(dataset_index, episode_index)[step])
            steps.append(step)
        # Forked workers must not share one RNG stream; RANK separates DDP
        # processes whose DataLoader workers have identical torch seeds.
        if self._augmentation_pid != os.getpid():
            self._augmentation_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.augmentation_seed, int(os.environ.get("RANK", "0")), int(initial_seed())]
                )
            )
            self._augmentation_pid = os.getpid()
        tokens, boundaries, _ = self.tokenizer.encode_augmented_batch(
            np.asarray(chunks),
            np.asarray(previous),
            np.asarray(steps),
            self.boundary_augmentation,
            self._augmentation_rng,
        )
        return [
            self._sample(position, stream, np.asarray(boundary, dtype=np.float32))
            for position, stream, boundary in zip(positions, tokens, boundaries, strict=True)
        ]
