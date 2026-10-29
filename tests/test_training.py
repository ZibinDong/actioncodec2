"""Dataset, token cache, checkpoint recovery, and distributed training contracts."""

from __future__ import annotations

import socket
import importlib.util
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from actioncodec2.vla.data import ActionChunkDataset, ChunkTokenStore
from actioncodec2.vla.data import lerobot as source_module
from actioncodec2.vla.data.lerobot import EpisodeRecord
from actioncodec2.vla.train import (
    DDP_OPTIONS, _accumulate_gradients, _make_scheduler, _restore_optimizer_state,
    _save_optimizer_state, _warm_start_model,
)
from actioncodec2.vla.training.config import DataConfig, RunConfig, TrainConfig
from actioncodec2.vla.training.distributed import Distributed
from actioncodec2.vla.training.tracking import tracking_config

HORIZON = 3
EPISODE_LENGTHS = (6, 8)
WORLD_SIZE = 2
MICROBATCHES = 3


@pytest.fixture(scope="session")
def lerobot_root(tmp_path_factory) -> Path:
    """A tiny LeRobot v3 dataset, written once per session.

    Episode ``e`` has actions ``[e, step]``, states ``[step, e]`` and the task
    ``"task e"``; it has no videos.  Tests that modify it must copy it first.
    """
    from lerobot.datasets import LeRobotDataset

    root = tmp_path_factory.mktemp("lerobot") / "dataset"
    features = {
        key: {"dtype": "float32", "shape": [2], "names": ["x", "y"]}
        for key in ("action", "observation.state")
    }
    writer = LeRobotDataset.create(
        "local/shared-fixture", fps=10, features=features, root=root, use_videos=False
    )
    for episode, length in enumerate(EPISODE_LENGTHS):
        for step in range(length):
            writer.add_frame({
                "action": np.array([episode, step], dtype=np.float32),
                "observation.state": np.array([step, episode], dtype=np.float32),
                "task": f"task {episode}",
            })
        writer.save_episode()
    writer.finalize()
    return root


class _Episodes:
    """In-memory stand-in for ``LeRobotEpisodes``: episode ``e`` has actions ``[e, step]``.

    Real LeRobot reading is checked separately below.
    """

    video_keys: tuple[str, ...] = ()
    action_keys = ("action",)
    video_backend = "none"
    fps = 10.0

    def __init__(self, root, **_) -> None:
        self.root = root
        self.episodes = [
            EpisodeRecord(
                index=index, length=length, task="move", data_range=(0, length),
                data_file=root, data_file_start=0, video_files={}, video_start_seconds={},
            )
            for index, length in enumerate(EPISODE_LENGTHS)
        ]
        self.by_index = {record.index: record for record in self.episodes}

    def frames(self, record):
        actions = np.array([[record.index, step] for step in range(record.length)], np.float32)
        return actions, actions.copy()


@pytest.fixture
def _in_memory_episodes(monkeypatch) -> None:
    monkeypatch.setattr("actioncodec2.vla.data.chunks.LeRobotEpisodes", _Episodes)


class _PositionTokenizer:
    """Encodes each chunk as ``[start step, boundary step]`` read from the actions."""

    name = "position"
    vocab_size = 64
    action_dim = 2
    time_horizon = HORIZON
    max_tokens_per_chunk = 2
    cache_batch_size = 4

    def __init__(self, stateful: bool) -> None:
        self.requires_previous_action = stateful

    def encode_batch(self, chunks, previous_actions=None):
        chunks = np.asarray(chunks)
        assert chunks.shape[1:] == (HORIZON, self.action_dim)
        if self.requires_previous_action:
            assert previous_actions is not None and len(previous_actions) == len(chunks)
            boundary = [int(row[1]) for row in previous_actions]
        else:
            assert previous_actions is None
            boundary = [0] * len(chunks)
        return [[int(chunk[0, 1]), step] for chunk, step in zip(chunks, boundary, strict=True)]

    def rollout_previous_actions(self, actions, *, execution_stride=1):
        return np.zeros_like(actions), np.ones(len(actions), dtype=bool)

    def to_json(self):
        return {"name": self.name, "stateful": self.requires_previous_action}


class _Processor:
    def __init__(self, requires_previous_action: bool) -> None:
        self.requires_previous_action = requires_previous_action

    def build_sample(self, frames, task, action_tokens, *, state=None, previous_action=None):
        return {"tokens": list(action_tokens), "previous_action": previous_action}


def _dataset(root, *, stateful: bool, **kwargs) -> ActionChunkDataset:
    return ActionChunkDataset(
        [root],
        _Processor(requires_previous_action=stateful),
        _PositionTokenizer(stateful),
        horizon=HORIZON,
        previous_action_source="ground_truth",
        **kwargs,
    )


@pytest.mark.usefixtures("_in_memory_episodes")
@pytest.mark.parametrize("stateful", [False, True])
def test_token_cache_matches_on_demand_encoding(tmp_path, stateful) -> None:
    on_demand = _dataset(tmp_path, stateful=stateful)
    cached = _dataset(tmp_path, stateful=stateful)
    store = ChunkTokenStore.ensure(cached, tmp_path, verbose=False)
    cached.use_token_cache(store)
    assert store.stateful == stateful
    assert len(cached) == sum(length - HORIZON + 1 for length in EPISODE_LENGTHS)
    for position in range(len(cached)):
        _, _, step = cached.locate(position)
        expected = [step, max(step - 1, 0) if stateful else 0]
        assert on_demand.tokens_at(position) == expected
        assert cached.tokens_at(position) == expected
        sample = cached[position]
        assert sample["tokens"] == expected
        if stateful:
            np.testing.assert_array_equal(
                cached.previous_action_at(position), on_demand.previous_action_at(position)
            )
            # The boundary is the recorded action before the chunk, zero at t=0.
            assert sample["previous_action"][1] == max(step - 1, 0)
        else:
            assert sample["previous_action"] is None
    reused = ChunkTokenStore.ensure(cached, tmp_path, verbose=False)
    assert reused.directory == store.directory
    if stateful:
        other = _dataset(tmp_path, stateful=True, execution_stride=2)
        assert ChunkTokenStore.directory_for(other, tmp_path)[1] != store.fingerprint



class _AugmentingTokenizer(_PositionTokenizer):
    """Shifts every non-initial boundary by a fixed amount and encodes against it."""

    def __init__(self) -> None:
        super().__init__(stateful=True)
        self.calls = 0

    def encode_augmented_batch(self, chunks, previous, steps, augmentation, rng):
        self.calls += 1
        shifted = np.asarray(previous, np.float32).copy()
        shifted[np.asarray(steps) > 0, 1] += 10
        return self.encode_batch(chunks, shifted), shifted, {}


@pytest.mark.usefixtures("_in_memory_episodes")
def test_boundary_augmentation_re_encodes_against_the_perturbed_boundary(tmp_path):
    tokenizer = _AugmentingTokenizer()
    dataset = ActionChunkDataset(
        [tmp_path],
        _Processor(requires_previous_action=True),
        tokenizer,
        horizon=HORIZON,
        previous_action_source="ground_truth",
        previous_action_augmentation={"sigmas": [0.1]},
    )
    positions = list(range(len(dataset)))
    samples = dataset.__getitems__(positions)
    assert tokenizer.calls == 1  # one batched encode per fetch
    for position, sample in zip(positions, samples, strict=True):
        _, _, step = dataset.locate(position)
        boundary = max(step - 1, 0) + (10 if step else 0)
        # Labels and the model's previous-action input use the same boundary.
        assert sample["tokens"] == [step, boundary]
        assert sample["previous_action"][1] == boundary
    with pytest.raises(ValueError, match="online boundary augmentation"):
        dataset.use_token_cache(ChunkTokenStore.ensure(_dataset(tmp_path, stateful=True), tmp_path, verbose=False))


@pytest.mark.slow
def test_lerobot_episode_actions_states_and_tasks(lerobot_root):
    source = source_module.LeRobotEpisodes(lerobot_root)
    assert len(source) == 2
    assert source.camera_keys == ()
    for episode, length in enumerate((6, 8)):
        record = source.by_index[episode]
        actions, states = source.frames(record)
        assert record.length == length and record.task == f"task {episode}"
        np.testing.assert_array_equal(actions[:, 0], episode)
        np.testing.assert_array_equal(actions[:, 1], np.arange(length))
        np.testing.assert_array_equal(states[:, 0], np.arange(length))
        np.testing.assert_array_equal(states[:, 1], episode)


@pytest.mark.slow
def test_cache_fingerprint_tracks_real_action_columns_and_contents(lerobot_root, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from actioncodec2.vla.data.chunk_tokens import _dataset_stamp

    root = tmp_path / "dataset"
    shutil.copytree(lerobot_root, root)
    source = source_module.LeRobotEpisodes(root)
    original = _dataset_stamp(source)
    other = source_module.LeRobotEpisodes(root, action_keys=["observation.state"])
    assert _dataset_stamp(other) != original
    path = source.episodes[0].data_file
    stat = path.stat()
    table = pq.read_table(path)
    values = [np.asarray(row) + 10 for row in table["action"].to_pylist()]
    changed = table.set_column(
        table.schema.get_field_index("action"), table.schema.field("action"),
        pa.array(values, type=table.schema.field("action").type),
    )
    pq.write_table(changed, path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert _dataset_stamp(source) != original


@pytest.mark.slow
def test_stats_fit_excludes_validation_episodes(lerobot_root, tmp_path):
    from actioncodec2.vla.data.splits import belongs_to_split

    script = Path(__file__).resolve().parents[1] / "scripts/fit_lerobot_stats.py"
    spec = importlib.util.spec_from_file_location("fit_lerobot_stats", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fraction = 0.1
    source = source_module.LeRobotEpisodes(lerobot_root)
    selected = [source.frames(record) for record in source.episodes
                if belongs_to_split(record.index, 0, fraction, "train")]
    assert selected and len(selected) < len(source.episodes)
    output = tmp_path / "stats"
    module.fit([lerobot_root], output, validation_fraction=fraction)
    states = np.concatenate([state for _, state in selected])
    actions = np.concatenate([action for action, _ in selected])
    proprio = json.loads((output / "proprio_stats.json").read_text())
    bounds = json.loads((output / "action_stats.json").read_text())
    np.testing.assert_allclose(proprio["train_mean"], states.mean(axis=0))
    np.testing.assert_allclose(bounds["q999"], np.quantile(actions, 0.999, axis=0))
    assert proprio["validation_fraction"] == bounds["validation_fraction"] == fraction


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_training_values_are_rejected(value):
    from actioncodec2.tokenizers.base import QuantileNormalizer
    from actioncodec2.tokenizers.actioncodec import ActionCodecNormalizer
    from actioncodec2.tokenizers.xtokenizer import ActionOutputCalibration

    for field in ("learning_rate", "weight_decay", "max_grad_norm", "min_lr_ratio",
                  "decay_fraction", "dataloader_timeout_seconds"):
        with pytest.raises(ValueError, match="finite"):
            TrainConfig(output_dir="unused", **{field: value})
    with pytest.raises(ValueError, match="finite"):
        DataConfig(roots=[], control_frequency_hz=value)
    with pytest.raises(ValueError, match="finite"):
        QuantileNormalizer([value], [1])
    with pytest.raises(ValueError, match="finite"):
        ActionCodecNormalizer([value, 0], [1, 1])
    with pytest.raises(ValueError, match="finite"):
        ActionOutputCalibration(
            np.full((1, 6, 6), value), np.zeros((1, 6)), np.array([0.5]),
            clip_low=-np.ones(6), clip_high=np.ones(6),
        )


def test_unknown_run_sections_do_not_silently_disable_tracking():
    config = RunConfig.load(
        Path(__file__).resolve().parents[1] / "configs/train/libero/fast.yaml", data_root="/tmp"
    ).to_json()
    config["traking"] = config.pop("tracking")
    with pytest.raises(ValueError, match="unknown run settings: traking"):
        RunConfig.from_dict(config)


def test_tracking_config_redacts_paths_and_credentials_without_changing_settings(tmp_path):
    source = {
        "model": {"path": str(tmp_path / "weights"), "dtype": "bfloat16"},
        "data": {"roots": [str(tmp_path / "data")]},
        "tokenizer": {"artifact": "example/public-model", "revision": "a" * 40},
        "train": {"learning_rate": 2.5e-5, "seed": 42, "output_dir": "outputs/run"},
        "auth": {"api_key": "sensitive", "credentials": {"value": "sensitive"}},
    }
    original = json.loads(json.dumps(source))
    public = tracking_config(source)
    assert source == original
    assert public["model.path"] == public["data.roots"] == public["train.output_dir"] == "<local-path>"
    assert public["auth.api_key"] == public["auth.credentials.value"] == "<redacted>"
    assert public["train.learning_rate"] == 2.5e-5 and public["train.seed"] == 42
    assert public["tokenizer.artifact"] == "example/public-model"
    assert public["tokenizer.revision"] == "a" * 40


def _linear_recipe(steps: int, schedule_steps: int | None = None) -> TrainConfig:
    return TrainConfig(
        output_dir="unused", steps=steps, schedule_steps=schedule_steps,
        warmup_steps=5, learning_rate=1e-3, min_lr_ratio=0.1,
    )


def test_a_resumed_schedule_continues_the_original_curve() -> None:
    def rates(resume_step: int, upto: int) -> list[float]:
        config = RunConfig.__new__(RunConfig)
        config.train = _linear_recipe(upto, schedule_steps=upto)
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=config.train.learning_rate)
        scheduler, _ = _make_scheduler(optimizer, config, resume_step=resume_step)
        seen = []
        for _ in range(resume_step, upto):
            seen.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return seen

    assert rates(30, 60) == pytest.approx(rates(0, 60)[30:])


def test_warm_start_copies_every_tensor_and_rejects_mismatches(tmp_path) -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    save_file(source.state_dict(), str(tmp_path / "model.safetensors"))
    (tmp_path / "_SUCCESS").write_text("ready\n")
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    _warm_start_model(target, tmp_path)
    for restored, original in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(restored, original)

    with pytest.raises(ValueError, match="does not match"):
        _warm_start_model(torch.nn.Sequential(torch.nn.Linear(3, 4)), tmp_path)
    with pytest.raises(ValueError, match="shape"):
        _warm_start_model(torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2)), tmp_path)


def test_optimizer_state_survives_a_checkpoint_round_trip(tmp_path) -> None:
    def trained() -> tuple[torch.nn.Linear, torch.optim.AdamW]:
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.ones(4, 3)).square().mean().backward()
            optimizer.step()
        return model, optimizer

    model, optimizer = trained()
    assert not _restore_optimizer_state(optimizer, tmp_path)  # nothing saved yet
    _save_optimizer_state(tmp_path, optimizer)

    fresh = torch.optim.AdamW(model.parameters(), lr=1e-2)
    assert _restore_optimizer_state(fresh, tmp_path)
    for parameter in model.parameters():
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(fresh.state[parameter][name], optimizer.state[parameter][name])


class _Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, features, action_labels, action_label_mask):
        logits = self.linear(features)
        mask = action_label_mask.bool()
        loss = F.cross_entropy(logits[mask], action_labels[mask])
        return SimpleNamespace(loss=loss, logits=logits)


def _data() -> list[list[dict[str, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(1)
    ranks = []
    for _ in range(WORLD_SIZE):
        batches = []
        for _ in range(MICROBATCHES):
            mask = torch.rand(4, 5, generator=generator) < 0.6
            mask[0, 0] = True
            batches.append({
                "features": torch.randn(4, 5, 3, generator=generator),
                "action_labels": torch.randint(0, 4, (4, 5), generator=generator),
                "action_label_mask": mask,
            })
        ranks.append(batches)
    return ranks


def _model() -> _Head:
    torch.manual_seed(0)
    return _Head()


def _flat_grad(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.flatten() for parameter in model.parameters()])


def _worker(rank: int, port: int, output: str) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD_SIZE
    )
    try:
        data = _data()
        model = _model()
        parallel = DistributedDataParallel(model, **DDP_OPTIONS)
        counts = [[int(b["action_label_mask"].sum()) for b in rank_data] for rank_data in data]
        _accumulate_gradients(
            parallel,
            data[rank],
            counts[rank],
            global_tokens=float(sum(map(sum, counts))),
            context=Distributed(rank=rank, world_size=WORLD_SIZE),
            device="cpu",
            compute=nullcontext,
        )
        torch.save(_flat_grad(model), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_two_ranks_accumulating_with_no_sync_match_the_global_token_mean(tmp_path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(port, str(tmp_path)), nprocs=WORLD_SIZE)
    gradients = [torch.load(tmp_path / f"rank{rank}.pt") for rank in range(WORLD_SIZE)]

    reference = _model()
    batches = [batch for rank_data in _data() for batch in rank_data]
    tokens = sum(int(batch["action_label_mask"].sum()) for batch in batches)
    total = sum(
        F.cross_entropy(
            reference.linear(b["features"])[b["action_label_mask"].bool()],
            b["action_labels"][b["action_label_mask"].bool()],
            reduction="sum",
        )
        for b in batches
    )
    (total / tokens).backward()
    expected = _flat_grad(reference)
    for gradient in gradients:
        torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-6)
