"""YAML-backed run configuration for G0.5 training and inference."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from actioncodec2.vla.training.tracking import TrackingConfig

__all__ = [
    "DataConfig",
    "ModelConfig",
    "RunConfig",
    "SUPPORTED_TOKENIZERS",
    "TrackingConfig",
    "TrainConfig",
    "resolve_base_checkpoint",
    "resolve_text_tokenizer",
]

SUPPORTED_TOKENIZERS = ("actioncodec2", "actioncodec", "fast", "x-tokenizer")
COMPUTE_DTYPES = ("float32", "bfloat16", "float16")
#: Width of G0.5's internal proprio vector; see ``g05_hf.configuration_g05``.
_G05_PROPRIO_SLOTS = 27
#: Where the released G0.5 base weights and Qwen3.5 text tokenizer live.
G05_HUB_REPO = "OpenGalaxea/G05"
G05_BASE_CHECKPOINT_FILE = "g05-base/checkpoints/model_state_dict.pt"
G05_TEXT_TOKENIZER_DIR = "qwen3_5_2b_base_processor"


def resolve_base_checkpoint(path: str | None) -> Path:
    """The G0.5 base ``model_state_dict.pt``: ``path`` if given, else the Hub release."""
    if path:
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"G0.5 base checkpoint is missing: {path}")
        return checkpoint
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(G05_HUB_REPO, G05_BASE_CHECKPOINT_FILE))


def resolve_text_tokenizer(path: str | Path | None) -> str:
    """G0.5's Qwen3.5 text tokenizer: ``path`` if given, else the Hub release."""
    if path:
        return str(path)
    from huggingface_hub import snapshot_download

    root = snapshot_download(G05_HUB_REPO, allow_patterns=[f"{G05_TEXT_TOKENIZER_DIR}/*"])
    return str(Path(root) / G05_TEXT_TOKENIZER_DIR)


@dataclass
class ModelConfig:
    """The G0.5 base model and how observations enter it."""

    #: G0.5 base ``model_state_dict.pt`` that initializes training; ``None``
    #: downloads the release from ``OpenGalaxea/G05`` (a gated Hub repository).
    path: str | None = None
    #: G0.5's Qwen3.5 text tokenizer (a local directory or Hub repository);
    #: ``None`` uses the copy in ``OpenGalaxea/G05``.
    tokenizer_path: str | None = None
    #: Proprio normalization statistics file or inline inference payload.
    proprio_stats: str | dict[str, object] | None = None
    #: ``observed`` feeds the recorded state, ``zero`` a constant zero state,
    #: and ``none`` omits the State slot and proprio adapter.
    proprio_mode: str = "observed"
    #: Text placed in G0.5's ``Embodiment: ...`` prompt.
    embodiment: str = "libero"
    #: Components of the stored state vector used as proprio input.
    proprio_source_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    #: G0.5 internal slot for each source index.  ``None`` places the arm at the
    #: head of ``right_control`` (slot 10) and the last index at slot 19.
    proprio_internal_indices: tuple[int, ...] | None = None
    #: Source indices normalized by the q01/q99 rule instead of a z-score
    #: (G0.5 applies it to the gripper).
    proprio_quantile_indices: tuple[int, ...] = (6,)
    #: Autocast compute dtype.  Parameters and optimizer state stay FP32.
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    image_size: tuple[int, int] = (256, 256)
    action_dim: int = 7
    #: Recompute the vision tower's activations in the backward pass.  The text
    #: tower is not checkpointed.
    vision_gradient_checkpointing: bool = True
    #: Normalize the action cross-entropy over the action rows only, matching
    #: grammar-constrained decoding.  ``False`` matches the released G0.5 loss;
    #: the two settings produce losses on different scales.
    action_only_loss: bool = False
    #: Override whether the model receives a continuous ``previous_action``
    #: input.  ``None`` follows the tokenizer's ``requires_previous_action``.
    #: This never changes what the tokenizer encodes.
    force_previous_action_input: bool | None = None

    def __post_init__(self) -> None:
        self.proprio_mode = str(self.proprio_mode).lower()
        self.embodiment = str(self.embodiment).strip()
        self.dtype = str(self.dtype).lower()
        if self.proprio_mode not in {"observed", "zero", "none"}:
            raise ValueError("model.proprio_mode must be 'observed', 'zero', or 'none'")
        if self.dtype not in COMPUTE_DTYPES:
            raise ValueError(f"model.dtype must be one of {COMPUTE_DTYPES}")
        if not self.embodiment:
            raise ValueError("model.embodiment must not be empty")
        self.image_size = tuple(int(value) for value in self.image_size)
        self.proprio_source_indices = tuple(int(value) for value in self.proprio_source_indices)
        self.proprio_quantile_indices = tuple(int(value) for value in self.proprio_quantile_indices)
        if self.proprio_internal_indices is not None:
            self.proprio_internal_indices = tuple(
                int(value) for value in self.proprio_internal_indices
            )
        if self.proprio_mode == "observed" and not self.proprio_stats:
            raise ValueError("model.proprio_stats is required when proprio_mode is 'observed'")
        if self.proprio_mode == "none":
            return
        width = len(self.proprio_source_indices)
        if not width:
            raise ValueError("model.proprio_source_indices must not be empty")
        internal = self.proprio_internal_indices
        if internal is not None:
            if len(internal) != width:
                raise ValueError(
                    "model.proprio_internal_indices needs one slot per proprio source index"
                )
            if len(set(internal)) != width or not all(
                0 <= slot < _G05_PROPRIO_SLOTS for slot in internal
            ):
                raise ValueError(
                    "model.proprio_internal_indices must be unique and inside G0.5's "
                    f"{_G05_PROPRIO_SLOTS} proprio slots"
                )
        if not set(self.proprio_quantile_indices) <= set(self.proprio_source_indices):
            raise ValueError(
                "model.proprio_quantile_indices must be a subset of model.proprio_source_indices"
            )


@dataclass
class DataConfig:
    """LeRobot roots, the action convention they store, and how to sample them."""

    #: LeRobot dataset directories.  Relative entries are resolved against the
    #: ``--data-root`` given at launch (``.`` is the data root itself).
    roots: list[str]
    horizon: int = 20
    stride: int = 1
    #: Physical meaning of the stored action.  Persisted in checkpoints and
    #: reported to simulator clients by the policy server.
    action_space: str = "delta_eef"
    action_frame: str = "robot_base"
    action_rotation_representation: str = "rotation_vector"
    #: Control rate of the stored actions.  ``None`` takes the dataset's own fps
    #: from ``info.json`` when training starts.
    control_frequency_hz: float | None = None
    #: Meters and radians that one stored unit of translation / rotation stands
    #: for (LIBERO stores normalized controller commands: 0.05 m and 0.5 rad).
    #: ``None`` when the dataset does not say.  Reported to simulator clients.
    translation_meters_per_unit: float | None = None
    rotation_radians_per_unit: float | None = None
    #: Stored values of the gripper command at its two extremes, ``[low, high]``.
    #: Only FAST supports a range other than ``[0, 1]``.
    gripper_range: tuple[float, float] = (0.0, 1.0)
    #: Boundary between open and closed inside ``gripper_range``.
    gripper_threshold: float = 0.5
    #: Whether the larger stored gripper value means "more open" in the dataset.
    #: Reported to simulator clients.  A tokenizer may take the opposite
    #: convention as its own input (``tokenizer.gripper_open_is_high``, which
    #: defaults to this value); the two need not agree.
    gripper_open_is_high: bool = True
    #: Columns concatenated into the action source; ``None`` reads ``action``.
    action_keys: list[str] | None = None
    #: Columns concatenated into the state; ``None`` reads ``observation.state``.
    state_keys: list[str] | None = None
    #: ``0`` uses the stored actions verbatim.  ``1`` treats ``action_keys`` as
    #: a state trajectory and trains on ``state[t+1] - state[t]``.
    action_offset: int = 0
    #: ``[[root_index, episode_index], ...]``; ``None`` uses every episode.
    episodes: list[list[int]] | None = None
    max_episodes: int | None = None
    #: Drop episodes without a task instruction.
    require_nonempty_task: bool = True
    #: Actions executed per plan at deployment.  ``None`` means the whole chunk
    #: for stateless tokenizers and one step for stateful ones.
    execution_stride: int | None = None
    #: Boundary action used by stateful tokenizers: ``reconstructed`` rolls the
    #: codec through the episode at ``execution_stride``; ``ground_truth`` uses
    #: the recorded preceding action (zero at the episode start).
    previous_action_source: str = "reconstructed"
    #: Online boundary-action jitter for stateful tokenizers that implement
    #: ``encode_augmented_batch``; see :class:`PreviousActionAugmentation`.
    #: Training split only, and incompatible with ``token_cache``.
    previous_action_augmentation: dict[str, Any] | None = None
    #: Cameras to use; ``None`` takes every camera in ``info.json`` order.
    video_keys: list[str] | None = None
    #: Episodes read concurrently.  Sizes the decoder, episode and table caches
    #: unless ``max_readers``/``cached_episodes``/``table_cache`` override them.
    lanes: int = 8
    #: ``random`` samples timesteps uniformly; ``balanced_replacement`` samples
    #: a root, then an episode, then a timestep, with replacement.
    sample_order: str = "random"
    #: Per-root weights for ``balanced_replacement``; ``None`` weighs roots equally.
    root_weights: list[float] | None = None
    #: Directory for the pre-encoded action-token cache, ``"auto"`` for
    #: ``outputs/cache``, or ``None`` to encode on demand.
    token_cache: str | None = None
    #: How long non-builder ranks wait for another rank to build the cache.
    token_cache_timeout_seconds: float = 43200.0
    max_readers: int | None = None
    cached_episodes: int | None = None
    table_cache: int | None = None
    validation_fraction: float = 0.0
    split: str = "all"

    def __post_init__(self) -> None:
        if not isinstance(self.roots, (list, tuple)):
            raise ValueError("data.roots must be a sequence of dataset paths")
        self.roots = [str(root) for root in self.roots]
        self.horizon = int(self.horizon)
        self.stride = int(self.stride)
        if self.horizon <= 0:
            raise ValueError("data.horizon must be positive")
        if self.stride <= 0:
            raise ValueError("data.stride must be positive")
        self.action_space = str(self.action_space).lower()
        self.action_frame = str(self.action_frame).lower()
        self.action_rotation_representation = str(self.action_rotation_representation).lower()
        if self.action_space not in {"delta_eef", "delta_joint", "absolute_eef", "absolute_joint"}:
            raise ValueError(
                "data.action_space must be 'delta_eef', 'delta_joint', "
                "'absolute_eef', or 'absolute_joint'"
            )
        if not self.action_frame:
            raise ValueError("data.action_frame must not be empty")
        if self.action_rotation_representation not in {"euler_xyz", "rotation_vector"}:
            raise ValueError(
                "data.action_rotation_representation must be 'euler_xyz' or 'rotation_vector'"
            )
        if self.control_frequency_hz is not None:
            self.control_frequency_hz = float(self.control_frequency_hz)
            if not math.isfinite(self.control_frequency_hz) or self.control_frequency_hz <= 0:
                raise ValueError("data.control_frequency_hz must be positive and finite")
        for name in ("translation_meters_per_unit", "rotation_radians_per_unit"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"data.{name} must be positive")
                setattr(self, name, value)
        self.gripper_range = tuple(float(value) for value in self.gripper_range)
        if len(self.gripper_range) != 2 or not all(map(math.isfinite, self.gripper_range)):
            raise ValueError("data.gripper_range must contain two finite values")
        low, high = self.gripper_range
        if not low < high:
            raise ValueError("data.gripper_range must be [low, high] with low < high")
        self.gripper_threshold = float(self.gripper_threshold)
        if not low <= self.gripper_threshold <= high:
            raise ValueError("data.gripper_threshold must lie inside data.gripper_range")
        self.gripper_open_is_high = bool(self.gripper_open_is_high)
        self.lanes = int(self.lanes)
        if self.lanes <= 0:
            raise ValueError("data.lanes must be positive")
        for name in ("max_readers", "cached_episodes", "table_cache"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"data.{name} must be positive")
                setattr(self, name, value)
        for name in ("action_keys", "state_keys"):
            values = getattr(self, name)
            if values is not None:
                values = [str(value).strip() for value in values]
                if not values or not all(values):
                    raise ValueError(f"data.{name} must contain non-empty column names")
                setattr(self, name, values)
        self.action_offset = int(self.action_offset)
        if self.action_offset not in {0, 1}:
            raise ValueError("data.action_offset must be 0 (stored actions) or 1 (adjacent delta)")
        if self.execution_stride is not None:
            self.execution_stride = int(self.execution_stride)
            if not 1 <= self.execution_stride <= self.horizon:
                raise ValueError("data.execution_stride must lie in [1, data.horizon]")
        self.previous_action_source = str(self.previous_action_source).lower()
        if self.previous_action_source not in {"reconstructed", "ground_truth"}:
            raise ValueError(
                "data.previous_action_source must be 'reconstructed' or 'ground_truth'"
            )
        if self.previous_action_augmentation is not None:
            from actioncodec2.vla.data.augmentation import PreviousActionAugmentation

            PreviousActionAugmentation(**self.previous_action_augmentation)
            if self.previous_action_source != "ground_truth" or self.action_offset:
                raise ValueError(
                    "data.previous_action_augmentation requires previous_action_source: "
                    "ground_truth and action_offset: 0"
                )
            if self.token_cache:
                raise ValueError(
                    "data.previous_action_augmentation re-encodes tokens online; "
                    "set data.token_cache: null"
                )
        self.sample_order = str(self.sample_order).lower()
        if self.sample_order not in {"random", "balanced_replacement"}:
            raise ValueError("data.sample_order must be 'random' or 'balanced_replacement'")
        if self.root_weights is not None:
            self.root_weights = [float(value) for value in self.root_weights]
            if len(self.root_weights) != len(self.roots):
                raise ValueError("data.root_weights must contain one weight per data root")
            if any(not math.isfinite(value) or value < 0 for value in self.root_weights) or not sum(self.root_weights) > 0:
                raise ValueError("data.root_weights must be non-negative and not all zero")
            if self.sample_order != "balanced_replacement":
                raise ValueError(
                    "data.root_weights requires data.sample_order: balanced_replacement"
                )
        self.split = str(self.split).lower()
        if self.split not in {"train", "validation", "all"}:
            raise ValueError("data.split must be 'train', 'validation', or 'all'")
        self.validation_fraction = float(self.validation_fraction)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("data.validation_fraction must be in [0, 1)")
        self.token_cache_timeout_seconds = float(self.token_cache_timeout_seconds)
        if not math.isfinite(self.token_cache_timeout_seconds) or self.token_cache_timeout_seconds <= 0:
            raise ValueError("data.token_cache_timeout_seconds must be positive and finite")


@dataclass
class TrainConfig:
    """Optimizer, schedule, checkpointing and data-loader settings."""

    output_dir: str
    steps: int = 2000
    batch_size: int = 8
    gradient_accumulation: int = 1
    learning_rate: float = 1.0e-4
    warmup_steps: int = 50
    #: Horizon of the LR schedule; ``None`` uses ``steps``.  Set it above
    #: ``steps`` to stop early on a longer reference schedule.
    schedule_steps: int | None = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1
    #: Fraction of the post-warmup steps spent in cosine decay; the rest holds
    #: the peak rate.
    decay_fraction: float = 1.0
    log_every: int = 10
    save_every: int = 0
    #: Extra global steps to checkpoint in addition to ``save_every``.
    save_steps: tuple[int, ...] = ()
    save_final: bool = True
    #: Also write ``optimizer.pt`` (AdamW moments, about 8 bytes per parameter) into
    #: each periodic checkpoint so ``--auto-resume`` continues with the optimizer
    #: state intact; without it the moments restart from zero.
    save_optimizer_state: bool = False
    num_workers: int = 4
    #: Seconds a DataLoader worker may take to produce a batch; ``0`` waits
    #: forever.  Keep it below the NCCL timeout so stalls report their cause.
    dataloader_timeout_seconds: float = 0.0
    seed: int = 42
    #: Checkpoint to continue an interrupted run from.  The LR schedule resumes at
    #: ``warm_start_step``; optimizer moments are restored from ``optimizer.pt`` when
    #: the checkpoint has one and restart at zero otherwise.
    warm_start_checkpoint: str | None = None
    warm_start_step: int = 0

    def __post_init__(self) -> None:
        self.save_steps = tuple(int(step) for step in self.save_steps)
        self.num_workers = int(self.num_workers)
        for name in ("learning_rate", "weight_decay", "max_grad_norm", "min_lr_ratio",
                     "decay_fraction", "dataloader_timeout_seconds"):
            try:
                setattr(self, name, float(getattr(self, name)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"train.{name} must be a number") from error
        self.validate()

    def validate(self) -> None:
        """Check the settings; call again after overriding fields on a loaded config."""
        for name in ("learning_rate", "weight_decay", "max_grad_norm", "min_lr_ratio",
                     "decay_fraction", "dataloader_timeout_seconds"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"train.{name} must be finite")
        for name in ("steps", "batch_size", "gradient_accumulation", "warmup_steps",
                     "log_every", "save_every", "num_workers", "warm_start_step", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"train.{name} must be an integer")
        if self.learning_rate <= 0:
            raise ValueError("train.learning_rate must be positive")
        if self.steps <= 0:
            raise ValueError("train.steps must be positive")
        if self.batch_size <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("train.batch_size and train.gradient_accumulation must be positive")
        if self.log_every <= 0:
            raise ValueError("train.log_every must be positive")
        if self.num_workers < 0:
            raise ValueError("train.num_workers must be non-negative")
        if self.dataloader_timeout_seconds < 0:
            raise ValueError("train.dataloader_timeout_seconds must be non-negative")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("train.weight_decay must be non-negative and max_grad_norm positive")
        if not 0 <= self.min_lr_ratio <= 1 or not 0 < self.decay_fraction <= 1:
            raise ValueError("train.min_lr_ratio must be in [0, 1] and decay_fraction in (0, 1]")
        if self.warmup_steps < 0 or self.save_every < 0 or self.warm_start_step < 0 or self.seed < 0:
            raise ValueError("warmup_steps, save_every, warm_start_step and seed must be non-negative")
        if self.schedule_steps is not None and (
            not isinstance(self.schedule_steps, int) or isinstance(self.schedule_steps, bool)
        ):
            raise ValueError("train.schedule_steps must be an integer")
        if self.schedule_steps is not None and self.schedule_steps < self.steps:
            raise ValueError("train.schedule_steps cannot be smaller than train.steps")
        if any(step <= 0 for step in self.save_steps):
            raise ValueError("train.save_steps must contain positive steps")
        if len(set(self.save_steps)) != len(self.save_steps):
            raise ValueError("train.save_steps must not contain duplicates")
        if self.warm_start_checkpoint:
            if self.warm_start_step <= 0:
                raise ValueError("train.warm_start_step must state the checkpoint's global step")
            if self.warm_start_step >= self.steps:
                raise ValueError("train.warm_start_step must be below train.steps")
            if self.schedule_steps is None:
                raise ValueError(
                    "train.warm_start_checkpoint requires train.schedule_steps so the "
                    "resumed run keeps the original LR schedule"
                )
        elif self.warm_start_step:
            raise ValueError("train.warm_start_step requires train.warm_start_checkpoint")


#: Keys written by older checkpoints, renamed to their current name.
_RENAMED_KEYS = {
    "model": {"gradient_checkpointing": "vision_gradient_checkpointing"},
    "data": {
        "chunk_cells": "token_cache",
        "chunk_token_timeout_seconds": "token_cache_timeout_seconds",
    },
}
_ANY = object()
#: Retired keys and the values under which dropping them changes nothing.
#: ``_ANY`` marks settings that never affected a loaded policy.
_RETIRED_KEYS: dict[str, dict[str, object]] = {
    "model": {"backend": ("g05",), "state_bins": (0,), "history_bins": (0,), "state_dim": _ANY},
    "data": {
        "group_by": _ANY,
        "split_root_indices": _ANY,
        "simulator_observations": (None, []),
    },
    "train": {
        "eval_every": _ANY,
        "stage1_steps": (0,),
        "stage1_learning_rate": _ANY,
        "stage1_warmup_steps": _ANY,
        "stage1_new_rows_only": _ANY,
    },
}


def _upgrade(section: str, values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    for old, new in _RENAMED_KEYS.get(section, {}).items():
        if old in values:
            values.setdefault(new, values.pop(old))
    for key, allowed in _RETIRED_KEYS.get(section, {}).items():
        if key not in values:
            continue
        value = values.pop(key)
        if allowed is not _ANY and value not in allowed:
            raise ValueError(f"{section}.{key}={value!r} is no longer supported")
    if section == "data" and values.get("sample_order") == "episode_interleaved":
        values["sample_order"] = "random"
    return values


def _build(cls, section: str, values: dict[str, Any] | None):
    values = _upgrade(section, values or {})
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown {section} settings: {', '.join(unknown)}")
    return cls(**values)


def _is_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~"))


@dataclass
class RunConfig:
    model: ModelConfig
    tokenizer: dict[str, Any]
    data: DataConfig
    train: TrainConfig
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    #: Settings holding file paths; ``./`` and ``../`` resolve against the
    #: repository root.
    PATH_KEYS = {
        "model": ("path", "tokenizer_path", "proprio_stats"),
        "tokenizer": (
            "artifact", "processor", "action_stats", "normalizer_path",
            "calibration_path", "statistics_path", "source",
        ),
        "train": ("warm_start_checkpoint",),
    }

    def __post_init__(self) -> None:
        name = str(self.tokenizer.get("name", "")).lower().replace("_", "-")
        if name.startswith("actioncodec2") and name.removeprefix("actioncodec2").strip("-") == "unified":
            self.tokenizer["name"] = "actioncodec2"
            self.tokenizer.setdefault("action_space", "delta_eef")

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> RunConfig:
        if not isinstance(document, dict):
            raise ValueError("run configuration must be a mapping")
        required = {"model", "tokenizer", "data", "train"}
        unknown = sorted(set(document) - required - {"tracking"})
        if unknown:
            raise ValueError(f"unknown run settings: {', '.join(unknown)}")
        missing = sorted(required - document.keys())
        if missing:
            raise ValueError(f"missing run sections: {', '.join(missing)}")
        for section, values in document.items():
            if not isinstance(values, dict) and not (section == "tracking" and values is None):
                raise ValueError(f"{section} must be a mapping")
        tokenizer = dict(document["tokenizer"])
        data = dict(document["data"])
        return cls(
            model=_build(ModelConfig, "model", document["model"]),
            tokenizer=tokenizer,
            data=_build(DataConfig, "data", data),
            train=_build(TrainConfig, "train", document["train"]),
            tracking=_build(TrackingConfig, "tracking", document.get("tracking")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        data_root: str | Path | None = None,
        base_checkpoint: str | Path | None = None,
        text_tokenizer: str | Path | None = None,
    ) -> RunConfig:
        """Load a YAML/JSON config and resolve its paths."""

        def read(source: Path, seen: set[Path]) -> dict[str, Any]:
            source = source.resolve()
            if source in seen:
                raise ValueError(f"configuration inheritance cycle at {source}")
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError(f"configuration must be a mapping: {source}")
            parent = document.pop("extends", None)
            if parent is None:
                return document
            base = read(source.parent / str(parent), seen | {source})
            for key, value in document.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key] = {**base[key], **value}
                else:
                    base[key] = value
            return base

        source = Path(path).resolve()
        document = read(source, set())
        config_root = next((parent for parent in source.parents if parent.name == "configs"), None)
        path_root = config_root.parent if config_root is not None else source.parent

        def expand(value: str) -> str:
            value = str(Path(value).expanduser()) if value.startswith("~") else value
            return str((path_root / value).resolve()) if value.startswith(("./", "../")) else value

        for section, keys in cls.PATH_KEYS.items():
            values = document.setdefault(section, {}) or {}
            for key in keys:
                if isinstance(values.get(key), str):
                    values[key] = expand(values[key])
        for key, override in (("path", base_checkpoint), ("tokenizer_path", text_tokenizer)):
            if override is not None:
                document.setdefault("model", {})[key] = str(Path(override).expanduser())
        data = document["data"]
        roots = []
        for root in data["roots"]:
            root = Path(str(root)).expanduser()
            if not root.is_absolute():
                if data_root is None:
                    raise ValueError(
                        f"{source.name}: data.roots contains the relative path {str(root)!r}; "
                        "pass --data-root (the directory that holds the LeRobot datasets)"
                    )
                root = Path(data_root).expanduser() / root
            roots.append(str(root.resolve()))
        data["roots"] = roots
        return cls.from_dict(document)

    def validate_for_training(self) -> None:
        """Catch recipe mistakes before loading G0.5 or building a token cache.

        Also fills ``data.control_frequency_hz`` from the datasets when unset.
        """
        self.train.validate()
        if self.data.gripper_range != (0.0, 1.0) and (
            str(self.tokenizer.get("name", "")).lower() != "fast"
        ):
            raise ValueError("data.gripper_range other than [0, 1] is supported for FAST only")
        if self.model.path and not Path(self.model.path).expanduser().is_file():
            raise FileNotFoundError(f"G0.5 base checkpoint is missing: {self.model.path}")
        tokenizer_path = self.model.tokenizer_path
        if tokenizer_path and _is_local_path(tokenizer_path) and not Path(tokenizer_path).is_dir():
            raise FileNotFoundError(f"G0.5 text tokenizer is missing: {tokenizer_path}")
        if not self.data.roots:
            raise ValueError("data.roots must contain at least one LeRobot dataset")
        rates: list[float] = []
        for root in self.data.roots:
            info_path = Path(root) / "meta" / "info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"LeRobot metadata is missing: {root}/meta/info.json")
            info = json.loads(info_path.read_text(encoding="utf-8"))
            fps = float(info["fps"])
            rates.append(fps)
            if self.data.control_frequency_hz is not None and not math.isclose(
                fps, self.data.control_frequency_hz, rel_tol=1e-6
            ):
                raise ValueError(
                    f"{root}: dataset fps {fps:g} does not match "
                    f"data.control_frequency_hz {self.data.control_frequency_hz:g}"
                )
            if self.data.video_keys is not None:
                missing = set(self.data.video_keys) - set(info["features"])
                if missing:
                    raise ValueError(f"{root}: missing configured camera features {sorted(missing)}")
        if self.data.control_frequency_hz is None:
            if not all(math.isclose(rate, rates[0], rel_tol=1e-6) for rate in rates):
                raise ValueError(
                    "the datasets have different frame rates; set data.control_frequency_hz"
                )
            self.data.control_frequency_hz = rates[0]
        expected_dim = 7 if self.data.action_space.endswith("eef") else 8
        if self.model.action_dim != expected_dim:
            raise ValueError(f"{self.data.action_space} requires model.action_dim={expected_dim}")
        horizon = self.tokenizer.get("time_horizon")
        if horizon is not None and int(horizon) != self.data.horizon:
            raise ValueError("tokenizer.time_horizon must match data.horizon")
        from actioncodec2.tokenizers.factory import check_tokenizer_settings

        check_tokenizer_settings(self.tokenizer)
        name = str(self.tokenizer.get("name", "")).lower().replace("_", "-")
        if name not in SUPPORTED_TOKENIZERS:
            raise ValueError(
                f"unsupported action tokenizer {name!r}; choose one of {SUPPORTED_TOKENIZERS}"
            )
        if name.startswith("actioncodec2"):
            profile = str(self.tokenizer.get("action_space", "single_eef_delta")).lower()
            joint = profile in {"delta_joint", "single_joint7_delta"}
            if joint != self.data.action_space.endswith("joint"):
                raise ValueError("ActionCodec2 profile does not match data.action_space")

    def to_json(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "tokenizer": self.tokenizer,
            "data": asdict(self.data),
            "train": asdict(self.train),
            "tracking": asdict(self.tracking),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
