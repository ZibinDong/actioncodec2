"""Assemble G0.5, a released action tokenizer, and LeRobot data; save and load runs."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from actioncodec2.tokenizers import build_tokenizer
from actioncodec2.tokenizers.factory import _canonical_name
from actioncodec2.vla.data import ActionChunkDataset, ChunkTokenStore
from actioncodec2.vla.training.config import DataConfig, ModelConfig, RunConfig

__all__ = [
    "DEFAULT_TOKEN_CACHE_ROOT",
    "DTYPES",
    "build_dataset",
    "build_model",
    "build_processor",
    "build_tokenizer_for",
    "load_policy",
    "load_run",
    "load_runtime_config",
    "resolve_checkpoint",
    "save_run",
    "tokenizer_settings",
]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
#: Where ``data.token_cache: auto`` stores pre-encoded tokens, relative to the
#: working directory.
DEFAULT_TOKEN_CACHE_ROOT = "outputs/cache"


@dataclass
class InferenceConfig:
    """Inputs needed to decode a published policy; training settings are absent."""

    model: ModelConfig
    tokenizer: dict
    data: DataConfig


def _check_gripper_threshold(config: RunConfig | InferenceConfig, settings: dict) -> None:
    """Fail if the run binarizes the gripper differently than the stats were fitted."""
    stats_path = settings.get("action_stats")
    if stats_path is None:
        return
    declared = json.loads(Path(stats_path).read_text(encoding="utf-8")).get("gripper_threshold")
    if declared is not None and float(declared) != float(config.data.gripper_threshold):
        raise ValueError(
            f"data.gripper_threshold={config.data.gripper_threshold} but "
            f"{stats_path} was fitted at {declared}"
        )


def tokenizer_settings(config: RunConfig | InferenceConfig) -> dict:
    """The run's ``tokenizer`` section completed with the action conventions of ``data``."""
    settings = dict(config.tokenizer)
    settings["name"] = _canonical_name(settings)
    settings.setdefault("time_horizon", config.data.horizon)
    settings.setdefault("action_dim", config.model.action_dim)
    # Training fills ``control_frequency_hz`` from the dataset; 20 Hz only serves
    # policies exported before that field was always recorded.
    settings.setdefault("fps", config.data.control_frequency_hz or 20.0)
    settings.setdefault("gripper_open_is_high", config.data.gripper_open_is_high)
    if str(settings.get("name", "")).startswith("actioncodec2"):
        settings.setdefault("gripper_threshold", config.data.gripper_threshold)
        settings.setdefault("action_space", config.data.action_space)
    if config.data.action_space.endswith("eef"):
        settings.setdefault("rotation_representation", config.data.action_rotation_representation)
    return settings


def build_tokenizer_for(config: RunConfig | InferenceConfig, *, device: str | None = None):
    """Build the run's action tokenizer (see :func:`tokenizer_settings`)."""
    settings = tokenizer_settings(config)
    _check_gripper_threshold(config, settings)
    tokenizer = build_tokenizer(settings, device=device or "cpu")
    if tokenizer.name == "actioncodec2-sparse":
        config.tokenizer["name"] = tokenizer.name
        config.tokenizer["artifact_fingerprint"] = tokenizer.artifact_fingerprint
    if tokenizer.name == "actioncodec2" and tokenizer.variant is not None:
        binding = tokenizer.to_json()
        for key in ("variant", "variant_fingerprint", "revision"):
            if binding.get(key):
                config.tokenizer[key] = binding[key]
    return tokenizer


def build_processor(config: RunConfig | InferenceConfig, tokenizer):
    from actioncodec2.vla.runtime.g05 import G05RunProcessor

    return G05RunProcessor(config, tokenizer)


def token_cache_root(config: RunConfig) -> str | None:
    """The directory ``data.token_cache`` names, or ``None`` when it is unset."""
    setting = config.data.token_cache
    if not setting:
        return None
    return DEFAULT_TOKEN_CACHE_ROOT if str(setting).lower() == "auto" else str(setting)


def build_dataset(config: RunConfig, processor, tokenizer) -> ActionChunkDataset:
    """Build the LeRobot chunk dataset and attach the token cache when configured.

    Every DDP rank calls this; one rank builds a missing cache while the others wait.
    """
    data = config.data
    dataset = ActionChunkDataset(
        data.roots,
        processor,
        tokenizer,
        horizon=data.horizon,
        video_keys=tuple(data.video_keys) if data.video_keys else None,
        action_keys=data.action_keys,
        state_keys=data.state_keys,
        action_offset=data.action_offset,
        action_space=data.action_space,
        action_rotation_representation=data.action_rotation_representation,
        gripper_threshold=data.gripper_threshold,
        episode_filter=(
            [(int(root), int(episode)) for root, episode in data.episodes]
            if data.episodes
            else None
        ),
        stride=data.stride,
        max_episodes=data.max_episodes,
        require_nonempty_task=data.require_nonempty_task,
        execution_stride=data.execution_stride,
        previous_action_source=data.previous_action_source,
        previous_action_augmentation=(
            data.previous_action_augmentation if data.split != "validation" else None
        ),
        augmentation_seed=config.train.seed,
        lanes=data.lanes,
        max_readers=data.max_readers,
        max_cached_episodes=data.cached_episodes,
        table_cache=data.table_cache,
        validation_fraction=data.validation_fraction,
        split=data.split,
    )
    cache_root = token_cache_root(config)
    if cache_root is not None:
        dataset.use_token_cache(
            ChunkTokenStore.ensure(dataset, cache_root, timeout=data.token_cache_timeout_seconds)
        )
    return dataset


def build_model(config: RunConfig, processor, *, device: str = "cuda"):
    """G0.5 initialized from the released base checkpoint, with FP32 parameters."""
    from actioncodec2.vla.runtime.g05 import build_g05_model

    return build_g05_model(config, processor, device=device)


def save_run(
    directory: str | Path,
    model,
    processor,
    config: RunConfig,
    *,
    execution_stride: int | None = None,
) -> None:
    """Save weights, processor, and the configs needed to rebuild the policy.

    ``scripts/export_policy_bundle.py`` resolves inference assets for sharing
    or upload. Released action tokenizers are referenced by Hub repository.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.processor.save_pretrained(path)
    config.save(path / "run_config.json")
    payload = {
        "action_vocab_size": processor.action_vocab_size,
        "action_token_begin": processor.action_token_begin,
        "total_vocab_size": processor.total_vocab_size,
        "action_dim": processor.action_dim,
        "image_size": list(processor.image_size),
        # Actions executed per plan when the training boundaries were built;
        # evaluation must execute the same number for stateful tokenizers.
        "execution_stride": execution_stride,
        "g05_proprio_stats": (
            {"mode": "none"}
            if processor.proprio_transform is None
            else processor.proprio_transform.to_json()
        ),
    }
    load_report = getattr(model, "_g05_load_report", None)
    if load_report is not None:
        payload["g05_load"] = load_report
    (path / "policy_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_checkpoint(
    directory: str | Path, *, revision: str | None = None, trust_remote_code: bool = False
) -> Path:
    """Resolve a local directory or Hub snapshot. Hub loading requires trust_remote_code=True; revision pins the snapshot."""
    path = Path(directory).expanduser()
    if path.is_dir():
        return path
    if str(directory).startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
    if not trust_remote_code:
        raise ValueError(
            f"{directory} is a Hub repository, and a policy can bundle tokenizer code that "
            "runs when it loads; pass trust_remote_code=True (--trust-remote-code) if you "
            "trust its source"
        )
    if revision is None:
        warnings.warn(
            f"{directory} is loaded without a pinned revision; its files may change",
            stacklevel=2,
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(directory), revision=revision))


def load_runtime_config(directory: str | Path) -> RunConfig | InferenceConfig:
    """Read a compact exported policy or an older full training checkpoint."""
    path = Path(directory)
    policy = json.loads((path / "policy_config.json").read_text(encoding="utf-8"))
    if "format_version" not in policy:
        return RunConfig.load(path / "run_config.json")
    if policy["format_version"] != 1:
        raise ValueError(f"unsupported policy_config format_version: {policy['format_version']}")
    model = dict(policy["model"])
    model.update(path=str(path), tokenizer_path=str(path))
    if model["proprio_mode"] == "observed":
        model["proprio_stats"] = policy["g05_proprio_stats"]
    data = dict(policy["action"])
    data["roots"] = []
    tokenizer = dict(policy["tokenizer"])
    name = str(tokenizer.get("name", "")).lower().replace("_", "-")
    if name.startswith("actioncodec2") and name.removeprefix("actioncodec2").strip("-") == "unified":
        tokenizer["name"] = "actioncodec2"
        tokenizer.setdefault("action_space", data["action_space"])
    for key in RunConfig.PATH_KEYS["tokenizer"]:
        value = tokenizer.get(key)
        if isinstance(value, str) and value.startswith(("./", "../")):
            tokenizer[key] = str((path / value).resolve())
    return InferenceConfig(
        model=ModelConfig(**model),
        tokenizer=tokenizer,
        data=DataConfig(**data),
    )


def load_run(
    directory: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    """Rebuild ``(model, processor, tokenizer, policy_config)`` from a checkpoint.

    ``directory`` is a saved run, an exported bundle, or a Hub repository holding one.
    ``dtype`` is the weight dtype; see :func:`load_g05_model`.
    """
    from actioncodec2.vla.runtime.g05 import G05RunProcessor, load_g05_model

    path = resolve_checkpoint(directory, revision=revision, trust_remote_code=trust_remote_code)
    policy_config = json.loads((path / "policy_config.json").read_text(encoding="utf-8"))
    config = load_runtime_config(path)
    # Neural tokenizers decode on the policy device at rollout.
    tokenizer = build_tokenizer_for(config, device=device)
    processor = G05RunProcessor(
        config,
        tokenizer,
        tokenizer_path=path,
        proprio_stats=policy_config.get("g05_proprio_stats"),
    )
    model = load_g05_model(path, device=device, dtype=dtype)
    interface = (
        "action_vocab_size", "action_token_begin_idx", "text_vocab_size", "image_size",
        "continuous_action_dim", "action_history_steps", "proprio_dim", "internal_proprio_dim",
        "proprio_indices",
    )
    mismatches = {key: (getattr(model.config, key), getattr(processor.model_config, key))
                  for key in interface
                  if getattr(model.config, key) != getattr(processor.model_config, key)}
    if mismatches:
        raise RuntimeError(f"policy model and processor interfaces differ: {mismatches}")
    expected = {
        "action_vocab_size": processor.action_vocab_size,
        "action_token_begin": processor.action_token_begin,
        "total_vocab_size": processor.total_vocab_size,
    }
    actual = {key: policy_config[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"saved G0.5 vocabulary layout differs: {actual} != {expected}")
    return model, processor, tokenizer, policy_config


def load_policy(
    directory: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a saved run as ``(ActionPolicy, policy_config)`` ready for rollout."""
    from actioncodec2.vla.runtime.policy import ActionPolicy

    model, processor, tokenizer, policy_config = load_run(
        directory, device=device, dtype=dtype, revision=revision,
        trust_remote_code=trust_remote_code,
    )
    policy = ActionPolicy(
        model, processor, tokenizer, device=device, compute_dtype=DTYPES[processor.compute_dtype]
    )
    return policy, policy_config
