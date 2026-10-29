"""Build a supported action tokenizer from a settings mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import ChunkTokenizer, QuantileNormalizer

__all__ = ["action_bounds", "build_tokenizer", "check_tokenizer_settings"]

DEFAULT_ACTIONCODEC = "ZibinDong/ActionCodec-5e"
DEFAULT_FAST = "physical-intelligence/fast"
DEFAULT_XTOKENIZER = "x-square-robot/X-Tokenizer"


def _local_or_hub(
    value: str | Path, revision: str = "", *, required_files: tuple[str, ...] = ()
) -> str:
    """A local path as-is, otherwise a snapshot of that Hugging Face repository."""
    path = Path(value).expanduser()
    if path.exists():
        return str(path)
    if str(value).startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"tokenizer artifact does not exist: {value}")
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    if revision and required_files:
        cached = [
            try_to_load_from_cache(str(value), name, revision=revision)
            for name in required_files
        ]
        if all(isinstance(item, str) and Path(item).is_file() for item in cached):
            return str(Path(cached[0]).parent)

    return snapshot_download(str(value), revision=revision or None)


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def action_bounds(settings: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Action bounds from ``action_clip`` or an ``action_stats`` file."""
    if "action_clip" in settings:
        clip = settings["action_clip"]
        return np.asarray(clip["low"], np.float32), np.asarray(clip["high"], np.float32)
    path = settings.get("action_stats")
    if not path:
        return None
    payload = _read_json(path)
    section = settings.get("action_stats_section")
    values = payload.get(section, payload) if section else payload
    for low, high in (("q001", "q999"), ("q01", "q99"), ("low", "high")):
        if low in values and high in values:
            return np.asarray(values[low], np.float32), np.asarray(values[high], np.float32)
    raise ValueError(f"{path} has no q001/q999, q01/q99 or low/high action bounds")


def _normalizer(settings: dict) -> QuantileNormalizer | None:
    if "normalizer" in settings:
        return QuantileNormalizer.from_json(settings["normalizer"])
    if settings.get("normalizer_path"):
        return QuantileNormalizer.from_json(_read_json(settings["normalizer_path"]))
    return None


_COMMON_KEYS = frozenset({
    "name", "time_horizon", "horizon", "action_dim", "fps", "gripper_open_is_high",
    "rotation_representation", "revision", "artifact", "device",
})
_BOUND_KEYS = frozenset({"action_stats", "action_stats_section", "action_clip"})
#: Every key each adapter reads.  A key outside this set is a typo, and a typo in
#: a convention such as ``gripper_open_is_high`` would otherwise fall back to its
#: default without a word.
_ALLOWED_KEYS = {
    "actioncodec2-sparse": _COMMON_KEYS | _BOUND_KEYS | {
        "action_space", "motion_scale", "rotation_multiplication", "gripper_threshold",
        "previous_action_resampling", "artifact_fingerprint",
    },
    "actioncodec2": _COMMON_KEYS | _BOUND_KEYS | {
        "action_space", "target_hz", "rotation_multiplication", "motion_scale",
        "gripper_threshold", "interpolation", "previous_action_resampling",
        "resampling_alignment",
        "variant", "variant_fingerprint",
    },
    "actioncodec": _COMMON_KEYS | _BOUND_KEYS | {"embodiment_id"},
    "fast": _COMMON_KEYS | _BOUND_KEYS | {
        "processor", "normalizer", "normalizer_path", "constrained_decoding", "source",
        "vocab_size",
    },
    "x-tokenizer": _COMMON_KEYS | {
        "statistics_path", "normalizer", "normalizer_path", "action_layout",
        "calibration_path", "robot_type", "source_revision", "condition_obs",
        "official_action_only", "anchor_mode",
    },
}


def _canonical_name(settings: dict) -> str:
    name = str(settings.get("name", "")).lower().replace("_", "-")
    if name in {"ac2sp", "actioncodec2sparse"}:
        return "actioncodec2-sparse"
    if name.startswith("actioncodec2") and name.removeprefix("actioncodec2").strip("-") == "unified":
        return "actioncodec2"
    return name


def check_tokenizer_settings(settings: dict) -> None:
    """Reject an unknown tokenizer name or a key the selected adapter does not read."""
    name = _canonical_name(settings)
    allowed = _ALLOWED_KEYS.get(name)
    if allowed is None:
        raise ValueError(
            f"unsupported tokenizer {name!r}; choose actioncodec2, "
            "actioncodec2-sparse, actioncodec, fast, or x-tokenizer"
        )
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"unknown tokenizer settings for {name}: {', '.join(unknown)}")


def build_tokenizer(settings: dict, *, device: str = "cpu") -> ChunkTokenizer:
    """Build a tokenizer from a run config's ``tokenizer`` section."""
    check_tokenizer_settings(settings)
    cfg = dict(settings)
    name = _canonical_name(cfg)
    if name == "actioncodec2":
        cfg["name"] = name
        cfg.setdefault("action_space", "delta_eef")
    horizon = int(cfg.get("time_horizon", cfg.get("horizon", 20)))
    gripper_open_is_high = bool(cfg.get("gripper_open_is_high", True))

    if name == "actioncodec2-sparse":
        from .actioncodec2_sparse import ActionCodec2SparseTokenizer

        artifact = cfg.get("artifact")
        if not artifact:
            raise ValueError("ActionCodec2Sparse requires a fitted tokenizer.artifact")
        options = {key: cfg[key] for key in (
            "action_space", "fps", "action_dim", "revision", "rotation_representation",
            "rotation_multiplication", "motion_scale", "gripper_threshold",
            "previous_action_resampling", "artifact_fingerprint",
        ) if key in cfg}
        return ActionCodec2SparseTokenizer.from_pretrained(
            artifact, time_horizon=horizon, bounds=action_bounds(cfg),
            gripper_open_is_high=gripper_open_is_high, **options,
        )

    if name == "actioncodec2":
        from .actioncodec2 import ActionCodec2Tokenizer

        path = cfg.get("artifact")
        if not path:
            raise ValueError(
                "ActionCodec2 requires tokenizer.artifact: a local codec artifact or "
                "Hugging Face repository"
            )
        bounds = action_bounds(cfg)
        action_space = str(cfg.get("action_space", "single_eef_delta"))
        default_dim = 8 if action_space.lower() in {"delta_joint", "single_joint7_delta"} else 7
        return ActionCodec2Tokenizer.from_pretrained(
            path,
            variant=cfg.get("variant"),
            variant_fingerprint=cfg.get("variant_fingerprint"),
            revision=cfg.get("revision") or None,
            time_horizon=horizon,
            fps=float(cfg.get("fps", 20.0)),
            action_space=action_space,
            configured_action_dim=int(cfg.get("action_dim", default_dim)),
            action_low=None if bounds is None else bounds[0],
            action_high=None if bounds is None else bounds[1],
            target_hz=float(cfg.get("target_hz", 15.0)),
            rotation_representation=str(cfg.get("rotation_representation", "rotation_vector")),
            rotation_multiplication=str(cfg.get("rotation_multiplication", "right")),
            motion_scale=cfg.get("motion_scale", 1.0),
            gripper_threshold=float(cfg.get("gripper_threshold", 0.8)),
            gripper_open_is_high=gripper_open_is_high,
            interpolation=str(cfg.get("interpolation", "cubic")),
            previous_action_resampling=str(cfg.get("previous_action_resampling", "strict")),
            resampling_alignment=str(cfg.get("resampling_alignment", "legacy")),
        )

    if name == "actioncodec":
        from .actioncodec import ActionCodecNormalizer, ActionCodecTokenizer

        bounds = action_bounds(cfg)
        if bounds is None:
            raise ValueError("ActionCodec needs tokenizer.action_stats or tokenizer.action_clip")
        revision = str(cfg.get("revision", ""))
        return ActionCodecTokenizer.from_pretrained(
            _local_or_hub(
                cfg.get("artifact", DEFAULT_ACTIONCODEC), revision,
                required_files=(
                    "config.json", "model.safetensors", "configuration_actioncodec.py",
                    "modeling_actioncodec.py", "modular_actioncodec.py", "rvq.py",
                ),
            ),
            ActionCodecNormalizer(*bounds, gripper_open_is_high=gripper_open_is_high),
            time_horizon=horizon,
            embodiment_id=int(cfg.get("embodiment_id", 0)),
            device=device,
            revision=revision,
        )

    if name == "fast":
        from .fast import FastTokenizer

        normalizer = _normalizer(cfg)
        if normalizer is None and cfg.get("action_stats"):
            normalizer = QuantileNormalizer(*action_bounds(cfg))
        if normalizer is None:
            raise ValueError("FAST needs tokenizer.normalizer, normalizer_path or action_stats")
        source = str(cfg.get("processor", cfg.get("source", DEFAULT_FAST)))
        revision = str(cfg.get("revision", ""))
        return FastTokenizer.from_pretrained(
            _local_or_hub(
                source, revision,
                required_files=(
                    "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "processing_action_tokenizer.py", "special_tokens_map.json",
                ),
            ),
            normalizer,
            time_horizon=horizon,
            action_dim=int(cfg.get("action_dim", 7)),
            constrained_decoding=bool(cfg.get("constrained_decoding", True)),
            source=source,
            revision=revision,
        )

    if name == "x-tokenizer":
        from .xtokenizer import DirectXTransform, LiberoXTransform, XTokenizerAdapter

        statistics_path = cfg.get("statistics_path")
        transform = None
        if not statistics_path:
            normalizer = _normalizer(cfg)
            if normalizer is None:
                raise ValueError("X-Tokenizer needs statistics_path, normalizer or normalizer_path")
            if cfg.get("action_layout") == "direct":
                transform = DirectXTransform(normalizer, action_dim=int(cfg.get("action_dim", 7)))
            else:
                transform = LiberoXTransform(
                    normalizer,
                    rotation_representation=str(
                        cfg.get("rotation_representation", "rotation_vector")
                    ),
                )
        return XTokenizerAdapter.from_pretrained(
            _local_or_hub(cfg.get("artifact", DEFAULT_XTOKENIZER), str(cfg.get("revision", ""))),
            transform,
            time_horizon=horizon,
            device=device,
            statistics_path=statistics_path,
            calibration_path=cfg.get("calibration_path"),
            robot_type=str(cfg.get("robot_type", "Franka")),
            source_revision=str(cfg.get("source_revision", "")),
            condition_obs=bool(cfg.get("condition_obs", False)),
        )

    raise ValueError(
        f"unsupported tokenizer {name!r}; choose actioncodec2, "
        "actioncodec, fast, or x-tokenizer"
    )
