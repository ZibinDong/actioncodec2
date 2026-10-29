from __future__ import annotations

import json

from pathlib import Path

from typing import Any

from .preprocessing import PreprocessingSpec


MANIFEST_NAME = "manifest.json"

SUPPORTED_FORMAT = 1

_USER_TO_STORAGE = {"2bin": "binary", "11bin": "eleven", "21bin": "twenty"}

_GRIPPER_LIMIT = {"2bin": 0.5, "11bin": 0.05, "21bin": 0.025}

_GRIPPER_BINS = {"2bin": 2, "11bin": 11, "21bin": 21}

_GRIPPER_COMPONENT = "gripper_open_command"


def normalize_gripper_mode(token: str) -> str:
    """Map user-facing and legacy gripper-mode tokens onto ``2bin``/``11bin``/``21bin``."""
    token = str(token).strip().lower()
    if token in _USER_TO_STORAGE:
        return token
    legacy = {v: k for k, v in _USER_TO_STORAGE.items()}
    if token in legacy:
        return legacy[token]
    raise ValueError(
        f"unknown gripper_mode {token!r}; expected one of "
        f"{sorted(_USER_TO_STORAGE)} (legacy tokens accepted)"
    )


def read_manifest(package: str | Path) -> dict[str, Any]:
    path = Path(package)
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{path} is not a release package: missing {MANIFEST_NAME}; "
            "expected a release directory containing manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fmt = manifest.get("format")
    if type(fmt) is not int or not 1 <= fmt <= SUPPORTED_FORMAT:
        raise ValueError(f"unsupported manifest format {fmt!r} in {manifest_path}")
    return manifest


def release_tokenizer(manifest: dict[str, Any]) -> str:
    """Tokenizer family of a release; manifests before ActionCodec2Sparse omit it."""
    return manifest.get("tokenizer", "actioncodec2")


def selected_variants(
    package: str | Path, *, gripper_mode: str, order: int, tokenizer: str = "actioncodec2"
):
    """Select one vocabulary per available physical profile from a release."""
    if type(order) is not int or order not in (1, 2):
        raise ValueError("primitive_order must be 1 or 2")
    mode = normalize_gripper_mode(gripper_mode)
    manifest = read_manifest(package)
    found = release_tokenizer(manifest)
    if found != tokenizer:
        frontend = {"actioncodec2": "ActionCodec2", "actioncodec2_sparse": "ActionCodec2Sparse"}
        raise ValueError(
            f"{package} is an {found!r} release; load it with {frontend.get(found, found)}"
        )
    selected = {}
    for variant in manifest["variants"]:
        if variant["order"] != order or variant["gripper_mode"] != mode:
            continue
        profile = variant["profile"]
        if profile not in {"joint", "eef"} or profile in selected:
            raise ValueError(f"invalid or duplicate release profile {profile!r}")
        selected[profile] = variant
    if not selected:
        raise ValueError(f"{package} has no release variants for {mode}/order{order}")
    return manifest, mode, selected


def check_gripper_consistency(
    entry: Path,
    processor,
    preprocessing: PreprocessingSpec,
    mode: str,
) -> None:
    expected_token = _USER_TO_STORAGE[mode]
    if preprocessing.gripper_mode != expected_token:
        raise ValueError(
            f"{entry}: preprocessing.json gripper_mode "
            f"{preprocessing.gripper_mode!r} does not match requested {mode!r}"
        )
    component = processor.config.physical.components.get(_GRIPPER_COMPONENT)
    if component is None:
        return
    limit = component.distortion_limit
    limit = float(limit[0]) if isinstance(limit, (list, tuple)) else float(limit)
    if abs(limit - _GRIPPER_LIMIT[mode]) > 1e-9:
        raise ValueError(
            f"{entry}: gripper distortion_limit {limit} does not match "
            f"{mode} ({_GRIPPER_LIMIT[mode]})"
        )
    if processor.bins[-1] != _GRIPPER_BINS[mode]:
        raise ValueError(
            f"{entry}: last primitive bin count {processor.bins[-1]} "
            f"does not match {mode} ({_GRIPPER_BINS[mode]})"
        )


def check_release_gates(entry: Path) -> None:
    validation_path = entry / "validation.json"
    derived_path = entry / "derived_report.json"
    if validation_path.is_file():
        report = json.loads(validation_path.read_text(encoding="utf-8"))
        additive = [x for x in report["additive_within_limit"] if x is not None]
        ok = (
            report["primitive_zero_exact"]
            and report["set_bpe_cell_roundtrip_exact"]
            and all(additive)
            and report["passed"]
        )
        if not ok:
            raise ValueError(f"{entry}: recorded validation gates failed")
    elif derived_path.is_file():
        report = json.loads(derived_path.read_text(encoding="utf-8"))
        if report["additive_within_limit_segments"] != report["segments"]:
            raise ValueError(f"{entry}: derived probe gates failed")
    else:
        raise FileNotFoundError(f"{entry}: no validation.json or derived_report.json")
