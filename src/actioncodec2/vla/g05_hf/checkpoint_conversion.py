# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Strict, auditable initialization from the released G0.5 Qwen3.5 checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from actioncodec2.vla.g05_hf.configuration_g05 import (
    G05_ACTION_BEGIN,
    G05_ACTION_ROWS,
    G05_EOV_ID,
    G05_INTERNAL_PROPRIO_DIM,
    G05_STATE_ID,
    G05_VOCAB_SIZE,
)
from actioncodec2.vla.g05_hf.modeling_g05 import G05Model


@dataclass(frozen=True)
class G05LoadReport:
    checkpoint: str
    loaded_tensors: int
    loaded_parameters: int
    target_parameters: int
    initialized_parameters: int
    ignored_source_tensors: int
    initialized_tensor_names: tuple[str, ...]
    action_history_initialization: str | None = None
    action_history_initialized_parameters: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _target_name(source_name: str) -> str | None:
    if source_name == "model.vlm.input_proj.weight":
        return "vlm.embed_tokens.weight"
    if source_name.startswith("model.vision_tower."):
        return "vision_tower." + source_name.removeprefix("model.vision_tower.")
    if source_name.startswith(("model.vlm.layers.", "model.vlm.norm.")):
        return "vlm." + source_name.removeprefix("model.vlm.")
    if source_name.startswith("model.proprio_embedder.mlp."):
        return "proprio_embedder." + source_name.removeprefix(
            "model.proprio_embedder.mlp."
        )
    return None


def load_g05_backbone(model: G05Model, checkpoint: str | Path) -> G05LoadReport:
    """Load every shape-compatible vision/text tensor and fail on any gap."""
    config = model.config
    expected = {
        "action_token_begin_idx": G05_ACTION_BEGIN,
        "eov_token_id": G05_EOV_ID,
        "state_token_id": G05_STATE_ID,
        "text_vocab_size": G05_VOCAB_SIZE,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"G0.5 base checkpoint requires {name}={value}, got {getattr(config, name)}"
            )
    if config.action_token_end_idx > (G05_ACTION_BEGIN + G05_ACTION_ROWS):
        raise ValueError("action rows exceed G0.5's exact 4096-token action interval")


    if config.proprio_dim == 0:
        expected_proprio = {
            "proprio_dim": 0,
            "internal_proprio_dim": 0,
            "proprio_indices": [],
        }
        for name, value in expected_proprio.items():
            if getattr(config, name) != value:
                raise ValueError(
                    f"G0.5 proprio mode requires {name}={value}, "
                    f"got {getattr(config, name)}"
                )
    else:
        if config.internal_proprio_dim != G05_INTERNAL_PROPRIO_DIM:
            raise ValueError(
                "the released G0.5 proprio adapter takes "
                f"{G05_INTERNAL_PROPRIO_DIM} internal slots, got "
                f"{config.internal_proprio_dim}"
            )
        slots = list(config.proprio_indices)
        if len(slots) != config.proprio_dim or len(set(slots)) != len(slots):
            raise ValueError("G0.5 proprio_indices must uniquely map every physical dimension")
        if slots and not 0 <= min(slots) <= max(slots) < G05_INTERNAL_PROPRIO_DIM:
            raise ValueError("a G0.5 proprio index lies outside the released adapter")

    path = Path(checkpoint)
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(raw) != {"model_state_dict"}:
        raise ValueError("expected the released G0.5 base model_state_dict wrapper")
    source = raw["model_state_dict"]
    input_weight = source.get("model.vlm.input_proj.weight")
    output_weight = source.get("model.vlm.output_proj.weight")
    if input_weight is None or output_weight is None:
        raise ValueError("G0.5 checkpoint is missing a VLM input/output projection")
    if input_weight.shape != (G05_VOCAB_SIZE, config.text_hidden_size):
        raise ValueError("G0.5 checkpoint has an unexpected shared-vocabulary shape")
    if not torch.equal(input_weight, output_weight):
        raise ValueError("G0.5 input/output vocabulary projections are not tied exactly")
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for source_name, tensor in source.items():
        target_name = _target_name(source_name)
        if target_name is None:
            continue
        if target_name not in target and target_name.startswith("proprio_embedder."):


            continue
        if target_name not in target:
            raise KeyError(f"mapped G0.5 tensor has no target: {target_name}")
        if tensor.shape != target[target_name].shape:
            raise ValueError(
                f"G0.5 shape mismatch for {source_name} -> {target_name}: "
                f"{tuple(tensor.shape)} != {tuple(target[target_name].shape)}"
            )
        mapped[target_name] = tensor

    required = {
        name
        for name in target
        if name.startswith(("vision_tower.", "vlm.", "proprio_embedder."))
    }
    missing_backbone = sorted(required - mapped.keys())
    if missing_backbone:
        raise ValueError(
            "G0.5 did not initialize the complete VLM/proprio backbone: "
            + ", ".join(missing_backbone[:8])
        )
    incompatible = model.load_state_dict(mapped, strict=False)
    if incompatible.unexpected_keys:
        raise AssertionError(f"unexpected mapped tensors: {incompatible.unexpected_keys}")
    initialized_names = tuple(sorted(set(target) - set(mapped)))
    action_history_parameters = 0
    action_history_initialization = None
    if model.action_history_embedder is not None:


        action_history_parameters = sum(
            parameter.numel() for parameter in model.action_history_embedder.parameters()
        )
        action_history_initialization = "default"
    loaded_parameters = sum(target[name].numel() for name in mapped)
    target_parameters = sum(tensor.numel() for tensor in target.values())
    mapped_source_names = {
        source_name for source_name in source if _target_name(source_name) is not None
    }
    mapped_source_names.add("model.vlm.output_proj.weight")
    return G05LoadReport(
        checkpoint=str(path.resolve()),
        loaded_tensors=len(mapped),
        loaded_parameters=loaded_parameters,
        target_parameters=target_parameters,
        initialized_parameters=target_parameters - loaded_parameters,
        ignored_source_tensors=len(source) - len(mapped_source_names),
        initialized_tensor_names=initialized_names,
        action_history_initialization=action_history_initialization,
        action_history_initialized_parameters=action_history_parameters,
    )
