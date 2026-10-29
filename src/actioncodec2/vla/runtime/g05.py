"""G0.5 input processing, proprio normalization, and model construction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from actioncodec2.vla.training.config import resolve_base_checkpoint, resolve_text_tokenizer

from actioncodec2.vla.g05_hf.configuration_g05 import (
    G05_ACTION_BEGIN,
    G05_ACTION_ROWS,
    G05_EOV_ID,
    G05_INTERNAL_PROPRIO_DIM,
    G05_STATE_ID,
    G05_VOCAB_SIZE,
    LEGACY_MODEL_TYPES,
)


G05_RIGHT_CONTROL_SLOT = 10
G05_RIGHT_GRIPPER_SLOT = 19


class G05ProprioTransform:
    """Normalize a state vector using the supplied statistics."""

    def __init__(
        self,
        source: str | Path | dict[str, object],
        *,
        quantile_indices: Sequence[int] | None = None,
    ) -> None:
        payload = (
            dict(source)
            if isinstance(source, dict)
            else json.loads(Path(source).read_text(encoding="utf-8"))
        )
        declared = payload.get("quantile_indices")
        if declared is not None:
            declared = tuple(int(index) for index in declared)
            if quantile_indices is not None and tuple(quantile_indices) != declared:
                raise ValueError(
                    f"the statistics declare quantile dimensions {declared} but the "
                    f"caller asked for {tuple(quantile_indices)}"
                )
            quantile_indices = declared
        elif quantile_indices is None:
            quantile_indices = (6,)
        self.mean = np.asarray(payload["train_mean"], dtype=np.float32)
        self.std = np.asarray(payload["train_std"], dtype=np.float32)
        self.q01 = np.asarray(payload["q01"], dtype=np.float32)
        self.q99 = np.asarray(payload["q99"], dtype=np.float32)
        self.dimension = int(self.mean.shape[0])
        if not (
            self.mean.shape == self.std.shape == self.q01.shape == self.q99.shape
        ) or self.mean.ndim != 1:
            raise ValueError("G0.5 proprio statistics must be four vectors of one width")
        if not all(np.isfinite(values).all() for values in (self.mean, self.std, self.q01, self.q99)):
            raise ValueError("G0.5 proprio statistics must be finite")
        self.quantile_indices = tuple(int(index) for index in quantile_indices)
        if any(not 0 <= index < self.dimension for index in self.quantile_indices):
            raise ValueError("a proprio quantile index lies outside the statistics")
        if np.any(self.q99[list(self.quantile_indices)] <= self.q01[list(self.quantile_indices)]):
            raise ValueError("a q01/q99 proprio dimension has an empty range")
        if np.any(self.std < 0):
            raise ValueError("invalid G0.5 proprio statistics")

    def to_json(self) -> dict[str, object]:
        return {
            "train_mean": self.mean.tolist(),
            "train_std": self.std.tolist(),
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "quantile_indices": list(self.quantile_indices),
        }

    def __call__(self, state: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"G0.5 proprio must have shape [{self.dimension}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("G0.5 proprio must be finite")
        scale = np.maximum(self.std, np.float32(1e-8))
        normalized = (values - self.mean) / scale
        constant = self.std < np.float32(1e-4)
        normalized[constant] = values[constant] - self.mean[constant]
        for index in self.quantile_indices:
            normalized[index] = (
                2.0 * (values[index] - self.q01[index]) / (self.q99[index] - self.q01[index])
                - 1.0
            )
        return np.clip(normalized, -5.0, 5.0).astype(np.float32, copy=False)


class G05ZeroProprioTransform:
    """Return the constant native state used by vision-only Bridge policies."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = int(dimension)

    def to_json(self) -> dict[str, object]:
        return {"mode": "zero", "dimension": self.dimension}

    def __call__(self, state: np.ndarray | None = None) -> np.ndarray:
        del state
        return np.zeros(self.dimension, dtype=np.float32)


class G05ActionHistoryTransform:
    """Transform a raw previous action into the model input format."""

    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        if self.low.ndim != 1 or self.high.shape != self.low.shape:
            raise ValueError("action-history bounds must be matching vectors")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
            raise ValueError("action-history bounds must be finite")
        if np.any(self.high <= self.low):
            raise ValueError("every action-history upper bound must exceed its lower bound")

    def __call__(self, action: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != self.low.shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"previous action must be a finite length-{len(self.low)} vector"
            )
        normalized = 2.0 * (values - self.low) / (self.high - self.low) - 1.0
        return normalized.astype(np.float32, copy=False)


class G05RunProcessor:
    """Turn raw observations and action tokens into G0.5 model inputs."""

    def __init__(
        self,
        config,
        action_tokenizer,
        *,
        tokenizer_path: str | Path | None = None,
        proprio_stats: dict[str, object] | None = None,
    ) -> None:
        from transformers import Qwen2Tokenizer

        from actioncodec2.vla.g05_hf.configuration_g05 import G05Config
        from actioncodec2.vla.g05_hf.modeling_g05 import liger_kernel_available
        from actioncodec2.vla.g05_hf.processing_g05 import G05Processor

        self.action_vocab_size = int(action_tokenizer.vocab_size)
        if self.action_vocab_size > G05_ACTION_ROWS:
            raise ValueError(
                f"G0.5 has {G05_ACTION_ROWS} native action rows; the tokenizer needs "
                f"{self.action_vocab_size}"
            )
        self.action_token_begin = G05_ACTION_BEGIN
        self.total_vocab_size = G05_VOCAB_SIZE
        self.image_size = tuple(config.model.image_size)
        self.action_dim = int(config.model.action_dim)

        self.compute_dtype = config.model.dtype


        override = config.model.force_previous_action_input
        self.requires_previous_action = (
            bool(action_tokenizer.requires_previous_action)
            if override is None
            else bool(override)
        )
        if self.requires_previous_action:
            low = getattr(action_tokenizer, "action_low", None)
            high = getattr(action_tokenizer, "action_high", None)
            if low is None or high is None:
                raise ValueError(
                    "a previous-action input needs action bounds; set "
                    "tokenizer.action_stats or tokenizer.action_clip"
                )
            self.action_history_transform = G05ActionHistoryTransform(low, high)
        else:
            self.action_history_transform = None
        self.source_indices = tuple(config.model.proprio_source_indices)
        self.state_dim = len(self.source_indices)


        self.internal_indices = tuple(
            config.model.proprio_internal_indices
            or (G05_RIGHT_CONTROL_SLOT + np.arange(self.state_dim - 1)).tolist()
            + [G05_RIGHT_GRIPPER_SLOT]
        )
        self.proprio_mode = config.model.proprio_mode
        if self.proprio_mode == "none":
            self.proprio_transform = None
        elif self.proprio_mode == "zero":
            self.proprio_transform = G05ZeroProprioTransform(1 + max(self.source_indices))
        else:
            self.proprio_transform = G05ProprioTransform(
                config.model.proprio_stats if proprio_stats is None else proprio_stats,
                quantile_indices=config.model.proprio_quantile_indices,
            )
        uses_proprio = self.proprio_mode != "none"
        self.model_config = G05Config(
            action_vocab_size=self.action_vocab_size,
            action_token_begin_idx=self.action_token_begin,
            text_vocab_size=G05_VOCAB_SIZE,
            eov_token_id=G05_EOV_ID,
            state_token_id=G05_STATE_ID,
            embodiment=config.model.embodiment,
            image_size=self.image_size,
            max_task_tokens=200,
            max_prompt_length=512,
            max_action_tokens=int(action_tokenizer.max_tokens_per_chunk),
            proprio_dim=self.state_dim if uses_proprio else 0,
            internal_proprio_dim=G05_INTERNAL_PROPRIO_DIM if uses_proprio else 0,
            proprio_indices=list(self.internal_indices) if uses_proprio else [],
            continuous_action_dim=self.action_dim if self.requires_previous_action else 0,
            action_history_steps=1 if self.requires_previous_action else 0,
            attn_implementation=config.model.attn_implementation,
            vision_attn_implementation=config.model.attn_implementation,
            use_flash_linear_attention=True,
            use_fused_linear_cross_entropy=liger_kernel_available(),
            action_only_cross_entropy=bool(config.model.action_only_loss),
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            resolve_text_tokenizer(
                config.model.tokenizer_path if tokenizer_path is None else tokenizer_path
            )
        )
        vocabulary = tokenizer.get_vocab()
        if len(tokenizer) != G05_ACTION_BEGIN or max(vocabulary.values()) != G05_ACTION_BEGIN - 1:
            raise ValueError("the configured tokenizer is not the G0.5 Qwen3.5 base tokenizer")
        if tokenizer.bos_token_id is not None:
            raise ValueError("the G0.5 base tokenizer must not inject a BOS token")
        special = {
            "image": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            "vision_start": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "vision_end": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        }
        expected_special = {
            "image": self.model_config.image_token_id,
            "vision_start": self.model_config.vision_start_token_id,
            "vision_end": self.model_config.vision_end_token_id,
        }
        if special != expected_special:
            raise ValueError(f"G0.5 tokenizer multimodal IDs differ: {special}")
        self.tokenizer = tokenizer
        self.processor = G05Processor(tokenizer, self.model_config)

    def build_sample(
        self,
        frames: Sequence[np.ndarray],
        task: str,
        action_tokens: Sequence[int],
        *,
        state: np.ndarray | None = None,
        previous_action: np.ndarray | None = None,
    ) -> dict[str, object]:
        if self.requires_previous_action and previous_action is None:
            raise ValueError("this policy requires the previous action")
        if not self.requires_previous_action and previous_action is not None:
            raise ValueError("this policy does not take a previous action")
        if state is None and self.proprio_mode == "observed":
            raise ValueError("observed-proprio mode requires the current robot state")
        images = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])
        sample = {
            "task": str(task),
            "current_image": torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),
            "action_labels": torch.as_tensor(action_tokens, dtype=torch.long),
        }
        if self.proprio_transform is not None:
            normalized = self.proprio_transform(
                None if state is None else np.asarray(state)
            )[list(self.source_indices)]
            sample["proprio"] = torch.from_numpy(normalized)
        if self.requires_previous_action:
            normalized_history = self.action_history_transform(previous_action)
            sample["continuous_action_history"] = torch.from_numpy(normalized_history[None, :])
            sample["continuous_action_history_mask"] = torch.ones(1, dtype=torch.bool)
        return sample

    def collator(self) -> G05Collator:
        return G05Collator(self.processor)

    def inference_inputs(
        self,
        frames: Sequence[np.ndarray],
        task: str,
        state: np.ndarray | None,
        previous_action: np.ndarray | None = None,
        *,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        return self.batch_inference_inputs([(frames, task, state, previous_action)], device=device)

    def batch_inference_inputs(
        self,
        rows: Sequence[
            tuple[Sequence[np.ndarray], str, np.ndarray | None, np.ndarray | None]
        ],
        *,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Pack ``(frames, task, state, previous_action)`` rows into one batch.

        Prompts are right-aligned, so rows whose tasks tokenize to different
        lengths still all end on ``<EOV>``, exactly as in training.
        """
        if not rows:
            raise ValueError("at least one observation is required")
        samples = [
            self.build_sample(frames, task, (), state=state, previous_action=previous)
            for frames, task, state, previous in rows
        ]
        return _collate(self.processor, samples, device=device)


def _collate(
    processor, rows: Sequence[dict[str, object]], *, device: torch.device | str | None = None
) -> dict[str, torch.Tensor]:
    kwargs = {}
    for key in ("proprio", "continuous_action_history", "continuous_action_history_mask"):
        if key in rows[0]:
            kwargs[key] = torch.stack([row[key] for row in rows])
    images = torch.stack([row["current_image"] for row in rows])
    if device is not None:
        images = images.to(device)
        kwargs = {key: value.to(device) for key, value in kwargs.items()}
    return processor(
        [row["task"] for row in rows], images, defer_image_normalization=True, **kwargs
    )


class G05Collator:
    """Batch :meth:`G05RunProcessor.build_sample` outputs for training."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, rows: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        from actioncodec2.vla.g05_hf.processing_g05 import G05Processor

        batch = _collate(self.processor, rows)
        labels, label_mask = G05Processor.pad_action_rows([row["action_labels"] for row in rows])
        batch["action_labels"] = labels
        batch["action_label_mask"] = label_mask
        return batch


def build_g05_model(config, processor: G05RunProcessor, *, device: str):
    """Build G0.5 in FP32 and load the released base checkpoint strictly.

    Parameters stay FP32; ``model.dtype`` selects the autocast compute dtype.
    """
    from actioncodec2.vla.g05_hf import register_huggingface_auto_classes
    from actioncodec2.vla.g05_hf.checkpoint_conversion import load_g05_backbone
    from actioncodec2.vla.g05_hf.modeling_g05 import G05Model, apply_liger_kernels

    register_huggingface_auto_classes()
    model = G05Model(processor.model_config)
    report = load_g05_backbone(model, resolve_base_checkpoint(config.model.path))
    apply_liger_kernels(model, device=device)
    if config.model.vision_gradient_checkpointing:
        model.vision_tower.gradient_checkpointing_enable({"use_reentrant": False})
    model._g05_load_report = report.to_dict()
    return model.to(device=device, dtype=torch.float32)


def load_g05_config(path: str | Path):
    """Read a saved :class:`G05Config`, including checkpoints with a legacy type."""
    from transformers.utils import cached_file

    from actioncodec2.vla.g05_hf.configuration_g05 import G05Config

    payload = json.loads(Path(cached_file(str(path), "config.json")).read_text(encoding="utf-8"))
    model_type = payload.get("model_type")
    if model_type in LEGACY_MODEL_TYPES:
        payload["model_type"] = G05Config.model_type
    elif model_type != G05Config.model_type:
        raise TypeError(f"{path} is not a G0.5 checkpoint (model_type={model_type!r})")
    return G05Config.from_dict(payload)


def load_g05_model(path: str | Path, *, device: str, dtype: torch.dtype = torch.float32):
    """Load a G0.5 policy with the requested weight dtype."""
    from actioncodec2.vla.g05_hf.modeling_g05 import G05Model, apply_liger_kernels

    model, loading = G05Model.from_pretrained(
        str(path), config=load_g05_config(path), dtype=dtype, output_loading_info=True,
    )
    failures = {key: loading[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if failures:
        raise RuntimeError(f"policy weights do not match the G0.5 implementation: {failures}")
    apply_liger_kernels(model, device=device)
    return model.to(device).eval()
