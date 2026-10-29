# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Raw text/image preparation for the one-step VLA.

This module deliberately does not depend on LeRobot. Its output is a plain
``dict[str, Tensor]`` that can be consumed by PyTorch or Hugging Face trainers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, ProcessorMixin

from actioncodec2.vla.g05_hf.configuration_g05 import G05Config


class G05Processor(ProcessorMixin):
    """Build G0.5 text/image prompts from tasks and current camera frames.

    Images use ``[batch, camera, channel, height, width]``. Floating-point
    inputs are in ``[0, 1]``; uint8 inputs are in ``[0, 255]``.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, tokenizer: Any, config: G05Config) -> None:
        if not hasattr(tokenizer, "encode"):
            raise TypeError("tokenizer must provide encode(text, add_special_tokens=False)")
        super().__init__(tokenizer=tokenizer)
        self.config = config

    def to_dict(self) -> dict[str, str]:
        # The model's config.json owns architectural settings; the processor
        # file only identifies this processor for AutoProcessor.
        return {"processor_class": self.__class__.__name__}

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config: G05Config | None = None,
        **kwargs,
    ) -> G05Processor:
        if config is None:
            config = G05Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(tokenizer, config)

    @staticmethod
    def pad_action_rows(
        rows: Sequence[Sequence[int] | Tensor],
        *,
        device: torch.device | str | None = None,
        width: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Right-pad ragged action-token rows and return IDs plus a validity mask."""
        if not rows:
            raise ValueError("at least one action-token row is required")
        normalized = [
            torch.as_tensor(row, dtype=torch.long, device=device).flatten() for row in rows
        ]
        required_width = max((row.numel() for row in normalized), default=0)
        width = required_width if width is None else width
        if width < required_width:
            raise ValueError(
                f"configured action-token width {width} is smaller than required {required_width}"
            )
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
        for index, row in enumerate(normalized):
            ids[index, : row.numel()] = row
            mask[index, : row.numel()] = True
        return ids, mask

    def _normalize_current_images(self, images: Tensor) -> Tensor:
        # [B,C,H,W] is a single camera; [B,Cam,C,H,W] is multi-camera.
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError("current_images must have shape [B,C,H,W] or [B,Cam,C,H,W]")
        return images.unsqueeze(2)

    def prepare_images(self, current_images: Tensor, *, defer_normalization: bool = False) -> Tensor:
        """Prepare [B, Cam, 1, 3, H, W] images. Deferred normalization preserves uint8 inputs at the configured size; resized outputs are floats in [-1, 1]."""
        images = self._normalize_current_images(torch.as_tensor(current_images))
        if images.shape[3] != 3:
            raise ValueError("the Qwen3.5 vision tower requires RGB images")

        batch, cameras, frames, channels, height, width = images.shape
        if (
            defer_normalization
            and images.dtype == torch.uint8
            and (height, width) == self.config.image_size
        ):
            return images
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        if (height, width) != self.config.image_size:
            images = F.interpolate(
                images.reshape(batch * cameras * frames, channels, height, width),
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(batch, cameras, frames, channels, *self.config.image_size)
        return images * 2.0 - 1.0

    def _encode_text(self, text: str) -> list[int]:
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if any(token < 0 or token >= self.config.text_vocab_size for token in ids):
            raise ValueError("the text tokenizer emitted an ID outside text_vocab_size")
        return ids

    def _prompt_row(
        self,
        task: str,
        num_frames: int,
        *,
        history_slots: int,
        state_slots: int,
    ) -> tuple[list[int], list[bool], list[bool]]:
        """Build one prompt row and its continuous-input position masks."""
        task = task.strip()
        row: list[int] = []
        for _ in range(num_frames):
            row.append(self.config.vision_start_token_id)
            row.extend([self.config.image_token_id] * self.config.image_tokens_per_frame)
            row.append(self.config.vision_end_token_id)
        row.extend(self._encode_text(f"Embodiment: {self.config.embodiment}; Task: "))
        row.extend(self._encode_text(task)[: self.config.max_task_tokens])
        history_positions = [False] * len(row)
        state_positions = [False] * len(row)
        if history_slots:
            memory_prefix = self._encode_text(" Memory: ")
            row.extend(memory_prefix)
            history_positions.extend([False] * len(memory_prefix))
            state_positions.extend([False] * len(memory_prefix))
            row.extend([self.config.pad_token_id] * history_slots)
            history_positions.extend([True] * history_slots)
            state_positions.extend([False] * history_slots)
        if state_slots:
            state_prefix = self._encode_text(" State: ")
            row.extend(state_prefix)
            history_positions.extend([False] * len(state_prefix))
            state_positions.extend([False] * len(state_prefix))
            row.extend([self.config.state_token_id] * state_slots)
            history_positions.extend([False] * state_slots)
            state_positions.extend([True] * state_slots)
        row.extend(self._encode_text(";Action: "))
        history_positions.extend([False] * (len(row) - len(history_positions)))
        state_positions.extend([False] * (len(row) - len(state_positions)))
        row.append(self.config.eov_token_id)
        history_positions.append(False)
        state_positions.append(False)
        if len(row) > self.config.max_prompt_length:
            raise ValueError("VLA prompt exceeds max_prompt_length")
        if not (len(row) == len(history_positions) == len(state_positions)):
            raise AssertionError("prompt slot bookkeeping is inconsistent")
        return row, history_positions, state_positions

    def __call__(
        self,
        tasks: str | Sequence[str],
        current_images: Tensor,
        *,
        action_labels: Tensor | None = None,
        action_label_mask: Tensor | None = None,
        continuous_action_history: Tensor | None = None,
        continuous_action_history_mask: Tensor | None = None,
        proprio: Tensor | None = None,
        defer_image_normalization: bool = False,
    ) -> dict[str, Tensor]:
        pixel_values = self.prepare_images(
            current_images, defer_normalization=defer_image_normalization
        )
        task_rows = [tasks] if isinstance(tasks, str) else list(tasks)
        if len(task_rows) != pixel_values.shape[0] or not all(
            isinstance(task, str) for task in task_rows
        ):
            raise ValueError("tasks must contain one string per batch item")

        history_slots = (
            0 if continuous_action_history is None
            else int(torch.as_tensor(continuous_action_history).shape[1])
        )
        if (
            continuous_action_history is not None
            and history_slots != self.config.action_history_steps
        ):
            raise ValueError(
                "continuous action history width must equal configured action_history_steps"
            )
        state_slots = int(proprio is not None)
        if state_slots != int(self.config.proprio_dim > 0):
            raise ValueError("proprio presence must match the model configuration")

        _, cameras, frames = pixel_values.shape[:3]
        packed_rows = [
            self._prompt_row(
                task,
                cameras * frames,
                history_slots=history_slots,
                state_slots=state_slots,
            )
            for task in task_rows
        ]
        rows = [item[0] for item in packed_rows]
        required_width = max(map(len, rows))
        width = self.config.pad_prompt_to_length or required_width
        if width < required_width:
            raise ValueError(
                f"configured prompt width {width} is smaller than required {required_width}"
            )
        input_ids = torch.full(
            (len(rows), width),
            self.config.pad_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_history_position_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        proprio_position_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            row_tensor = torch.tensor(row, dtype=torch.long, device=pixel_values.device)
            input_ids[index, -len(row) :] = row_tensor
            attention_mask[index, -len(row) :] = True
            action_history_position_mask[index, -len(row) :] = torch.tensor(
                packed_rows[index][1], dtype=torch.bool, device=pixel_values.device
            )
            proprio_position_mask[index, -len(row) :] = torch.tensor(
                packed_rows[index][2], dtype=torch.bool, device=pixel_values.device
            )

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action_history_position_mask": action_history_position_mask,
            "proprio_position_mask": proprio_position_mask,
        }
        optional = {
            "action_labels": action_labels,
            "action_label_mask": action_label_mask,
            "continuous_action_history": continuous_action_history,
            "continuous_action_history_mask": continuous_action_history_mask,
            "proprio": proprio,
        }
        for name, value in optional.items():
            if value is not None:
                output[name] = torch.as_tensor(value, device=pixel_values.device)
        return output
