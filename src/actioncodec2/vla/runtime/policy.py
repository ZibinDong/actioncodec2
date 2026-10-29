"""G0.5 policy inference with a released action tokenizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import math
from typing import Any

import numpy as np
import torch

__all__ = ["ActionPolicy"]


class ActionPolicy:
    """Predict raw-unit action chunks from camera frames and a task prompt.

    ``temperature == 0`` decodes greedily; a positive value samples with that
    temperature (top-k/top-p come from the model config).
    """

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        *,
        device: str = "cuda",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype

    @property
    def requires_previous_action(self) -> bool:
        """Whether the action tokenizer needs the previous action to decode."""
        return bool(self.tokenizer.requires_previous_action)

    @property
    def needs_previous_action_input(self) -> bool:
        """Whether callers must pass ``previous_action``: for the tokenizer's boundary,
        the model's action-history input, or both."""
        return bool(self.requires_previous_action or self.processor.requires_previous_action)

    def _previous_action(self, value: np.ndarray | None) -> np.ndarray | None:
        if not self.needs_previous_action_input:
            return None
        if value is None:
            raise ValueError(f"{self.tokenizer.name} policy requires previous_action")
        previous = np.asarray(value, np.float32)
        if previous.shape != (self.processor.action_dim,) or not np.isfinite(previous).all():
            raise ValueError(f"previous_action must be finite [{self.processor.action_dim}]")
        return previous

    def _autocast(self):
        if self.device.type != "cuda" or self.compute_dtype == torch.float32:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.compute_dtype)

    def _generate(self, inputs: dict[str, torch.Tensor], *, temperature: float):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        with self._autocast():
            return self.model.generate_action_tokens(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["pixel_values"],
                self.tokenizer.grammar,
                continuous_action_history=inputs.get("continuous_action_history"),
                continuous_action_history_mask=inputs.get("continuous_action_history_mask"),
                proprio=inputs.get("proprio"),
                action_history_position_mask=inputs["action_history_position_mask"],
                proprio_position_mask=inputs["proprio_position_mask"],
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                max_new_tokens=self.tokenizer.max_tokens_per_chunk,
            )

    def _decode(self, streams, previous: list[np.ndarray | None]):
        rows = [stream.cpu().tolist() for stream in streams]
        if not all(self.tokenizer.grammar.is_complete(tokens) for tokens in rows):
            raise RuntimeError("G0.5 emitted an incomplete action chunk")
        if self.requires_previous_action:
            actions = self.tokenizer.decode_batch(rows, np.stack(previous))
        else:
            actions = self.tokenizer.decode_batch(rows)
        return list(zip(actions, rows, strict=True))

    @torch.inference_mode()
    def predict(
        self,
        frames: Sequence[np.ndarray],
        task: str,
        *,
        state: np.ndarray | None = None,
        previous_action: np.ndarray | None = None,
        temperature: float = 0.0,
    ) -> tuple[np.ndarray, list[int]]:
        """Return ``(actions [horizon, action_dim], tokens)`` for one observation."""
        return self.predict_batch(
            [{"frames": frames, "task": task, "state": state, "previous_action": previous_action}],
            temperature=temperature,
        )[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> list[tuple[np.ndarray, list[int]]]:
        """Batched :meth:`predict`; each observation is a mapping of its arguments."""
        rows = list(observations)
        if not rows:
            return []
        if self.processor.proprio_mode == "observed" and any(
            row.get("state") is None for row in rows
        ):
            raise ValueError("this policy requires the current robot state")
        previous = [self._previous_action(row.get("previous_action")) for row in rows]
        model_previous = previous if self.processor.requires_previous_action else [None] * len(rows)
        inputs = self.processor.batch_inference_inputs(
            [
                (row["frames"], row["task"], row.get("state"), action)
                for row, action in zip(rows, model_previous, strict=True)
            ],
            device=self.device,
        )
        streams = self._generate(inputs, temperature=temperature)
        return self._decode(streams, previous)
