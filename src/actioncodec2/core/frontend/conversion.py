from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import torch

from ..backends.actioncodec2 import (
    PresetConfig,
)


def _numpy(value):
    return value.detach().cpu().float().numpy() if isinstance(value, torch.Tensor) else value


def _numpy_token_rows(token_ids):
    if isinstance(token_ids, torch.Tensor):
        if (
            token_ids.dtype == torch.bool
            or token_ids.dtype.is_floating_point
            or token_ids.dtype.is_complex
        ):
            raise ValueError("token ids must be integers")
        values = token_ids.detach().cpu().long().numpy()
        if values.ndim == 2:
            return values.tolist()
    elif isinstance(token_ids, np.ndarray) and token_ids.ndim == 2:
        if token_ids.dtype.kind not in {"i", "u"}:
            raise ValueError("token ids must be integers")
        return np.asarray(token_ids, dtype=np.int64).tolist()
    rows = []
    for row in token_ids:
        if isinstance(row, torch.Tensor):
            if row.dtype == torch.bool or row.dtype.is_floating_point or row.dtype.is_complex:
                raise ValueError("token ids must be integers")
            values = row.detach().cpu().long().numpy()
        else:
            values = np.asarray(row)
            if values.dtype.kind not in {"i", "u"} and values.size:
                raise ValueError("token ids must be integers")
            values = np.asarray(values, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError(f"token id row must be 1D, got {values.shape}")
        rows.append(values.tolist())
    return rows


def restore_source_layout(preset: PresetConfig, decoded: Mapping[str, np.ndarray]) -> np.ndarray:
    """Place ``{component: (B,T,D_c)}`` outputs back into the preset's ``(B,T,D)`` columns."""
    first = next(iter(decoded.values()))
    batch, steps = first.shape[:2]
    if any(values.shape[:2] != (batch, steps) for values in decoded.values()):
        raise RuntimeError("decoded components disagree on batch or sequence length")
    actions = np.empty((batch, steps, preset.input_dim), dtype=np.float32)
    for component in preset.components:
        values = decoded[component.name]
        motion_width = len(component.source_dims)
        actions[:, :, component.source_dims] = values[:, :, :motion_width]
        if component.gripper_dim is not None:
            actions[:, :, component.gripper_dim] = values[:, :, motion_width]
    return actions
