"""Shared lightweight types for the frontend layer."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

ActionIdxBatch = list[list[int]]
TokenIds = list[int] | np.ndarray | torch.Tensor


class DecodeResult(NamedTuple):
    action: torch.Tensor
    absent_keys: set


__all__ = ["ActionIdxBatch", "TokenIds", "DecodeResult"]
