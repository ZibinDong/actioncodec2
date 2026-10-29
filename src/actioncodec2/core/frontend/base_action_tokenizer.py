"""Abstract base for frontend action tokenizers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch

from ._types import ActionIdxBatch, DecodeResult, TokenIds

logger = logging.getLogger(__name__)


class BaseActionTokenizer(ABC):
    """Base action tokenizer: maps actions ↔ discrete action-token ids."""

    BAR_SENTINEL_TOKENS = ("<bos_blk>", "<eos_blk>", "<pad_action_token>")

    encode_kwargs: frozenset[str] = frozenset()

    drops_noop_parts: bool = False

    parts_meta: dict[str, int] | None = None

    def __init__(self, *, append_bar_sentinels: bool = True) -> None:
        required = (
            "action_tokenizer",
            "_action_vocab_size",
            "_new_action_tokens",
            "max_action_token_len",
            "fixed_action_token_len",
        )
        missing = [a for a in required if not hasattr(self, a)]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.__init__ must set {missing} before super().__init__()"
            )
        if self.max_action_token_len is not None:
            self.max_action_token_len = int(self.max_action_token_len)
            if self.max_action_token_len <= 0:
                raise ValueError(
                    f"{type(self).__name__}: max_action_token_len must be positive, "
                    f"got {self.max_action_token_len}"
                )
        if self.fixed_action_token_len is not None:
            self.fixed_action_token_len = int(self.fixed_action_token_len)
            if self.max_action_token_len is None:
                raise ValueError(
                    f"{type(self).__name__}: fixed_action_token_len requires a finite "
                    "max_action_token_len"
                )
            if self.fixed_action_token_len != self.max_action_token_len:
                raise ValueError(
                    f"{type(self).__name__}: fixed_action_token_len="
                    f"{self.fixed_action_token_len} must equal max_action_token_len="
                    f"{self.max_action_token_len} — a fixed-length row is its own maximum"
                )
        base = int(self._action_vocab_size)
        self._new_action_tokens = list(self._new_action_tokens)
        if append_bar_sentinels:
            self._new_action_tokens += list(self.BAR_SENTINEL_TOKENS)
            self.bos_blk_id: int | None = base
            self.eos_blk_id: int | None = base + 1
            self.pad_action_id: int | None = base + 2
            self._action_vocab_size = base + len(self.BAR_SENTINEL_TOKENS)
        else:
            self.bos_blk_id = self.eos_blk_id = self.pad_action_id = None

    @property
    def new_action_tokens(self) -> list[str]:
        """Token strings in id order — downstream registers these into its VLM vocab."""
        return list(self._new_action_tokens)

    def to_token_strs(self, token_ids: TokenIds) -> list[str]:
        """Render one id row as its token strings, e.g. ``<right_arm_0>`` / ``<action0042>``."""
        row = self._to_id_row(token_ids).tolist()
        n = len(self._new_action_tokens)
        bad = [i for i in row if not (0 <= i < n)]
        if bad:
            raise ValueError(f"token ids {bad[:5]} out of range [0, {n})")
        return [self._new_action_tokens[i] for i in row]

    def to(self, device_id):
        """Move the backend to ``device_id`` and record it as ``self.device``."""
        self.device = torch.device(device_id)
        self.action_tokenizer = self.action_tokenizer.to(device_id)
        return self

    def _check_encode_kwargs(self, encode_kwargs: dict | None) -> dict:
        kw = dict(encode_kwargs or {})
        unknown = sorted(set(kw) - self.encode_kwargs)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} does not consume encode_kwargs {unknown}; "
                f"it declares {sorted(self.encode_kwargs)}"
            )
        return kw

    def __call__(
        self,
        action: torch.Tensor,
        *,
        encode_kwargs: dict | None = None,
    ) -> ActionIdxBatch:
        return self._encode_action_indices(
            action, encode_kwargs=self._check_encode_kwargs(encode_kwargs)
        )

    def _strip_bar_sentinels(self, row: torch.Tensor) -> torch.Tensor:
        if self.bos_blk_id is None:
            return row
        keep = (row != self.bos_blk_id) & (row != self.eos_blk_id) & (row != self.pad_action_id)
        return row[keep]

    @staticmethod
    def _to_id_row(token_ids: TokenIds) -> torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            row = token_ids.detach().cpu()
            if row.dtype == torch.bool or row.dtype.is_floating_point or row.dtype.is_complex:
                raise ValueError("token ids must be integers")
            row = row.to(torch.long)
        else:
            values = np.asarray(token_ids)
            if values.dtype.kind not in {"i", "u"} and values.size:
                raise ValueError("token ids must be integers")
            row = torch.from_numpy(np.asarray(values, dtype=np.int64))
        if row.ndim != 1:
            raise ValueError(f"token id row must be 1D, got {tuple(row.shape)}")
        return row

    def _checked_decode_row(self, token_ids: TokenIds) -> torch.Tensor:
        row = self._to_id_row(token_ids)
        invalid = (row < 0) | (row >= int(self._action_vocab_size))
        if torch.any(invalid):
            bad = row[invalid][:5].tolist()
            raise ValueError(f"token ids {bad} out of range [0, {int(self._action_vocab_size)})")
        return self._strip_bar_sentinels(row)

    def decode_token_ids_to_actions(
        self,
        action_token_ids: TokenIds,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> DecodeResult:
        """Decode one action-token id row back to a ``(T, D)`` action chunk."""
        row = self._checked_decode_row(action_token_ids)
        actions_np, absent_keys_list = self._decode_action_indices(
            [row.tolist()],
            time_horizon=time_horizon,
            action_dim=action_dim,
            decode_kwargs=decode_kwargs,
        )
        return DecodeResult(
            action=torch.from_numpy(actions_np)[0],
            absent_keys=absent_keys_list[0] if absent_keys_list else set(),
        )

    def decode_token_ids_to_actions_batch(
        self,
        action_token_ids: Sequence[TokenIds],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Decode a batch of id rows to actions ``(B, T, D)``."""
        batch_action_idx: ActionIdxBatch = [
            self._checked_decode_row(a).tolist() for a in action_token_ids
        ]
        actions_np, _ = self._decode_action_indices(
            batch_action_idx,
            time_horizon=time_horizon,
            action_dim=action_dim,
            decode_kwargs=decode_kwargs,
        )
        return torch.from_numpy(actions_np)

    @property
    def vocab_size(self) -> int:
        return int(self._action_vocab_size)

    @abstractmethod
    def _encode_action_indices(
        self, action: torch.Tensor, encode_kwargs: dict | None = None
    ) -> ActionIdxBatch: ...

    @abstractmethod
    def _decode_action_indices(
        self,
        action_indices: ActionIdxBatch,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, list[set[str]]]: ...
