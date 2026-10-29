"""Explicit ActionCodec2Sparse token constraints for Hugging Face generate()."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from ...backends.actioncodec2.cache import BoundedCache


class ActionCodec2SparseLogitsProcessor(LogitsProcessor):
    """Constrain generated action IDs to the sparse grammar; allow EOS once complete."""

    def __init__(
        self, codec, horizon, *, fps, prompt_length, eos_token_id, token_offset=0, pad_token_id=None
    ):
        for name, value in {
            "prompt_length": prompt_length,
            "eos_token_id": eos_token_id,
            "token_offset": token_offset,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if pad_token_id is not None and (type(pad_token_id) is not int or pad_token_id < 0):
            raise ValueError("pad_token_id must be a nonnegative integer")
        self.grammar = codec.grammar(horizon, fps=fps)
        self.prompt_length, self.token_offset = prompt_length, token_offset
        self.eos_token_id, self.pad_token_id = eos_token_id, pad_token_id
        stop = token_offset + self.grammar.vocab_size
        if token_offset <= eos_token_id < stop:
            raise ValueError("eos_token_id must be outside the action token interval")
        if pad_token_id is not None and token_offset <= pad_token_id < stop:
            raise ValueError("pad_token_id must be outside the action token interval")
        self._mask_cache = BoundedCache(64)

    def _state(self, row):
        if self.eos_token_id in row:
            end = row.index(self.eos_token_id)
            if any(t not in (self.eos_token_id, self.pad_token_id) for t in row[end:]):
                raise ValueError("only EOS/padding may follow completed action generation")
            if not self.grammar.is_complete([t - self.token_offset for t in row[:end]]):
                raise ValueError("EOS occurred before the action chunk was complete")
            return "done"
        state = self.grammar._consume([t - self.token_offset for t in row])
        if state is None:
            raise ValueError(
                "generated prefix violates the action grammar; check prompt_length and token_offset"
            )
        return state

    def _mask(self, state, scores):
        key = (state, str(scores.device), scores.shape[-1])
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.zeros(scores.shape[-1], device=scores.device, dtype=torch.bool)
            if state == "done":
                mask[self.eos_token_id] = True
                if self.pad_token_id is not None:
                    mask[self.pad_token_id] = True
            else:
                allowed = self.grammar._mask_from_state(state)
                stop = self.token_offset + len(allowed)
                mask[self.token_offset : stop] = torch.as_tensor(allowed, device=scores.device)
                mask[self.eos_token_id] = self.grammar._state_complete(state)
            self._mask_cache[key] = mask
        return mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0]:
            raise ValueError("expected input_ids(B,L) and scores(B,V) with the same batch size")
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("prompt_length exceeds the supplied input length")
        required = max(
            self.token_offset + self.grammar.vocab_size,
            self.eos_token_id + 1,
            0 if self.pad_token_id is None else self.pad_token_id + 1,
        )
        if scores.shape[1] < required:
            raise ValueError("model vocabulary does not cover configured action/EOS/padding IDs")
        self.grammar.reserve_states(int(input_ids.shape[0]))
        rows = input_ids[:, self.prompt_length :].detach().cpu().tolist()
        output = scores.clone()
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(self._state(row), []).append(index)
        for state, indices in groups.items():
            output[indices] = scores[indices].masked_fill(~self._mask(state, scores), -torch.inf)
        if not torch.isfinite(output).any(dim=1).all():
            raise ValueError(
                "all legal action logits were suppressed; check other generation processors"
            )
        return output
