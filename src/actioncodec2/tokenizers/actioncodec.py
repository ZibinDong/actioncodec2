"""The released ``ZibinDong/ActionCodec-5e`` behind the chunk interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from actioncodec2.tokenizers.base import ChunkTokenizer, FixedLengthGrammar

__all__ = ["ActionCodecNormalizer", "ActionCodecTokenizer"]

BATCH = 256


class ActionCodecNormalizer:
    """The published ActionCodec LIBERO normalization and its inverse."""

    def __init__(
        self, low: Sequence[float], high: Sequence[float], *, gripper_open_is_high: bool = True
    ) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        if self.low.shape != self.high.shape or self.low.ndim != 1:
            raise ValueError("ActionCodec bounds must be matching vectors")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
            raise ValueError("ActionCodec bounds must be finite")
        self.scale = np.maximum(np.abs(self.low), np.abs(self.high))
        self.gripper_open_is_high = bool(gripper_open_is_high)
        if np.any(self.scale[:-1] <= 0):
            raise ValueError("ActionCodec continuous coordinates need nonzero scale")

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32).copy()
        values[..., :-1] /= self.scale[:-1]
        openness = values[..., -1] if self.gripper_open_is_high else 1.0 - values[..., -1]
        values[..., -1] = openness * 2.0 - 1.0
        return np.clip(values, -1.0, 1.0)

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32).copy()
        values[..., :-1] *= self.scale[:-1]
        openness = (values[..., -1] + 1.0) / 2.0
        values[..., -1] = openness if self.gripper_open_is_high else 1.0 - openness
        return values

    def to_json(self) -> dict[str, object]:
        return {
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "gripper_open_is_high": self.gripper_open_is_high,
        }


class ActionCodecTokenizer(ChunkTokenizer):
    """Inference-only wrapper around the official pretrained ActionCodec."""

    name = "actioncodec"
    is_exact = False

    def __init__(
        self,
        model,
        normalizer: ActionCodecNormalizer,
        *,
        time_horizon: int,
        embodiment_id: int = 0,
        artifact: str | Path | None = None,
        device: str = "cpu",
        revision: str = "",
    ) -> None:
        self.model = model.to(device).eval().requires_grad_(False)
        self.normalizer = normalizer
        self.time_horizon = int(time_horizon)
        self.embodiment_id = int(embodiment_id)
        self.artifact = None if artifact is None else str(artifact)
        self.device = str(device)
        self.revision = str(revision)
        configurations = list(model.config.embodiment_config.values())
        if not 0 <= self.embodiment_id < len(configurations):
            raise ValueError("ActionCodec embodiment_id is out of range")
        embodiment = configurations[self.embodiment_id]
        expected_horizon = round(float(embodiment["freq"]) * float(embodiment["duration"]))
        self.action_dim = int(embodiment["action_dim"])
        if self.time_horizon != expected_horizon:
            raise ValueError(
                f"ActionCodec embodiment {self.embodiment_id} requires horizon "
                f"{expected_horizon}, got {self.time_horizon}"
            )
        if self.normalizer.low.shape != (self.action_dim,):
            raise ValueError("ActionCodec bounds disagree with its action dimension")
        self.vocab_size = int(model.vocab_size)
        self._tokens_per_chunk = int(model.config.n_tokens)
        self._grammar = FixedLengthGrammar(self._tokens_per_chunk, self.vocab_size)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        normalizer: ActionCodecNormalizer,
        *,
        time_horizon: int,
        embodiment_id: int = 0,
        device: str = "cpu",
        revision: str = "",
    ) -> ActionCodecTokenizer:
        from safetensors.torch import load_model
        from transformers import AutoConfig, AutoModel

        # The released remote code targets Transformers 4.57.  Transformers 5
        # reaches its new ``all_tied_weights_keys`` finalizer after loading and
        # rejects the otherwise compatible model class.  Constructing from the
        # same official config and strictly loading the same safetensors avoids
        # that version-specific convenience layer while retaining exact weights.
        root = Path(path)
        config = AutoConfig.from_pretrained(
            str(root), trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_config(config, trust_remote_code=True)
        missing, unexpected = load_model(
            model, str(root / "model.safetensors"), strict=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"ActionCodec checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        return cls(
            model,
            normalizer,
            time_horizon=time_horizon,
            embodiment_id=embodiment_id,
            artifact=path,
            device=device,
            revision=revision,
        )

    @property
    def max_tokens_per_chunk(self) -> int:
        return self._tokens_per_chunk

    @property
    def grammar(self) -> FixedLengthGrammar:
        return self._grammar

    def encode(self, chunk: np.ndarray) -> list[int]:
        return self.encode_batch(self._check_chunk(chunk)[None])[0]

    def encode_batch(
        self, chunks: np.ndarray, previous_actions: np.ndarray | None = None
    ) -> list[list[int]]:
        if previous_actions is not None:
            raise TypeError("ActionCodec does not use a previous action")
        values = self.normalizer(self._check_chunks(chunks))
        streams: list[list[int]] = []
        for start in range(0, len(values), BATCH):
            streams.extend(
                [list(map(int, row)) for row in self.model.encode(
                    values[start : start + BATCH], embodiment_ids=self.embodiment_id
                )]
            )
        if any(not self._grammar.is_complete(row) for row in streams):
            raise ValueError("the official ActionCodec encoder returned the wrong token count")
        return streams

    def decode(self, tokens: Sequence[int]) -> np.ndarray:
        return self.decode_batch([tokens])[0]

    def decode_batch(self, streams: Sequence[Sequence[int]]) -> np.ndarray:
        rows = [[int(token) for token in stream] for stream in streams]
        if not rows or any(not self._grammar.is_complete(row) for row in rows):
            raise ValueError(
                f"ActionCodec requires exactly {self._tokens_per_chunk} tokens per chunk"
            )
        decoded = []
        for start in range(0, len(rows), BATCH):
            actions, mask = self.model.decode(
                rows[start : start + BATCH], embodiment_ids=self.embodiment_id
            )
            actions = np.asarray(actions, dtype=np.float32)[..., : self.action_dim]
            mask = np.asarray(mask, dtype=bool)
            if actions.shape[1:] != (self.time_horizon, self.action_dim) or not mask.all():
                raise ValueError("the official ActionCodec decoder returned an invalid chunk")
            decoded.append(actions)
        values = np.concatenate(decoded)
        if not np.isfinite(values).all():
            raise ValueError("the official ActionCodec decoder returned non-finite actions")
        return self.normalizer.inverse(values)

    def to_json(self) -> dict[str, object]:
        if self.artifact is None:
            raise ValueError("this ActionCodec tokenizer has no artifact path")
        return {
            "name": self.name,
            "artifact": self.artifact,
            "time_horizon": self.time_horizon,
            "embodiment_id": self.embodiment_id,
            "device": self.device,
            "revision": self.revision,
            "action_clip": self.normalizer.to_json(),
            "gripper_open_is_high": self.normalizer.gripper_open_is_high,
        }
