"""Adapter for the released X-Tokenizer with explicit action-layout transforms."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from actioncodec2.tokenizers.base import (
    ChunkTokenizer,
    FixedLengthGrammar,
    QuantileNormalizer,
)

__all__ = [
    "ActionOutputCalibration",
    "DirectXTransform",
    "LiberoXTransform",
    "XTokenizerAdapter",
]

ENCODE_BATCH = 2048
DECODE_BATCH = 128


LIBERO_ACTION_DIM = 7
METERS_PER_TRANSLATION_UNIT = 0.05
RADIANS_PER_ROTATION_UNIT = 0.5

_POSITION_KEY = "follow_left_ee_cartesian_pos"
_ROTATION_KEY = "follow_left_ee_rotation_6D"
_GRIPPER_KEY = "follow_left_gripper"


def _matrix_to_rotation_6d(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices)
    return np.concatenate((values[..., 0, :], values[..., 1, :]), axis=-1)


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64)
    first = values[..., :3]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = values[..., 3:6]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def absolute_batch_from_libero_chunks(
    chunks: np.ndarray,
    dof_dims: Mapping[str, int],
    *,
    inactive_fill: float = 0.5,
) -> np.ndarray:
    """Convert LIBERO delta chunks to absolute X-Tokenizer trajectories.

    Returns ``[B, H + 1, D]`` with a zero/identity anchor for each chunk.
    """
    values = np.asarray(chunks, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != LIBERO_ACTION_DIM:
        raise ValueError(f"LIBERO chunks must have shape [B, H, 7], got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("LIBERO chunks must be non-empty and finite")
    batch, horizon, _ = values.shape

    position = np.zeros((batch, horizon + 1, 3), dtype=np.float64)
    position[:, 1:] = np.cumsum(
        values[:, :, :3].astype(np.float64) * METERS_PER_TRANSLATION_UNIT,
        axis=1,
    )
    matrices = np.empty((batch, horizon + 1, 3, 3), dtype=np.float64)
    matrices[:, 0] = np.eye(3, dtype=np.float64)
    increments = Rotation.from_rotvec(
        values[:, :, 3:6].astype(np.float64).reshape(-1, 3)
        * RADIANS_PER_ROTATION_UNIT
    ).as_matrix().reshape(batch, horizon, 3, 3)
    for step in range(horizon):
        matrices[:, step + 1] = increments[:, step] @ matrices[:, step]

    gripper = np.empty((batch, horizon + 1, 1), dtype=np.float32)


    gripper[:, 0, 0] = 0.5
    gripper[:, 1:, 0] = values[:, :, 6]
    active = {
        _POSITION_KEY: position.astype(np.float32),
        _ROTATION_KEY: _matrix_to_rotation_6d(matrices).astype(np.float32),
        _GRIPPER_KEY: gripper,
    }
    output = np.full(
        (batch, horizon + 1, sum(int(dim) for dim in dof_dims.values())),
        float(inactive_fill),
        dtype=np.float32,
    )
    offset = 0
    for key, dim_value in dof_dims.items():
        dim = int(dim_value)
        stripped = key.replace("_relative", "")
        if stripped in active:
            output[:, :, offset : offset + dim] = active[stripped]
        elif "_relative" in key and "rotation_6D" not in key:


            output[:, 1:, offset : offset + dim] = float(inactive_fill) + 0.5
        offset += dim
    return output


def libero_actions_from_absolute_matrix(
    absolute_actions: np.ndarray,
    observation_anchor: np.ndarray,
) -> np.ndarray:
    """Invert decoded left-arm absolute trajectories back to raw LIBERO action."""
    values = np.asarray(absolute_actions, dtype=np.float64)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] < 10:
        raise ValueError(f"absolute actions must have shape [B, T, D>=10], got {values.shape}")

    anchors = np.asarray(observation_anchor, dtype=np.float64)
    if anchors.ndim == 1:
        anchors = anchors[None]
    if anchors.shape != (values.shape[0], values.shape[2]):
        raise ValueError(
            f"anchors must have shape {(values.shape[0], values.shape[2])}, got {anchors.shape}"
        )

    previous_position = np.concatenate((anchors[:, None, :3], values[:, :-1, :3]), axis=1)
    translation = (
        values[:, :, :3] - previous_position
    ) / METERS_PER_TRANSLATION_UNIT

    rotations = _rotation_6d_to_matrix(values[:, :, 3:9])
    anchor_rotations = _rotation_6d_to_matrix(anchors[:, 3:9])
    previous_rotations = np.concatenate(
        (anchor_rotations[:, None], rotations[:, :-1]), axis=1
    )
    increments = rotations @ np.swapaxes(previous_rotations, -1, -2)
    rotation = Rotation.from_matrix(increments.reshape(-1, 3, 3)).as_rotvec()
    rotation = rotation.reshape(values.shape[0], values.shape[1], 3)
    rotation /= RADIANS_PER_ROTATION_UNIT

    output = np.empty((values.shape[0], values.shape[1], LIBERO_ACTION_DIM), dtype=np.float32)
    output[:, :, :3] = translation
    output[:, :, 3:6] = rotation
    output[:, :, 6] = values[:, :, 9]
    return output[0] if squeeze else output


class ActionOutputCalibration:
    """Apply the configured action output calibration."""

    SCHEMA = "actioncodec2.x-tokenizer-output-calibration/2"
    FEATURE_DIMS = {"linear": 6, "quadratic": 27, "cubic": 83}

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray,
        gripper_threshold: np.ndarray,
        *,
        clip_low: np.ndarray,
        clip_high: np.ndarray,
        feature_type: str = "linear",
    ) -> None:
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.gripper_threshold = np.asarray(gripper_threshold, dtype=np.float32)
        self.clip_low = np.asarray(clip_low, dtype=np.float32)
        self.clip_high = np.asarray(clip_high, dtype=np.float32)
        self.feature_type = str(feature_type)
        if not all(np.isfinite(value).all() for value in (
            self.weight, self.bias, self.gripper_threshold, self.clip_low, self.clip_high
        )):
            raise ValueError("X-Tokenizer calibration values must be finite")
        feature_dim = self.FEATURE_DIMS.get(self.feature_type)
        if feature_dim is None:
            raise ValueError("X-Tokenizer calibration feature_type is unsupported")
        if self.weight.ndim != 3 or self.weight.shape[1:] != (feature_dim, 6):
            raise ValueError(
                f"X-Tokenizer calibration weight must have shape [H, {feature_dim}, 6]"
            )
        horizon = self.weight.shape[0]
        if self.bias.shape != (horizon, 6):
            raise ValueError("X-Tokenizer calibration bias must have shape [H, 6]")
        if self.gripper_threshold.shape != (horizon,):
            raise ValueError("X-Tokenizer gripper threshold must have shape [H]")
        if self.clip_low.shape != (6,) or self.clip_high.shape != (6,):
            raise ValueError("X-Tokenizer calibration clip bounds must have shape [6]")
        if np.any(self.clip_low >= self.clip_high):
            raise ValueError("X-Tokenizer calibration clip bounds must be ordered")

    @property
    def horizon(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (self.horizon, 7):
            raise ValueError(
                f"X-Tokenizer calibration expects [B, {self.horizon}, 7], got {values.shape}"
            )
        output = np.empty_like(values)
        features = self.feature_matrix(values[..., :6], self.feature_type)
        output[..., :6] = np.einsum(
            "bti,tij->btj", features, self.weight, optimize=True
        ) + self.bias[None]
        output[..., :6] = np.clip(output[..., :6], self.clip_low, self.clip_high)
        output[..., 6] = values[..., 6] >= self.gripper_threshold[None]
        return output

    @classmethod
    def feature_matrix(cls, values: np.ndarray, feature_type: str) -> np.ndarray:
        continuous = np.asarray(values, dtype=np.float32)
        if continuous.shape[-1] != 6:
            raise ValueError("X-Tokenizer calibration features require six coordinates")
        if feature_type == "linear":
            return continuous
        if feature_type not in {"quadratic", "cubic"}:
            raise ValueError("X-Tokenizer calibration feature_type is unsupported")
        products = [continuous]
        for degree in range(2, 4 if feature_type == "cubic" else 3):
            products.extend(
                np.prod(continuous[..., indices], axis=-1, keepdims=True)
                for indices in combinations_with_replacement(range(6), degree)
            )
        return np.concatenate(products, axis=-1)

    @classmethod
    def from_json(cls, path: str | Path) -> ActionOutputCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported X-Tokenizer calibration schema in {path}")
        return cls(
            payload["weight"],
            payload["bias"],
            payload["gripper_threshold"],
            clip_low=payload["clip_low"],
            clip_high=payload["clip_high"],
            feature_type=payload["feature_type"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "horizon": self.horizon,
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
            "gripper_threshold": self.gripper_threshold.tolist(),
            "clip_low": self.clip_low.tolist(),
            "clip_high": self.clip_high.tolist(),
            "feature_type": self.feature_type,
        }


class LiberoXTransform:
    """Seven-dimensional delta EEF commands <-> normalized X ``[..., 26]``."""

    MODEL_DIM = 26
    ACTIVE_DIM = 10
    ACTION_DIM = 7

    ROTATION_REPRESENTATIONS = ("rotvec", "euler_xyz")

    ALIASES = {"rotation_vector": "rotvec"}

    def __init__(
        self,
        normalizer: QuantileNormalizer,
        *,
        rotation_representation: str = "rotvec",
    ) -> None:
        if normalizer.low.shape != (self.ACTIVE_DIM,):
            raise ValueError("X-Tokenizer projected normalizer must hold 10 coordinates")
        self.normalizer = normalizer
        self.action_dim = self.ACTION_DIM
        self.active_dim = self.ACTIVE_DIM
        representation = str(rotation_representation).lower()
        self.rotation_representation = self.ALIASES.get(representation, representation)
        if self.rotation_representation not in self.ROTATION_REPRESENTATIONS:
            raise ValueError(
                "X-Tokenizer rotation_representation must be 'rotvec' or 'euler_xyz'"
            )

    @staticmethod
    def project(
        actions: np.ndarray, *, rotation_representation: str = "rotvec"
    ) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.shape[-1] != LiberoXTransform.ACTION_DIM:
            raise ValueError("LIBERO X-Tokenizer input must have seven action coordinates")
        representation = str(rotation_representation).lower()
        representation = LiberoXTransform.ALIASES.get(representation, representation)
        if representation == "rotvec":
            matrices = Rotation.from_rotvec(values[..., 3:6].reshape(-1, 3)).as_matrix()
        elif representation == "euler_xyz":
            matrices = Rotation.from_euler(
                "xyz", values[..., 3:6].reshape(-1, 3)
            ).as_matrix()
        else:
            raise ValueError(
                "X-Tokenizer rotation_representation must be 'rotvec' or 'euler_xyz'"
            )
        rotation_6d = np.concatenate([matrices[:, 0], matrices[:, 1]], axis=-1).reshape(
            values.shape[:-1] + (6,)
        )
        projected = np.zeros(values.shape[:-1] + (LiberoXTransform.MODEL_DIM,), np.float32)
        projected[..., :3] = values[..., :3]
        projected[..., 3:9] = rotation_6d
        projected[..., 9] = values[..., 6]
        return projected

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        projected = self.project(
            actions, rotation_representation=self.rotation_representation
        )
        projected[..., : self.ACTIVE_DIM] = self.normalizer(
            projected[..., : self.ACTIVE_DIM]
        )
        return projected

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32)
        if values.shape[-1] != self.MODEL_DIM:
            raise ValueError("decoded X-Tokenizer actions must have 26 coordinates")
        active = self.normalizer.inverse(values[..., : self.ACTIVE_DIM])
        r6 = active[..., 3:9].reshape(-1, 6)
        first = r6[:, :3]
        first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
        second = r6[:, 3:]
        second -= (first * second).sum(axis=-1, keepdims=True) * first
        second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
        third = np.cross(first, second)
        matrices = np.stack([first, second, third], axis=1)
        rotation = Rotation.from_matrix(matrices)
        if self.rotation_representation == "rotvec":
            restored_rotation = rotation.as_rotvec()
        else:
            restored_rotation = rotation.as_euler("xyz")
        restored_rotation = restored_rotation.reshape(values.shape[:-1] + (3,))
        actions = np.empty(values.shape[:-1] + (self.ACTION_DIM,), dtype=np.float32)
        actions[..., :3] = active[..., :3]
        actions[..., 3:6] = restored_rotation
        actions[..., 6] = active[..., 9]
        return actions

    @classmethod
    def fit(
        cls,
        actions: np.ndarray,
        *,
        quantile: float = 0.001,
        rotation_representation: str = "rotvec",
    ) -> LiberoXTransform:
        projected = cls.project(
            np.asarray(actions, dtype=np.float32),
            rotation_representation=rotation_representation,
        )
        return cls(
            QuantileNormalizer.fit(projected[..., : cls.ACTIVE_DIM], quantile=quantile),
            rotation_representation=rotation_representation,
        )


class DirectXTransform:
    """Map a compact action vector directly into X's canonical 26-DoF layout."""

    MODEL_DIM = 26
    ACTION_LAYOUT = "direct"

    def __init__(self, normalizer: QuantileNormalizer, *, action_dim: int) -> None:
        self.normalizer = normalizer
        self.action_dim = int(action_dim)
        self.active_dim = self.action_dim
        if not 1 <= self.action_dim <= self.MODEL_DIM:
            raise ValueError(f"direct X action_dim must lie in [1, {self.MODEL_DIM}]")
        if self.normalizer.low.shape != (self.action_dim,):
            raise ValueError(
                f"direct X-Tokenizer normalizer must hold {self.action_dim} coordinates"
            )

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.shape[-1] != self.action_dim:
            raise ValueError(
                f"direct X-Tokenizer input must have {self.action_dim} coordinates"
            )
        projected = np.zeros(values.shape[:-1] + (self.MODEL_DIM,), np.float32)
        projected[..., : self.action_dim] = self.normalizer(values)
        return projected

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32)
        if values.shape[-1] != self.MODEL_DIM:
            raise ValueError("decoded X-Tokenizer actions must have 26 coordinates")
        return self.normalizer.inverse(values[..., : self.action_dim])

    @classmethod
    def fit(
        cls, actions: np.ndarray, *, quantile: float, action_dim: int
    ) -> DirectXTransform:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != int(action_dim):
            raise ValueError(f"direct X-Tokenizer actions must be [T, {action_dim}]")
        return cls(
            QuantileNormalizer.fit(values, quantile=quantile),
            action_dim=int(action_dim),
        )


class XTokenizerAdapter(ChunkTokenizer):
    """Inference-only wrapper around ``x-square-robot/X-Tokenizer``."""

    name = "x-tokenizer"
    is_exact = False


    cache_batch_size = ENCODE_BATCH

    def __init__(
        self,
        backend,
        transform: LiberoXTransform | DirectXTransform | None,
        *,
        time_horizon: int,
        artifact: str | Path | None = None,
        device: str = "cpu",
        robot_type: str = "Franka",
        source_revision: str = "",
        statistics_path: str | Path | None = None,
        condition_obs: bool = False,
        calibration: ActionOutputCalibration | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.transform = transform
        self.time_horizon = int(time_horizon)
        self.action_dim = (
            LiberoXTransform.ACTION_DIM
            if transform is None
            else int(transform.action_dim)
        )
        self.active_dim = (
            LiberoXTransform.ACTIVE_DIM
            if transform is None
            else int(getattr(transform, "active_dim", LiberoXTransform.ACTIVE_DIM))
        )
        self.artifact = None if artifact is None else str(artifact)
        self.device = str(device)
        self.robot_type = str(robot_type)
        self.source_revision = str(source_revision)
        self.statistics_path = None if statistics_path is None else str(statistics_path)
        self.condition_obs = bool(condition_obs)
        self.calibration = calibration
        self.calibration_path = (
            None if calibration_path is None else str(calibration_path)
        )
        self.official_action_only = transform is None
        if self.official_action_only and self.statistics_path is None:
            raise ValueError("official action-only X-Tokenizer requires statistics_path")
        if int(backend.get_action_dim()) != LiberoXTransform.MODEL_DIM:
            raise ValueError("the X-Tokenizer checkpoint is not the released 26-DoF model")
        maximum = int(backend.get_max_chunk_length())
        if not 8 <= self.time_horizon <= maximum:
            raise ValueError(f"X-Tokenizer horizon must lie in [8, {maximum}]")
        if self.calibration is not None and self.calibration.horizon != self.time_horizon:
            raise ValueError(
                "X-Tokenizer output calibration horizon does not match time_horizon"
            )
        self.vocab_size = int(backend.get_vocab_size())
        self.num_quantizers = int(backend.get_num_quantizers())
        self.compression_ratio = int(backend.get_compression_ratio())
        latents = (self.time_horizon + self.compression_ratio - 1) // self.compression_ratio
        self._tokens_per_chunk = latents * self.num_quantizers
        self._grammar = FixedLengthGrammar(self._tokens_per_chunk, self.vocab_size)
        self._dof_mask = np.zeros(
            (self.time_horizon, LiberoXTransform.MODEL_DIM), dtype=bool
        )
        self._dof_mask[:, : self.active_dim] = True

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        transform: LiberoXTransform | DirectXTransform | None = None,
        *,
        time_horizon: int,
        device: str = "cpu",
        robot_type: str = "Franka",
        source_revision: str = "",
        statistics_path: str | Path | None = None,
        condition_obs: bool = False,
        calibration_path: str | Path | None = None,
    ) -> XTokenizerAdapter:
        from xtokenizer import XTokenizer

        backend = XTokenizer.from_pretrained(
            path, statistics_path=statistics_path, device=device
        )
        calibration = (
            None
            if calibration_path is None
            else ActionOutputCalibration.from_json(calibration_path)
        )
        return cls(
            backend,
            transform,
            time_horizon=time_horizon,
            artifact=path,
            device=device,
            robot_type=robot_type,
            source_revision=source_revision,
            statistics_path=statistics_path,
            condition_obs=condition_obs,
            calibration=calibration,
            calibration_path=calibration_path,
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
            raise TypeError("X-Tokenizer does not use a previous action")
        if self.official_action_only:
            checked = self._check_chunks(chunks)
            absolute = absolute_batch_from_libero_chunks(
                checked, self.backend.get_dof_layout()
            )
            streams: list[list[int]] = []
            official_batch = min(128, ENCODE_BATCH)
            for start in range(0, len(absolute), official_batch):
                block = absolute[start : start + official_batch]
                masks = np.broadcast_to(
                    self._dof_mask, (len(block),) + self._dof_mask.shape
                ).copy()
                indices = self.backend.encode_from_absolute(
                    block[:, 1:],
                    block[:, :1],
                    robot_type=self.robot_type,
                    dof_mask=masks,
                    token_order="time_major",
                ).detach().cpu().numpy()
                streams.extend(
                    indices.reshape(len(indices), -1).astype(np.int64).tolist()
                )
            if any(not self._grammar.is_complete(row) for row in streams):
                raise ValueError("X-Tokenizer returned the wrong token count")
            return streams
        values = self.transform(self._check_chunks(chunks))
        streams: list[list[int]] = []
        for start in range(0, len(values), ENCODE_BATCH):
            block = values[start : start + ENCODE_BATCH]
            obs = (
                np.zeros((len(block), 1, LiberoXTransform.MODEL_DIM), np.float32)
                if self.condition_obs
                else None
            )
            output = self.backend.encode(
                block,
                obs_state=obs,
                dof_mask=np.broadcast_to(
                    self._dof_mask, (len(block),) + self._dof_mask.shape
                ).copy(),
                robot_type=self.robot_type,
                token_order="time_major",
            )
            indices = output["indices"].detach().cpu().numpy()
            streams.extend(indices.reshape(len(indices), -1).astype(np.int64).tolist())
        if any(not self._grammar.is_complete(row) for row in streams):
            raise ValueError("X-Tokenizer returned the wrong token count")
        return streams

    def decode(self, tokens: Sequence[int]) -> np.ndarray:
        return self.decode_batch([tokens])[0]

    def decode_batch(self, streams: Sequence[Sequence[int]]) -> np.ndarray:
        rows = np.asarray(streams, dtype=np.int64)
        if rows.ndim != 2 or len(rows) == 0 or rows.shape[1] != self._tokens_per_chunk:
            raise ValueError(
                f"X-Tokenizer requires exactly {self._tokens_per_chunk} tokens per chunk"
            )
        if np.any((rows < 0) | (rows >= self.vocab_size)):
            raise ValueError(f"X-Tokenizer ids must lie in [0, {self.vocab_size})")
        latents = self._tokens_per_chunk // self.num_quantizers
        if self.official_action_only:
            decoded = []
            official_batch = min(128, DECODE_BATCH)
            for start in range(0, len(rows), official_batch):
                block = rows[start : start + official_batch].reshape(
                    -1, latents, self.num_quantizers
                )
                anchors = absolute_batch_from_libero_chunks(
                    np.zeros((len(block), self.time_horizon, self.action_dim), np.float32),
                    self.backend.get_dof_layout(),
                )[:, :1]
                masks = np.broadcast_to(
                    self._dof_mask, (len(block),) + self._dof_mask.shape
                ).copy()
                absolute = self.backend.decode_to_absolute(
                    block,
                    target_length=self.time_horizon,
                    obs_state_absolute=anchors,
                    robot_type=self.robot_type,
                    dof_mask=masks,
                    token_order="time_major",
                ).detach().float().cpu().numpy()
                decoded.append(
                    libero_actions_from_absolute_matrix(absolute, anchors[:, 0])
                )
            values = np.concatenate(decoded).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("X-Tokenizer returned non-finite actions")
            return values
        decoded = []
        for start in range(0, len(rows), DECODE_BATCH):
            block = rows[start : start + DECODE_BATCH].reshape(
                -1, latents, self.num_quantizers
            )
            masks = np.broadcast_to(
                self._dof_mask, (len(block),) + self._dof_mask.shape
            ).copy()
            obs = (
                np.zeros((len(block), 1, LiberoXTransform.MODEL_DIM), np.float32)
                if self.condition_obs
                else None
            )
            actions = self.backend.decode(
                block,
                target_length=self.time_horizon,
                obs_state=obs,
                dof_mask=masks,
                robot_type=self.robot_type,
                token_order="time_major",
            )
            decoded.append(actions.detach().float().cpu().numpy())
        values = np.concatenate(decoded).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("X-Tokenizer returned non-finite actions")
        restored = self.transform.inverse(values)
        return self.calibration(restored) if self.calibration is not None else restored

    def to_json(self) -> dict[str, object]:
        if self.artifact is None:
            raise ValueError("this X-Tokenizer has no artifact path")
        payload: dict[str, object] = {
            "name": self.name,
            "artifact": self.artifact,
            "time_horizon": self.time_horizon,
            "device": self.device,
            "robot_type": self.robot_type,
            "source_revision": self.source_revision,
        }
        if self.official_action_only:
            payload["statistics_path"] = self.statistics_path
            payload["official_action_only"] = True
            payload["anchor_mode"] = "chunk_zero"
        else:
            if getattr(self.transform, "ACTION_LAYOUT", None) == "direct":
                payload["action_layout"] = "direct"
                payload["action_dim"] = self.action_dim
            else:
                payload["rotation_representation"] = self.transform.rotation_representation
            payload["normalizer"] = self.transform.normalizer.to_json()
            payload["condition_obs"] = self.condition_obs
            if self.calibration_path is not None:
                payload["calibration_path"] = self.calibration_path
        return payload
