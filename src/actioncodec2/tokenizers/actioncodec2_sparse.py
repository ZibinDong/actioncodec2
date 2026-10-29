"""Fixed-horizon VLA adapter for fitted ActionCodec2Sparse artifacts."""

from __future__ import annotations

import numpy as np

from actioncodec2.sparse import ActionCodec2Sparse
from actioncodec2.core.backends.actioncodec2.batch_semantics import externalize_batch
from actioncodec2.core.backends.actioncodec2.semantic import resampled_horizon

from .actioncodec2 import ActionCodec2Tokenizer
from .base import ChunkGrammar, ChunkTokenizer


class SparseChunkGrammar(ChunkGrammar):
    """Reserve one local EOS ID; a decodable sparse prefix may still continue."""

    def __init__(self, grammar, offset, size):
        self.inner, self.offset = grammar, offset
        self.eos_token_id, self.vocab_size = size, size + 1

    def _prefix(self, tokens):
        row = []
        for token in tokens:
            if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
                return None
            if not 0 <= token < self.eos_token_id:
                return None
            row.append(int(token) + self.offset)
        return row

    def next_token_mask(self, tokens):
        mask = np.zeros(self.vocab_size, dtype=bool)
        row = self._prefix(tokens)
        if row is not None:
            mask[:-1] = self.inner.next_token_mask(row)[
                self.offset : self.offset + self.eos_token_id
            ]
            mask[-1] = self.inner.is_complete(row)
        return mask

    def is_complete(self, tokens):
        if not len(tokens) or isinstance(tokens[-1], (bool, np.bool_)):
            return False
        if not isinstance(tokens[-1], (int, np.integer)) or tokens[-1] != self.eos_token_id:
            return False
        row = self._prefix(tokens[:-1])
        return row is not None and self.inner.is_complete(row)


class ActionCodec2SparseTokenizer(ChunkTokenizer):
    """One delta joint/EEF profile with explicit EOS and raw-unit boundaries."""

    name = "actioncodec2-sparse"
    _canonical_input = ActionCodec2Tokenizer._canonical_input
    _external_output = ActionCodec2Tokenizer._external_output
    _left_to_right = ActionCodec2Tokenizer._left_to_right
    _right_to_left = ActionCodec2Tokenizer._right_to_left
    _boundary = ActionCodec2Tokenizer._boundary
    _profile_bounds = ActionCodec2Tokenizer._profile_bounds
    encode = ActionCodec2Tokenizer.encode
    decode = ActionCodec2Tokenizer.decode
    rollout_previous_actions = ActionCodec2Tokenizer.rollout_previous_actions

    def __init__(
        self,
        codec,
        *,
        artifact,
        time_horizon,
        fps=15.0,
        action_dim=None,
        rotation_representation="rotation_vector",
        rotation_multiplication="right",
        motion_scale=1.0,
        gripper_threshold=0.8,
        gripper_open_is_high=True,
        previous_action_resampling="strict",
        revision=None,
        bounds=None,
        artifact_fingerprint=None,
    ):
        codec._require_fitted()
        self.codec, self.processor = codec, codec.action_tokenizer
        self.preset = codec.action_space
        self.action_space = self.preset.name
        if self.action_space not in {
            "single_eef_delta",
            "single_joint6_delta",
            "single_joint7_delta",
        }:
            raise ValueError("the Sparse VLA adapter requires a built-in single-arm delta layout")
        self.action_dim = codec.action_dim
        if action_dim is not None and int(action_dim) != self.action_dim:
            raise ValueError("action_dim does not match the Sparse action space")
        self.component = self.preset.components[0]
        self.profile_name = self.component.space
        self.profile = self.processor.profiles[self.profile_name]
        if any(
            c.input_order != 0 for c in self.profile.processor.config.physical.components.values()
        ):
            raise ValueError("the Sparse VLA adapter requires canonical absolute physical profiles")
        self.time_horizon, self.fps, self.target_hz = int(time_horizon), float(fps), 15.0
        self.internal_horizon = resampled_horizon(
            self.time_horizon, source_hz=self.fps, target_hz=15.0
        )
        if rotation_representation not in {"rotation_vector", "euler_xyz"}:
            raise ValueError("rotation_representation must be rotation_vector or euler_xyz")
        if rotation_multiplication not in {"left", "right"}:
            raise ValueError("rotation_multiplication must be left or right")
        self.rotation_representation, self.rotation_multiplication = (
            rotation_representation,
            rotation_multiplication,
        )
        self.motion_scale = np.broadcast_to(
            np.asarray(motion_scale, np.float32), (self.action_dim - 1,)
        ).copy()
        if not np.isfinite(self.motion_scale).all() or np.any(self.motion_scale <= 0):
            raise ValueError("motion_scale must contain positive finite values")
        self.gripper_threshold, self.gripper_open_is_high = (
            float(gripper_threshold),
            bool(gripper_open_is_high),
        )
        if not np.isfinite(self.gripper_threshold):
            raise ValueError("gripper_threshold must be finite")
        self.previous_action_resampling = previous_action_resampling
        if previous_action_resampling not in {"strict", "constant_velocity"}:
            raise ValueError("previous_action_resampling must be strict or constant_velocity")
        self.requires_previous_action = codec.requires_previous_first_order
        if (
            self.requires_previous_action
            and self.fps != 15.0
            and previous_action_resampling == "strict"
        ):
            raise ValueError(
                "second-order Sparse needs 15 Hz or explicit constant_velocity boundary resampling"
            )
        self.delta_native, self._native = False, None
        self.action_low, self.action_high = self._profile_bounds() if bounds is None else bounds
        if (
            np.shape(self.action_low) != (self.action_dim,)
            or np.shape(self.action_high) != (self.action_dim,)
            or not np.isfinite(self.action_low).all()
            or not np.isfinite(self.action_high).all()
            or np.any(self.action_low >= self.action_high)
        ):
            raise ValueError("action bounds must be finite, ordered vectors matching action_dim")
        size = self.profile.processor.vocab_size
        self.vocab_size = size + 1
        self._grammar = SparseChunkGrammar(
            codec.grammar(self.time_horizon, fps=self.fps), self.profile.spec.token_offset, size
        )
        self.artifact, self.revision = str(artifact), revision
        self.artifact_fingerprint = self.profile.processor.artifact_fingerprint
        if artifact_fingerprint is not None and artifact_fingerprint != self.artifact_fingerprint:
            raise ValueError("Sparse artifact fingerprint does not match the saved policy")

    @classmethod
    def from_pretrained(cls, artifact, *, action_space="single_eef_delta", revision=None, **kwargs):
        action_space = {"delta_eef": "single_eef_delta", "delta_joint": "single_joint7_delta"}.get(
            action_space, action_space
        )
        codec = ActionCodec2Sparse.from_pretrained(
            artifact, action_space=action_space, revision=revision
        )
        return cls(codec, artifact=artifact, revision=revision, **kwargs)

    @property
    def grammar(self):
        return self._grammar

    @property
    def max_tokens_per_chunk(self):
        return self.internal_horizon * (len(self.component.active_dims) + 1) + 1

    def encode_batch(self, chunks, previous_actions=None):
        values = self._canonical_input(self._check_chunks(chunks))
        if self.profile.processor.bins[-1] == 2:
            opened = values[..., -1] > self.gripper_threshold
            values[..., -1] = opened if self.gripper_open_is_high else ~opened
        elif not self.gripper_open_is_high:
            values[..., -1] = 1.0 - values[..., -1]
        trajectories = self.preset.canonicalize(values, source_hz=self.fps)[self.component.name]
        rows = self.profile.processor.encode_batch(
            np.stack([row.values for row in trajectories]),
            active_dims=self.component.active_dims,
            previous_first_order=self._boundary(previous_actions, len(values)),
        )
        return [row + [self.grammar.eos_token_id] for row in rows]

    def decode_batch(self, streams, previous_actions=None):
        rows = list(streams)
        if not rows:
            raise ValueError("at least one Sparse token row is required")
        if any(not self.grammar.is_complete(row) for row in rows):
            raise ValueError("Sparse rows must end with EOS after a valid complete prefix")
        canonical = self.profile.processor.decode_batch(
            [list(row[:-1]) for row in rows],
            horizon=self.internal_horizon,
            active_dims=self.component.active_dims,
            previous_first_order=self._boundary(previous_actions, len(rows)),
        )
        restored = externalize_batch(
            canonical,
            self.component,
            source_hz=15.0,
            target_hz=self.fps,
            target_length=self.time_horizon,
            method=self.preset.interpolation,
            current_state=None,
        )
        if not self.gripper_open_is_high:
            restored[..., -1] = 1.0 - restored[..., -1]
        return self._external_output(restored)

    def to_json(self):
        return dict(
            name=self.name,
            artifact=self.artifact,
            revision=self.revision,
            artifact_fingerprint=self.artifact_fingerprint,
            time_horizon=self.time_horizon,
            action_dim=self.action_dim,
            fps=self.fps,
            action_space=self.action_space,
            motion_scale=self.motion_scale.tolist(),
            rotation_representation=self.rotation_representation,
            rotation_multiplication=self.rotation_multiplication,
            gripper_threshold=self.gripper_threshold,
            gripper_open_is_high=self.gripper_open_is_high,
            previous_action_resampling=self.previous_action_resampling,
            action_clip={
                "low": np.asarray(self.action_low).tolist(),
                "high": np.asarray(self.action_high).tolist(),
            },
        )
