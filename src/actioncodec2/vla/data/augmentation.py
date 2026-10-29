"""Target-preserving augmentation of a stateful tokenizer's boundary action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PreviousActionAugmentation"]


@dataclass
class PreviousActionAugmentation:
    """Configuration for optional previous-action augmentation during training."""

    clean_probability: float = 0.5
    sigmas: tuple[float, ...] = (0.05, 0.1)
    max_abs_noise: float = 0.25
    max_resamples: int = 16
    rotation_sigmas: tuple[float, ...] | None = None
    rotation_max_abs_noise: float | None = None

    def __post_init__(self):
        self.sigmas = tuple(float(x) for x in self.sigmas)
        if not np.isfinite(self.clean_probability) or not 0 <= self.clean_probability <= 1:
            raise ValueError('clean_probability must be in [0,1]')
        if not self.sigmas or not all(np.isfinite(x) and x > 0 for x in self.sigmas):
            raise ValueError('sigmas must be nonempty, finite and positive')
        if not np.isfinite(self.max_abs_noise) or self.max_abs_noise <= 0:
            raise ValueError('max_abs_noise must be finite and positive')
        if not isinstance(self.max_resamples, int) or self.max_resamples < 1:
            raise ValueError('max_resamples must be a positive integer')
        if self.rotation_sigmas is None:
            if self.rotation_max_abs_noise is not None:
                raise ValueError('rotation_max_abs_noise requires rotation_sigmas')
        else:
            self.rotation_sigmas = tuple(float(x) for x in self.rotation_sigmas)
            if not self.rotation_sigmas or not all(
                np.isfinite(x) and x > 0 for x in self.rotation_sigmas
            ):
                raise ValueError('rotation_sigmas must be nonempty, finite and positive')
            if len(self.rotation_sigmas) != len(self.sigmas):
                raise ValueError(
                    'rotation_sigmas must give one level per translation sigma'
                )
            if self.rotation_max_abs_noise is None or not np.isfinite(
                self.rotation_max_abs_noise
            ) or self.rotation_max_abs_noise <= 0:
                raise ValueError(
                    'rotation_max_abs_noise must be finite and positive when rotation_sigmas is set'
                )

    def sample_validated(
        self,
        previous,
        steps,
        rng,
        *,
        validate,
        rotation_representation,
        rotation_scale,
        rotation_multiplication,
        action_space="delta_eef",
    ):
        """Return augmented previous actions [B, D] and counters. The validation callback accepts indices and candidate actions and returns an acceptance mask."""
        from scipy.spatial.transform import Rotation

        previous = np.asarray(previous, np.float32)
        steps = np.asarray(steps)
        if previous.ndim != 2 or steps.shape != (len(previous),):
            raise ValueError("expected previous [B,D] and steps [B]")
        if not np.isfinite(previous).all():
            raise ValueError("boundaries must be finite")
        if action_space not in {"delta_eef", "delta_joint"}:
            raise ValueError("action_space must be delta_eef or delta_joint")
        joint_layout = action_space == "delta_joint"
        if (joint_layout and not 2 <= previous.shape[1] <= 8) or (
            not joint_layout and previous.shape[1] != 7
        ):
            raise ValueError("boundary width does not match action_space")
        if joint_layout and self.rotation_sigmas is not None:
            raise ValueError("rotation_sigmas requires the delta_eef action space")
        result = previous.copy()
        selected = (rng.random(len(previous)) >= self.clean_probability) & (steps > 0)
        pending = np.flatnonzero(selected)
        levels = rng.integers(len(self.sigmas), size=len(previous))
        xyz_scale = np.asarray(self.sigmas)[levels]
        rot_scale = (
            None if self.rotation_sigmas is None else np.asarray(self.rotation_sigmas)[levels]
        )
        draws = 0
        for _ in range(self.max_resamples):
            if not len(pending):
                break
            candidates = previous[pending].copy()
            if joint_layout:
                noise = rng.normal(size=(len(pending), previous.shape[1] - 1)) * xyz_scale[
                    pending, None
                ]
                candidates[:, :-1] += noise
                valid = (np.abs(noise) <= self.max_abs_noise).all(axis=1)
            else:
                noise = rng.normal(size=(len(pending), 3)) * xyz_scale[pending, None]
                candidates[:, :3] += noise
                valid = (np.abs(noise) <= self.max_abs_noise).all(axis=1)
            if rot_scale is not None:
                noise = rng.normal(size=(len(pending), 3)) * rot_scale[pending, None]
                raw = previous[pending, 3:6] * rotation_scale
                rotations = (
                    Rotation.from_euler("xyz", raw)
                    if rotation_representation == "euler_xyz"
                    else Rotation.from_rotvec(raw)
                )
                perturbation = Rotation.from_rotvec(noise)
                augmented = (
                    perturbation * rotations
                    if rotation_multiplication == "left"
                    else rotations * perturbation
                )
                candidates[:, 3:6] = (
                    augmented.as_euler("xyz")
                    if rotation_representation == "euler_xyz"
                    else augmented.as_rotvec()
                ) / rotation_scale
                valid &= (np.abs(noise) <= self.rotation_max_abs_noise).all(axis=1)
            valid &= validate(pending, candidates)
            result[pending[valid]] = candidates[valid]
            draws += len(pending)
            pending = pending[~valid]
        stats = {
            "selected": int(selected.sum()),
            "draws": draws,
            "fallback": len(pending),
            "changed": int(np.any(result != previous, axis=1).sum()),
        }
        if rot_scale is not None:
            stats["changed_rotation"] = int(
                np.any(result[:, 3:6] != previous[:, 3:6], axis=1).sum()
            )
        return result, stats
