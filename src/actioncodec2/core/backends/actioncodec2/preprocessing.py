"""Versioned preprocessing of continuous segments, after discontinuity QC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation

from .native_backend import native_kernel
from .semantic import discretize_gripper_actions


def step_norms(values: np.ndarray) -> np.ndarray:
    """L2 norm of per-step deltas, ``[T-1]`` for ``[T, D]`` values."""
    kernel = native_kernel("step_norms")
    if kernel is not None:
        values = np.asarray(values)
        result = kernel(values, values.dtype == np.float32)
        return result.astype(np.float32) if values.dtype == np.float32 else result
    return np.linalg.norm(np.diff(values, axis=0), axis=1)


def _legacy_boundary_trim_indices(
    values: np.ndarray,
    *,
    tau: float,
    min_idle_steps: int,
) -> tuple[int, int]:
    kernel = native_kernel("boundary_trim")
    if kernel is not None:
        return tuple(
            kernel(values, tau, min_idle_steps, 1, True, np.asarray(values).dtype == np.float32)
        )
    if len(values) < 2:
        return 0, len(values)
    idle = step_norms(values) <= tau
    moving = ~idle
    if not moving.any():
        return len(values), len(values)
    head = int(np.argmax(moving))
    trailing = int(np.argmax(moving[::-1]))
    tail = len(idle) - trailing

    start = head + 1 if head >= min_idle_steps else 0
    stop = tail + 1 if trailing >= min_idle_steps else len(values)
    if stop <= start:
        return start, start
    return start, stop


def boundary_trim_indices(
    values: np.ndarray,
    *,
    tau: float,
    min_idle_steps: int,
    context_frames: int = 1,
) -> tuple[int, int]:
    """Propose a boundary slice retaining the first moving edge and context."""
    if context_frames < 1:
        raise ValueError("context_frames must be positive")
    kernel = native_kernel("boundary_trim")
    if kernel is not None:
        return tuple(
            kernel(
                values,
                tau,
                min_idle_steps,
                context_frames,
                False,
                np.asarray(values).dtype == np.float32,
            )
        )
    if len(values) < 2:
        return 0, len(values)
    moving = step_norms(values) > tau
    if not moving.any():
        return 0, len(values)
    head = int(np.argmax(moving))
    trailing = int(np.argmax(moving[::-1]))
    start = max(0, head - context_frames + 1) if head >= min_idle_steps else 0
    stop = len(values) - trailing if trailing >= min_idle_steps else len(values)
    return start, max(stop, start + 1)


def trim_deviation(
    values: np.ndarray,
    start: int,
    stop: int,
    *,
    profile: str,
    gripper_column: int | None = None,
) -> dict[str, object]:
    """Maximum physical error from constant-filling removed boundary rows."""
    if not 0 <= start < stop <= len(values):
        raise ValueError("trim slice must retain at least one row")
    kernel = native_kernel("trim_deviation")
    if kernel is not None:
        return kernel(
            values,
            start,
            stop,
            profile,
            -1 if gripper_column is None else gripper_column % values.shape[1],
        )
    width = values.shape[1]
    motion = [i for i in range(width) if gripper_column is None or i != gripper_column % width]
    result: dict[str, object] = {
        "motion_max_error": [0.0] * len(motion),
        "translation_max_error_m": 0.0,
        "rotation_max_error_rad": 0.0,
        "gripper_changed": False,
    }
    for removed, anchor in ((values[:start], values[start]), (values[stop:], values[stop - 1])):
        if not len(removed):
            continue
        delta = np.abs(removed[:, motion].astype(np.float64) - anchor[motion])
        result["motion_max_error"] = np.maximum(
            result["motion_max_error"], delta.max(axis=0)
        ).tolist()
        if profile == "eef":
            result["translation_max_error_m"] = max(
                result["translation_max_error_m"], float(np.linalg.norm(delta[:, :3], axis=1).max())
            )
            rotations = Rotation.from_rotvec(removed[:, motion[3:6]])
            relative = Rotation.from_rotvec(anchor[motion[3:6]]).inv() * rotations
            result["rotation_max_error_rad"] = max(
                result["rotation_max_error_rad"], float(relative.magnitude().max())
            )
        if gripper_column is not None:
            result["gripper_changed"] |= bool(
                np.any(removed[:, gripper_column] != anchor[gripper_column])
            )
    return result


def savgol_smooth(
    values: np.ndarray,
    *,
    window: int,
    polyorder: int = 3,
) -> np.ndarray:
    """Zero-phase Savitzky-Golay smoothing along time, per dimension."""
    length = len(values)
    if length < 3 or window < 3:
        return np.asarray(values, dtype=np.float32)
    if window >= length:
        window = length if length % 2 == 1 else length - 1
    if polyorder >= window:
        polyorder = window - 2 if window >= 3 else 1
    kernel = native_kernel("savgol_smooth")
    if kernel is not None:
        return kernel(values, window, polyorder)
    smoothed = savgol_filter(np.asarray(values, dtype=np.float64), window, polyorder, axis=0)
    return smoothed.astype(np.float32)


def ease_overspeed(
    values: np.ndarray,
    *,
    limits: np.ndarray,
    horizon: int | None = None,
    max_passes: int | None = None,
) -> np.ndarray:
    """Slew-rate-limit per-step deltas, repaying the deficit downstream."""
    del horizon, max_passes
    out = np.array(values, dtype=np.float64, copy=True)
    if len(out) < 2:
        return out.astype(np.float32)
    limit = np.broadcast_to(np.asarray(limits, dtype=np.float64), (out.shape[1],))
    kernel = native_kernel("ease_overspeed")
    if kernel is not None:
        return kernel(out, limit)
    steps = np.diff(out, axis=0)
    violating_dims = np.flatnonzero(np.any(np.abs(steps) > limit, axis=0))
    for j in violating_dims:
        debt = 0.0
        for k in range(len(steps)):
            candidate = steps[k, j] + debt
            if abs(candidate) <= limit[j]:
                steps[k, j] = candidate
                debt = 0.0
            else:
                sign = 1.0 if candidate > 0 else -1.0
                steps[k, j] = sign * limit[j]
                debt = candidate - sign * limit[j]
    out[1:] = out[0] + np.cumsum(steps, axis=0)
    return out.astype(np.float32)


def _median(values: np.ndarray, window: int, *, nearest: bool) -> np.ndarray:
    kernel = native_kernel("median_filter")
    if kernel is not None:
        return kernel(values, window, nearest).astype(values.dtype)
    return (
        median_filter(values, size=window, mode="nearest")
        if nearest
        else medfilt(values, kernel_size=window)
    )


def _motion_change(values, reference, columns):
    kernel = native_kernel("max_change")
    if kernel is not None:
        return kernel(
            values, reference, columns, values.dtype == np.float32 and reference.dtype == np.float32
        )
    return (
        np.abs(values[:, columns] - reference[:, columns]).max(axis=0).tolist() if columns else []
    )


def _overspeed_steps(values, limits):
    kernel = native_kernel("overspeed_steps")
    if kernel is not None:
        return kernel(values, limits, values.dtype == np.float32)
    return int(np.count_nonzero(np.any(np.abs(np.diff(values, axis=0)) > limits, axis=1)))


@dataclass(frozen=True)
class PreprocessingSpec:
    """Parameters for the shared preprocessing hook."""

    tau: float = 0.002
    min_idle_steps: int = 15
    savgol_window: int = 0
    savgol_polyorder: int = 3
    ease_horizon: int = 0
    smooth_columns: tuple[int, ...] | None = None
    gripper_mode: str = "binary"
    gripper_threshold: float = 0.8
    gripper_median_window: int = 0
    version: int = 2
    trim_mode: str = "shadow"
    profile: str = "joint"
    context_frames: int = 1
    motion_limits: tuple[float, ...] | None = None
    translation_limit: float | None = None
    rotation_limit: float | None = None

    def __post_init__(self) -> None:
        if self.version not in (1, 2):
            raise ValueError("unsupported preprocessing version")
        if self.trim_mode not in ("off", "shadow", "bounded"):
            raise ValueError("trim_mode must be off, shadow or bounded")
        if self.profile not in ("joint", "eef") or self.context_frames < 1:
            raise ValueError("invalid profile or context_frames")
        if self.savgol_window != 0 and (self.savgol_window < 3 or self.savgol_window % 2 == 0):
            raise ValueError("savgol_window must be zero or odd and at least three")
        if self.savgol_polyorder < 1 or (
            self.savgol_window and self.savgol_polyorder >= self.savgol_window
        ):
            raise ValueError("invalid Savitzky-Golay polynomial order")
        if self.ease_horizon < 0 or (
            self.gripper_median_window >= 3 and self.gripper_median_window % 2 == 0
        ):
            raise ValueError("invalid ease or gripper median window")
        if not np.isfinite(self.tau) or self.tau < 0 or self.min_idle_steps < 1:
            raise ValueError("invalid idle candidate parameters")
        budgets = [*(self.motion_limits or ()), self.translation_limit, self.rotation_limit]
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in budgets):
            raise ValueError("physical trim budgets must be finite and positive")
        if self.version == 2 and self.trim_mode == "bounded":
            if self.profile == "joint" and not self.motion_limits:
                raise ValueError("bounded joint trimming requires motion_limits")
            if self.profile == "eef" and (
                self.translation_limit is None or self.rotation_limit is None
            ):
                raise ValueError("bounded EEF trimming requires translation and rotation budgets")
        if self.gripper_mode not in ("binary", "eleven", "twenty"):
            raise ValueError("gripper_mode must be 'binary', 'eleven' or 'twenty'")
        if not 0.0 < self.gripper_threshold < 1.0:
            raise ValueError("gripper_threshold must lie strictly inside (0, 1)")

    def apply(
        self,
        values: np.ndarray,
        *,
        ease_limits: np.ndarray | None = None,
        smooth_columns: tuple[int, ...] | None = None,
        gripper_column: int | None = None,
        stats: dict[str, object] | None = None,
        active_dims: tuple[int, ...] | None = None,
        gripper_is_discrete: bool = False,
    ) -> np.ndarray:
        """Transform one finite continuous segment; record alignment and loss."""
        values = np.asarray(values)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("preprocessing requires finite [T,D] values")
        log = {} if stats is None else stats
        if not len(values):
            log.update(start=0, stop=0, ease_steps=0)
            return values.copy()
        if self.version == 1:
            return self._legacy_apply_recorded(
                values, ease_limits, smooth_columns, gripper_column, log
            )
        start, stop = self._trim_slice(values, gripper_column, active_dims, log)
        original = values[start:stop]
        out = np.array(original, dtype=np.result_type(original.dtype, np.float32), copy=True)
        columns = self.smooth_columns if smooth_columns is None else smooth_columns
        columns = tuple(range(out.shape[1])) if columns is None else tuple(columns)
        if self.profile == "eef":
            columns = tuple(c for c in columns if c in (0, 1, 2))
        columns = tuple(
            c
            for c in columns
            if gripper_column is None or c % out.shape[1] != gripper_column % out.shape[1]
        )
        if self.savgol_window >= 3 and columns:
            out[:, columns] = savgol_smooth(
                out[:, columns], window=self.savgol_window, polyorder=self.savgol_polyorder
            )
            out[: self.context_frames, columns] = original[: self.context_frames, columns]
        log["motion_max_change"] = _motion_change(out, original, columns)
        log["ease_steps"] = 0
        if self.ease_horizon >= 1 and ease_limits is not None:
            if self.profile == "eef":
                raise ValueError("additive easing cannot modify EEF rotations")
            ease_limits = np.broadcast_to(ease_limits, (out.shape[1],)).copy()
            if gripper_column is not None:
                ease_limits[gripper_column] = np.inf
            log["ease_steps"] = _overspeed_steps(out, ease_limits)
            eased = ease_overspeed(out, limits=ease_limits)
            log["ease_endpoint_reverted"] = not bool(
                np.allclose(eased[-1], out[-1], atol=1e-6, rtol=0)
                and np.allclose(
                    eased[: self.context_frames], out[: self.context_frames], atol=1e-6, rtol=0
                )
            )
            if not log["ease_endpoint_reverted"]:
                out = eased
        if gripper_column is not None:
            column = gripper_column % out.shape[1]
            raw = original[:, column : column + 1]
            reference = discretize_gripper_actions(
                raw, mode=self.gripper_mode, threshold=self.gripper_threshold
            )
            if (
                self.gripper_mode in ("eleven", "twenty")
                and not gripper_is_discrete
                and self.gripper_median_window >= 3
            ):
                out[:, column] = _median(raw[:, 0], self.gripper_median_window, nearest=True)
            out[:, column] = discretize_gripper_actions(
                out[:, column : column + 1],
                mode=self.gripper_mode,
                threshold=self.gripper_threshold,
            )[:, 0]
            log["gripper_changed_frames"] = int(np.count_nonzero(out[:, column] != reference[:, 0]))
        log["motion_max_change"] = _motion_change(out, original, columns)
        return out

    def _legacy_apply_recorded(self, values, ease_limits, smooth_columns, gripper_column, log):
        start, stop = _legacy_boundary_trim_indices(
            values, tau=self.tau, min_idle_steps=self.min_idle_steps
        )
        log.update(start=start, stop=stop, version=1)
        out = self._apply_v1(
            values,
            ease_limits=ease_limits,
            smooth_columns=smooth_columns,
            gripper_column=gripper_column,
            stats=log,
        )
        if gripper_column is not None and len(out):
            reference = discretize_gripper_actions(
                values[start:stop, gripper_column : gripper_column + 1]
                if gripper_column >= 0
                else values[start:stop, -1:],
                mode=self.gripper_mode,
                threshold=self.gripper_threshold,
            )
            log["gripper_changed_frames"] = int(
                np.count_nonzero(out[:, gripper_column] != reference[:, 0])
            )
        return out

    def _trim_slice(self, values, gripper_column, active_dims, log):
        start, stop = boundary_trim_indices(
            values,
            tau=self.tau,
            min_idle_steps=self.min_idle_steps,
            context_frames=self.context_frames,
        )
        metrics = trim_deviation(
            values, start, stop, profile=self.profile, gripper_column=gripper_column
        )
        allowed = not metrics["gripper_changed"]
        if self.profile == "joint" and self.motion_limits is not None:
            dims = tuple(range(values.shape[1])) if active_dims is None else active_dims
            local = [
                i
                for i in range(values.shape[1])
                if gripper_column is None or i != gripper_column % values.shape[1]
            ]
            limits = np.asarray(self.motion_limits)[[dims[i] for i in local]]
            allowed &= bool(np.all(np.asarray(metrics["motion_max_error"]) <= limits))
        elif (
            self.profile == "eef"
            and self.translation_limit is not None
            and self.rotation_limit is not None
        ):
            allowed &= metrics["translation_max_error_m"] <= self.translation_limit
            allowed &= metrics["rotation_max_error_rad"] <= self.rotation_limit
        else:
            allowed = False
        log.update(
            version=2,
            candidate_start=start,
            candidate_stop=stop,
            candidate_deviation=metrics,
            trim_approved=bool(allowed),
            trim_mode=self.trim_mode,
        )
        if self.trim_mode != "bounded" or not allowed:
            start, stop = 0, len(values)
        log.update(start=start, stop=stop)
        return start, stop

    def _apply_v1(
        self,
        values: np.ndarray,
        *,
        ease_limits: np.ndarray | None = None,
        smooth_columns: tuple[int, ...] | None = None,
        gripper_column: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> np.ndarray:
        smooth_columns = smooth_columns if smooth_columns is not None else self.smooth_columns
        out = np.asarray(values)
        start, stop = _legacy_boundary_trim_indices(
            out, tau=self.tau, min_idle_steps=self.min_idle_steps
        )
        if stop <= start:
            if stats is not None:
                stats["ease_steps"] = 0
            return out[start:start]
        out = out[start:stop]
        if self.savgol_window >= 3:
            if smooth_columns is None:
                out = savgol_smooth(out, window=self.savgol_window, polyorder=self.savgol_polyorder)
            else:
                out = np.array(out, copy=True)
                columns = tuple(smooth_columns)
                out[:, columns] = savgol_smooth(
                    out[:, columns], window=self.savgol_window, polyorder=self.savgol_polyorder
                )
        if stats is not None:
            stats["ease_steps"] = 0
        if self.ease_horizon >= 1 and ease_limits is not None:
            limit = np.broadcast_to(np.asarray(ease_limits, dtype=np.float64), (out.shape[1],))
            if stats is not None:
                stats["ease_steps"] = _overspeed_steps(out, limit)
            out = ease_overspeed(out, limits=ease_limits, horizon=self.ease_horizon)
        if gripper_column is not None:
            out = np.array(out, copy=True)
            column = gripper_column % out.shape[1]
            if (
                self.gripper_mode in ("eleven", "twenty")
                and self.gripper_median_window >= 3
                and len(out) >= self.gripper_median_window
            ):
                out[:, column] = _median(out[:, column], self.gripper_median_window, nearest=False)
            out[:, column] = discretize_gripper_actions(
                out[:, column : column + 1],
                mode=self.gripper_mode,
                threshold=self.gripper_threshold,
            )[:, 0]
        return out

    def to_dict(self) -> dict[str, object]:
        """Serialize for ``preprocessing.json`` alongside a saved tokenizer."""
        payload = {
            "tau": self.tau,
            "min_idle_steps": self.min_idle_steps,
            "savgol_window": self.savgol_window,
            "savgol_polyorder": self.savgol_polyorder,
            "ease_horizon": self.ease_horizon,
            "smooth_columns": list(self.smooth_columns)
            if self.smooth_columns is not None
            else None,
            "gripper_mode": self.gripper_mode,
            "gripper_threshold": self.gripper_threshold,
            "gripper_median_window": self.gripper_median_window,
        }
        if self.version == 2:
            payload.update(
                version=2,
                trim_mode=self.trim_mode,
                profile=self.profile,
                context_frames=self.context_frames,
                motion_limits=list(self.motion_limits) if self.motion_limits is not None else None,
                translation_limit=self.translation_limit,
                rotation_limit=self.rotation_limit,
            )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PreprocessingSpec:
        """Inverse of :meth:`to_dict`; raises on unknown keys."""
        fields = dict(payload)
        smooth = fields.pop("smooth_columns", None)
        version = fields.pop("version", 1)
        limits = fields.pop("motion_limits", None)
        spec = cls(
            tau=float(fields.pop("tau")),
            min_idle_steps=int(fields.pop("min_idle_steps")),
            savgol_window=int(fields.pop("savgol_window")),
            savgol_polyorder=int(fields.pop("savgol_polyorder")),
            ease_horizon=int(fields.pop("ease_horizon")),
            smooth_columns=tuple(int(c) for c in smooth) if smooth is not None else None,
            gripper_mode=str(fields.pop("gripper_mode", "binary")),
            gripper_threshold=float(fields.pop("gripper_threshold", 0.8)),
            gripper_median_window=int(
                fields.pop("gripper_median_window", 5 if version == 1 else 0)
            ),
            version=version,
            trim_mode=fields.pop("trim_mode", "shadow"),
            profile=fields.pop("profile", "joint"),
            context_frames=int(fields.pop("context_frames", 1)),
            motion_limits=None if limits is None else tuple(float(v) for v in limits),
            translation_limit=fields.pop("translation_limit", None),
            rotation_limit=fields.pop("rotation_limit", None),
        )
        if fields:
            raise ValueError(f"unknown preprocessing keys: {sorted(fields)}")
        return spec
