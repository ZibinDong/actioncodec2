"""Explicit tokenizer pretraining and cleaning shared by both token layers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from .fitting import ProfileFitter

class TrainingMixin:
    def preprocess(
        self,
        values: np.ndarray,
        *,
        profile: str | None = None,
        gripper_column: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Clean a canonical ``(T,D_active)`` profile trajectory for explicit tokenizer pretraining."""
        if profile is None:
            profiles = {component.space for component in self.action_space.components}
            if len(profiles) != 1:
                raise ValueError("profile is required for layouts combining joint and EEF")
            profile = profiles.pop()
        if profile not in self.preprocessing:
            raise ValueError(f"no recorded preprocessing for profile {profile!r}")
        values = np.asarray(values)
        if "ease_limits" not in kwargs and values.ndim == 2:
            kwargs["ease_limits"] = self._joint_ease_limits(
                profile, values.shape[1], gripper_column, kwargs.get("active_dims")
            )
        return self.preprocessing[profile].apply(values, gripper_column=gripper_column, **kwargs)

    def _joint_ease_limits(self, profile, width, gripper_column, active_dims):
        processor = self.action_tokenizer.profiles[profile].processor
        component = processor.config.physical.components.get("generalized_joint_position")
        if profile != "joint" or component is None or component.primitive_order != 1:
            return None
        domain = component.constraints["order_1"]
        limits = np.full(processor.action_dim, np.inf)
        limits[list(component.dims)] = np.minimum(np.abs(domain.minimum), np.abs(domain.maximum))
        dims = tuple(range(width)) if active_dims is None else tuple(active_dims)
        limits = limits[list(dims)]
        if gripper_column is not None:
            limits[gripper_column] = np.inf
        return limits

    def fit(
        self,
        episodes,
        *,
        fps: float | None = None,
        current_state=None,
        previous_first_order=None,
        bootstrap_context: bool = False,
        apply_preprocessing: bool = False,
        preprocessing=None,
        backend: str = "auto",
        threads: int = 0,
        batch_size: int = 256,
        progress: bool = False,
        max_cell_bytes: int | None = None,
        work_dir: str | Path | None = None,
    ) -> TrainingMixin:
        """Fit vocabularies from episodes in declared source layouts."""
        fitter = ProfileFitter(self, bootstrap_context=bootstrap_context,
                      apply_preprocessing=apply_preprocessing, preprocessing=preprocessing)
        router, report = fitter.fit(
            episodes,
            work_dir=work_dir,
            metadata=dict(
                fps=fps, current_state=current_state, previous_first_order=previous_first_order
            ),
            backend=backend,
            threads=threads,
            batch_size=batch_size,
            progress=progress,
            max_cell_bytes=max_cell_bytes,
        )
        replacement = type(self)(router, action_space=self.action_space.name)
        replacement._fitted_profiles = self.fitted_profiles | report["profiles"].keys()
        replacement.training_report = report
        replacement.preprocessing = dict(fitter.preprocessing)
        replacement._warned_default_fps = self._warned_default_fps
        self.__dict__.update(replacement.__dict__)
        return self
