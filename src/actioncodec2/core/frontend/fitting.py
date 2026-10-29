"""Replayable, bounded source staging for the public ActionCodec2.fit interface."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..backends.actioncodec2.boundary import bootstrap_second_order_trajectory
from ..backends.actioncodec2.trajectory import ActiveTrajectory
from ..backends.actioncodec2.preprocessing import PreprocessingSpec
from .conversion import _numpy


class TrajectorySpool:
    """Stage canonical episodes as one concatenated .npy stream per profile."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.counts = {}
        self._stack = ExitStack()
        self._writers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def append(self, profile, trajectory):
        if profile not in self._writers:
            self._writers[profile] = self._stack.enter_context(
                (self.directory / f"{profile}.episodes").open("wb")
            )
        stream = self._writers[profile]
        for value in (
            trajectory.values,
            trajectory.active_dims,
            [] if trajectory.previous_first_order is None else trajectory.previous_first_order,
        ):
            np.save(stream, np.asarray(value), allow_pickle=False)
        self.counts[profile] = self.counts.get(profile, 0) + 1

    def trajectories(self, profile):
        self._writers[profile].flush()
        with (self.directory / f"{profile}.episodes").open("rb") as stream:
            for _ in range(self.counts[profile]):
                values = np.load(stream, allow_pickle=False)
                dims = tuple(int(d) for d in np.load(stream, allow_pickle=False))
                previous = np.load(stream, allow_pickle=False)
                yield ActiveTrajectory(
                    values, dims, previous_first_order=previous if previous.size else None
                )


class ProfileFitter:
    """Route user layouts and fit replacement profiles without mutating the caller."""

    def __init__(self, codec, *, bootstrap_context=False, apply_preprocessing=False, preprocessing=None):
        if not isinstance(bootstrap_context, bool):
            raise TypeError("bootstrap_context must be a boolean")
        if type(apply_preprocessing) is not bool:
            raise TypeError("apply_preprocessing must be a boolean")
        if preprocessing is not None and not apply_preprocessing:
            raise ValueError("preprocessing requires apply_preprocessing=True during tokenizer fitting")
        self.preprocessing = dict(codec.preprocessing)
        if preprocessing is not None:
            if not isinstance(preprocessing, Mapping) or set(preprocessing) - set(codec.action_tokenizer.profiles):
                raise ValueError("preprocessing must map available physical profiles to cleaning specs")
            self.preprocessing = {
                name: spec if isinstance(spec, PreprocessingSpec) else PreprocessingSpec.from_dict(spec)
                for name, spec in preprocessing.items()
            }
        self.apply_preprocessing = apply_preprocessing
        if apply_preprocessing:
            for name, spec in self.preprocessing.items():
                if spec.version == 2 and spec.profile != name:
                    raise ValueError(f"preprocessing profile {spec.profile!r} does not match {name!r}")
        self.trainer_class = codec.trainer_class
        self.codec = codec
        self.router = codec.action_tokenizer
        self.bootstrap_context = bootstrap_context

    def _route(self, record):
        unknown = set(record) - {
            "actions",
            "action_space",
            "fps",
            "current_state",
            "previous_first_order",
        }
        if unknown:
            raise ValueError(f"unknown training record fields: {sorted(unknown)}")
        if "actions" not in record:
            raise ValueError("training record requires an 'actions' array")
        preset = self.router.preset(record.get("action_space", self.codec.action_space.name))
        values = _numpy(torch.as_tensor(record["actions"]))
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("training actions must have nonempty shape (T,D) or (B,T,D)")
        current = _numpy(record.get("current_state"))
        if preset.input == "absolute" and current is None:
            raise ValueError(f"absolute action space {preset.name!r} requires current_state(B,D)")
        if preset.input == "delta" and current is not None:
            raise ValueError(f"delta action space {preset.name!r}: omit current_state")
        previous = self.codec._numpy_boundary(record.get("previous_first_order"))
        kwargs = dict(
            preset=preset.name,
            fps=self.codec._source_fps(record.get("fps"), preset),
            current_state=current,
        )
        if not self.bootstrap_context and not self.apply_preprocessing:
            return self.router.route_trajectories(values, previous_first_order=previous, **kwargs)
        if self.bootstrap_context and previous is not None:
            raise ValueError("bootstrap_context and previous_first_order are mutually exclusive")
        canonical = preset.canonicalize(values, source_hz=kwargs["fps"], current_state=current)
        routed = {}
        for component in preset.components:
            processor = self.router.profiles[component.space].processor
            trajectories = canonical[component.name]
            if self.apply_preprocessing:
                if component.space not in self.preprocessing:
                    raise ValueError(f"no preprocessing spec for profile {component.space!r}")
                spec = self.preprocessing[component.space]
                if processor.requires_previous_first_order and not self.bootstrap_context:
                    raise ValueError("second-order cleaning requires full trajectories and bootstrap_context=True")
                column = -1 if component.gripper_dim is not None else None
                limits = self.codec._joint_ease_limits(component.space, len(component.active_dims), column, component.active_dims)
                cleaned = []
                for trajectory in trajectories:
                    values = spec.apply(trajectory.values, ease_limits=limits,
                                        gripper_column=column, active_dims=component.active_dims)
                    if len(values):
                        cleaned.append(ActiveTrajectory(values, component.active_dims))
                trajectories = cleaned
            if processor.requires_previous_first_order:
                trajectories = [
                    bootstrap_second_order_trajectory(t, processor.config) for t in trajectories
                ]
            routed.setdefault(component.space, []).extend(trajectories)
        return routed

    def fit(self, episodes, *, work_dir=None, metadata=None, **options):
        """Fit observed profiles from arrays or records, replaying staged episodes per profile."""
        metadata = {k: v for k, v in (metadata or {}).items() if v is not None}
        if isinstance(episodes, (np.ndarray, torch.Tensor, Mapping)):
            episodes = (episodes,)
        reports = {}
        processors = {name: runtime.processor for name, runtime in self.router.profiles.items()}
        records = 0
        with TemporaryDirectory(prefix="actioncodec2-fit-", dir=work_dir) as directory:
            with TrajectorySpool(directory) as spool:
                for episode in episodes:
                    record = dict(episode) if isinstance(episode, Mapping) else {"actions": episode}
                    duplicates = set(record).intersection(metadata)
                    if duplicates:
                        raise ValueError(f"training arguments provided twice: {sorted(duplicates)}")
                    record.update(metadata)
                    for profile, trajectories in self._route(record).items():
                        for trajectory in trajectories:
                            spool.append(profile, trajectory)
                    records += 1
                if not spool.counts:
                    raise ValueError("fit requires at least one nonempty episode")
                for name in spool.counts:
                    trainer = self.trainer_class(processors[name].config)
                    processors[name], reports[name] = trainer.fit(
                        spool.trajectories(name), **options
                    )
        router = type(self.router).from_processors(
            processors,
            token_budgets={
                name: runtime.spec.token_budget for name, runtime in self.router.profiles.items()
            },
            presets=list(self.router.action_spaces.values()),
        )
        return router, {
            "records": records,
            "bootstrap_context": self.bootstrap_context,
            "apply_preprocessing": self.apply_preprocessing,
            "profiles": reports,
        }
