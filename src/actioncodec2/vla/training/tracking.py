"""Experiment tracking: four scalars per logged step, two interchangeable backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

__all__ = ["TrackingConfig", "RunTracker", "flatten"]

BACKENDS = ("none", "swanlab", "wandb")


@dataclass
class TrackingConfig:
    """Where a run reports to.  ``backend: none`` is a complete, working run."""

    backend: str = "none"
    project: str = "actioncodec2"
    #: Defaults to the basename of ``train.output_dir``, which is already the
    #: name the run is stored under on disk.
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    #: Passed through to the backend: swanlab takes online/local/offline/disabled,
    #: wandb takes online/offline/disabled.  ``None`` uses the backend default.
    mode: str | None = None
    #: Formal runs may require the remote record to exist.  When true, an init
    #: failure aborts before optimizer step one instead of silently running
    #: without the experiment the user asked for.
    required: bool = False

    def __post_init__(self) -> None:
        self.backend = (self.backend or "none").lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"tracking.backend must be one of {BACKENDS}, got {self.backend!r}")


def flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config to ``a.b.c`` keys, which both backends table well.

    Lists are left whole: ``data.roots`` is one fact about the run, not five.
    """
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def tracking_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare configuration values for remote tracking."""
    flat = flatten(payload)
    for key, value in flat.items():
        leaf = key.rsplit(".", 1)[-1].lower()
        if any(part.lower() in {"token", "api_key", "access_token", "secret", "password", "credential", "credentials"} or part.lower().endswith("_token")
               for part in key.split(".")):
            flat[key] = "<redacted>"
        elif leaf in {"roots", "output_dir", "warm_start_checkpoint"}:
            flat[key] = "<local-path>"
        elif isinstance(value, str):
            if value.startswith(("/", "./", "../", "~")) or (
                leaf in {"path", "tokenizer_path", "proprio_stats", "artifact", "processor",
                         "source", "action_stats", "normalizer_path", "statistics_path",
                         "calibration_path"} and Path(value).exists()
            ):
                flat[key] = "<local-path>"
            else:
                flat[key] = re.sub(
                    r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                    r"github_pat_[A-Za-z0-9_]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b",
                    "<redacted>", value,
                )
    return flat


class RunTracker:
    """A logging handle that is safe to call whether or not anything is behind it."""

    def __init__(self, backend: str = "none", run: Any = None) -> None:
        self.backend = backend
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @classmethod
    def start(
        cls,
        config: TrackingConfig | None,
        *,
        run_config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> RunTracker:
        """Open a run, or return a disabled tracker if that is not possible."""
        if config is None or config.backend == "none":
            return cls()
        flat = tracking_config(run_config or {})
        run_name = config.name or name
        try:
            if config.backend == "swanlab":
                import swanlab

                run = swanlab.init(
                    project=config.project,
                    name=run_name,
                    description=config.notes,
                    tags=list(config.tags) or None,
                    config=flat,
                    mode=config.mode,
                )
            else:
                import wandb

                run = wandb.init(
                    project=config.project,
                    name=run_name,
                    notes=config.notes,
                    tags=list(config.tags) or None,
                    config=flat,
                    mode=config.mode,
                )
        except Exception as error:  # noqa: BLE001 - logging must never stop training
            if config.required:
                raise RuntimeError(
                    f"required {config.backend} tracking failed to initialize"
                ) from error
            print(f"[tracking] {config.backend} unavailable ({error}); running unlogged",
                  flush=True)
            return cls()
        print(f"[tracking] {config.backend} run {run_name!r} in project {config.project!r}",
              flush=True)
        return cls(config.backend, run)

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Send one point.  ``step`` is the global optimizer step."""
        if self._run is None:
            return
        try:
            self._run.log({key: float(value) for key, value in metrics.items()}, step=step)
        except Exception as error:  # noqa: BLE001
            print(f"[tracking] dropped step {step} ({error})", flush=True)

    def summarize(self, values: Mapping[str, Any]) -> None:
        """Record end-of-run facts: single numbers, not a series."""
        if self._run is None:
            return
        flat = flatten(dict(values), "summary")
        try:
            if self.backend == "wandb":
                self._run.summary.update(flat)
            else:
                self._run.config.update(flat)
        except Exception as error:  # noqa: BLE001
            print(f"[tracking] summary not recorded ({error})", flush=True)

    def finish(self, *, state: str = "success", error: str | None = None) -> None:
        """Close the backend with the real process outcome (SwanLab would otherwise
        report every run as ``success``)."""
        if self._run is None:
            return
        try:
            if self.backend == "swanlab":
                self._run.finish(state=state, error=error)
            elif self.backend == "wandb":
                self._run.finish(exit_code=0 if state == "success" else 1)
            else:
                self._run.finish()
        except Exception as error:  # noqa: BLE001
            print(f"[tracking] finish failed ({error})", flush=True)
        finally:
            self._run = None

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(self, exc_type: object, exc: object, _: object) -> None:
        if exc is None:
            self.finish()
            return
        state = "aborted" if isinstance(exc, KeyboardInterrupt) else "crashed"
        self.finish(state=state, error=f"{type(exc).__name__}: {exc}")
