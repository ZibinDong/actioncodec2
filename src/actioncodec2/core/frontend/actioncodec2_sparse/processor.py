"""Action-space-bound ActionCodec2Sparse frontend for independent action chunks."""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from ...backends.actioncodec2 import ActionCodec2Config, PresetConfig
from ...backends.actioncodec2.defaults import builtin_action_spaces
from ...backends.actioncodec2.hf import (
    GENERATED_CARD_MARKER,
    PROCESSOR_CONFIG_NAME,
    resolve_pretrained_directory,
)
from ...backends.actioncodec2.preprocessing import PreprocessingSpec
from ...backends.actioncodec2.release import MANIFEST_NAME
from ...exporting import save_runtime
from ...backends.actioncodec2.semantic import resampled_horizon
from ...backends.actioncodec2.sparse import (
    ActionCodec2SparseProcessor,
    ActionCodec2SparseRouter,
    SparseLayout,
    SparseProfileTrainer,
    SparseVocabulary,
    sparse_profile_config,
)
from ...backends.actioncodec2.sparse.routing import SPARSE_ROUTER_CONFIG_NAME, save_router_metadata
from ...backends.actioncodec2.training import build_primitives
from ..conversion import (
    _numpy,
    _numpy_token_rows,
    restore_source_layout,
)
from ..base_action_tokenizer import BaseActionTokenizer
from ..layout import ActionSpaceMixin
from ..training import TrainingMixin
from .loading import describe_release, load_release_router

ALLOW_PATTERNS = [
    SPARSE_ROUTER_CONFIG_NAME,
    PROCESSOR_CONFIG_NAME,
    "config.json",
    MANIFEST_NAME,
    "profiles/**",
    "joint/**",
    "eef/**",
    "fit_report.json",
    "README.md",
]
_ASSETS = Path(__file__).absolute().parents[2] / "backends" / "actioncodec2" / "assets"


def _profile_config(raw) -> ActionCodec2Config:
    if isinstance(raw, ActionCodec2Config):
        return raw
    if isinstance(raw, Mapping):
        return ActionCodec2Config.from_dict(raw)
    return ActionCodec2Config.from_yaml(raw)


def build_fresh_router(
    profile_configs=None, primitive_order: int = 1, vocab_size: int = 4096
) -> ActionCodec2SparseRouter:
    """Unfitted joint/EEF profiles; defaults derive from the packaged profiles of that order."""
    if type(primitive_order) is not int or primitive_order not in (1, 2):
        raise ValueError("primitive_order must be 1 or 2")
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    if profile_configs is not None and vocab_size != 4096:
        raise ValueError("set vocabulary budgets in profile_configs when providing custom profiles")
    configs = dict(profile_configs or {})
    if configs.keys() - {"joint", "eef"}:
        raise ValueError("profile_configs keys must be 'joint' or 'eef'")
    suffix = "" if primitive_order == 1 else "_second_order"
    processors, budgets = {}, {}
    for name in ("joint", "eef"):
        if name in configs:
            config = _profile_config(configs[name])
        else:
            default = ActionCodec2Config.from_yaml(_ASSETS / "profiles" / f"{name}{suffix}.yaml")
            config = sparse_profile_config(default, vocab_size=vocab_size)
        if config.train is None or config.train.sparse is None:
            raise ValueError(f"{name} profile config requires train.sparse")
        primitives = build_primitives(config)
        layout = SparseLayout.from_config(config, primitives)
        vocabulary = SparseVocabulary(layout, (), (), config.set_bpe.lookahead)
        processors[name] = ActionCodec2SparseProcessor(config, primitives, vocabulary)
        budgets[name] = config.train.bpe.vocab_size
    return ActionCodec2SparseRouter.from_processors(
        processors, token_budgets=budgets, presets=list(builtin_action_spaces().values())
    )


class ActionCodec2Sparse(ActionSpaceMixin, TrainingMixin, BaseActionTokenizer):
    """ActionCodec2 physics with a zero-free token layer, bound to one action space."""

    trainer_class = SparseProfileTrainer
    model_type = "actioncodec2_sparse"
    encode_kwargs = frozenset({"fps", "current_state", "previous_first_order"})

    def __init__(
        self,
        backend: ActionCodec2SparseRouter | None = None,
        *,
        action_space: str,
        primitive_order: int = 1,
        vocab_size: int = 4096,
        profile_configs: Mapping[str, object] | None = None,
    ) -> None:
        fresh = backend is None
        if fresh:
            backend = build_fresh_router(profile_configs, primitive_order, vocab_size)
        elif profile_configs is not None or primitive_order != 1 or vocab_size != 4096:
            raise ValueError("profile construction options cannot accompany an existing backend")
        if not isinstance(backend, ActionCodec2SparseRouter):
            raise TypeError(
                f"ActionCodec2Sparse requires ActionCodec2SparseRouter, got {type(backend).__name__}"
            )
        preset = backend.preset(action_space)
        if preset.target_hz != 15.0:
            raise ValueError("ActionCodec2Sparse artifacts require target_hz=15")
        self.action_tokenizer = backend
        self._action_space = preset
        self._action_vocab_size = int(backend.vocab_size)
        self._new_action_tokens = backend.new_action_tokens
        self.action_dim = preset.input_dim
        self.requires_previous_first_order = bool(backend.second_order_boundaries(preset))
        self.max_action_token_len = None
        self.fixed_action_token_len = None
        self._warned_default_fps = False
        super().__init__(append_bar_sentinels=False)
        self._fitted_profiles = frozenset() if fresh else frozenset(backend.profiles)
        self.training_report = None
        self.preprocessing: dict[str, PreprocessingSpec] = {}
        self.release_metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"ActionCodec2Sparse(action_space={self.action_space.name!r}, "
            f"D={self.action_dim}, vocab={self.vocab_size})"
        )


    def _describe_action_spaces(self, action_dim: int | None = None) -> str:
        fitted = ", ".join(sorted(self.fitted_profiles)) or "none; call codec.fit(episodes)"
        layouts = self.action_tokenizer.describe_action_spaces(
            selected=self.action_space.name, action_dim=action_dim
        )
        return f"Fitted profiles: {fitted}\n{layouts}"

    @staticmethod
    def builtin_action_spaces():
        """Return the immutable named layout definitions shipped in the package."""
        return builtin_action_spaces()



    def print_action_spaces(self, *, action_dim: int | None = None, file: TextIO | None = None):
        """Print all layouts or candidates matching ``action_dim``."""
        print(self._describe_action_spaces(action_dim), file=sys.stdout if file is None else file)


    def for_action_space(self, name: str) -> ActionCodec2Sparse:
        """Share current weights with a new binding."""
        codec = type(self)(self.action_tokenizer, action_space=name)
        codec._fitted_profiles = self.fitted_profiles
        codec.training_report = self.training_report
        codec.preprocessing = dict(self.preprocessing)
        codec.release_metadata = self.release_metadata
        return codec

    @property
    def fitted_profiles(self) -> frozenset[str]:
        """Names of profiles fitted on data; persisted by save_pretrained()."""
        return self._fitted_profiles

    @property
    def is_fitted(self) -> bool:
        """Whether every profile used by the selected action space is fitted."""
        return all(c.space in self.fitted_profiles for c in self.action_space.components)

    def _require_fitted(self) -> None:
        missing = {c.space for c in self.action_space.components} - self.fitted_profiles
        if missing:
            raise RuntimeError(
                f"Unfitted profiles {sorted(missing)}; call codec.fit(episodes) first"
            )

    def _source_fps(self, fps, preset=None) -> float:
        preset = self.action_space if preset is None else preset
        if fps is None:
            if not self._warned_default_fps:
                warnings.warn(
                    f"fps was omitted; assuming {preset.source_hz:g} Hz from action_space "
                    f"{preset.name!r}. Pass the actual fps explicitly to avoid changing action timing.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_default_fps = True
            return float(preset.source_hz)
        return float(fps)



    def codec_horizon(self, horizon: int, *, fps: float | None = None) -> int:
        """Codec frames (15 Hz) of a ``horizon``-frame source chunk at ``fps``."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return resampled_horizon(
            horizon, source_hz=self._source_fps(fps), target_hz=self.action_space.target_hz
        )

    def max_tokens_for_horizon(self, horizon: int, *, fps: float | None = None) -> int:
        """Upper bound on the tokens of one encoded ``horizon``-frame source chunk."""
        frames = self.codec_horizon(horizon, fps=fps)
        components = self.action_space.components
        width = sum(len(c.active_dims) + 1 for c in components)
        return frames * width + len(components) - 1

    def grammar(self, horizon: int, *, fps: float | None = None):
        """Constrained-generation grammar for one ``horizon``-frame source chunk."""
        self._require_fitted()
        return self.action_tokenizer.grammar(
            preset=self.action_space.name, horizon=self.codec_horizon(horizon, fps=fps)
        )

    def logits_processor(
        self, horizon, *, fps, prompt_length, eos_token_id, token_offset=0, pad_token_id=None
    ):
        """Create a Transformers LogitsProcessor for one ``horizon``-frame chunk."""
        from .generation import ActionCodec2SparseLogitsProcessor

        return ActionCodec2SparseLogitsProcessor(
            self,
            horizon,
            fps=fps,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
            token_offset=token_offset,
            pad_token_id=pad_token_id,
        )

    def _check_state(self, current_state, action: str):
        name = self.action_space.name
        if self.action_space.input == "absolute" and current_state is None:
            raise ValueError(
                f"action space {name!r} is absolute; {action} requires current_state with shape [B, D]"
            )
        if self.action_space.input == "delta" and current_state is not None:
            raise ValueError(f"action space {name!r} is delta; omit current_state")
        return _numpy(current_state)

    def encode(
        self, action, *, fps=None, current_state=None, previous_first_order=None, encode_kwargs=None
    ) -> list[list[int]]:
        """Encode ``(B,T,D)`` or ``(T,D)`` source chunks into ragged token rows."""
        kwargs = self._check_encode_kwargs(encode_kwargs)
        direct = {
            "fps": fps,
            "current_state": current_state,
            "previous_first_order": previous_first_order,
        }
        duplicates = sorted(k for k, v in direct.items() if v is not None and k in kwargs)
        if duplicates:
            raise ValueError(f"encode arguments provided twice: {duplicates}")
        kwargs.update({k: v for k, v in direct.items() if v is not None})
        values = torch.as_tensor(action)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        return self._encode_action_indices(values, encode_kwargs=kwargs)

    def __call__(
        self,
        action,
        *,
        fps=None,
        current_state=None,
        previous_first_order=None,
        encode_kwargs=None,
        return_tensors=None,
    ) -> BatchFeature:
        rows = self.encode(
            action,
            fps=fps,
            current_state=current_state,
            previous_first_order=previous_first_order,
            encode_kwargs=encode_kwargs,
        )
        return BatchFeature({"input_ids": rows}, tensor_type=return_tensors)

    def _encode_action_indices(self, action, encode_kwargs=None):
        self._require_fitted()
        kwargs = dict(encode_kwargs or {})
        current_state = self._check_state(kwargs.get("current_state"), "encode")
        values = _numpy(action)
        if values.ndim != 3 or values.shape[2] != self.action_dim:
            raise ValueError(f"actions must have shape [B, T_source, {self.action_dim}]")
        return self.action_tokenizer.encode_batch(
            values,
            preset=self.action_space.name,
            fps=self._source_fps(kwargs.get("fps")),
            current_state=current_state,
            previous_first_order=self._numpy_boundary(kwargs.get("previous_first_order")),
        )

    def decode(
        self,
        action_token_ids,
        *,
        horizon: int,
        fps=None,
        current_state=None,
        previous_first_order=None,
    ) -> torch.Tensor:
        """Decode rows of one ``horizon``-frame source chunk to a CPU ``(B,T,D)`` tensor."""
        return self.decode_token_ids_to_actions_batch(
            action_token_ids,
            time_horizon=horizon,
            decode_kwargs={
                "fps": fps,
                "current_state": current_state,
                "previous_first_order": previous_first_order,
            },
        )

    def decode_token_ids_to_actions_batch(
        self, action_token_ids, *, time_horizon=None, action_dim=None, decode_kwargs=None
    ) -> torch.Tensor:
        """Decode integer rows (ragged or ``(B,L)``) of one source ``time_horizon``."""
        rows = [
            self._checked_decode_row(row).tolist() for row in _numpy_token_rows(action_token_ids)
        ]
        actions, _ = self._decode_action_indices(
            rows, time_horizon=time_horizon, action_dim=action_dim, decode_kwargs=decode_kwargs
        )
        return torch.from_numpy(actions)

    def _decode_action_indices(
        self, action_indices, *, time_horizon=None, action_dim=None, decode_kwargs=None
    ):
        self._require_fitted()
        if time_horizon is None:
            raise ValueError("ActionCodec2Sparse decode requires the chunk horizon")
        if action_dim is not None and int(action_dim) != self.action_dim:
            raise ValueError(
                f"action_dim={action_dim} disagrees with action space D={self.action_dim}"
            )
        kwargs = dict(decode_kwargs or {})
        unknown = sorted(set(kwargs) - self.encode_kwargs)
        if unknown:
            raise ValueError(f"ActionCodec2Sparse does not consume decode_kwargs {unknown}")
        fps = self._source_fps(kwargs.get("fps"))
        decoded = self.action_tokenizer.decode_batch(
            action_indices,
            preset=self.action_space.name,
            horizon=self.codec_horizon(time_horizon, fps=fps),
            fps=fps,
            current_state=self._check_state(kwargs.get("current_state"), "decode"),
            previous_first_order=self._numpy_boundary(kwargs.get("previous_first_order")),
        )
        actions = restore_source_layout(self.action_space, decoded)
        return actions, [set() for _ in range(len(actions))]

    def batch_decode(
        self,
        token_rows,
        *,
        horizon: int,
        fps=None,
        current_state=None,
        previous_first_order=None,
        errors="raise",
    ):
        """Decode independent rows of one horizon, isolating invalid rows."""
        if errors not in ("raise", "return"):
            raise ValueError("errors must be 'raise' or 'return'")
        rows = list(token_rows)
        current = _numpy(current_state)
        if current is not None and np.shape(current) != (len(rows), self.action_dim):
            raise ValueError(f"current_state must have shape {(len(rows), self.action_dim)}")
        previous = self._numpy_boundary(previous_first_order)
        if previous is not None and any(len(v) != len(rows) for v in previous.values()):
            raise ValueError(
                "previous_first_order arrays must have batch dimension equal to token rows"
            )
        result = []
        for index, row in enumerate(rows):
            try:
                state = None if current is None else current[index : index + 1]
                boundary = (
                    None
                    if previous is None
                    else {k: v[index : index + 1] for k, v in previous.items()}
                )
                result.append(
                    self.decode(
                        [row],
                        horizon=horizon,
                        fps=fps,
                        current_state=state,
                        previous_first_order=boundary,
                    )
                )
            except ValueError as error:
                error = ValueError(f"token row {index}: {error}")
                if errors == "raise":
                    raise error from None
                result.append(error)
        return result

    def decode_prefix(
        self,
        action_token_ids,
        *,
        horizon: int,
        steps: int,
        fps=None,
        current_state=None,
        previous_first_order=None,
    ):
        """Decode the first ``steps`` 15 Hz frames of streaming prefixes to ``(B,steps,D)``."""
        self._require_fitted()
        fps = self._source_fps(fps)
        if fps != self.action_space.target_hz:
            raise ValueError("prefix decode requires fps equal to the 15 Hz codec clock")
        frames = self.codec_horizon(horizon, fps=fps)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= frames:
            raise ValueError("steps must be an integer within the chunk horizon")
        rows = [self._checked_decode_row(row).tolist() for row in action_token_ids]
        grammar = self.action_tokenizer.grammar(preset=self.action_space.name, horizon=frames)
        if any(grammar.covered_steps(row) < steps for row in rows):
            raise ValueError(f"action prefix does not fix the first {steps} frames")
        decoded = self.action_tokenizer.decode_batch(
            rows,
            preset=self.action_space.name,
            horizon=frames,
            fps=fps,
            current_state=self._check_state(_numpy(current_state), "decode"),
            previous_first_order=self._numpy_boundary(previous_first_order),
        )
        return torch.from_numpy(restore_source_layout(self.action_space, decoded)[:, :steps])

    def final_first_order(
        self,
        action_token_ids,
        *,
        horizon: int,
        previous_first_order,
        fps=None,
        executed_steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Second-order component boundaries ``{component: (B,D_active)}`` for the next chunk."""
        self._require_fitted()
        return self.action_tokenizer.final_first_order(
            [self._checked_decode_row(row).tolist() for row in action_token_ids],
            preset=self.action_space.name,
            horizon=self.codec_horizon(horizon, fps=fps),
            previous_first_order=self._numpy_boundary(previous_first_order),
            executed_steps=executed_steps,
        )




    @classmethod
    def describe_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        action_dim=None,
        gripper_mode="2bin",
        primitive_order: int = 1,
        **kwargs,
    ) -> str:
        """Report a release's or saved artifact's layouts from manifests alone."""
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, allow_patterns=ALLOW_PATTERNS, **kwargs
        )
        if (directory / MANIFEST_NAME).is_file():
            return describe_release(
                directory,
                action_dim=action_dim,
                gripper_mode=gripper_mode,
                primitive_order=primitive_order,
            )
        specs, presets = ActionCodec2SparseRouter.read_manifest(directory)
        metadata = _saved_metadata(directory)
        fitted = ", ".join(metadata.get("fitted_profiles", [s.name for s in specs]))
        from ...backends.actioncodec2.presets import format_action_spaces

        layouts = format_action_spaces(
            {p.name: p for p in presets},
            selected=metadata.get("action_space"),
            action_dim=action_dim,
            codec_name="ActionCodec2Sparse",
            chunk_horizon_decode=True,
        )
        return f"Fitted profiles: {fitted}\n{layouts}"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        action_space=None,
        gripper_mode: str | None = None,
        primitive_order: int | None = None,
        verify_gates: bool = False,
        **kwargs,
    ) -> ActionCodec2Sparse:
        """Load a release package or saved frontend and bind an action space."""
        if cls is ActionCodec2Sparse and ("variant" in kwargs or (Path(pretrained_model_name_or_path) / "variants.json").is_file()
                                   or (Path(pretrained_model_name_or_path) / "artifact_manifest.json").is_file()):
            from ...catalog import ActionCodec2 as Catalog
            return Catalog.from_pretrained(pretrained_model_name_or_path, action_space=action_space, **kwargs)
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, allow_patterns=ALLOW_PATTERNS, **kwargs
        )
        metadata = _saved_metadata(directory)
        action_space = metadata.get("action_space") if action_space is None else action_space
        if action_space is None:
            raise ValueError(
                "action_space is required; inspect layouts with "
                "print(ActionCodec2Sparse.describe_pretrained(path))"
            )
        if (directory / MANIFEST_NAME).is_file():
            router, preprocessing, release = load_release_router(
                directory,
                gripper_mode=gripper_mode or "2bin",
                primitive_order=1 if primitive_order is None else primitive_order,
                verify_gates=verify_gates,
            )
            codec = cls(router, action_space=_register(router, action_space))
            codec.preprocessing, codec.release_metadata = preprocessing, release
            return codec
        if gripper_mode is not None or primitive_order is not None or verify_gates:
            raise ValueError(
                "gripper_mode, primitive_order and verify_gates select release variants only"
            )
        router = ActionCodec2SparseRouter.from_pretrained(directory)
        codec = cls(router, action_space=_register(router, action_space))
        fitted = metadata.get("fitted_profiles", list(router.profiles))
        if not isinstance(fitted, list) or set(fitted) - router.profiles.keys():
            raise ValueError("invalid fitted_profiles in processor_config.json")
        codec._fitted_profiles = frozenset(fitted)
        codec.preprocessing = {
            name: PreprocessingSpec.from_dict(spec)
            for name, spec in metadata.get("preprocessing", {}).items()
        }
        codec.release_metadata = metadata.get("release")
        report = directory / "fit_report.json"
        if report.is_file():
            codec.training_report = json.loads(report.read_text(encoding="utf-8"))
        return codec

    def to_dict(self) -> dict[str, object]:
        """Lightweight discovery metadata saved as ``processor_config.json``."""
        payload = {
            "action_space": self.action_space.name,
            "fitted_profiles": sorted(self.fitted_profiles),
        }
        if self.preprocessing:
            payload["preprocessing"] = {k: v.to_dict() for k, v in self.preprocessing.items()}
        return payload

    def save_pretrained(
        self, save_directory: str | Path, *, include_training_report=False
    ) -> list[str]:
        """Save inference assets; include the local fit report only when explicitly requested."""
        directory = Path(save_directory)
        files = self.action_tokenizer.save_pretrained(directory)
        files.extend(map(str, save_router_metadata(directory, **self.to_dict())))
        files.append(str(_save_model_card(directory, self)))
        report = directory / "fit_report.json"
        if include_training_report and self.training_report is not None:
            report.write_text(json.dumps(self.training_report, indent=2) + "\n", encoding="utf-8")
            files.append(str(report))
        elif report.exists():
            report.unlink()
        files.extend(save_runtime(directory))
        return files

    def eval(self) -> ActionCodec2Sparse:
        return self


def _save_model_card(directory: Path, codec: ActionCodec2Sparse) -> Path:
    path = directory / "README.md"
    if path.exists() and GENERATED_CARD_MARKER not in path.read_text(encoding="utf-8"):
        return path
    context = 'context = {"fps": 15.0}\n'
    if codec.action_space.input == "absolute":
        context += 'context["current_state"] = current_state  # observed (B, D) source state\n'
    if codec.requires_previous_first_order:
        context += 'context["previous_first_order"] = previous_first_order  # canonical 15 Hz increments by component\n'
    path.write_text(
        "---\nlibrary_name: actioncodec2\nlicense: apache-2.0\ntags:\n- robotics\n- action-tokenizer\n---\n\n"
        f"{GENERATED_CARD_MARKER}\n\n# ActionCodec2Sparse\n\n"
        "Install actioncodec2 >= 0.1.0 before loading this artifact.\n\n"
        "```python\nfrom actioncodec2 import ActionCodec2Sparse\n\n"
        "codec = ActionCodec2Sparse.from_pretrained('./my_codec')\n"
        + context
        + "rows = codec.encode(actions, **context)\n"
        "restored = codec.decode(rows, horizon=actions.shape[-2], **context)\n```\n\n"
        "Supply actions and any required boundary observations in the declared layout.\n"
        "Rows are independent chunks; decode and grammar take the chunk horizon.\n\n"
        "## Layouts\n\n```text\n" + str(codec) + "\n```\n",
        encoding="utf-8",
    )
    return path


def _saved_metadata(directory: Path) -> dict:
    path = directory / PROCESSOR_CONFIG_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _register(router: ActionCodec2SparseRouter, action_space) -> str:
    if isinstance(action_space, (PresetConfig, Mapping, Path)):
        return router.register_action_space(action_space).name
    if isinstance(action_space, str):
        if Path(action_space).is_file():
            return router.register_action_space(Path(action_space)).name
        return action_space
    raise TypeError("action_space must be a registered name, PresetConfig, mapping, or YAML path")
