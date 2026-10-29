"""Hugging Face processor over the shared ActionCodec2 frontend implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ...backends.actioncodec2 import ActionCodec2Router, PresetConfig
from ...backends.actioncodec2.defaults import (
    CodecBuilder,
    ProfileBundleBuilder,
    builtin_action_spaces,
)
from ...backends.actioncodec2.hf import (
    PROCESSOR_CONFIG_NAME,
    ROUTER_ALLOW_PATTERNS,
    ROUTER_MANIFEST_PATTERNS,
    ActionCodec2FrontendHFConfig,
    resolve_pretrained_directory,
    save_huggingface_metadata,
    save_model_card,
    upload_pretrained_directory,
)
from ...backends.actioncodec2.preprocessing import PreprocessingSpec
from ...backends.actioncodec2.release import MANIFEST_NAME
from ...exporting import save_runtime
from .loading import describe_release, load_release_router
from .tokenizer import ActionCodec2Tokenizer
from ..training import TrainingMixin
from ...backends.actioncodec2.training import ProfileTrainer


def _saved_metadata(directory: Path) -> dict:
    metadata = directory / PROCESSOR_CONFIG_NAME
    return json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}


def _fitted_status(profiles) -> str:
    status = ", ".join(sorted(profiles)) or "none; call codec.fit(episodes)"
    return f"Fitted profiles: {status}\n"


class ActionCodec2(TrainingMixin, ActionCodec2Tokenizer, ProcessorMixin):
    'Action-space-bound processor with lossless token serialization.'

    trainer_class = ProfileTrainer
    model_type = ActionCodec2FrontendHFConfig.model_type
    model_input_names = ["input_ids"]
    # Numerical artifacts are owned by the router, not modality sub-processors.
    attributes = []

    def __init__(
        self,
        backend: ActionCodec2Router | None = None,
        *,
        action_space: str,
        primitive_order: int = 1,
        bpe_scheme: str | None = None,
        profile_configs: Mapping[str, object] | None = None,
    ) -> None:
        fresh = backend is None
        if fresh:
            backend = CodecBuilder(
                primitive_order=primitive_order,
                bpe_scheme=bpe_scheme,
                profile_configs=profile_configs,
            ).build()
        elif primitive_order != 1 or bpe_scheme is not None or profile_configs is not None:
            raise ValueError("profile construction options cannot accompany an existing backend")
        super().__init__(backend, action_space=action_space)
        ProcessorMixin.__init__(self)
        self._fitted_profiles = frozenset() if fresh else frozenset(backend.profiles)
        self.training_report = None
        self.preprocessing: dict[str, PreprocessingSpec] = {}
        self.release_metadata: dict[str, Any] | None = None

    def __dir__(self):
        inapplicable = {
            "audio_tokenizer",
            "chat_template",
            "feature_extractor_class",
            "apply_chat_template",
            "post_process_image_text_to_text",
            "pad_action_id",
            "bos_blk_id",
            "parts_meta",
            "BAR_SENTINEL_TOKENS",
        }
        return [name for name in super().__dir__() if name not in inapplicable]

    def __repr__(self) -> str:
        return (
            f"ActionCodec2(action_space={self.action_space.name!r}, "
            f"D={self.action_dim}, vocab={self.action_tokenizer.vocab_size})"
        )

    @staticmethod
    def builtin_action_spaces():
        """Return the immutable named layout definitions shipped in the package."""
        return builtin_action_spaces()

    @classmethod
    def from_profiles(cls, joint, eef, *, action_space: str, token_budgets=None):
        'Bind an artifact assembled from existing joint/EEF profile directories.'
        router = ProfileBundleBuilder(joint, eef, token_budgets=token_budgets).build()
        return cls(router, action_space=action_space)

    def logits_processor(
        self,
        horizon: int,
        *,
        fps: float,
        prompt_length: int,
        eos_token_id: int,
        token_offset: int = 0,
        pad_token_id=None,
    ):
        'Create a Transformers LogitsProcessor for a fixed action horizon.'
        from .generation import ActionCodec2LogitsProcessor

        return ActionCodec2LogitsProcessor(
            self,
            horizon,
            fps=fps,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
            token_offset=token_offset,
            pad_token_id=pad_token_id,
        )

    def batch_decode(
        self, token_rows, *, fps=None, current_state=None, previous_first_order=None, errors="raise"
    ):
        'Decode independent token rows, including different valid horizons.'
        self._require_fitted()
        if errors not in ("raise", "return"):
            raise ValueError("errors must be 'raise' or 'return'")
        rows = list(token_rows)
        current = (
            None
            if current_state is None
            else np.asarray(
                current_state.detach().cpu()
                if isinstance(current_state, torch.Tensor)
                else current_state
            )
        )
        if current is not None and current.shape != (len(rows), self.action_dim):
            raise ValueError(f"current_state must have shape {(len(rows), self.action_dim)}")
        previous = self._numpy_boundary(previous_first_order)
        if previous is not None and any(
            np.ndim(v) != 2 or len(v) != len(rows) for v in previous.values()
        ):
            raise ValueError(
                "previous_first_order arrays must have batch dimension equal to token rows"
            )
        result = []
        for index, row in enumerate(rows):
            try:
                decoded = self.decode(
                    [row],
                    fps=fps,
                    current_state=None if current is None else current[index : index + 1],
                    previous_first_order=None
                    if previous is None
                    else {k: v[index : index + 1] for k, v in previous.items()},
                )
            except ValueError as error:
                error = ValueError(f"token row {index}: {error}")
                if errors == "raise":
                    raise error from None
                decoded = error
            result.append(decoded)
        return result

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError(
            "ActionCodec2 accepts action tensors; use codec.encode(actions, fps=...) or codec(actions, fps=...)"
        )

    def post_process_image_text_to_text(self, *args, **kwargs):
        raise NotImplementedError(
            "ActionCodec2 returns actions; use codec.decode(token_rows, fps=...)"
        )

    @property
    def fitted_profiles(self) -> frozenset[str]:
        """Names of profiles fitted on data; persisted by save_pretrained()."""
        return self._fitted_profiles

    @property
    def is_fitted(self) -> bool:
        """Whether every profile used by the selected action space is fitted."""
        return all(c.space in self.fitted_profiles for c in self.action_space.components)

    def _require_fitted(self):
        missing = {c.space for c in self.action_space.components} - self.fitted_profiles
        if missing:
            raise RuntimeError(
                f"Unfitted profiles {sorted(missing)}; call codec.fit(episodes) first"
            )

    def _describe_action_spaces(self, action_dim: int | None = None) -> str:
        return _fitted_status(self.fitted_profiles) + super()._describe_action_spaces(action_dim)

    def for_action_space(self, name: str) -> ActionCodec2:
        """Share current weights with a new binding; subsequent fit replaces only that instance."""
        codec = super().for_action_space(name)
        codec._fitted_profiles = self.fitted_profiles
        codec.training_report = self.training_report
        codec.preprocessing = dict(self.preprocessing)
        codec.release_metadata = self.release_metadata
        return codec




    def _encode_action_indices(self, action, encode_kwargs=None):
        self._require_fitted()
        return super()._encode_action_indices(action, encode_kwargs)

    def _decode_action_indices(self, action_indices, **kwargs):
        self._require_fitted()
        return super()._decode_action_indices(action_indices, **kwargs)

    def decode_prefix(self, action_token_ids, **kwargs):
        self._require_fitted()
        return super().decode_prefix(action_token_ids, **kwargs)

    def grammar(self, horizon: int, *, fps: float | None = None):
        self._require_fitted()
        return super().grammar(horizon, fps=fps)

    def final_first_order(self, action_token_ids, *, previous_first_order, executed_steps=None):
        self._require_fitted()
        return super().final_first_order(
            action_token_ids,
            previous_first_order=previous_first_order,
            executed_steps=executed_steps,
        )

    @classmethod
    def describe_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        action_dim: int | None = None,
        gripper_mode: str = "2bin",
        primitive_order: int = 1,
        **kwargs: Any,
    ) -> str:
        'Inspect a local/Hub artifact before choosing an action space.'
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, allow_patterns=ROUTER_MANIFEST_PATTERNS, **kwargs
        )
        if (directory / MANIFEST_NAME).is_file():
            return describe_release(
                directory,
                action_dim=action_dim,
                gripper_mode=gripper_mode,
                primitive_order=primitive_order,
            )
        specs, _ = ActionCodec2Router.read_manifest(directory)
        metadata = _saved_metadata(directory)
        fitted = metadata.get("fitted_profiles", [spec.name for spec in specs])
        return _fitted_status(fitted) + ActionCodec2Router.describe_pretrained(
            directory, selected=metadata.get("action_space"), action_dim=action_dim
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        action_space: str | Path | PresetConfig | Mapping[str, object] | None = None,
        gripper_mode: str | None = None,
        primitive_order: int | None = None,
        verify_gates: bool = False,
        **kwargs: Any,
    ) -> ActionCodec2:
        'Load a local/Hub artifact and restore or override its action space.'
        if cls is ActionCodec2 and ("variant" in kwargs or (Path(pretrained_model_name_or_path) / "variants.json").is_file()
                                   or (Path(pretrained_model_name_or_path) / "artifact_manifest.json").is_file()):
            from ...catalog import ActionCodec2 as Catalog
            return Catalog.from_pretrained(pretrained_model_name_or_path, action_space=action_space, **kwargs)
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, allow_patterns=ROUTER_ALLOW_PATTERNS, **kwargs
        )
        if action_space is None:
            action_space = _saved_metadata(directory).get("action_space")
        if action_space is None:
            raise ValueError(
                "action_space is required when loading an unbound router artifact. "
                "First inspect its layouts with print(ActionCodec2.describe_pretrained(path)), "
                "then pass action_space='NAME'; shape alone cannot identify physical meaning."
            )
        if (directory / MANIFEST_NAME).is_file():
            router, preprocessing, metadata = load_release_router(
                directory,
                gripper_mode="2bin" if gripper_mode is None else gripper_mode,
                primitive_order=1 if primitive_order is None else primitive_order,
                verify_gates=verify_gates,
            )
            codec = cls._from_router(router, action_space=action_space)
            codec.preprocessing, codec.release_metadata = preprocessing, metadata
            return codec
        if gripper_mode is not None or primitive_order is not None or verify_gates:
            raise ValueError(
                "gripper_mode, primitive_order and verify_gates select release variants only"
            )
        codec = super().from_pretrained(directory, action_space=action_space)
        metadata = _saved_metadata(directory)
        fitted = metadata.get("fitted_profiles", list(codec.action_tokenizer.profiles))
        if not isinstance(fitted, list) or any(
            not isinstance(name, str) or name not in codec.action_tokenizer.profiles
            for name in fitted
        ):
            raise ValueError("invalid fitted_profiles in processor_config.json")
        codec._fitted_profiles = frozenset(fitted)
        policies = metadata.get("preprocessing", {})
        if (
            not isinstance(policies, dict)
            or policies.keys() - codec.action_tokenizer.profiles.keys()
        ):
            raise ValueError("invalid preprocessing profiles in processor_config.json")
        codec.preprocessing = {
            name: PreprocessingSpec.from_dict(spec) for name, spec in policies.items()
        }
        codec.release_metadata = metadata.get("release")
        report_path = directory / "fit_report.json"
        if report_path.is_file():
            codec.training_report = json.loads(report_path.read_text(encoding="utf-8"))
        return codec

    def to_dict(self, legacy_serialization: bool = True) -> dict[str, object]:
        'Return lightweight discovery metadata.'
        del legacy_serialization
        payload = {
            "processor_class": type(self).__name__,
            "model_type": self.model_type,
            "action_space": self.action_space.name,
            "fitted_profiles": sorted(self.fitted_profiles),
        }
        if self.preprocessing:
            payload["preprocessing"] = {
                name: spec.to_dict() for name, spec in self.preprocessing.items()
            }
        return payload

    def save_pretrained(
        self, save_directory: str | Path, push_to_hub: bool = False, *, include_training_report=False, **kwargs: Any
    ) -> list[str]:
        """Save the complete router and the selected action-space binding."""
        if not push_to_hub and kwargs:
            raise TypeError(f"unused save_pretrained keyword(s): {', '.join(sorted(kwargs))}")
        directory = Path(save_directory)
        files = self.action_tokenizer.save_pretrained(directory)
        # After the router, so the bound description replaces its generic card.
        save_model_card(directory, str(self))
        save_huggingface_metadata(
            directory,
            **self.to_dict(),
        )
        report_path = directory / "fit_report.json"
        if include_training_report and self.training_report is not None:
            report_path.write_text(
                json.dumps(self.training_report, indent=2) + "\n", encoding="utf-8"
            )
            files.append(str(report_path))
        elif report_path.exists():
            report_path.unlink()
        files.extend(save_runtime(directory))
        if push_to_hub:
            upload_pretrained_directory(directory, **kwargs)
        return files

    def __call__(
        self,
        action: torch.Tensor | np.ndarray,
        *,
        fps: float | None = None,
        current_state: torch.Tensor | np.ndarray | None = None,
        previous_first_order: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        encode_kwargs: dict | None = None,
        return_tensors: str | None = None,
    ) -> BatchFeature:
        'Prepare ``input_ids`` from ``(B,T,D)`` or a single ``(T,D)`` action.'
        rows = self.encode(
            action,
            fps=fps,
            current_state=current_state,
            previous_first_order=previous_first_order,
            encode_kwargs=encode_kwargs,
        )
        return BatchFeature({"input_ids": rows}, tensor_type=return_tensors)
