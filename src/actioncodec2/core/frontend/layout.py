"""Action-space bindings and boundary conventions shared by both codecs."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from pathlib import Path
from ..backends.actioncodec2.presets import PresetConfig
from .conversion import _numpy

class ActionSpaceMixin:
    def __str__(self) -> str:
        """Show supported tensor layouts, physical units, and invocation requirements."""
        return self._describe_action_spaces()

    @property
    def action_space(self) -> PresetConfig:
        """The immutable tensor semantics selected for this tokenizer."""
        return self._action_space

    @property
    def action_spaces(self) -> Mapping[str, PresetConfig]:
        """Every built-in or user-registered option in the loaded artifact."""
        return self.action_tokenizer.action_spaces

    def register_action_space(
        self,
        preset: PresetConfig | Mapping[str, object] | str | Path,
    ) -> PresetConfig:
        """Register a custom action space without changing this instance's binding."""
        return self.action_tokenizer.register_action_space(preset)

    def first_order_template(self, batch_size: int) -> dict[str, np.ndarray]:
        'Explicit zero boundary templates, shaped {component: (B,D_active)}.'
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return {
            name: np.zeros((batch_size, width), dtype=np.float32)
            for name, width in self.action_tokenizer.second_order_boundaries(
                self.action_space
            ).items()
        }

    @staticmethod
    def _numpy_boundary(previous_first_order):
        if previous_first_order is None:
            return None
        if not isinstance(previous_first_order, Mapping):
            raise ValueError("previous_first_order must map component names to [B,D_active] arrays")
        return {name: _numpy(value) for name, value in previous_first_order.items()}
