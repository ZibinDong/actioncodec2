"""Optimizer parameter grouping."""

from __future__ import annotations

import torch

__all__ = ["parameter_groups", "no_decay_parameter_ids"]


def no_decay_parameter_ids(model) -> set[int]:
    """Ids of parameters that must never be weight-decayed."""
    excluded: set[int] = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            excluded.add(id(module.weight))
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, accessor, None)
        layer = getter() if callable(getter) else None
        weight = getattr(layer, "weight", None)
        if weight is not None:
            excluded.add(id(weight))
    for parameter in model.parameters():
        if parameter.ndim < 2:
            excluded.add(id(parameter))
    return excluded


def parameter_groups(model, *, weight_decay: float) -> list[dict[str, object]]:
    """Split trainable parameters into decayed and non-decayed groups."""
    excluded = no_decay_parameter_ids(model)
    decayed: list[torch.nn.Parameter] = []
    plain: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (plain if id(parameter) in excluded else decayed).append(parameter)
    return [
        {"params": decayed, "weight_decay": float(weight_decay)},
        {"params": plain, "weight_decay": 0.0},
    ]
