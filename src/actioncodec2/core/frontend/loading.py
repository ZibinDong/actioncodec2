"""Load ActionCodec2Sparse release packages into a router."""

from __future__ import annotations

import json
from pathlib import Path

from ..backends.actioncodec2.defaults import builtin_action_spaces
from ..backends.actioncodec2.preprocessing import PreprocessingSpec
from ..backends.actioncodec2.presets import format_action_spaces
from ..backends.actioncodec2.release import (
    check_gripper_consistency,
    check_release_gates,
    selected_variants,
)
from ..backends.actioncodec2.sparse import ActionCodec2SparseProcessor, ActionCodec2SparseRouter

TOKENIZER = "actioncodec2_sparse"


def release_layouts(profiles):
    """Built-in layouts whose components are all available in ``profiles``."""
    return {
        name: preset
        for name, preset in builtin_action_spaces().items()
        if {component.space for component in preset.components} <= set(profiles)
    }


def load_release_router(directory, *, gripper_mode, primitive_order=1, verify_gates, tokenizer=TOKENIZER):
    """Load one gripper-mode/order vocabulary per profile with its cleaning policy."""
    manifest, mode, variants = selected_variants(
        directory, gripper_mode=gripper_mode, order=primitive_order, tokenizer=tokenizer
    )
    from ..backends.actioncodec2 import ActionCodec2Processor, ActionCodec2Router

    processor_class, router_class = ((ActionCodec2SparseProcessor, ActionCodec2SparseRouter)
                                     if tokenizer == TOKENIZER else (ActionCodec2Processor, ActionCodec2Router))
    profiles, preprocessing = {}, {}
    for name, variant in variants.items():
        entry = (Path(directory) / variant["path"]).resolve()
        if not entry.is_relative_to(Path(directory).resolve()):
            raise ValueError("release profile path escapes its artifact directory")
        processor = processor_class.from_pretrained(entry)
        spec = PreprocessingSpec.from_dict(
            json.loads((entry / "preprocessing.json").read_text(encoding="utf-8"))
        )
        check_gripper_consistency(entry, processor, spec, mode)
        if verify_gates:
            if variant.get("gates_passed") is False:
                raise ValueError(f"{entry}: recorded release gates failed")
            check_release_gates(entry)
        profiles[name], preprocessing[name] = processor, spec
    router = router_class.from_processors(
        profiles,
        token_budgets={name: manifest["codebook_size"] for name in profiles},
        presets=list(release_layouts(profiles).values()),
    )
    metadata = {
        "package": manifest.get("package", Path(directory).name),
        "tokenizer": tokenizer,
        "precision": manifest["precision"],
        "codebook_size": manifest["codebook_size"],
        "gripper_mode": mode,
        "primitive_order": primitive_order,
        "variants": list(variants.values()),
    }
    return router, preprocessing, metadata


def describe_release(directory, *, action_dim, gripper_mode, primitive_order=1, tokenizer=TOKENIZER):
    """Report a release's layouts from its manifest alone."""
    _, _, variants = selected_variants(
        directory, gripper_mode=gripper_mode, order=primitive_order, tokenizer=tokenizer
    )
    presets = release_layouts(variants)
    return format_action_spaces(
        presets, action_dim=action_dim, codec_name="ActionCodec2Sparse" if tokenizer == TOKENIZER else "ActionCodec2", chunk_horizon_decode=tokenizer == TOKENIZER
    )
