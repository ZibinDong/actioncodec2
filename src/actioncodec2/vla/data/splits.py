"""Deterministic episode-level train/validation split."""

from __future__ import annotations


def belongs_to_split(episode: int, root_index: int, validation_fraction: float, split: str) -> bool:
    """Whether ``episode`` of root ``root_index`` belongs to ``split``.

    The assignment is a fixed hash of the two indices, so it does not depend on
    the order or number of episodes being read.
    """
    if split not in {"train", "validation", "all"}:
        raise ValueError("split must be train, validation, or all")
    if not 0 <= float(validation_fraction) < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if split == "all":
        return True
    bucket = (int(episode) * 2654435761 + int(root_index) * 805459861) % 1_000_003
    validation = bucket < round(float(validation_fraction) * 1_000_003)
    return validation == (split == "validation")
