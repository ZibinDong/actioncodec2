"""Runtime opt-in controls for lossless fixed-vocabulary Set-BPE search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingSearch:
    'Independent post-encoding stages; defaults retain ranked merge replay.'

    spatial_search: bool = False
    temporal_search: bool = False
    search_window: int = 32
    search_beam: int = 4

    def __post_init__(self):
        for name in ("spatial_search", "temporal_search"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("search_window", "search_beam"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.search_beam > 64:
            raise ValueError("search_beam must be at most 64")

    @property
    def enabled(self):
        return self.spatial_search or self.temporal_search

    def validate_codec(self, scheme, bins):
        if not self.enabled:
            return
        if scheme != "set_bpe":
            raise ValueError("packing search requires set_bpe")
        if len(bins) > 8 or max(bins) > 255:
            raise ValueError("packing search supports at most eight slots and 255 bins per slot")
        if self.search_window * len(bins) > 256:
            raise ValueError("search_window times profile width must be at most 256")
