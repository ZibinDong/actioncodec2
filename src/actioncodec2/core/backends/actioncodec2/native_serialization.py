"""Cached C++ serializers shared by scalar and batched inference paths."""

from __future__ import annotations

import numpy as np

from .native_backend import native_kernel


class NativeCacheMixin:
    """Serialize numerical state while rebuilding process-local C++ caches."""

    def __getstate__(self):
        state = vars(self).copy()
        for key in ("_compiled_serializer", "_compiled_spatial", "_coverage_metadata"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        vars(self).update(state)


def compiled_serializer(owner):
    """Return a codec for a serializer/vocabulary, or the Python override."""
    spatial = getattr(owner, "spatial", None)
    temporal = getattr(owner, "temporal_merges", getattr(owner, "merges", ()))
    scheme = getattr(owner, "scheme", "set_bpe" if spatial is not None else "bpe")
    constructor = native_kernel("SetBPE" if scheme == "set_bpe" else "PlainBPE")
    if constructor is None:
        return None
    cached = getattr(owner, "_compiled_serializer", None)
    if cached is None:
        bins = getattr(owner, "bins", spatial.bins if spatial is not None else ())
        if scheme == "set_bpe":
            merges = getattr(owner, "spatial_merges", spatial.merges)
            cached = constructor(bins, owner.lookahead, merges, temporal)
        else:
            cached = constructor(bins, owner.merges)
        owner._compiled_serializer = cached
    return cached


def encode_native(owner, dense: np.ndarray):
    codec = compiled_serializer(owner)
    options = getattr(owner, "encoding_search", None)
    if options is not None and options.enabled:
        if codec is None or not hasattr(codec, "encode_search_batch"):
            raise RuntimeError(
                "packing search requires the rebuilt actioncodec2 native extension; "
                "run 'python setup.py build_ext --inplace'"
            )
        return codec.encode_search_batch(
            np.asarray(dense, dtype=np.int64),
            options.spatial_search,
            options.temporal_search,
            options.search_window,
            options.search_beam,
        )
    return None if codec is None else codec.encode_batch(np.asarray(dense, dtype=np.int64))


def decode_native(owner, rows, dims, *, validate=False):
    codec = compiled_serializer(owner)
    if codec is None:
        return None
    bins = getattr(owner, "bins", owner.spatial.bins if hasattr(owner, "spatial") else ())
    if validate:
        if not hasattr(codec, "decode_checked_batch"):
            return None
        support = dims if hasattr(codec, "decode_active_batch") else sum(1 << d for d in dims)
        return codec.decode_checked_batch(rows, len(bins), support)
    if hasattr(codec, "decode_active_batch"):
        return codec.decode_active_batch(rows, len(bins), dims)
    return codec.decode_batch(rows, len(bins), sum(1 << d for d in dims))
