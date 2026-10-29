"""Minimal open-source G0.5 VLA and external action-tokenizer adapters."""

from importlib.metadata import PackageNotFoundError, version

from .tokenizers.factory import build_tokenizer

try:
    __version__ = version("actioncodec2")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

def __getattr__(name):
    if name == "PreprocessingSpec":
        from .core.backends.actioncodec2.preprocessing import PreprocessingSpec
        return PreprocessingSpec
    if name in {"ActionCodec2", "ActionCodec2LogitsProcessor"}:
        from .core.frontend import actioncodec2
        return getattr(actioncodec2, name)
    if name in {"ActionCodec2Sparse", "ActionCodec2SparseLogitsProcessor"}:
        from . import sparse

        return getattr(sparse, name)
    raise AttributeError(name)


__all__ = ["__version__", "build_tokenizer", "ActionCodec2", "ActionCodec2LogitsProcessor", "ActionCodec2Sparse", "ActionCodec2SparseLogitsProcessor", "PreprocessingSpec"]
