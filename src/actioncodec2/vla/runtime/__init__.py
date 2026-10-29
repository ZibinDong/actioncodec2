"""Build, save, load, and serve G0.5 action-tokenizer policies.

Names resolve on first use so that ``actioncodec2.vla.runtime.server`` (the wire
format and the batching thread) can be imported without PyTorch.
"""

from importlib import import_module

_EXPORTS = {
    name: "actioncodec2.vla.runtime.run"
    for name in (
        "DTYPES",
        "build_dataset",
        "build_model",
        "build_processor",
        "build_tokenizer_for",
        "load_policy",
        "load_run",
        "resolve_checkpoint",
        "save_run",
    )
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
