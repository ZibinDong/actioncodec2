"""Dispatch AC2 numerical work to the optional C++ runtime."""

from __future__ import annotations

import os

__all__ = ["native", "have_native", "require_native", "resolve_backend", "native_kernel"]

try:  # pragma: no cover - exercised by whichever branch the build state picks
    from . import _native as native
except ImportError:  # pragma: no cover
    try:
        from actioncodec2.core.backends.actioncodec2 import _native as native
    except ImportError:
        try:
            from actioncodec2_kernels import _native as native
        except ImportError:
            native = None


def have_native() -> bool:
    """True when the compiled kernels are importable and not disabled."""
    return native is not None and not os.environ.get("ACTIONCODEC2_NO_NATIVE")


def native_kernel(name: str):
    """Resolve an optional kernel, honoring the Python reference override."""
    return getattr(native, name, None) if have_native() else None


def require_native():
    """Return the native module or explain how to get one."""
    if not have_native():
        raise RuntimeError(
            "the actioncodec2 native extension is not available; run "
            "'python setup.py build_ext --inplace' from the repository root or "
            "select backend='python'"
        )
    return native


def resolve_backend(backend: str) -> str:
    """Turn ``'auto'`` into a concrete backend name."""
    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native' or 'python'")
    if backend == "auto":
        return "native" if have_native() else "python"
    if backend == "native":
        require_native()
    return backend
