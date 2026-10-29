"""Configure the G0.5 adapter native thread budget."""
import functools
import os

@functools.cache
def _configure(native) -> None:
    """Honor an explicit thread budget; otherwise use one thread after a fork."""
    configured = os.environ.get("ACTIONCODEC2_NUM_THREADS")
    if configured is not None:
        if not configured.isdigit() or int(configured) < 1:
            raise ValueError("ACTIONCODEC2_NUM_THREADS must be a positive integer")
        native.set_thread_budget(int(configured))
    else:
        native.set_thread_budget(0)
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=lambda: native.set_thread_budget(1))
