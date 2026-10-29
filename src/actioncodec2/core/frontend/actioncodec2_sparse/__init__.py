"""ActionCodec2Sparse frontend: action-space binding, release loading and generation constraints."""

from transformers import AutoConfig, AutoProcessor, PretrainedConfig

from .generation import ActionCodec2SparseLogitsProcessor
from .processor import ActionCodec2Sparse


class ActionCodec2SparseHFConfig(PretrainedConfig):
    """AutoConfig discovery key for :class:`ActionCodec2Sparse` artifacts."""

    model_type = ActionCodec2Sparse.model_type

    def __init__(self, **kwargs):
        kwargs.setdefault("processor_class", "ActionCodec2Sparse")
        super().__init__(**kwargs)


AutoConfig.register(ActionCodec2Sparse.model_type, ActionCodec2SparseHFConfig, exist_ok=True)
AutoProcessor.register(ActionCodec2SparseHFConfig, ActionCodec2Sparse, exist_ok=True)

__all__ = ["ActionCodec2Sparse", "ActionCodec2SparseLogitsProcessor"]
