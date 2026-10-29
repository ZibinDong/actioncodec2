"""Arbitrary-horizon frontend bound to one ActionCodec2 action space."""

from transformers import AutoConfig, AutoProcessor

from ...backends.actioncodec2.hf import ActionCodec2FrontendHFConfig
from .generation import ActionCodec2LogitsProcessor
from .processor import ActionCodec2
from .tokenizer import ActionCodec2Tokenizer

AutoConfig.register(ActionCodec2.model_type, ActionCodec2FrontendHFConfig, exist_ok=True)
AutoProcessor.register(ActionCodec2FrontendHFConfig, ActionCodec2, exist_ok=True)

__all__ = ["ActionCodec2", "ActionCodec2Tokenizer", "ActionCodec2LogitsProcessor"]
