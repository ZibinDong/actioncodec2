"""Shared release loader bindings."""
from functools import partial
from ..loading import load_release_router as _load, describe_release as _describe

load_release_router = partial(_load, tokenizer='actioncodec2_sparse')
describe_release = partial(_describe, tokenizer='actioncodec2_sparse')
