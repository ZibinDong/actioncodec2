"""LeRobot reading, action-chunk sampling, and the pre-encoded token cache.

Names resolve on first use, so importing one submodule does not load the rest.
"""

from importlib import import_module

_EXPORTS = {
    "ActionChunkDataset": "chunks",
    "BalancedReplacementSampler": "sampler",
    "ChunkTokenError": "chunk_tokens",
    "ChunkTokenStore": "chunk_tokens",
    "LeRobotEpisodes": "lerobot",
    "TokenTable": "chunks",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)
