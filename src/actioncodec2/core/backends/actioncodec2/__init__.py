"""Shared physical models, layouts, and dense and sparse routers."""
from .configuration import ActionCodec2Config, ConfigError
from .presets import ComponentSpec, PresetConfig
from .processor import ActionCodec2Processor
from .routing import ActionCodec2Router
from .trajectory import ActiveTrajectory

__all__ = ["ActionCodec2Config", "ConfigError", "ComponentSpec", "PresetConfig", "ActionCodec2Processor", "ActionCodec2Router", "ActiveTrajectory"]
