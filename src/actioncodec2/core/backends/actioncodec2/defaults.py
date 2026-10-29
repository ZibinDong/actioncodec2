"""Package-owned configuration templates for fresh ActionCodec2 instances."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from .bpe import BPESerializer, SetBPESerializer
from .configuration import ActionCodec2Config
from .presets import PresetConfig
from .processor import ActionCodec2Processor
from .routing import ActionCodec2Router
from .training import build_primitives


class CodecBuilder:
    'Build unfitted physical profiles from packaged defaults or explicit configs.'

    def __init__(self, *, primitive_order=1, bpe_scheme=None, profile_configs=None):
        if type(primitive_order) is not int or primitive_order not in (1, 2):
            raise ValueError("primitive_order must be 1 or 2")
        if bpe_scheme not in (None, "set_bpe", "unified_set_bpe", "bpe"):
            raise ValueError("bpe_scheme must be 'set_bpe', 'unified_set_bpe' or 'bpe'")
        if profile_configs is not None and not isinstance(profile_configs, Mapping):
            raise TypeError("profile_configs must map 'joint'/'eef' to physical configurations")
        self.profile_configs = dict(profile_configs or {})
        if set(self.profile_configs) - {"joint", "eef"}:
            raise ValueError("profile_configs keys must be 'joint' or 'eef'")
        self.primitive_order = primitive_order
        self.bpe_scheme = bpe_scheme
        self.assets = Path(__file__).with_name("assets")

    def _config(self, name: str) -> ActionCodec2Config:
        suffix = "" if self.primitive_order == 1 else "_second_order"
        raw = self.profile_configs.get(name, self.assets / "profiles" / f"{name}{suffix}.yaml")
        if isinstance(raw, ActionCodec2Config):
            config = raw
        elif isinstance(raw, Mapping):
            config = ActionCodec2Config.from_dict(raw)
        else:
            config = ActionCodec2Config.from_yaml(raw)
        if config.train is None:
            raise ValueError(f"{name} profile config requires a train section")
        if self.bpe_scheme is not None:
            payload = config.to_dict()
            payload["set_bpe"]["scheme"] = self.bpe_scheme
            if self.bpe_scheme == "bpe":
                payload["train"]["bpe"]["spatial_vocab_size"] = payload["train"]["bpe"][
                    "vocab_size"
                ]
            config = ActionCodec2Config.from_dict(payload)
        return config

    def build(self) -> ActionCodec2Router:
        """Create primitive-only profiles; the public facade tracks fitted state."""
        processors = {}
        budgets = {}
        for name in ("joint", "eef"):
            config = self._config(name)
            primitives = build_primitives(config)
            if config.set_bpe.scheme == "bpe":
                serializer = BPESerializer(
                    bins=primitives.bins, lookahead=config.set_bpe.lookahead, merges=()
                )
            elif config.set_bpe.scheme == "unified_set_bpe":
                from .unified_set_bpe import UnifiedSetBPESerializer

                serializer = UnifiedSetBPESerializer(
                    bins=primitives.bins, lookahead=config.set_bpe.lookahead, merges=()
                )
            else:
                serializer = SetBPESerializer(
                    bins=primitives.bins,
                    lookahead=config.set_bpe.lookahead,
                    spatial_merges=(),
                    temporal_merges=(),
                )
            processors[name] = ActionCodec2Processor(config, primitives, serializer)
            budgets[name] = config.train.bpe.vocab_size
        presets = list(builtin_action_spaces().values())
        return ActionCodec2Router.from_processors(
            processors, token_budgets=budgets, presets=presets
        )


@lru_cache(maxsize=1)
def builtin_action_spaces():
    'Return a read-only name-to-preset mapping shipped in the installed package.'
    presets = [
        PresetConfig.from_yaml(path)
        for path in sorted(Path(__file__).with_name("assets").joinpath("presets").glob("*.yaml"))
    ]
    return MappingProxyType({preset.name: preset for preset in presets})


class ProfileBundleBuilder:
    'Assemble two existing physical artifacts using their recorded vocabulary budgets.'

    def __init__(self, joint, eef, *, token_budgets=None):
        self.paths = {"joint": joint, "eef": eef}
        self.budgets = dict(token_budgets or {})
        if self.budgets.keys() - self.paths.keys():
            raise ValueError("token_budgets accepts only 'joint' and 'eef'")

    def build(self):
        processors = {
            name: ActionCodec2Processor.from_pretrained(path) for name, path in self.paths.items()
        }
        budgets = {
            name: self.budgets.get(
                name,
                processor.config.train.bpe.vocab_size
                if processor.config.train is not None
                else processor.vocab_size,
            )
            for name, processor in processors.items()
        }
        return ActionCodec2Router.from_processors(
            processors, token_budgets=budgets, presets=list(builtin_action_spaces().values())
        )
