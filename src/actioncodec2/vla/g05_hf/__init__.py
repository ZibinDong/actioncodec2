# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Hugging Face G05 model components; see the repository's LICENSE-G0.5."""


def register_huggingface_auto_classes() -> None:
    """Register G0.5 with Transformers' standard ``Auto*`` loaders.

    Registration is explicit and lazy so importing the package does not pull
    in the Qwen3.5 implementation or optional CUDA kernels.
    """
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from .configuration_g05 import G05Config
    from .modeling_g05 import G05Model
    from .processing_g05 import G05Processor

    try:
        AutoConfig.register(G05Config.model_type, G05Config)
    except ValueError as error:
        if "already used" not in str(error):
            raise
    try:
        AutoModel.register(G05Config, G05Model)
    except ValueError as error:
        if "already used" not in str(error):
            raise
    AutoProcessor.register(G05Config, G05Processor, exist_ok=True)


__all__ = ["register_huggingface_auto_classes"]
