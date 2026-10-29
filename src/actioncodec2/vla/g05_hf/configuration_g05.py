# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Configuration for the G0.5 action-token policy."""

from __future__ import annotations

from transformers import PretrainedConfig

from actioncodec2.vla.g05_hf.constants import (  # noqa: F401  (re-exported)
    G05_ACTION_BEGIN,
    G05_ACTION_ROWS,
    G05_EOV_ID,
    G05_INTERNAL_PROPRIO_DIM,
    G05_MODEL_TYPE,
    G05_STATE_ID,
    G05_VOCAB_SIZE,
    LEGACY_MODEL_TYPES,
)


def _default_layer_types(num_layers: int) -> list[str]:
    """Match the hybrid Qwen3.5/G0.5 decoder layout."""
    return [
        "linear_attention" if index % 4 != 3 else "full_attention"
        for index in range(num_layers)
    ]


class G05Config(PretrainedConfig):
    """Serializable architecture and inference contract for :class:`G05Model`."""

    model_type = G05_MODEL_TYPE

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> G05Config | tuple[G05Config, dict]:
        # Older Transformers overwrites a saved attention backend with None.
        kwargs["attn_implementation"] = (
            kwargs.get("attn_implementation") or config_dict.get("attn_implementation", "eager")
        )
        return super().from_dict(dict(config_dict), **kwargs)

    def __init__(
        self,
        *,
        action_vocab_size: int = 2048,
        action_token_begin_idx: int = G05_ACTION_BEGIN,
        text_vocab_size: int = G05_VOCAB_SIZE,
        pad_token_id: int = 0,
        eos_token_id: int = 248_044,
        image_token_id: int = 248_056,
        vision_start_token_id: int = 248_053,
        vision_end_token_id: int = 248_054,
        eov_token_id: int = G05_EOV_ID,
        state_token_id: int = G05_STATE_ID,
        embodiment: str = "libero",
        image_size: tuple[int, int] | list[int] = (256, 256),
        max_task_tokens: int = 256,
        max_prompt_length: int = 4096,
        pad_prompt_to_length: int | None = None,
        max_action_tokens: int = 64,
        proprio_dim: int = 0,
        internal_proprio_dim: int | None = None,
        proprio_indices: list[int] | None = None,
        continuous_action_dim: int = 0,
        action_history_steps: int = 0,
        text_hidden_size: int = 2048,
        text_intermediate_size: int = 6144,
        text_num_layers: int = 24,
        text_num_heads: int = 8,
        text_num_kv_heads: int = 2,
        text_head_dim: int = 256,
        text_layer_types: list[str] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        rope_theta: float = 10_000_000.0,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
        vision_depth: int = 24,
        vision_hidden_size: int = 1024,
        vision_intermediate_size: int = 4096,
        vision_num_heads: int = 16,
        vision_patch_size: int = 16,
        vision_temporal_patch_size: int = 2,
        vision_spatial_merge_size: int = 2,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        attn_implementation: str = "eager",
        vision_attn_implementation: str = "sdpa",
        use_flash_linear_attention: bool = False,
        use_fused_linear_cross_entropy: bool = False,
        action_only_cross_entropy: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        image_size = tuple(int(value) for value in image_size)
        mrope_section = tuple(int(value) for value in mrope_section)
        layer_types = (
            _default_layer_types(text_num_layers)
            if text_layer_types is None
            else list(text_layer_types)
        )

        if action_vocab_size <= 0 or text_vocab_size <= 0:
            raise ValueError("text and action vocabulary sizes must be positive")
        if action_token_begin_idx < 0:
            raise ValueError("action_token_begin_idx cannot be negative")
        if action_vocab_size > G05_ACTION_ROWS:
            raise ValueError(f"G0.5 provides exactly {G05_ACTION_ROWS} native action-token rows")
        if action_token_begin_idx + action_vocab_size > text_vocab_size:
            raise ValueError("the action-token range lies outside the G0.5 vocabulary")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain two positive values")
        if min(max_task_tokens, max_prompt_length, max_action_tokens) <= 0:
            raise ValueError("token limits must be positive")
        if pad_prompt_to_length is not None and not 0 < pad_prompt_to_length <= max_prompt_length:
            raise ValueError("pad_prompt_to_length must be positive and at most max_prompt_length")
        internal_proprio_dim = (
            proprio_dim if internal_proprio_dim is None else internal_proprio_dim
        )
        proprio_indices = (
            list(range(proprio_dim)) if proprio_indices is None else list(proprio_indices)
        )
        if min(proprio_dim, internal_proprio_dim, continuous_action_dim, action_history_steps) < 0:
            raise ValueError("proprio and action-history dimensions cannot be negative")
        if (proprio_dim == 0) != (internal_proprio_dim == 0):
            raise ValueError(
                "proprio_dim and internal_proprio_dim must both be enabled or disabled"
            )
        if proprio_dim:
            if len(proprio_indices) != proprio_dim or len(set(proprio_indices)) != proprio_dim:
                raise ValueError("proprio_indices must uniquely map every physical dimension")
            if min(proprio_indices) < 0 or max(proprio_indices) >= internal_proprio_dim:
                raise ValueError("a proprio index lies outside internal_proprio_dim")
        elif proprio_indices:
            raise ValueError("proprio_indices require proprio_dim > 0")
        if (continuous_action_dim == 0) != (action_history_steps == 0):
            raise ValueError(
                "continuous_action_dim and action_history_steps must both be enabled or disabled"
            )
        if text_hidden_size != text_num_heads * text_head_dim:
            raise ValueError("text_hidden_size must equal text_num_heads * text_head_dim")
        if text_num_kv_heads <= 0 or text_num_heads % text_num_kv_heads:
            raise ValueError("text_num_kv_heads must divide text_num_heads")
        if len(layer_types) != text_num_layers:
            raise ValueError("text_layer_types must contain one entry per text layer")
        if any(kind not in {"linear_attention", "full_attention"} for kind in layer_types):
            raise ValueError("unsupported Qwen3.5 layer type")
        # Transformers 5.14 still assumes that Qwen3.5 has at least one recurrent
        # layer when it constructs its hybrid cache and attention masks.
        if "linear_attention" not in layer_types:
            raise ValueError("Qwen3.5 currently requires at least one linear_attention layer")
        if len(mrope_section) != 3 or min(mrope_section) <= 0:
            raise ValueError("mrope_section must contain three positive values")
        if vision_hidden_size % vision_num_heads:
            raise ValueError("vision_hidden_size must be divisible by vision_num_heads")
        image_stride = vision_patch_size * vision_spatial_merge_size
        if any(size % image_stride for size in image_size):
            raise ValueError(
                "image dimensions must be divisible by patch_size * spatial_merge_size"
            )
        if vision_temporal_patch_size <= 0:
            raise ValueError("vision_temporal_patch_size must be positive")
        valid_ids = (
            pad_token_id,
            eos_token_id,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
            eov_token_id,
            state_token_id,
        )
        if any(token_id < 0 or token_id >= text_vocab_size for token_id in valid_ids):
            raise ValueError("all text special-token IDs must be inside the text vocabulary")
        supported_attention = {"eager", "sdpa", "flash_attention_2", "flash_attention_4"}
        if attn_implementation not in supported_attention:
            raise ValueError("unsupported text attention implementation")
        if vision_attn_implementation not in supported_attention:
            raise ValueError("unsupported vision attention implementation")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid top-k/top-p sampling configuration")

        self.action_vocab_size = action_vocab_size
        self.action_token_begin_idx = action_token_begin_idx
        self.text_vocab_size = text_vocab_size
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.eov_token_id = eov_token_id
        self.state_token_id = state_token_id
        self.embodiment = str(embodiment).strip()
        if not self.embodiment:
            raise ValueError("embodiment must be a non-empty G0.5 template value")
        self.image_size = image_size
        self.max_task_tokens = max_task_tokens
        self.max_prompt_length = max_prompt_length
        self.pad_prompt_to_length = pad_prompt_to_length
        self.max_action_tokens = max_action_tokens
        self.proprio_dim = proprio_dim
        self.internal_proprio_dim = internal_proprio_dim
        self.proprio_indices = proprio_indices
        self.continuous_action_dim = continuous_action_dim
        self.action_history_steps = action_history_steps

        self.text_hidden_size = text_hidden_size
        self.text_intermediate_size = text_intermediate_size
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_num_kv_heads = text_num_kv_heads
        self.text_head_dim = text_head_dim
        self.text_layer_types = layer_types
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section

        self.vision_depth = vision_depth
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_heads = vision_num_heads
        self.vision_patch_size = vision_patch_size
        self.vision_temporal_patch_size = vision_temporal_patch_size
        self.vision_spatial_merge_size = vision_spatial_merge_size

        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.attn_implementation = attn_implementation
        self.vision_attn_implementation = vision_attn_implementation
        self.use_flash_linear_attention = use_flash_linear_attention
        self.use_fused_linear_cross_entropy = use_fused_linear_cross_entropy
        self.action_only_cross_entropy = action_only_cross_entropy
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @property
    def image_tokens_per_frame(self) -> int:
        stride = self.vision_patch_size * self.vision_spatial_merge_size
        return (self.image_size[0] // stride) * (self.image_size[1] // stride)

    @property
    def action_token_end_idx(self) -> int:
        """Exclusive end of the active action rows in G0.5's vocabulary."""
        return self.action_token_begin_idx + self.action_vocab_size
