# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Image packing and multimodal positions are adapted from the released G0.5 policy.

"""G05 vision-language backbone with grammar-constrained action decoding."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from types import FunctionType, MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.utils import ModelOutput

from actioncodec2.vla.g05_hf.configuration_g05 import G05Config

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5RMSNormGated,
        Qwen3_5TextModel,
        Qwen3_5VisionModel,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_kernels

    torch_causal_conv1d_update = getattr(qwen_kernels, "torch_causal_conv1d_update", None)
    if torch_causal_conv1d_update is None:
        torch_causal_conv1d_update = qwen_kernels.causal_conv1d_update
except ImportError as error:  # pragma: no cover - exercised only with an old dependency
    raise ImportError(
        "G05 requires a Transformers release containing Qwen3.5"
    ) from error


def liger_kernel_available() -> bool:
    """Whether the optional Liger CUDA kernels are installed."""
    return importlib.util.find_spec("liger_kernel") is not None


def apply_liger_kernels(model: "G05Model", *, device: torch.device | str) -> None:
    """Enable optional Liger kernels on this CUDA model instance."""
    if torch.device(device).type != "cuda" or not liger_kernel_available():
        return
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    norms = [model.vlm.norm]
    for layer in model.vlm.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
        layer.mlp.forward = MethodType(LigerQwen3MoeSwiGLUMLP.forward, layer.mlp)
    for norm in norms:
        norm.offset = 1.0
        norm.casting_mode = "gemma"
        norm.variance_epsilon = getattr(norm, "variance_epsilon", None) or norm.eps
        norm.in_place = False
        norm.row_mode = None
        norm.forward = MethodType(LigerRMSNormForQwen3Next.forward, norm)


class ActionGrammar(Protocol):
    """Minimal generation contract implemented by the action tokenizers."""

    @property
    def vocab_size(self) -> int: ...

    def next_token_mask(self, tokens: Sequence[int]) -> np.ndarray | Tensor: ...

    def is_complete(self, tokens: Sequence[int]) -> bool: ...


@dataclass
class G05Output(ModelOutput):
    loss: Tensor | None = None
    logits: Tensor | None = None


class G05Model(PreTrainedModel):
    """Qwen3.5 VLM using G0.5's native tied action-token vocabulary rows.

    One observation autoregressively predicts a grammar-delimited action chunk.
    History and proprio replace explicit slots in the released G0.5 prompt.
    """

    config_class = G05Config
    model_type = G05Config.model_type
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def __init__(self, config: G05Config) -> None:
        super().__init__(config)
        text_config = Qwen3_5TextConfig(
            vocab_size=config.text_vocab_size,
            hidden_size=config.text_hidden_size,
            intermediate_size=config.text_intermediate_size,
            num_hidden_layers=config.text_num_layers,
            num_attention_heads=config.text_num_heads,
            num_key_value_heads=config.text_num_kv_heads,
            head_dim=config.text_head_dim,
            max_position_embeddings=262_144,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": 0.25,
                "mrope_section": list(config.mrope_section),
                "mrope_interleaved": True,
            },
            layer_types=config.text_layer_types,
            linear_conv_kernel_dim=config.linear_conv_kernel_dim,
            linear_key_head_dim=config.linear_key_head_dim,
            linear_value_head_dim=config.linear_value_head_dim,
            linear_num_key_heads=config.linear_num_key_heads,
            linear_num_value_heads=config.linear_num_value_heads,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )
        vision_config = Qwen3_5VisionConfig(
            depth=config.vision_depth,
            hidden_size=config.vision_hidden_size,
            intermediate_size=config.vision_intermediate_size,
            num_heads=config.vision_num_heads,
            patch_size=config.vision_patch_size,
            temporal_patch_size=config.vision_temporal_patch_size,
            spatial_merge_size=config.vision_spatial_merge_size,
            out_hidden_size=config.text_hidden_size,
        )
        text_config._attn_implementation = config.attn_implementation
        vision_config._attn_implementation = config.vision_attn_implementation


        self.validate_inputs = True
        self.vision_tower = Qwen3_5VisionModel(vision_config)
        self.vlm = Qwen3_5TextModel(text_config)
        self._configure_gated_delta_backend(config.use_flash_linear_attention)


        self.action_history_embedder = (
            nn.Linear(config.continuous_action_dim, config.text_hidden_size)
            if config.continuous_action_dim
            else None
        )
        self.proprio_embedder = (
            nn.Sequential(
                nn.Linear(config.internal_proprio_dim, config.text_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.text_hidden_size),
                nn.Linear(config.text_hidden_size, config.text_hidden_size),
            )
            if config.proprio_dim
            else None
        )
        self.post_init()


        if self.proprio_embedder is not None:
            init.zeros_(self.proprio_embedder[3].weight)
            init.zeros_(self.proprio_embedder[3].bias)
        if self.action_history_embedder is not None:


            init.normal_(
                self.action_history_embedder.bias,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.vlm.embed_tokens

    def get_output_embeddings(self) -> nn.Module:

        return self.vlm.embed_tokens


    def _action_weight(self) -> Tensor:
        """A view of the active rows in G0.5's tied vocabulary matrix."""
        return self.vlm.embed_tokens.weight.narrow(
            0,
            self.config.action_token_begin_idx,
            self.config.action_vocab_size,
        )

    def _embed_action_tokens(self, local_ids: Tensor) -> Tensor:
        global_ids = local_ids + self.config.action_token_begin_idx
        return F.embedding(global_ids, self.vlm.embed_tokens.weight)

    def _action_logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self._action_weight())

    def _action_cross_entropy(
        self,
        prediction_hidden: Tensor,
        local_labels: Tensor,
        label_mask: Tensor,
    ) -> Tensor:
        """Compute next-token loss over the full or active action vocabulary.

        Action-only loss matches grammar-constrained decoding and avoids a
        full-vocabulary projection; its scale differs from full-vocabulary CE.
        """
        hidden = prediction_hidden[label_mask]
        if self.config.action_only_cross_entropy:
            targets = local_labels[label_mask]
            weight = self._action_weight()
        else:
            targets = local_labels[label_mask] + self.config.action_token_begin_idx
            weight = self.vlm.embed_tokens.weight
        if (
            self.config.use_fused_linear_cross_entropy
            and hidden.is_cuda
            and liger_kernel_available()
        ):
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            loss_fn = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)


            return loss_fn(weight, hidden, targets).float()
        full_logits = F.linear(hidden, weight)
        return F.cross_entropy(full_logits.float(), targets)

    def _configure_gated_delta_backend(self, use_flash: bool) -> None:
        """Keep CPU/default models independent of optional CUDA-only FLA kernels."""
        if use_flash:
            return
        for layer in self.vlm.layers:
            module = getattr(layer, "linear_attn", None)
            if module is None:
                continue
            if hasattr(module, "chunk_gated_delta_rule"):
                module.causal_conv1d_fn = None
                module.causal_conv1d_update = torch_causal_conv1d_update
                module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            else:


                forward = inspect.unwrap(module.forward)
                namespace = dict(forward.__globals__)
                for name in ("causal_conv1d_fn", "causal_conv1d_update",
                             "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                    namespace[name] = inspect.unwrap(getattr(qwen_kernels, name))
                reference_forward = FunctionType(
                    forward.__code__, namespace, forward.__name__,
                    forward.__defaults__, forward.__closure__,
                )
                reference_forward.__kwdefaults__ = forward.__kwdefaults__
                accelerate_hooks = getattr(qwen_kernels, "force_accelerate_hooks", None)
                if accelerate_hooks is not None:
                    reference_forward = accelerate_hooks("conv1d")(reference_forward)
                module.forward = MethodType(reference_forward, module)
            module.norm = Qwen3_5RMSNormGated(
                module.head_v_dim, eps=module.layer_norm_epsilon
            )


    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)


    @staticmethod
    def _patchify(images: Tensor, *, patch: int, merge: int, temporal: int) -> Tensor:
        """Pack images with shape [B, Cam, Time, C, H, W] for the vision tower."""
        batch, cameras, frames, channels, height, width = images.shape
        grid_h, grid_w = height // patch, width // patch
        packed = images.reshape(batch * cameras * frames, channels, height, width)
        packed = packed.unsqueeze(2).expand(-1, -1, temporal, -1, -1).contiguous()
        legacy = packed.reshape(-1, channels * temporal, height, width)
        legacy = legacy.reshape(
            -1,
            temporal,
            channels,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        return legacy.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
            -1, channels * temporal * patch * patch
        )

    def encode_images(self, images: Tensor) -> Tensor:
        """Encode RGB images supplied as uint8 or floats in [-1, 1]."""
        if images.ndim == 5:
            images = images.unsqueeze(2)
        if images.dtype == torch.uint8:

            images = images.float() / 255.0 * 2.0 - 1.0
        if images.ndim != 6:
            raise ValueError("pixel_values must have shape [B,Cam,Time,C,H,W]")
        batch, cameras, frames, channels, height, width = images.shape
        if channels != 3 or (height, width) != self.config.image_size:
            raise ValueError("pixel_values do not match the configured RGB image shape")

        patch = self.config.vision_patch_size
        merge = self.config.vision_spatial_merge_size
        temporal = self.config.vision_temporal_patch_size
        if height % (patch * merge) or width % (patch * merge):
            raise ValueError("image size is incompatible with the vision patch layout")
        grid_h, grid_w = height // patch, width // patch

        patches = self._patchify(images, patch=patch, merge=merge, temporal=temporal)

        num_images = batch * cameras * frames
        grid_thw = torch.tensor(
            (1, grid_h, grid_w), device=images.device, dtype=torch.long
        ).expand(num_images, -1)
        features = self.vision_tower(patches, grid_thw).pooler_output
        return features.reshape(batch, -1, self.config.text_hidden_size)

    def build_mrope_position_ids(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Build Qwen3.5 temporal/height/width positions for packed image blocks."""
        merged_h = self.config.image_size[0] // (
            self.config.vision_patch_size * self.config.vision_spatial_merge_size
        )
        merged_w = self.config.image_size[1] // (
            self.config.vision_patch_size * self.config.vision_spatial_merge_size
        )
        image_length = merged_h * merged_w
        image_span = max(merged_h, merged_w)
        mask = attention_mask.bool()
        image_mask = input_ids.eq(self.config.image_token_id) & mask
        if self.validate_inputs and torch.any(image_mask.sum(dim=-1).remainder(image_length)):
            raise ValueError(f"every image must occupy exactly {image_length} prompt tokens")

        valid_rank = mask.long().cumsum(dim=-1) - 1
        image_rank = image_mask.long().cumsum(dim=-1) - 1
        within_image = image_rank.remainder(image_length)
        image_number = torch.div(image_rank, image_length, rounding_mode="floor")
        completed = torch.div(image_mask.long().cumsum(dim=-1), image_length, rounding_mode="floor")
        text_position = valid_rank - completed * (image_length - image_span)
        image_position = valid_rank - within_image - image_number * (image_length - image_span)
        temporal = torch.where(image_mask, image_position, text_position)
        height = torch.where(
            image_mask,
            image_position + torch.div(within_image, merged_w, rounding_mode="floor"),
            text_position,
        )
        width = torch.where(
            image_mask,
            image_position + within_image.remainder(merged_w),
            text_position,
        )
        positions = torch.stack((temporal, height, width))
        return positions.masked_fill(~mask.unsqueeze(0), 0)

    def _embed_prompt(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [B,L]")
        if input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError("text and image batch sizes must match")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long")
        if self.validate_inputs and torch.any(
            input_ids.lt(0) | input_ids.ge(self.config.text_vocab_size)
        ):
            raise ValueError("input_ids contain an ID outside the text vocabulary")
        mask = attention_mask.bool()
        embeddings = self.vlm.embed_tokens(input_ids)
        image_features = self.encode_images(pixel_values).to(embeddings.dtype)
        image_mask = input_ids.eq(self.config.image_token_id) & mask
        expected = image_features.shape[1]
        if self.validate_inputs and torch.any(image_mask.sum(dim=-1).ne(expected)):
            raise ValueError("each prompt must contain one image-token block per input frame")
        embeddings = embeddings.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(embeddings), image_features.reshape(-1)
        )
        return embeddings, self.build_mrope_position_ids(input_ids, mask)


    @staticmethod
    def _validate_right_padded(mask: Tensor, name: str) -> None:
        if mask.ndim != 2:
            raise ValueError(f"{name} must have shape [B,L]")
        if mask.shape[1] > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError(f"{name} must be right-padded")

    def _continuous_history(
        self,
        batch_size: int,
        device: torch.device,
        continuous_action_history: Tensor | None,
        continuous_action_history_mask: Tensor | None,
    ) -> Tensor | None:
        if not self.config.continuous_action_dim:
            if continuous_action_history is not None or continuous_action_history_mask is not None:
                raise ValueError("continuous action history is disabled by the model config")
            return None
        if continuous_action_history is None or continuous_action_history_mask is None:
            raise ValueError("the configured model requires continuous action history and its mask")
        values = continuous_action_history.to(device=device, dtype=torch.float32)
        expected = (
            batch_size,
            self.config.action_history_steps,
            self.config.continuous_action_dim,
        )
        if values.shape != expected:
            raise ValueError(f"continuous_action_history must have shape {expected}")
        mask = continuous_action_history_mask.to(device=device, dtype=torch.bool)
        if mask.shape != expected[:2]:
            raise ValueError("continuous_action_history_mask has the wrong shape")
        if self.validate_inputs and not torch.isfinite(values).all():
            raise ValueError("continuous action history must be finite")
        if self.validate_inputs and torch.any(values.masked_select(~mask.unsqueeze(-1)).ne(0)):
            raise ValueError("masked action-history positions must contain exact zero padding")
        assert self.action_history_embedder is not None


        embedder_dtype = self.vlm.embed_tokens.weight.dtype
        history = self.action_history_embedder(values.to(embedder_dtype))

        history = history.masked_fill(~mask.unsqueeze(-1), 0)
        return history.to(self.vlm.embed_tokens.weight.dtype)

    def _proprio_token(
        self, batch_size: int, device: torch.device, proprio: Tensor | None
    ) -> Tensor | None:
        if not self.config.proprio_dim:
            if proprio is not None:
                raise ValueError("proprio is disabled by the model config")
            return None
        if proprio is None:
            raise ValueError("the configured model requires current proprio")
        values = proprio.to(device=device, dtype=torch.float32)
        if values.shape != (batch_size, self.config.proprio_dim):
            raise ValueError(f"proprio must have shape [B,{self.config.proprio_dim}]")
        if self.validate_inputs and not torch.isfinite(values).all():
            raise ValueError("proprio must be finite")
        padded = values.new_zeros(batch_size, self.config.internal_proprio_dim)
        padded[:, self.config.proprio_indices] = values
        assert self.proprio_embedder is not None


        embedder_dtype = self.vlm.embed_tokens.weight.dtype
        embedded = self.proprio_embedder(padded.to(embedder_dtype))
        return embedded.to(self.vlm.embed_tokens.weight.dtype).unsqueeze(1)

    @staticmethod
    def _replace_prompt_slots(
        prompt: Tensor,
        prompt_mask: Tensor,
        slot_mask: Tensor,
        values: Tensor,
        *,
        name: str,
        validate: bool = True,
    ) -> Tensor:
        if slot_mask.shape != prompt_mask.shape:
            raise ValueError(f"{name} must match input_ids")
        slots = slot_mask.to(device=prompt.device, dtype=torch.bool)
        if validate and torch.any(slots & ~prompt_mask):
            raise ValueError(f"{name} cannot select padding positions")
        if values.ndim != 3 or values.shape[0] != prompt.shape[0]:
            raise ValueError(f"{name} replacement values must have shape [B,N,D]")
        expected = values.shape[1]
        if validate and torch.any(slots.sum(dim=1).ne(expected)):
            raise ValueError(f"{name} must select exactly {expected} positions per sample")
        if expected == 0:
            return prompt
        expanded = slots.unsqueeze(-1).expand_as(prompt)
        return prompt.masked_scatter(expanded, values.to(prompt.dtype).reshape(-1))

    def _build_context(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        continuous_action_history: Tensor | None,
        continuous_action_history_mask: Tensor | None,
        proprio: Tensor | None,
        action_history_position_mask: Tensor | None,
        proprio_position_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prompt, prompt_positions = self._embed_prompt(input_ids, attention_mask, pixel_values)
        prompt_mask = attention_mask.bool()
        batch = input_ids.shape[0]
        history = self._continuous_history(
            batch,
            input_ids.device,
            continuous_action_history,
            continuous_action_history_mask,
        )
        if history is None:
            history = prompt.new_empty(batch, 0, self.config.text_hidden_size)
        proprio_token = self._proprio_token(batch, input_ids.device, proprio)
        if action_history_position_mask is None:
            if history.shape[1]:
                raise ValueError("action history requires an explicit prompt position mask")
            action_history_position_mask = torch.zeros_like(prompt_mask)
        if proprio_position_mask is None:
            if proprio_token is not None:
                raise ValueError("proprio requires an explicit prompt position mask")
            proprio_position_mask = torch.zeros_like(prompt_mask)
        prompt = self._replace_prompt_slots(
            prompt,
            prompt_mask,
            action_history_position_mask,
            history,
            name="action_history_position_mask",
            validate=self.validate_inputs,
        )
        state_values = (
            prompt.new_empty(batch, 0, self.config.text_hidden_size)
            if proprio_token is None
            else proprio_token
        )
        prompt = self._replace_prompt_slots(
            prompt,
            prompt_mask,
            proprio_position_mask,
            state_values,
            name="proprio_position_mask",
            validate=self.validate_inputs,
        )
        if self.validate_inputs and torch.any(input_ids[:, -1].ne(self.config.eov_token_id)):
            raise ValueError("the right-aligned G0.5 base context must end in <EOV>")
        return prompt, prompt_mask, prompt_positions


    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        action_labels: Tensor,
        action_label_mask: Tensor | None = None,
        continuous_action_history: Tensor | None = None,
        continuous_action_history_mask: Tensor | None = None,
        proprio: Tensor | None = None,
        action_history_position_mask: Tensor | None = None,
        proprio_position_mask: Tensor | None = None,
    ) -> G05Output:
        """Teacher-forced action-token loss for one batch.

        Inference goes through :meth:`generate_action_tokens`, which applies the
        tokenizer grammar at every step.
        """
        labels = action_labels.to(device=input_ids.device, dtype=torch.long)
        if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0] or labels.shape[1] < 1:
            raise ValueError("action_labels must have shape [B,A] with A >= 1")
        label_mask = (
            labels.new_ones(labels.shape, dtype=torch.bool)
            if action_label_mask is None
            else action_label_mask.to(device=input_ids.device, dtype=torch.bool)
        )
        if label_mask.shape != labels.shape:
            raise ValueError("action_label_mask must match action_labels")
        if self.validate_inputs:
            self._validate_right_padded(label_mask, "action_label_mask")
            if torch.any(label_mask.sum(dim=1).eq(0)):
                raise ValueError("every sample needs at least one target action token")
            if torch.any((labels.lt(0) | labels.ge(self.config.action_vocab_size)) & label_mask):
                raise ValueError("action labels contain an ID outside the action vocabulary")
        labels = labels.masked_fill(~label_mask, 0)

        context, context_mask, context_positions = self._build_context(
            input_ids,
            attention_mask,
            pixel_values,
            continuous_action_history,
            continuous_action_history_mask,
            proprio,
            action_history_position_mask,
            proprio_position_mask,
        )
        teacher_ids = labels[:, :-1]
        teacher_mask = label_mask[:, :-1]
        teacher = self._embed_action_tokens(teacher_ids)
        context_last = context_positions[:, :, -1]
        teacher_offsets = teacher_mask.long().cumsum(dim=1)
        teacher_positions = context_last[:, :, None] + teacher_offsets[None]
        embeddings = torch.cat((context, teacher), dim=1)
        full_mask = torch.cat((context_mask, teacher_mask), dim=1)
        positions = torch.cat((context_positions, teacher_positions), dim=2)
        positions = positions.masked_fill(~full_mask.unsqueeze(0), 0)

        outputs = self.vlm(
            inputs_embeds=embeddings,
            attention_mask=full_mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        context_width = context.shape[1]


        prediction_hidden = outputs.last_hidden_state[:, context_width - 1 :]
        logits = self._action_logits(prediction_hidden)
        loss = self._action_cross_entropy(prediction_hidden, labels, label_mask)
        return G05Output(loss=loss, logits=logits)


    def _sample(
        self, logits: Tensor, allowed: Tensor, *, do_sample: bool, temperature: float
    ) -> Tensor:
        scores = logits.float().masked_fill(~allowed, -torch.inf)
        if torch.any(~torch.isfinite(scores).any(dim=-1)):
            raise RuntimeError("the action grammar left no legal next token")
        if not do_sample:
            return scores.argmax(dim=-1)
        scores = scores / temperature
        top_k = min(self.config.top_k, scores.shape[-1])
        if top_k:
            threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
            scores.masked_fill_(scores < threshold, -torch.inf)
        if self.config.top_p < 1:
            sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
            cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > self.config.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_scores.masked_fill_(remove, -torch.inf)
            scores = torch.full_like(scores, -torch.inf).scatter(1, sorted_indices, sorted_scores)
        return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(1)


    @staticmethod
    def _snapshot_linear_cache(cache, inactive: Tensor) -> list[tuple[object, Tensor, Tensor]]:
        snapshots = []
        if not torch.any(inactive):
            return snapshots
        for layer in cache.layers:
            conv = getattr(layer, "conv_states", None)
            recurrent = getattr(layer, "recurrent_states", None)
            if isinstance(conv, Tensor) and isinstance(recurrent, Tensor):
                snapshots.append((layer, conv[inactive].clone(), recurrent[inactive].clone()))
        return snapshots

    @staticmethod
    def _restore_linear_cache(
        snapshots: list[tuple[object, Tensor, Tensor]], inactive: Tensor
    ) -> None:
        for layer, conv, recurrent in snapshots:
            layer.conv_states[inactive] = conv
            layer.recurrent_states[inactive] = recurrent

    @torch.inference_mode()
    def generate_action_tokens(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        grammar: ActionGrammar,
        continuous_action_history: Tensor | None = None,
        continuous_action_history_mask: Tensor | None = None,
        proprio: Tensor | None = None,
        action_history_position_mask: Tensor | None = None,
        proprio_position_mask: Tensor | None = None,
        *,
        do_sample: bool | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[Tensor]:
        """Generate one complete action chunk per row under ``grammar``.

        ``do_sample`` and ``temperature`` default to the model config; top-k and
        top-p always come from the config.
        """
        if grammar.vocab_size != self.config.action_vocab_size:
            raise ValueError("grammar and model action vocabulary sizes differ")
        limit = self.config.max_action_tokens if max_new_tokens is None else max_new_tokens
        if limit <= 0:
            raise ValueError("max_new_tokens must be positive")
        sample = self.config.do_sample if do_sample is None else do_sample
        temperature = self.config.temperature if temperature is None else float(temperature)
        if temperature <= 0:
            raise ValueError("temperature must be positive; use do_sample=False for greedy decoding")
        context, full_mask, positions = self._build_context(
            input_ids,
            attention_mask,
            pixel_values,
            continuous_action_history,
            continuous_action_history_mask,
            proprio,
            action_history_position_mask,
            proprio_position_mask,
        )
        outputs = self.vlm(
            inputs_embeds=context,
            attention_mask=full_mask,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1]
        last_position = positions[:, :, -1]
        batch = input_ids.shape[0]
        generated: list[list[int]] = [[] for _ in range(batch)]


        finished_host = [False] * batch
        for _ in range(limit):
            allowed_host = np.stack(
                [
                    np.asarray(grammar.next_token_mask(row), dtype=bool).reshape(-1)
                    for row in generated
                ]
            )
            if allowed_host.shape[1] != self.config.action_vocab_size:
                raise ValueError("grammar returned a mask with the wrong vocabulary size")


            for index, done in enumerate(finished_host):
                if done:
                    allowed_host[index] = False
                    allowed_host[index, 0] = True
            allowed = torch.from_numpy(allowed_host).to(input_ids.device, non_blocking=True)
            next_token = self._sample(
                self._action_logits(last_hidden),
                allowed,
                do_sample=sample,
                temperature=temperature,
            )
            sampled = next_token.tolist()
            for index, done in enumerate(finished_host):
                if not done:
                    generated[index].append(int(sampled[index]))

            for index, row in enumerate(generated):
                if not finished_host[index] and grammar.is_complete(row):
                    finished_host[index] = True
            if all(finished_host):
                break

            active = torch.tensor(
                [not done for done in finished_host],
                dtype=torch.bool,
                device=input_ids.device,
            )
            input_token = next_token.masked_fill(~active, 0)
            full_mask = torch.cat((full_mask, active[:, None]), dim=1)
            next_position = last_position + active.long().unsqueeze(0)
            snapshots = self._snapshot_linear_cache(cache, ~active)
            outputs = self.vlm(
                inputs_embeds=self._embed_action_tokens(input_token).unsqueeze(1),
                attention_mask=full_mask,
                position_ids=next_position.unsqueeze(-1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            self._restore_linear_cache(snapshots, ~active)
            last_hidden = torch.where(
                active[:, None], outputs.last_hidden_state[:, -1], last_hidden
            )
            last_position = torch.where(active.unsqueeze(0), next_position, last_position)
        else:
            incomplete = [
                index for index, row in enumerate(generated) if not grammar.is_complete(row)
            ]
            if incomplete:
                raise RuntimeError(
                    f"action generation reached max_new_tokens before rows {incomplete} completed"
                )

        return [input_ids.new_tensor(row) for row in generated]
