# coding=utf-8
""" Minimalistic implementation of Qwen3Loop via direct inheritance from HF Qwen3 """

from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3ForCausalLM,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .configuration_qwen3loop import Qwen3LoopConfig


class Qwen3LoopMinimalModel(Qwen3Model):
    """
    Minimal Qwen3Loop model that inherits directly from HF Qwen3Model,
    reusing all attention, MLP, RMSNorm, and DecoderLayer classes unchanged.
    Overriding only forward() for looped execution and virtual layer caching.
    """
    config_class = Qwen3LoopConfig

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        num_loops = getattr(self.config, "num_loops", 2)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            target_virtual_layers = num_loops * self.config.num_hidden_layers
            if hasattr(past_key_values, "layers"):
                while len(past_key_values.layers) < target_virtual_layers:
                    layer_cls = getattr(past_key_values, "layer_class_to_replicate", None)
                    if layer_cls is None:
                        from transformers.cache_utils import DynamicLayer
                        layer_cls = DynamicLayer
                    past_key_values.layers.append(layer_cls())

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Prepare causal mask using native HF create_causal_mask
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if getattr(self, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        for loop_idx in range(num_loops):
            for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Set virtual layer index for KV Cache updating across loops
                virtual_layer_idx = i + loop_idx * self.config.num_hidden_layers
                orig_layer_idx = getattr(decoder_layer, "layer_idx", i)
                orig_attn_layer_idx = getattr(decoder_layer.self_attn, "layer_idx", i)

                decoder_layer.layer_idx = virtual_layer_idx
                decoder_layer.self_attn.layer_idx = virtual_layer_idx

                try:
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        **kwargs,
                    )
                finally:
                    # Restore physical layer indices
                    decoder_layer.layer_idx = orig_layer_idx
                    decoder_layer.self_attn.layer_idx = orig_attn_layer_idx

            # Apply RMSNorm per loop iteration
            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class Qwen3LoopMinimalForCausalLM(Qwen3ForCausalLM):
    """
    Minimal CausalLM wrapper inheriting directly from Qwen3ForCausalLM.
    """
    config_class = Qwen3LoopConfig

    def __init__(self, config: Qwen3LoopConfig):
        super().__init__(config)
        self.model = Qwen3LoopMinimalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
