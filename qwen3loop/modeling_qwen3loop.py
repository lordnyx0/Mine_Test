# coding=utf-8
# Copyright 2026 Qwen3Loop team and HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

""" Qwen3Loop Model Architecture - Looped Transformer for Qwen3 via Direct HF Inheritance

Paridade de arquitetura com o Nanbeige4.2-3B (modeling_nanbeige.py). O laço de
loop, a indexação virtual do KV cache e a norm entre passagens seguem o mesmo
esquema; os defaults reproduzem os pesos publicados do Nanbeige4.2-3B, que NÃO
ativam LoopSplit, mHC, depth attention nem n-gram embeddings.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3ForCausalLM,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .configuration_qwen3loop import Qwen3LoopConfig


def get_loop_cache_layer_idx(layer_idx: int, loop_idx: int, num_hidden_layers: int,
                             cache_layer_idx: Optional[int] = None) -> int:
    """Índice virtual de camada no KV cache. Igual a _get_loop_cache_layer_idx do Nanbeige."""
    if cache_layer_idx is not None:
        return cache_layer_idx
    return layer_idx + loop_idx * num_hidden_layers


def get_double_loop_split_layer_order(num_hidden_layers: int,
                                      loop_middle_layers: Optional[int] = None
                                      ) -> List[Tuple[int, Optional[int]]]:
    """
    LoopSplit. Porte fiel de _get_double_loop_split_layer_order_with_mhc_loop_indices.

    Mantém um prefixo e um sufixo executados uma única vez e repete só o miolo.
    Para N=22, M=11: [0..4] uma vez, [5..15] tres vezes, [16..21] uma vez = 44
    execuções — mesmo custo de 2 loops completos, distribuído de outra forma.

    Devolve pares (layer_idx, repeat_idx), com repeat_idx=None nas camadas não iteradas.
    """
    if num_hidden_layers <= 0:
        raise ValueError("enable_double_loop_split requer num_hidden_layers > 0.")
    if loop_middle_layers is None:
        if num_hidden_layers % 2 != 0:
            raise ValueError(
                "enable_double_loop_split requer num_hidden_layers divisível por 2 "
                "quando loop_middle_layers não é definido."
            )
        loop_middle_layers = num_hidden_layers // 2
    if loop_middle_layers <= 0:
        raise ValueError("loop_middle_layers deve ser > 0.")
    if num_hidden_layers % loop_middle_layers != 0:
        raise ValueError("loop_middle_layers deve ser divisor de num_hidden_layers.")

    first_unlooped = (num_hidden_layers - loop_middle_layers) // 2
    middle_start = first_unlooped
    middle_end = middle_start + loop_middle_layers
    middle_repeats = (num_hidden_layers + loop_middle_layers) // loop_middle_layers
    return (
        [(idx, None) for idx in range(0, middle_start)]
        + [
            (idx, repeat_idx)
            for repeat_idx in range(middle_repeats)
            for idx in range(middle_start, middle_end)
        ]
        + [(idx, None) for idx in range(middle_end, num_hidden_layers)]
    )


class Qwen3LoopModel(Qwen3Model):
    """
    Qwen3LoopModel inherits directly from HF's Qwen3Model.
    Reuses all native Qwen3 modules (Qwen3Attention, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3MLP)
    and overrides forward() to execute the layer stack num_loops times with loop-virtual KV cache indexing.
    """
    config_class = Qwen3LoopConfig

    # ------------------------------------------------------------------
    # Helpers de loop (espelham NanbeigeModel)
    # ------------------------------------------------------------------
    def get_num_loops(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False):
            return 1
        weights = getattr(self.config, "loop_loss_weights", None)
        if weights:
            return len(weights) + 1
        return getattr(self.config, "num_loops", 1)

    def get_layer_execution_order(self) -> List[Tuple[int, Optional[int]]]:
        if getattr(self.config, "enable_double_loop_split", False):
            return get_double_loop_split_layer_order(
                self.config.num_hidden_layers,
                getattr(self.config, "loop_middle_layers", None),
            )
        return [(i, None) for i in range(self.config.num_hidden_layers)]

    def get_cache_seq_length(self, past_key_values: Optional[Cache]) -> int:
        """
        O comprimento do cache tem que sair de uma camada que de fato escreveu nele.
        Com LoopSplit é a camada 0; caso contrário, o máximo entre as camadas
        virtuais que iniciam cada passagem. Espelha _get_cache_seq_length do Nanbeige.
        """
        if past_key_values is None:
            return 0
        if getattr(self.config, "enable_double_loop_split", False):
            return past_key_values.get_seq_length(0)
        max_len = 0
        for loop_idx in range(self.get_num_loops()):
            max_len = max(max_len, past_key_values.get_seq_length(loop_idx * self.config.num_hidden_layers))
        return max_len

    def _total_virtual_layers(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False):
            return len(self.get_layer_execution_order())
        return self.get_num_loops() * self.config.num_hidden_layers

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
        num_loops = self.get_num_loops()
        layer_order = self.get_layer_execution_order()
        double_split = getattr(self.config, "enable_double_loop_split", False)
        share_kv = getattr(self.config, "loop_share_kv", False)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Gradient checkpointing e cache são incompatíveis (o cache seria reescrito
        # na recomputação do backward).
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if share_kv:
            # Compartilhar o KV da 1a passagem exige interceptar o forward da
            # atenção (o Nanbeige usa _apply_loop_shared_kv dentro do NanbeigeAttention).
            # Aqui reusamos a Qwen3Attention nativa do HF, que não tem esse gancho,
            # então a opção falha alto em vez de silenciosamente não fazer nada.
            # O relatório do Nanbeige4.2 mede ganho consistentemente menor com KV
            # compartilhado, então isto não bloqueia a paridade com os pesos publicados.
            raise NotImplementedError(
                "loop_share_kv=True exige uma subclasse de Qwen3Attention; não implementado. "
                "Os pesos publicados do Nanbeige4.2-3B usam loop_share_kv=False."
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            target_virtual_layers = self._total_virtual_layers()
            if hasattr(past_key_values, "layers"):
                while len(past_key_values.layers) < target_virtual_layers:
                    layer_cls = getattr(past_key_values, "layer_class_to_replicate", None)
                    if layer_cls is None:
                        from transformers.cache_utils import DynamicLayer
                        layer_cls = DynamicLayer
                    past_key_values.layers.append(layer_cls())

        if position_ids is None:
            past_seen_tokens = self.get_cache_seq_length(past_key_values)
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
            for execution_idx, (i, _repeat_idx) in enumerate(layer_order):
                decoder_layer = self.layers[i]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Índice virtual para o KV cache. Com LoopSplit a posição na ordem de
                # execução é que define o slot; sem ele, o deslocamento por passagem.
                if double_split:
                    virtual_layer_idx = execution_idx
                else:
                    virtual_layer_idx = get_loop_cache_layer_idx(i, loop_idx, self.config.num_hidden_layers)

                orig_layer_idx = getattr(decoder_layer, "layer_idx", i)
                orig_attn_layer_idx = getattr(decoder_layer.self_attn, "layer_idx", i)
                decoder_layer.layer_idx = virtual_layer_idx
                decoder_layer.self_attn.layer_idx = virtual_layer_idx

                attn_mask = causal_mask_mapping[self.config.layer_types[i]]
                try:
                    if self.gradient_checkpointing and self.training:
                        # Sem isto o gradient_checkpointing_enable() do HF é um no-op:
                        # o laço chama decoder_layer diretamente e as ativações de
                        # todas as execuções (num_loops x num_layers) ficam retidas.
                        hidden_states = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            attn_mask,
                            position_ids,
                            past_key_values,
                            use_cache,
                            position_embeddings,
                        )
                    else:
                        hidden_states = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
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

            # Norm ao fim de cada passagem (skip_loop_final_norm=False, como no
            # Nanbeige4.2 publicado). Com True, só uma vez ao final de tudo.
            if not getattr(self.config, "skip_loop_final_norm", False):
                hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "skip_loop_final_norm", False):
            hidden_states = self.norm(hidden_states)
            if output_hidden_states and all_hidden_states:
                all_hidden_states = all_hidden_states[:-1] + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class Qwen3LoopForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3LoopForCausalLM inherits directly from HF's Qwen3ForCausalLM.
    100% compatible with Qwen3 checkpoints and HF ecosystem.
    """
    config_class = Qwen3LoopConfig

    def __init__(self, config: Qwen3LoopConfig):
        super().__init__(config)
        self.model = Qwen3LoopModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _supports_default_dynamic_cache(self) -> bool:
        # Com mais de uma passagem o cache tem num_loops x num_hidden_layers
        # camadas virtuais, então o cache padrão do HF não serve.
        return self.config.num_loops == 1 and super()._supports_default_dynamic_cache()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        loop_overrides = {
            key: kwargs.pop(key)
            for key in (
                "num_loops",
                "skip_loop_final_norm",
                "enable_double_loop_split",
                "loop_middle_layers",
                "loop_share_kv",
                "loop_loss_weights",
            )
            if key in kwargs
        }

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if not isinstance(config, Qwen3LoopConfig):
            config_dict = config.to_dict()
            config_dict.update(loop_overrides)
            config_dict.setdefault("num_loops", 2)
            config_dict["model_type"] = "qwen3loop"
            config = Qwen3LoopConfig.from_dict(config_dict)
        else:
            for key, value in loop_overrides.items():
                setattr(config, key, value)

        kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


# Register with Auto classes
try:
    AutoConfig.register("qwen3loop", Qwen3LoopConfig)
    AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)
except Exception:
    pass
