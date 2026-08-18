# coding=utf-8
# Copyright 2026 Qwen3Loop team and HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

""" Qwen3Loop model configuration inheriting directly from HF Qwen3Config """

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3LoopConfig(Qwen3Config):
    """
    Configuration class for Qwen3Loop, inheriting directly from HF Qwen3Config
    and adding the looped-transformer hyperparameters.

    Os nomes e os defaults seguem o Nanbeige4.2-3B (configuration_nanbeige.py), de
    modo que um config do Nanbeige possa ser lido aqui sem tradução. Os defaults
    reproduzem exatamente o comportamento anterior do Qwen3Loop, que por sua vez é
    o mesmo dos pesos publicados do Nanbeige4.2-3B (Base e Instruct):

        num_loops=2, skip_loop_final_norm=False, enable_double_loop_split=False,
        loop_share_kv=False, loop_loss_weights=[]

    Args:
        num_loops (int): número de passagens sobre a pilha de camadas.
        skip_loop_final_norm (bool): se False (padrão), a norm final é aplicada ao
            fim de CADA passagem. Se True, apenas uma vez, no fim de todas.
        enable_double_loop_split (bool): LoopSplit. Em vez de repetir a pilha
            inteira, mantém um prefixo e um sufixo executados uma única vez e
            itera apenas as camadas do meio.
        loop_middle_layers (int|None): quantas camadas formam o miolo iterado.
            None = num_hidden_layers // 2.
        loop_share_kv (bool): reaproveita o KV cache da primeira passagem nas
            demais. Reduz o cache pela metade; o relatório do Nanbeige4.2 mede
            ganho consistentemente menor, por isso o padrão é False.
        loop_loss_weights (list): pesos de supervisão auxiliar por passagem.
            Quando não vazio, o número de passagens passa a ser len(weights)+1.
    """

    model_type = "qwen3loop"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_loops=2,
        skip_loop_final_norm=False,
        enable_double_loop_split=False,
        loop_middle_layers=None,
        loop_share_kv=False,
        loop_loss_weights=None,
        **kwargs,
    ):
        self.num_loops = num_loops
        self.skip_loop_final_norm = skip_loop_final_norm
        self.enable_double_loop_split = enable_double_loop_split
        self.loop_middle_layers = loop_middle_layers
        self.loop_share_kv = loop_share_kv
        self.loop_loss_weights = loop_loss_weights if loop_loss_weights is not None else []
        super().__init__(**kwargs)

        if self.enable_double_loop_split and self.loop_middle_layers is None:
            if self.num_hidden_layers % 2 != 0:
                raise ValueError(
                    "enable_double_loop_split exige num_hidden_layers par quando "
                    "loop_middle_layers não é definido."
                )
            self.loop_middle_layers = self.num_hidden_layers // 2
