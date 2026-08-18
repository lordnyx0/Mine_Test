# coding=utf-8
# Copyright 2026 Qwen3Loop team and HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer as Qwen3LoopDecoderLayer,
    Qwen3PreTrainedModel as Qwen3LoopPreTrainedModel,
)

from .configuration_qwen3loop import Qwen3LoopConfig
from .modeling_qwen3loop import (
    Qwen3LoopForCausalLM,
    Qwen3LoopModel,
)

# Register with Hugging Face Auto classes
try:
    AutoConfig.register("qwen3loop", Qwen3LoopConfig)
    AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)
except Exception:
    pass

__all__ = [
    "Qwen3LoopConfig",
    "Qwen3LoopPreTrainedModel",
    "Qwen3LoopModel",
    "Qwen3LoopDecoderLayer",
    "Qwen3LoopForCausalLM",
]
