# coding=utf-8
"""
generation.py — Deterministic response generation.

Every model sees an identically-built prompt and identical decoding settings
(greedy, fixed seed), so any difference in the transcripts comes from the
weights alone.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

import torch
from transformers import set_seed

from .config import EvalConfig
from .models import LoadedModel
from .types import EvalItem, GenerationRecord

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_messages(item: EvalItem) -> list[dict[str, str]]:
    """System prompt, then conversation history, then the question itself."""
    messages: list[dict[str, str]] = []
    if item.system:
        messages.append({"role": "system", "content": item.system})
    messages.extend(turn.as_message() for turn in item.history)
    messages.append({"role": "user", "content": item.prompt})
    return messages


def render_prompt(tokenizer: Any, item: EvalItem, enable_thinking: bool) -> str:
    """
    Apply the model's chat template.

    Qwen3 templates accept `enable_thinking`; older tokenizers do not, and
    tokenizers without a chat template at all fall back to a plain transcript.
    """
    messages = build_messages(item)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    parts = [f"{m['role']}: {m['content']}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def strip_thinking(text: str) -> str:
    """Remove `<think>…</think>` spans so reports show the actual answer."""
    cleaned = _THINK_BLOCK.sub("", text)
    # An unterminated block means the budget ran out mid-thought; keep the tail.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


class Generator:
    """Runs one item against one loaded model and returns a `GenerationRecord`."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.gen_cfg = cfg.generation
        self.seed = cfg.run.seed

    def _generate_kwargs(self, max_new_tokens: int, tokenizer: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.gen_cfg.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        # temperature/top_p are meaningless under greedy decoding and make
        # transformers emit a warning on every call.
        if self.gen_cfg.do_sample:
            kwargs["temperature"] = self.gen_cfg.temperature
            kwargs["top_p"] = self.gen_cfg.top_p
        if self.gen_cfg.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty
        return kwargs

    def generate(self, loaded: LoadedModel, item: EvalItem) -> GenerationRecord:
        tokenizer = loaded.tokenizer
        max_new_tokens = self.gen_cfg.max_new_tokens_for(item.category, item.max_new_tokens)
        rendered = render_prompt(tokenizer, item, self.cfg.model.enable_thinking)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        def record(**overrides: Any) -> GenerationRecord:
            base: dict[str, Any] = {
                "checkpoint": loaded.spec.name,
                "category": item.category,
                "question_id": item.id,
                "prompt": item.prompt,
                "response": "",
                "generation_time": 0.0,
                "tokens_generated": 0,
                "prompt_tokens": 0,
                "temperature": self.gen_cfg.temperature,
                "top_p": self.gen_cfg.top_p,
                "seed": self.seed,
                "timestamp": now,
            }
            base.update(overrides)
            return GenerationRecord(**base)

        try:
            inputs = tokenizer(rendered, return_tensors="pt")
            target_device = getattr(loaded.model, "device", None) or loaded.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            prompt_tokens = int(inputs["input_ids"].shape[-1])

            set_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                outputs = loaded.model.generate(
                    **inputs, **self._generate_kwargs(max_new_tokens, tokenizer)
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            new_tokens = outputs[0][prompt_tokens:]
            tokens_generated = int(new_tokens.shape[-1])
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            truncated = tokens_generated >= max_new_tokens

            return record(
                response=strip_thinking(raw),
                generation_time=elapsed,
                tokens_generated=tokens_generated,
                prompt_tokens=prompt_tokens,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001 — one bad item must not kill the run
            return record(error=f"{type(exc).__name__}: {exc}")
