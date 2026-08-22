# coding=utf-8
"""
fase7/politica_cot_autoregressiva.py — Raciocínio Profundo de Longo Alcance (Até 500 Tokens CoT).
"""
from __future__ import annotations
import os
import sys
import re
import io
import math
import base64
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.acoes_taticas import MODOS, YAW_BINS_9, decodificar_acao_fatorada

YAW_BIN_NAMES = {
    -120: "yaw_neg120",
    -60:  "yaw_neg60",
    -25:  "yaw_neg25",
    -5:   "yaw_neg5",
    0:    "yaw_0",
    5:    "yaw_pos5",
    25:   "yaw_pos25",
    60:   "yaw_pos60",
    120:  "yaw_pos120"
}
NAME_TO_YAW_BIN = {v: k for k, v in YAW_BIN_NAMES.items()}

# System Prompt para Raciocínio Visual Completo e Profundo
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "Você é um agente robótico inteligente no Minecraft com visão em primeira pessoa.\n"
    "Ao receber uma missão, analise detalhadamente a cena visual capturada pelos seus olhos:\n"
    "1. Localize visualmente o pilar alvo colorido e determine sua orientação angular relativa.\n"
    "2. Examine a topografia natural ao redor (árvores bloqueando o caminho, colinas, barrancos, desníveis de terra ou caminhos desobstruídos).\n"
    "3. Formule um plano de locomoção espacial detalhado para alcançar o objetivo com máxima eficiência.\n"
    "Feche seu pensamento em </think> e conclua com a ação motora na tag <action>modo_yaw</action>.\n\n"
    "Ações motoras disponíveis:\n"
    "- Modos: sprint, andar, parado, pular, sprint_pular, alinhar\n"
    "- Yaw: yaw_neg120, yaw_neg60, yaw_neg25, yaw_neg5, yaw_0, yaw_pos5, yaw_pos25, yaw_pos60, yaw_pos120\n"
    "<|im_end|>\n"
)


def decodificar_frame_rgb(obs_item: Dict[str, Any]) -> torch.Tensor:
    """Decodifica imagem RGB de base64 ou tensor direto."""
    if "frame_b64" in obs_item and obs_item["frame_b64"]:
        try:
            img_bytes = base64.b64decode(obs_item["frame_b64"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return torch.from_numpy(arr)
        except Exception:
            pass
    elif "frame" in obs_item and obs_item["frame"] is not None:
        try:
            img = torch.as_tensor(obs_item["frame"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            return F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
        except Exception:
            pass
    return torch.zeros(3, 224, 224, dtype=torch.float32)


def formatar_acao_texto(modo_idx: int, yaw_idx: int) -> str:
    modo_nome = MODOS[modo_idx]
    yaw_val = YAW_BINS_9[yaw_idx]
    yaw_nome = YAW_BIN_NAMES[yaw_val]
    return f"{modo_nome}_{yaw_nome}"


def extrair_acao_texto(texto_gerado: str) -> Tuple[int, int, Dict[str, Any]]:
    """Extrai ação através da tag <action> ou do contexto final de decisão."""
    modo_idx = 0  # Default: sprint
    yaw_idx = 4   # Default: yaw_0 (0°)

    match = re.search(r"<action>(.*?)</action>", texto_gerado, re.IGNORECASE | re.DOTALL)
    if match:
        conteudo = match.group(1).strip().lower()
        partes = conteudo.split("_yaw_")
        if len(partes) == 2:
            modo_str = partes[0].strip()
            yaw_str = "yaw_" + partes[1].strip()
            if modo_str in MODOS:
                modo_idx = MODOS.index(modo_str)
            if yaw_str in NAME_TO_YAW_BIN:
                yaw_val = NAME_TO_YAW_BIN[yaw_str]
                yaw_idx = YAW_BINS_9.index(yaw_val)
        else:
            for m_i, m_nome in enumerate(MODOS):
                if m_nome in conteudo:
                    modo_idx = m_i
                    break
            for y_i, (y_val, y_nome) in enumerate(YAW_BIN_NAMES.items()):
                if y_nome in conteudo:
                    yaw_idx = y_i
                    break
        acao_mc = decodificar_acao_fatorada(modo_idx, yaw_idx)
        return modo_idx, yaw_idx, acao_mc

    txt_lower = texto_gerado.lower()

    # Busca modo nos últimos 200 caracteres
    trecho_final = txt_lower[-200:] if len(txt_lower) > 200 else txt_lower

    for m_i, m_nome in enumerate(MODOS):
        if m_nome in trecho_final:
            modo_idx = m_i
            break

    if "yaw_neg120" in trecho_final:
        yaw_idx = 0
    elif "yaw_neg60" in trecho_final:
        yaw_idx = 1
    elif "yaw_neg25" in trecho_final:
        yaw_idx = 2
    elif "yaw_neg5" in trecho_final:
        yaw_idx = 3
    elif "yaw_pos120" in trecho_final:
        yaw_idx = 8
    elif "yaw_pos60" in trecho_final:
        yaw_idx = 7
    elif "yaw_pos25" in trecho_final:
        yaw_idx = 6
    elif "yaw_pos5" in trecho_final:
        yaw_idx = 5
    elif "yaw_0" in trecho_final:
        yaw_idx = 4
    elif "esquerda" in trecho_final:
        yaw_idx = 1
    elif "direita" in trecho_final:
        yaw_idx = 7
    else:
        yaw_idx = 4

    acao_mc = decodificar_acao_fatorada(modo_idx, yaw_idx)
    return modo_idx, yaw_idx, acao_mc


class PoliticaCoTAutoregressiva:
    def __init__(self, vla, tokenizer, device="cuda"):
        self.vla = vla
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_size = getattr(vla, "hidden_size", 896)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def obter_embed_tokens(self):
        causal_lm = self.vla.qwen_model if hasattr(self.vla, "qwen_model") else self.vla
        if hasattr(causal_lm, "model") and hasattr(causal_lm.model, "embed_tokens"):
            return causal_lm.model.embed_tokens
        elif hasattr(causal_lm, "embed_tokens"):
            return causal_lm.embed_tokens
        else:
            return causal_lm.get_input_embeddings()

    def preparar_prefixo_multimodal(
        self,
        pixel_tensor: Optional[torch.Tensor],
        state_tensor: torch.Tensor,
        prompts: List[str],
        precomputed_v_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state_tensor.size(0)
        
        if precomputed_v_emb is not None:
            v_emb = precomputed_v_emb.to(dtype=torch.bfloat16)
        elif pixel_tensor is not None:
            dtype_target = self.vla.resampler.latents.dtype if hasattr(self.vla, "resampler") else torch.bfloat16
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_feats = self.vla.vision_encoder(pixel_tensor.to(dtype=dtype_target)).last_hidden_state
                v_res = self.vla.resampler(v_feats.to(dtype=dtype_target))
                v_emb = self.vla.projector(v_res).to(dtype=torch.bfloat16)
        else:
            v_emb = torch.zeros(B, 32, self.hidden_size, device=self.device, dtype=torch.bfloat16)

        s_emb = self.vla.state_encoder(state_tensor.to(self.device).float()).to(dtype=torch.bfloat16)

        prompts_chatml = [
            f"{SYSTEM_PROMPT}<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            for p in prompts
        ]

        tok_out = self.tokenizer(prompts_chatml, return_tensors="pt", padding=True).to(self.device)
        p_ids = tok_out["input_ids"]
        p_mask = tok_out["attention_mask"]
        
        embed_fn = self.obter_embed_tokens()
        p_emb = embed_fn(p_ids).to(dtype=torch.bfloat16)

        inputs_embeds = torch.cat([v_emb, s_emb, p_emb], dim=1)
        prefix_mask = torch.ones(B, v_emb.size(1) + s_emb.size(1), device=self.device, dtype=p_mask.dtype)
        attention_mask = torch.cat([prefix_mask, p_mask], dim=1)

        return inputs_embeds, attention_mask

    @torch.inference_mode()
    def gerar_cot_e_acoes(
        self,
        obs: List[Dict[str, Any]],
        prompts: List[str],
        alvos_abs: Optional[List[Tuple[float, float]]] = None,
        max_new_tokens: int = 500,
        temperaturas_grupo: List[float] = [0.6, 0.8, 1.0, 1.2],
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        num_amostras_por_env: int = 4
    ) -> List[Dict[str, Any]]:
        N = len(obs)
        estados = [o["estado"] for o in obs]

        sv_list = []
        for i in range(N):
            e = estados[i]
            sv = [
                e.get("vel_x", 0.0), e.get("vel_y", 0.0), e.get("vel_z", 0.0),
                e.get("pitch", 0.0), 1.0 if e.get("on_ground", True) else 0.0
            ] + [0.0] * 27
            sv_list.append(sv[:32])

        state_t = torch.as_tensor(sv_list, dtype=torch.float32, device=self.device)
        pixel_list = [decodificar_frame_rgb(o) for o in obs]
        pixel_t = torch.stack(pixel_list).to(self.device, dtype=torch.bfloat16)

        inputs_embeds, attention_mask = self.preparar_prefixo_multimodal(
            pixel_tensor=pixel_t,
            state_tensor=state_t,
            prompts=prompts
        )

        inputs_embeds = inputs_embeds.repeat_interleave(num_amostras_por_env, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_amostras_por_env, dim=0)

        temps = []
        for _ in range(N):
            temps.extend(temperaturas_grupo[:num_amostras_por_env])
        temp_tensor = torch.tensor(temps, dtype=torch.float32, device=self.device).unsqueeze(1)

        causal_lm = self.vla.qwen_model if hasattr(self.vla, "qwen_model") else self.vla
        embed_fn = self.obter_embed_tokens()
        embed_weights = embed_fn.weight
        
        gen_tokens = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_mask = attention_mask

        for step in range(max_new_tokens):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if past_key_values is None:
                    out = causal_lm(inputs_embeds=cur_embeds, attention_mask=cur_mask, use_cache=True)
                else:
                    out = causal_lm(inputs_embeds=cur_embeds, attention_mask=cur_mask, past_key_values=past_key_values, use_cache=True)

                past_key_values = out.past_key_values if hasattr(out, "past_key_values") else None
                hidden_last = out.last_hidden_state[:, -1, :] if hasattr(out, "last_hidden_state") else out[0][:, -1, :]
                
                if hasattr(causal_lm, "lm_head"):
                    next_logits = causal_lm.lm_head(hidden_last).float()
                else:
                    next_logits = F.linear(hidden_last, embed_weights).float()

            if gen_tokens and repetition_penalty != 1.0:
                past_tokens = torch.cat(gen_tokens, dim=1)
                for b_i in range(past_tokens.size(0)):
                    unique_tokens = torch.unique(past_tokens[b_i])
                    for tok_id in unique_tokens:
                        if next_logits[b_i, tok_id] > 0:
                            next_logits[b_i, tok_id] /= repetition_penalty
                        else:
                            next_logits[b_i, tok_id] *= repetition_penalty

            next_logits = next_logits / torch.clamp(temp_tensor, min=1e-3)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, -float("Inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            gen_tokens.append(next_token)

            cur_embeds = embed_fn(next_token).to(dtype=torch.bfloat16)
            cur_mask = torch.cat([cur_mask, torch.ones(cur_mask.size(0), 1, device=self.device, dtype=cur_mask.dtype)], dim=1)

            # Early stopping assim que todas as amostras fecharem a ação
            if step > 40 and (step % 5 == 0):
                token_ids_so_far = torch.cat(gen_tokens, dim=1)
                textos_parciais = self.tokenizer.batch_decode(token_ids_so_far, skip_special_tokens=True)
                if all("</action>" in t or "<|im_end|>" in t for t in textos_parciais):
                    break

        todos_token_ids = torch.cat(gen_tokens, dim=1)
        textos_completos = self.tokenizer.batch_decode(todos_token_ids, skip_special_tokens=True)

        resultados = []
        for idx, txt in enumerate(textos_completos):
            texto_formatado = "<think>\n" + txt.replace("<think>", "").strip()
            modo_idx, yaw_idx, acao_mc = extrair_acao_texto(texto_formatado)
            
            resultados.append({
                "env_id": idx // num_amostras_por_env,
                "amostra_id": idx % num_amostras_por_env,
                "temperatura": temps[idx],
                "texto_gerado": texto_formatado,
                "token_ids": todos_token_ids[idx].cpu(),
                "modo_idx": modo_idx,
                "yaw_idx": yaw_idx,
                "acao": acao_mc
            })

        return resultados
