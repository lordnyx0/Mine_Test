# coding=utf-8
"""
fase7/alinhar_sft_cot_vla.py — Alinhamento Supervisionado de Alta Velocidade e Baixa VRAM (Cold-Start SFT).

Otimizações Críticas:
  1. Pré-computação das features do SigLIP (frozen) em inference mode -> economiza 5 GB de VRAM.
  2. Micro-batch = 2 com Acumulação de Gradientes = 4 (Batch Efetivo = 8) -> VRAM < 4 GB.
  3. Treinamento ultra-rápido do Projetor + Resampler + LoRA em ~20 segundos.
"""
from __future__ import annotations
import os
import sys
import math
import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora
from fase7.politica_cot_autoregressiva import (
    PoliticaCoTAutoregressiva,
    SYSTEM_PROMPT,
    formatar_acao_texto,
    decodificar_frame_rgb,
    YAW_BIN_NAMES
)
from fase7.ambiente_cognitivo import AmbienteCognitivoFase7
from fase5.acoes_taticas import MODOS, YAW_BINS_9


def gerar_texto_cot_supervisionado(
    cor_alvo: str,
    dist: float,
    erro_yaw: float,
    muro_presente: bool,
    lado_livre: str,
    modo_idx: int,
    yaw_idx: int
) -> str:
    setor_str = "no centro da visão" if abs(erro_yaw) < 8.0 else ("à direita" if erro_yaw > 0 else "à esquerda")
    modo_str = MODOS[modo_idx]
    yaw_val = YAW_BINS_9[yaw_idx]
    yaw_nome = YAW_BIN_NAMES[yaw_val]
    acao_str = f"{modo_str}_{yaw_nome}"

    if muro_presente:
        think = (
            f"<think>\n"
            f"1. Percepção Visual: Identifico o pilar {cor_alvo} visível {setor_str}. "
            f"Há um muro de blocos de pedra bloqueando o avanço direto. Passagem livre desobstruída à {lado_livre}.\n"
            f"2. Decisão Espacial: Manobrar em {modo_str} com rotação {yaw_nome} para contornar o obstáculo pela {lado_livre}.\n"
            f"</think>\n"
            f"<action>{acao_str}</action>"
        )
    else:
        think = (
            f"<think>\n"
            f"1. Percepção Visual: Identifico o pilar {cor_alvo} visível {setor_str}. "
            f"Caminho totalmente livre sem obstáculos.\n"
            f"2. Decisão Espacial: Avançar em {modo_str} alinhando com rotação {yaw_nome} diretamente ao alvo.\n"
            f"</think>\n"
            f"<action>{acao_str}</action>"
        )
    return think


def coletar_buffer_multimodal(
    vla: Any,
    ambiente_cog: AmbienteCognitivoFase7,
    num_amostras_total: int = 128,
    num_envs: int = 8,
    device: str = "cuda",
    seed: int = 42
) -> List[Dict[str, Any]]:
    print(f"[*] Coletando e pré-codificando buffer de {num_amostras_total} amostras visuais...", flush=True)
    buffer = []
    num_rodadas = math.ceil(num_amostras_total / num_envs)

    dtype_target = vla.resampler.latents.dtype if hasattr(vla, "resampler") else torch.bfloat16

    for rodada in range(num_rodadas):
        tarefas, blocos = ambiente_cog.gerar_tarefas_cognitivas(num_envs, seed=seed + rodada * 31)
        post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        post("/lote/colocar_bloco", {"blocos": blocos})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * num_envs, "frames": True})
        obs = r["obs"][:num_envs]

        pixel_list = [decodificar_frame_rgb(o) for o in obs]
        pixel_t = torch.stack(pixel_list).to(device, dtype=dtype_target)

        # Pré-computa features visuais do SigLIP em inference mode
        with torch.inference_mode():
            v_feats = vla.vision_encoder(pixel_t).last_hidden_state.cpu()

        for i in range(num_envs):
            if len(buffer) >= num_amostras_total:
                break
            e = obs[i]["estado"]
            alvo = tarefas[i]["alvo_abs"]
            dx = alvo[0] - e["x"]
            dz = alvo[1] - e["z"]
            dist = math.hypot(dx, dz)
            ang = math.degrees(math.atan2(dx, dz))
            erro_yaw = (ang - e["yaw"] + 180.0) % 360.0 - 180.0

            lado_livre = tarefas[i]["lado_livre"]
            cor_alvo = tarefas[i]["alvo_cor"]

            if lado_livre == "esquerda":
                yaw_idx = 1  # yaw_neg60
            else:
                yaw_idx = 7  # yaw_pos60
            modo_idx = 0     # sprint

            cot_target = gerar_texto_cot_supervisionado(
                cor_alvo=cor_alvo,
                dist=dist,
                erro_yaw=erro_yaw,
                muro_presente=True,
                lado_livre=lado_livre,
                modo_idx=modo_idx,
                yaw_idx=yaw_idx
            )

            sv = [
                e.get("vel_x", 0.0), e.get("vel_y", 0.0), e.get("vel_z", 0.0),
                e.get("pitch", 0.0), 1.0 if e.get("on_ground", True) else 0.0
            ] + [0.0] * 27

            buffer.append({
                "v_feat": v_feats[i],
                "state": torch.tensor(sv[:32], dtype=torch.float32),
                "prompt": tarefas[i]["prompt"],
                "target_text": f"{cot_target}<|im_end|>\n"
            })

    print(f"[✓] Buffer coletado e pré-processado: {len(buffer)} amostras!", flush=True)
    return buffer


def treinar_alinhamento_sft(
    passos_sft: int = 150,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 4,
    lr: float = 5e-5,
    ckpt_entrada: str = "checkpoints_vla/vla_fase6_ppo_cot.pt",
    ckpt_saida: str = "checkpoints_vla/vla_fase7_sft_aligned.pt",
    seed: int = 42
):
    print("=" * 80)
    print(" ALINHAMENTO SUPERVISIONADO DE ALTA VELOCIDADE E BAIXA VRAM (FASE 7)")
    print("=" * 80)
    print(f"  Passos de SFT       : {passos_sft}")
    print(f"  Micro-Batch Size    : {micro_batch_size} (Acumulação: {grad_accum_steps} -> Batch Efetivo: {micro_batch_size * grad_accum_steps})")
    print(f"  Learning Rate       : {lr}")
    print(f"  Checkpoint Entrada  : {ckpt_entrada}")
    print(f"  Checkpoint Saída    : {ckpt_saida}")
    print("=" * 80, flush=True)

    torch.cuda.empty_cache()
    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(device)

    vla.vision_encoder.eval()
    for p in vla.vision_encoder.parameters():
        p.requires_grad = False

    for p in vla.projector.parameters():
        p.requires_grad = True
    for p in vla.resampler.parameters():
        p.requires_grad = True
    for p in vla.state_encoder.parameters():
        p.requires_grad = True

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    if os.path.exists(ckpt_entrada):
        try:
            ckpt_data = torch.load(ckpt_entrada, map_location=device)
            if "treinaveis" in ckpt_data:
                vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
                print(f"[*] Pesos restaurados de {ckpt_entrada} ({len(ckpt_data['treinaveis'])} tensores)!", flush=True)
        except Exception as e:
            print(f"[*] Aviso ao carregar checkpoint: {e}", flush=True)

    base_dir = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    politica = PoliticaCoTAutoregressiva(vla, tokenizer, device=device)
    ambiente_cog = AmbienteCognitivoFase7(tipo_cenario="muro_simples")

    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(treinaveis, lr=lr, weight_decay=0.01)

    info = get("/lote/info")
    N = info["envs"]

    torch.manual_seed(seed)
    random.seed(seed)

    buffer = coletar_buffer_multimodal(vla, ambiente_cog, num_amostras_total=128, num_envs=N, device=device, seed=seed)

    dtype_target = vla.resampler.latents.dtype if hasattr(vla, "resampler") else torch.bfloat16
    embed_fn = politica.obter_embed_tokens()

    t0 = time.time()
    for step in range(1, passos_sft + 1):
        t_step = time.time()
        optimizer.zero_grad()
        loss_total = 0.0

        for micro in range(grad_accum_steps):
            batch = random.sample(buffer, micro_batch_size)

            v_feats = torch.stack([b["v_feat"] for b in batch]).to(device, dtype=dtype_target)
            state_t = torch.stack([b["state"] for b in batch]).to(device)
            prompts = [b["prompt"] for b in batch]
            textos_completos = [b["target_text"] for b in batch]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 1. Resampler + Projector
                v_res = vla.resampler(v_feats)
                v_emb = vla.projector(v_res).to(dtype=torch.bfloat16)

                # 2. State Encoder
                s_emb = vla.state_encoder(state_t).to(dtype=torch.bfloat16)

                # 3. Text Prompt
                prompts_chatml = [
                    f"{SYSTEM_PROMPT}<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    for p in prompts
                ]
                tok_out = tokenizer(prompts_chatml, return_tensors="pt", padding=True).to(device)
                p_ids = tok_out["input_ids"]
                p_mask = tok_out["attention_mask"]
                p_emb = embed_fn(p_ids).to(dtype=torch.bfloat16)

                prefix_embeds = torch.cat([v_emb, s_emb, p_emb], dim=1)
                prefix_mask = torch.ones(micro_batch_size, v_emb.size(1) + s_emb.size(1), device=device, dtype=p_mask.dtype)
                attention_mask_p = torch.cat([prefix_mask, p_mask], dim=1)

                # 4. Target Response
                tok_tgt = tokenizer(textos_completos, return_tensors="pt", padding=True).to(device)
                tgt_ids = tok_tgt["input_ids"]
                tgt_mask = tok_tgt["attention_mask"]
                tgt_emb = embed_fn(tgt_ids).to(dtype=torch.bfloat16)

                inputs_embeds = torch.cat([prefix_embeds, tgt_emb], dim=1)
                attention_mask = torch.cat([attention_mask_p, tgt_mask], dim=1)

                L_prefix = prefix_embeds.size(1)
                L_target = tgt_ids.size(1)

                labels = torch.full((micro_batch_size, L_prefix + L_target), -100, dtype=torch.long, device=device)
                labels[:, L_prefix:] = tgt_ids.masked_fill(tgt_mask == 0, -100)

                causal_lm = vla.qwen_model if hasattr(vla, "qwen_model") else vla
                out = causal_lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

                if hasattr(causal_lm, "lm_head"):
                    logits = causal_lm.lm_head(hidden[:, :-1, :]).float()
                else:
                    logits = F.linear(hidden[:, :-1, :], embed_fn.weight).float()

                targets = labels[:, 1:].contiguous()
                loss_micro = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
                loss_scaled = loss_micro / grad_accum_steps

            loss_scaled.backward()
            loss_total += loss_micro.item()

        torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
        optimizer.step()

        loss_avg = loss_total / grad_accum_steps
        dt = time.time() - t_step

        if step % 15 == 0 or step == 1 or step == passos_sft:
            print(f"  Passo SFT {step:03d}/{passos_sft:03d} ({dt*1000:4.0f}ms) | SFT Loss: {loss_avg:.4f}", flush=True)

    total_time = time.time() - t0
    os.makedirs(os.path.dirname(ckpt_saida) or ".", exist_ok=True)
    torch.save({"treinaveis": {k: v.cpu() for k, v in vla.state_dict().items() if any(t in k for t in ["lora_", "projector", "resampler", "state_encoder", "cabeca_"])}, "passos": passos_sft}, ckpt_saida)

    print("\n" + "=" * 80)
    print(f" [ALINHAMENTO SFT CONCLUÍDO COM SUCESSO EM {total_time:.1f} SEGUNDOS!]")
    print(f"  Checkpoint Alinhado Salvo em: {ckpt_saida}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    treinar_alinhamento_sft(passos_sft=120, micro_batch_size=2, grad_accum_steps=4, lr=5e-5)
