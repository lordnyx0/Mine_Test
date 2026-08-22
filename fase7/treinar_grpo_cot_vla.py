# coding=utf-8
"""
fase7/treinar_grpo_cot_vla.py — Treinamento GRPO Token-Level com Raciocínio Profundo de 500 Tokens em Terreno Natural.
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
    formatar_acao_texto,
    decodificar_frame_rgb,
    YAW_BIN_NAMES
)
from fase5.acoes_taticas import MODOS, YAW_BINS_9
from fase7.ambiente_cognitivo import AmbienteCognitivoFase7

LOG_JSON_PATH = os.path.join(_ROOT, "fase7", "raciocinios_tempo_real.json")
LOG_JSONL_PATH = os.path.join(_ROOT, "fase7", "raciocinios_stream.jsonl")


def treinar_grpo(
    iteracoes: int = 30,
    num_amostras_grupo: int = 4,
    temperaturas_grupo: List[float] = [0.6, 0.8, 1.0, 1.2],
    lr: float = 3e-5,
    max_tokens_cot: int = 500,
    ckpt_entrada: str = "checkpoints_vla/vla_fase7_sft_aligned.pt",
    ckpt_saida: str = "checkpoints_vla/vla_fase7_grpo_cot.pt",
    seed: int = 42
):
    print("=" * 90)
    print(" 🌲 TREINAMENTO GRPO COM RACIOCÍNIO PROFUNDO (MAX 500 TOKENS) EM TERRENO NATURAL")
    print("=" * 90)
    print(f"  Iterações           : {iteracoes}")
    print(f"  Amostras por Grupo G: {num_amostras_grupo} caminhos por robô")
    print(f"  Max Tokens CoT      : {max_tokens_cot} (Cadeia de Raciocínio Completa)")
    print(f"  Temperaturas        : {temperaturas_grupo} (Exploração Multimodal)")
    print(f"  Ambiente            : 100% Terreno Procedural Natural do Minecraft")
    print(f"  Checkpoint Entrada  : {ckpt_entrada}")
    print(f"  Checkpoint Saída    : {ckpt_saida}")
    print(f"  Log JSON Tempo Real : {LOG_JSON_PATH}")
    print("=" * 90, flush=True)

    os.makedirs(os.path.dirname(LOG_JSON_PATH), exist_ok=True)
    with open(LOG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"iteracoes": []}, f, indent=2)

    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(device)

    vla.vision_encoder.eval()
    for p in vla.vision_encoder.parameters():
        p.requires_grad = False

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
    ambiente_cog = AmbienteCognitivoFase7(tipo_cenario="natural")

    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(treinaveis, lr=lr, weight_decay=0.01)

    info = get("/lote/info")
    N = info["envs"]
    print(f"[ENV] Conectado ao simulador Mineflayer com {N} ambientes paralelos.", flush=True)

    torch.manual_seed(seed)
    random.seed(seed)

    historico_completo = []
    melhor_taxa_sucesso = 0.0

    for it in range(1, iteracoes + 1):
        t_it = time.time()
        
        tarefas, blocos = ambiente_cog.gerar_tarefas_cognitivas(N, seed=seed + it * 41)
        post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        if blocos:
            post("/lote/colocar_bloco", {"blocos": blocos})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * N, "frames": True})
        obs = r["obs"][:N]

        alvos_abs = [t["alvo_abs"] for t in tarefas]
        prompts = [t["prompt"] for t in tarefas]

        vla.eval()
        dist_anteriores = [math.hypot(tarefas[i]["alvo_abs"][0] - obs[i]["estado"]["x"], tarefas[i]["alvo_abs"][1] - obs[i]["estado"]["z"]) for i in range(N)]
        sucessos = [False] * N
        recompensas_acumuladas = [0.0] * N

        # Geração CoT Autoregressiva Profunda (até 500 tokens)
        with torch.inference_mode():
            amostras = politica.gerar_cot_e_acoes(
                obs=obs,
                prompts=prompts,
                alvos_abs=alvos_abs,
                max_new_tokens=max_tokens_cot,
                temperaturas_grupo=temperaturas_grupo,
                top_p=0.95,
                repetition_penalty=1.15,
                num_amostras_por_env=num_amostras_grupo
            )

        acoes_execucao = []
        for i in range(N):
            amostras_env = [a for a in amostras if a["env_id"] == i]
            acoes_execucao.append(amostras_env[0]["acao"] if amostras_env else {"hold": ["w", "sprint"], "mouse": [0, 0], "duration_ms": 50})

        rr = post("/lote/passo", {"acoes": acoes_execucao, "frames": True})
        obs_prox = rr["obs"][:N]

        recompensas_grupo = []
        detalhes_ambientes = []

        for i in range(N):
            e_prox = obs_prox[i]["estado"]
            alvo = tarefas[i]["alvo_abs"]
            d_novo = math.hypot(alvo[0] - e_prox["x"], alvo[1] - e_prox["z"])
            d_ant = dist_anteriores[i]
            
            delta_d = d_ant - d_novo
            rec_deslocamento = delta_d * 6.0
            rec_terminal = 20.0 if d_novo <= 1.8 else 0.0
            penalidade_estagnacao = -1.5 if delta_d <= 0.0 else 0.0
            
            rec_total = rec_deslocamento + rec_terminal + penalidade_estagnacao
            recompensas_acumuladas[i] += rec_total
            if d_novo <= 1.8:
                sucessos[i] = True

            amostras_env = [a for a in amostras if a["env_id"] == i]
            rec_g = []
            detalhes_g = []

            for a_idx, a in enumerate(amostras_env):
                yaw_val = YAW_BINS_9[a["yaw_idx"]]
                
                # Bônus para locomoção ativa (sprint/andar/pular)
                bonus_modo = 2.0 if a["modo_idx"] in [0, 1, 4] else -1.5
                
                r_amostra = rec_total + bonus_modo
                rec_g.append(r_amostra)

                detalhes_g.append({
                    "amostra_id": a_idx,
                    "temperatura": a["temperatura"],
                    "raciocinio_completo": a["texto_gerado"],
                    "modo_escolhido": a["modo_idx"],
                    "yaw_escolhido": a["yaw_idx"],
                    "yaw_graus": yaw_val,
                    "modo_nome": MODOS[a["modo_idx"]],
                    "recompensa_amostra": round(r_amostra, 3)
                })

            recompensas_grupo.append(rec_g)
            detalhes_ambientes.append({
                "env_id": i,
                "prompt": tarefas[i]["prompt"],
                "alvo_cor": tarefas[i]["alvo_cor"],
                "dist_inicial": round(d_ant, 2),
                "dist_final": round(d_novo, 2),
                "delta_dist": round(delta_d, 2),
                "sucesso": d_novo <= 1.8,
                "amostras_grupo": detalhes_g
            })

        # Cálculo das Vantagens Relativas GRPO
        vantagens_lista = []
        for env_i, rec_g in enumerate(recompensas_grupo):
            rg_tensor = torch.tensor(rec_g, dtype=torch.float32)
            mu = rg_tensor.mean()
            std = rg_tensor.std()
            if std < 1e-4:
                vant_g = rg_tensor - mu
            else:
                vant_g = (rg_tensor - mu) / (std + 1e-4)

            for a_i in range(len(rec_g)):
                v_val = round(vant_g[a_i].item(), 3)
                detalhes_ambientes[env_i]["amostras_grupo"][a_i]["vantagem_grpo"] = v_val
                vantagens_lista.append(vant_g[a_i])

        vla.train()
        optimizer.zero_grad()

        loss_grpo_val = 0.0
        causal_lm = vla.qwen_model if hasattr(vla, "qwen_model") else vla
        embed_fn = politica.obter_embed_tokens()

        flat_token_ids = [a["token_ids"].to(device) for a in amostras]
        all_vants = torch.stack(vantagens_lista).to(device)

        if flat_token_ids:
            micro_size = 8
            num_chunks = math.ceil(len(flat_token_ids) / micro_size)
            
            for mb in range(0, len(flat_token_ids), micro_size):
                sub_tokens = flat_token_ids[mb:mb+micro_size]
                sub_vants = all_vants[mb:mb+micro_size]
                
                batch_tokens = torch.nn.utils.rnn.pad_sequence(sub_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
                
                if batch_tokens.size(1) > 1:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = causal_lm(input_ids=batch_tokens)
                        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                        logits = F.linear(hidden[:, :-1, :], embed_fn.weight).float()
                        targets = batch_tokens[:, 1:].contiguous()
                        
                        log_probs = F.log_softmax(logits, dim=-1)
                        token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                        mask = (targets != tokenizer.pad_token_id).float()
                        
                        seq_log_prob = (token_log_probs * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)
                        loss_policy = - (sub_vants * seq_log_prob).mean()
                        
                        probs = F.softmax(logits, dim=-1)
                        entropy = - (probs * log_probs).sum(dim=-1).mean()
                        
                        loss_chunk = (loss_policy - 0.01 * entropy) / num_chunks

                    loss_chunk.backward()
                    loss_grpo_val += loss_policy.item() * num_chunks

            torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
            optimizer.step()

        taxa_sucesso = 100.0 * sum(sucessos) / max(1, N)
        rec_media = sum(recompensas_acumuladas) / max(1, N)
        dt = time.time() - t_it

        iter_log = {
            "iteracao": it,
            "tempo_s": round(dt, 2),
            "taxa_sucesso": round(taxa_sucesso, 1),
            "recompensa_media": round(rec_media, 2),
            "grpo_loss": round(loss_grpo_val, 4),
            "ambientes": detalhes_ambientes
        }
        historico_completo.append(iter_log)

        with open(LOG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"total_iteracoes": len(historico_completo), "historico": historico_completo}, f, indent=2, ensure_ascii=False)

        with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(iter_log, ensure_ascii=False) + "\n")

        print("\n" + "=" * 90)
        print(f" 🌲 [ITERAÇÃO {it:02d}/{iteracoes:02d}] ({dt:4.1f}s) | Taxa Sucesso: {taxa_sucesso:5.1f}% | Rec Média: {rec_media:+5.2f} | Loss GRPO: {loss_grpo_val:+.4f}")
        print("=" * 90)

        for amb in detalhes_ambientes[:3]:
            env_i = amb["env_id"]
            alvo = amb["alvo_cor"]
            d_ini = amb["dist_inicial"]
            d_fim = amb["dist_final"]
            delta_d = amb["delta_dist"]
            print(f"  🔹 [Robô {env_i}] Alvo: {alvo.upper()} | Dist: {d_ini}m -> {d_fim}m (Δ: {delta_d:+.2f}m)")
            for a_s in amb["amostras_grupo"]:
                a_id = a_s["amostra_id"]
                t_val = a_s["temperatura"]
                adv = a_s["vantagem_grpo"]
                rec = a_s["recompensa_amostra"]
                modo_n = a_s["modo_nome"]
                yaw_g = a_s["yaw_graus"]
                txt_limpo = a_s["raciocinio_completo"].replace("\n", " ").strip()
                print(f"     • [Amostra {a_id} (T={t_val:.1f})] (Adv: {adv:+5.2f} | Rec: {rec:+5.1f}) -> Ação: {modo_n} | Yaw: {yaw_g:+4d}°")
                print(f"       Raciocínio: \"{txt_limpo}\"")
            print("  " + "-" * 86)
        print("=" * 90 + "\n", flush=True)

        if taxa_sucesso >= melhor_taxa_sucesso:
            melhor_taxa_sucesso = taxa_sucesso
            os.makedirs(os.path.dirname(ckpt_saida) or ".", exist_ok=True)
            torch.save({"treinaveis": {k: v.cpu() for k, v in vla.state_dict().items() if any(t in k for t in ["lora_", "cabeca_"])}, "taxa_sucesso": taxa_sucesso, "iteracao": it}, ckpt_saida)

    print("\n" + "=" * 90)
    print(" [TREINAMENTO GRPO (500 TOKENS) CONCLUÍDO COM SUCESSO!]")
    print(f"   Log JSON em Tempo Real: {LOG_JSON_PATH}")
    print(f"   Log JSONL Stream: {LOG_JSONL_PATH}")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    treinar_grpo(iteracoes=30, max_tokens_cot=500)
