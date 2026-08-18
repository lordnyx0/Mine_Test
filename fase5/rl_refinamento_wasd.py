# coding=utf-8
"""
fase5/rl_refinamento_wasd.py — Refinamento por Aprendizado por Reforço (Warm-Start RL).
"""
from __future__ import annotations

import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import (
    montar_tarefas_logicas,
    RAIO_CHEGADA_SUBMETA,
    BONUS_SUBMETA,
    BONUS_FINAL
)
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from modelo.lora_vla import aplicar_lora
from infra.gpu_utils import compactar_backbone
from infra.run_vla_agent import load_vla_agent
from fase5.acoes_taticas import decodificar_acao_36

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def retornos(R: np.ndarray, VIVO: np.ndarray, gamma: float = 0.98) -> np.ndarray:
    """Calcula os retornos descontados G_t = sum gamma^k R_{t+k}."""
    T, N = R.shape
    G = np.zeros_like(R)
    acum = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        acum = R[t] + gamma * acum * VIVO[t]
        G[t] = acum
    return G


def rollout_rl_wasd(pol: PoliticaRaciocinioLoop, tarefas: list, passos: int = 50):
    n = len(tarefas)
    
    # 1. Reset para largadas secas
    r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in tarefas]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    estagio_atual = [0] * n
    submeta_atingida = [False] * n
    vivo = [True] * n
    concluiu_em = [None] * n

    dant = [
        math.hypot(
            t["estagios"][0]["alvo_abs"][0] - est[i]["x"],
            t["estagios"][0]["alvo_abs"][1] - est[i]["z"]
        ) for i, t in enumerate(tarefas)
    ]

    SV_L, IDS_L, IDX_L, R_L, VIVO_L = [], [], [], [], []

    for p in range(passos):
        prompts_ativos = [
            t["estagios"][min(estagio_atual[i], len(t["estagios"]) - 1)].get(
                "prompt_estagio", t["prompt"]
            )
            for i, t in enumerate(tarefas)
        ]
        alvos_ativos_abs = [
            t["estagios"][min(estagio_atual[i], len(t["estagios"]) - 1)]["alvo_abs"]
            for i, t in enumerate(tarefas)
        ]

        acoes = pol.agir(est, alvos_ativos_abs, obs, prompts=prompts_ativos, estagios=estagio_atual)
        for i in range(n):
            if not vivo[i]:
                acoes[i] = {"hold": [], "mouse": [0, 0], "duration_ms": 50}

        u = pol.ultimo
        sv_passo  = u["sv"]   # numpy array [N, 32]
        ids_passo = u["ids"]  # numpy array [N, 48]
        idx_passo = u["idx"]  # numpy array [N]

        rr = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = rr["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        rec = np.zeros(n, dtype=np.float32)
        vivo_passo = np.zeros(n, dtype=np.float32)

        for i in range(n):
            if not vivo[i]:
                continue
            
            vivo_passo[i] = 1.0
            e = est[i]

            if e.get("in_water") or e.get("in_lava"):
                rec[i] -= 3.0
                vivo[i] = False
                continue

            t_i = tarefas[i]
            alvo_atual = t_i["estagios"][min(estagio_atual[i], len(t_i["estagios"]) - 1)]["alvo_abs"]
            d_atual = math.hypot(alvo_atual[0] - e["x"], alvo_atual[1] - e["z"])

            # 1. Custo por passo (anti-órbita / anti-ciclo infinito)
            rec[i] -= 0.05

            # 2. Recompensa de avanço vetorial
            delta_d = dant[i] - d_atual
            rec[i] += np.clip(delta_d * 1.5, -0.5, 1.5)
            dant[i] = d_atual

            # 3. Verificação de Submetas
            if d_atual <= RAIO_CHEGADA_SUBMETA:
                if estagio_atual[i] == 0:
                    # Submeta 1 concluída
                    rec[i] += BONUS_SUBMETA
                    submeta_atingida[i] = True
                    estagio_atual[i] = 1
                    novo_alvo = t_i["estagios"][1]["alvo_abs"]
                    dant[i] = math.hypot(novo_alvo[0] - e["x"], novo_alvo[1] - e["z"])
                elif estagio_atual[i] == 1:
                    # Submeta 2 concluída (Sucesso Total)
                    rec[i] += BONUS_FINAL
                    concluiu_em[i] = p
                    vivo[i] = False

        SV_L.append(sv_passo)
        IDS_L.append(ids_passo)
        IDX_L.append(idx_passo)
        R_L.append(rec)
        VIVO_L.append(vivo_passo)

        if not any(vivo):
            break

    met = []
    for i in range(n):
        met.append({
            "submeta_ok": submeta_atingida[i],
            "concluiu": concluiu_em[i] is not None,
            "passos": concluiu_em[i] if concluiu_em[i] is not None else passos,
            "estagio_final": estagio_atual[i]
        })

    SV_T   = np.array(SV_L)    # [T, N, 32]
    IDS_T  = np.array(IDS_L)   # [T, N, 48]
    IDX_T  = np.array(IDX_L)   # [T, N]
    R_T    = np.array(R_L)     # [T, N]
    VIVO_T = np.array(VIVO_L)  # [T, N]

    return (SV_T, IDS_T, IDX_T, R_T, VIVO_T), met


def treinar_rl_refinamento(
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_wasd_tatico.pt",
    ckpt_saida:   str = "checkpoints_vla/vla_fase5_rl_wasd.pt",
    iteracoes:    int = 15,
    passos_ep:    int = 50,
    lr:         float = 3e-5,
    gamma:      float = 0.98,
    seed:         int = 42
):
    print("=" * 80)
    print(" [FASE 5.3] REFINAMENTO DE RACIOCÍNIO VIA RL HÍBRIDO (WARM-START)")
    print(f"    Checkpoint Base : {ckpt_entrada}")
    print(f"    Checkpoint Fim  : {ckpt_saida}")
    print(f"    Iterações       : {iteracoes} | Passos/Ep: {passos_ep} | LR: {lr}")
    print("=" * 80)

    # 1. Carrega modelo VLA e Tokenizer
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    pol = PoliticaRaciocinioLoop(None, amostrar=True, device=dev, vla=vla, loops_pensamento=3, num_acoes=36)

    # Restaura pesos do checkpoint treinado
    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            state_filtrado = {k: v for k, v in ckpt_data["treinaveis"].items()}
            msg = vla.load_state_dict(state_filtrado, strict=False)
            print(f"[VLA] Pesos restaurados com sucesso de '{ckpt_entrada}' ({len(state_filtrado)} tensores).")

    vla.to(dev)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(treinaveis, lr=lr, weight_decay=1e-4)

    N = get("/lote/info")["envs"]
    print(f"[Simulador] Conectado a {N} ambientes paralelos.")
    print(f"[Otimizador] {len(treinaveis)} tensores ativos para refinamento RL.")
    print("--- Iniciando Iterações de Refinamento RL ---")

    t_ini = time.time()

    for it in range(1, iteracoes + 1):
        # 1. Coleta Rollout nos 8 robôs
        vla.eval()
        pol.amostrar = True
        tarefas = montar_tarefas_logicas(N, seed=seed + it * 13, nivel_curriculo=2)

        (SV_T, IDS_T, IDX_T, R_T, VIVO_T), met = rollout_rl_wasd(pol, tarefas, passos=passos_ep)

        taxa_sub1 = sum(m["submeta_ok"] for m in met) / len(met) * 100.0
        taxa_tot  = sum(m["concluiu"] for m in met) / len(met) * 100.0
        recompensa_media = float(R_T.sum() / max(1, len(met)))

        # 2. Calcula Vantagem Descontada
        G = retornos(R_T, VIVO_T, gamma=gamma)
        mascara = VIVO_T.reshape(-1) > 0
        g_ativo = G.reshape(-1)[mascara]
        
        g_std = float(g_ativo.std()) if len(g_ativo) > 1 else 1.0
        adv = (g_ativo - g_ativo.mean()) / (max(g_std, 1.0) + 1e-6)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=dev)

        # Prepara tensores para forward
        T_len, N_len = VIVO_T.shape
        b_sv  = torch.tensor(SV_T.reshape(T_len * N_len, -1)[mascara], dtype=torch.float32, device=dev)
        b_ids = torch.tensor(IDS_T.reshape(T_len * N_len, -1)[mascara], dtype=torch.long, device=dev)
        b_idx = torch.tensor(IDX_T.reshape(-1)[mascara], dtype=torch.long, device=dev)

        # 3. Atualização de Gradiente de Política em Minilotes (cabe com folga em 12GB VRAM)
        vla.train()
        torch.cuda.empty_cache()
        num_transicoes = len(b_idx)
        minilote = 16
        indices = torch.randperm(num_transicoes)
        otimizador.zero_grad()
        total_pg_loss = 0.0
        entropia_media = 0.0

        for mb in range(0, num_transicoes, minilote):
            mb_idx = indices[mb:mb + minilote]
            mb_sv = b_sv[mb_idx]
            mb_ids = b_ids[mb_idx]
            mb_a = b_idx[mb_idx]
            mb_adv = adv_t[mb_idx]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = vla.state_encoder(mb_sv)
                t_embeds = vla.qwen_model.get_input_embeddings()(mb_ids)
                inputs_embeds = torch.cat([s_embeds, t_embeds], dim=1)

                outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                last_hidden = outputs.last_hidden_state[:, -1, :]
                logits = vla.cabeca_acao_36(last_hidden)

                LOGIT_CLIP = 3.0
                logits = torch.tanh(logits / LOGIT_CLIP) * LOGIT_CLIP

                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(mb_a)
                entropia = dist.entropy().mean()

                # Perda ponderada pelo tamanho do minilote
                pg_loss = -(log_probs * mb_adv).mean()
                loss = (pg_loss - 0.005 * entropia) * (len(mb_idx) / max(1, num_transicoes))

            loss.backward()
            total_pg_loss += pg_loss.item() * len(mb_idx)
            entropia_media += entropia.item() * len(mb_idx)

        torch.nn.utils.clip_grad_norm_(treinaveis, max_norm=1.0)
        otimizador.step()

        pg_loss_final = total_pg_loss / max(1, num_transicoes)
        ent_final = entropia_media / max(1, num_transicoes)

        print(
            f"  Iteração {it:2d}/{iteracoes} | "
            f"Recompensa: {recompensa_media:+6.2f} | "
            f"Submeta 1: {taxa_sub1:5.1f}% | "
            f"Sucesso Total: {taxa_tot:5.1f}% | "
            f"PG Loss: {pg_loss_final:+.4f} | "
            f"Entropia: {ent_final:.3f}",
            flush=True
        )

        # Salva o checkpoint atualizado a cada iteração
        os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
        tensores_treinaveis = {
            k: v for k, v in vla.state_dict().items()
            if any(t in k for t in ["lora_", "state_encoder", "cabeca_acao_36"])
        }
        torch.save({
            "treinaveis": tensores_treinaveis,
            "iteracao": it,
            "taxa_submeta1": taxa_sub1,
            "taxa_total": taxa_tot,
            "recompensa": recompensa_media,
            "num_acoes": 36
        }, ckpt_saida)

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Refinamento RL concluído em {duracao:.1f}s.")
    print(f"[OK] Checkpoint final salvo em: {ckpt_saida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",       default="checkpoints_vla/vla_fase5_wasd_tatico.pt")
    ap.add_argument("--saida",      default="checkpoints_vla/vla_fase5_rl_wasd.pt")
    ap.add_argument("--iteracoes",  type=int,   default=15)
    ap.add_argument("--passos",     type=int,   default=50)
    ap.add_argument("--lr",         type=float, default=3e-5)
    args = ap.parse_args()

    treinar_rl_refinamento(
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        iteracoes=args.iteracoes,
        passos_ep=args.passos,
        lr=args.lr
    )
