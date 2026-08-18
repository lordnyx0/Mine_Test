# coding=utf-8
"""
fase5/treinar_ppo_bc_hibrido.py — Treinamento Híbrido PPO + Ancoragem Supervisionada (BC 70/30).
Une a alta precisão de tiro na Submeta 1 (BC Offline 70%) com a capacidade de transição/frenagem (RL Online 30%).
"""
from __future__ import annotations

import os
import sys
import time
import math
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

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


def gerar_tarefas_busca_ativa(num_ambientes: int, seed: int = 42, nivel: int = 2) -> list:
    """Gera tarefas com dispersão angular ampla no Pilar 1 (±90°) e Pilar 2 (±120°) para forçar busca ativa."""
    from ambiente.tarefas_logicas import CORES_MAP
    caminho_secas = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "largadas_secas.json")
    if os.path.exists(caminho_secas):
        banco = json.load(open(caminho_secas, "r", encoding="utf-8"))
    else:
        from ambiente.tarefas_logicas import BANCO_LARGADAS
        banco = BANCO_LARGADAS

    rng = random.Random(seed)
    coords = rng.sample(banco, min(num_ambientes, len(banco)))
    
    post("/lote/reset", {"posicoes": [[c[0], c[1]] for c in coords]})
    r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * num_ambientes, "frames": False})
    
    tarefas = []
    blocos = []
    for env_id, o in enumerate(r["obs"][:num_ambientes]):
        e = o["estado"]
        lx, ly, lz = e["x"], e["y"], e["z"]
        lyaw = math.radians(e["yaw"])
        
        cor1, cor2 = rng.sample(list(CORES_MAP.keys()), 2)
        id1, id2 = CORES_MAP[cor1], CORES_MAP[cor2]
        
        # Pilar 1 com dispersão ampla (±75° no nível 2, ±120° no nível 3)
        lim_p1 = 75.0 if nivel == 2 else 120.0
        desvio1 = math.radians(rng.uniform(-lim_p1, lim_p1))
        y1 = lyaw + desvio1
        d1 = rng.uniform(6.5, 9.5)
        tx1 = round(lx - math.sin(y1) * d1, 1)
        tz1 = round(lz - math.cos(y1) * d1, 1)
        ty1 = math.floor(ly)
        
        # Pilar 2 com dispersão de transição (±110°)
        desvio2 = math.radians(rng.uniform(-110.0, 110.0))
        y2 = y1 + desvio2
        d2 = rng.uniform(7.0, 10.0)
        tx2 = round(tx1 - math.sin(y2) * d2, 1)
        tz2 = round(tz1 - math.cos(y2) * d2, 1)
        ty2 = math.floor(ly)
        
        tarefas.append({
            "tipo": "sequencia",
            "env": env_id,
            "largada": (round(lx, 2), round(ly, 2), round(lz, 2), lyaw),
            "prompt": f"Vá até a coluna {cor1} por terra firme sem entrar na água, e depois vá até a coluna {cor2}.",
            "estagios": [
                {"cor": cor1, "bloco_id": id1, "alvo_abs": (tx1, tz1), "alvo_y": ty1, "prompt_estagio": f"Objetivo: vá até a coluna {cor1} por terra firme (evite água) [Etapa 1/2]"},
                {"cor": cor2, "bloco_id": id2, "alvo_abs": (tx2, tz2), "alvo_y": ty2, "prompt_estagio": f"Objetivo: vá até a coluna {cor2} por terra firme (evite água) [Etapa 2/2]"}
            ]
        })
        blocos.append({"env": env_id, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": id1, "altura": 50})
        blocos.append({"env": env_id, "x": math.floor(tx2), "y": ty2, "z": math.floor(tz2), "id": id2, "altura": 50})
        
    return tarefas, blocos


def rollout_rl_wasd(pol: PoliticaRaciocinioLoop, tarefas: list, blocos: list = None, passos: int = 100):
    n = len(tarefas)
    
    # 1. Reset para largadas secas
    r = post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
    
    # 2. Garante colocação das torres de 50 blocos no mundo após o reset
    if blocos:
        post("/lote/colocar_bloco", {"blocos": blocos})
        
    # 3. Publica os alvos para o radar e visualizador do navegador (/ver)
    alvos_web = []
    for t in tarefas:
        a0 = t["estagios"][0]["alvo_abs"]
        alvos_web.append({"x": a0[0], "z": a0[1], "dist": 8.0, "graus": 0})
    post("/lote/alvos", {"alvos": alvos_web})

    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    estagio_atual = [0] * n
    submeta_atingida = [False] * n
    vivo = [True] * n
    concluiu_em = [None] * n
    cooldown_frenagem = [0] * n
    alvo_avistado = [False] * n
    passos_sem_foco = [0] * n

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
            dx = alvo_atual[0] - e["x"]
            dz = alvo_atual[1] - e["z"]
            d_atual = math.hypot(dx, dz)

            # Cálculo de Alinhamento Visual (Gaze on Target)
            ang_alvo_rad = math.atan2(-dx, -dz)
            yaw_rad = math.radians(e["yaw"])
            delta_ang = (yaw_rad - ang_alvo_rad + math.pi) % (2 * math.pi) - math.pi
            alinhamento = math.cos(delta_ang)  # 1.0 = mirando o alvo, -1.0 = alvo nas costas

            acao_exec = acoes[i]
            corre_frente = "W" in acao_exec.get("hold", [])

            # 1. Custo básico de tempo (anti-loop)
            rec[i] -= 0.05

            # 2. Recompensa de Alinhamento Visual / Penalidade de Perda de Foco / Corrida Cega
            if alinhamento > 0.70:
                # Mirando o alvo (cone de ±45°) -> marca contato visual estabelecido e reseta perda de foco
                alvo_avistado[i] = True
                passos_sem_foco[i] = 0
                rec[i] += 0.12 * alinhamento
            else:
                if alvo_avistado[i]:
                    passos_sem_foco[i] += 1
                    # Penalidade progressiva se passar mais de 5 ticks (1.25s) sem ver o alvo após tê-lo avistado
                    if passos_sem_foco[i] > 5:
                        rec[i] -= 0.05 * (passos_sem_foco[i] - 5)

                if alinhamento < 0.20 and corre_frente:
                    # Correndo para frente sem ver o alvo (>78° fora de mira) -> PENALIDADE DE CORRIDA CEGA
                    rec[i] -= 0.35
                elif alinhamento < 0.50 and not corre_frente:
                    # Girando / buscando sem correr cegamente -> busca ativa (ajustada para +0.04 anti-farming)
                    rec[i] += 0.04

            # Melhoria 3: Bônus de Frenagem e Transição Limpa pós-Submeta 1
            if cooldown_frenagem[i] > 0:
                cooldown_frenagem[i] -= 1
                if not corre_frente:
                    rec[i] += 0.25  # Recompensa soltar W para frear a inércia
                if not corre_frente and alinhamento > 0.25:
                    rec[i] += 0.20  # Recompensa rotacionar mirando o Pilar 2

            # 3. Recompensa de avanço vetorial
            delta_d = dant[i] - d_atual
            rec[i] += np.clip(delta_d * 1.5, -0.5, 1.5)
            dant[i] = d_atual

            # 4. Verificação de Submetas
            if d_atual <= RAIO_CHEGADA_SUBMETA:
                if estagio_atual[i] == 0:
                    # Submeta 1 concluída
                    rec[i] += BONUS_SUBMETA
                    submeta_atingida[i] = True
                    estagio_atual[i] = 1
                    cooldown_frenagem[i] = 4  # Ativa 4 passos de amortecimento e busca
                    alvo_avistado[i] = False   # Reseta para permitir busca livre do Pilar 2 sem penalidade prévia
                    passos_sem_foco[i] = 0
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


def treinar_ppo_bc_hibrido(
    dataset_path: str = "fase5/dados/dataset_wasd_tatico_36.pt",
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_wasd_tatico.pt",
    ckpt_saida:   str = "checkpoints_vla/vla_fase5_ppo_bc.pt",
    iteracoes:    int = 15,
    passos_ep:    int = 50,
    lr:         float = 3e-5,
    gamma:      float = 0.98,
    lambda_bc:  float = 1.0,
    seed:         int = 42
):
    print("=" * 80)
    print(" [FASE 5.4] PPO-BC HÍBRIDO COM ANCORAGEM CAUSAL (70% BC / 30% RL)")
    print(f"    Dataset Base    : {dataset_path}")
    print(f"    Checkpoint Base : {ckpt_entrada}")
    print(f"    Checkpoint Fim  : {ckpt_saida}")
    print(f"    Iterações       : {iteracoes} | Passos/Ep: {passos_ep} | LR: {lr} | Lambda BC: {lambda_bc}")
    print("=" * 80)

    # 1. Carrega modelo VLA e Tokenizer
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)

    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    pol = PoliticaRaciocinioLoop(None, amostrar=True, device=dev, vla=vla, loops_pensamento=3, num_acoes=36)

    # Restaura pesos do checkpoint treinado da Época 12
    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            state_filtrado = {k: v for k, v in ckpt_data["treinaveis"].items()}
            msg = vla.load_state_dict(state_filtrado, strict=False)
            print(f"[VLA] Pesos restaurados com sucesso de '{ckpt_entrada}' ({len(state_filtrado)} tensores).")

    vla.to(dev)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(treinaveis, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=iteracoes, eta_min=lr * 0.1)
    loss_ce = nn.CrossEntropyLoss()

    # 2. Carrega Dataset Offline para Ancoragem Causal (70%)
    dados_offline = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset BC] {len(dados_offline)} amostras carregadas para ancoragem contínua.")

    todos_sv_bc    = torch.stack([d["sv"] for d in dados_offline]).to(dev)
    todas_acoes_bc = torch.tensor([int(d["acao_otima"]) for d in dados_offline], dtype=torch.long, device=dev)
    prompts_bc     = [d.get("prompt", "Objetivo: vá até o bloco amarelo [Etapa 1/2]") for d in dados_offline]
    enc_bc         = tokenizer(prompts_bc, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    tokens_bc      = enc_bc["input_ids"].to(dev)
    num_bc_total   = len(dados_offline)

    N = get("/lote/info")["envs"]
    print(f"[Simulador] Conectado a {N} ambientes paralelos.")
    print(f"[Otimizador] {len(treinaveis)} tensores ativos para otimização conjunta PPO + BC.")
    print("--- Iniciando Iterações PPO-BC Híbridas ---")

    t_ini = time.time()
    melhor_score = -999.0
    ckpt_melhor = ckpt_saida.replace(".pt", "_melhor.pt")



    for it in range(1, iteracoes + 1):
        lr_atual = otimizador.param_groups[0]["lr"]
        # 1. Coleta Rollout RL nos 8 robôs (30% do sinal de treino)
        vla.eval()
        pol.amostrar = True
        tarefas, blocos_tarefas = gerar_tarefas_busca_ativa(N, seed=seed + it * 17, nivel=2)

        (SV_T, IDS_T, IDX_T, R_T, VIVO_T), met = rollout_rl_wasd(pol, tarefas, blocos=blocos_tarefas, passos=passos_ep)

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

        # Prepara tensores RL
        T_len, N_len = VIVO_T.shape
        b_sv_rl  = torch.tensor(SV_T.reshape(T_len * N_len, -1)[mascara], dtype=torch.float32, device=dev)
        b_ids_rl = torch.tensor(IDS_T.reshape(T_len * N_len, -1)[mascara], dtype=torch.long, device=dev)
        b_idx_rl = torch.tensor(IDX_T.reshape(-1)[mascara], dtype=torch.long, device=dev)

        # 3. Amostra 2.33x mais amostras do Buffer BC Especialista Offline (70% Ancoragem Causal Limpa)
        num_rl = len(b_idx_rl)
        num_bc_amostra = min(int(num_rl * 2.33), num_bc_total)
        idx_bc_rand = torch.randperm(num_bc_total, device=dev)[:num_bc_amostra]

        b_sv_bc_sub  = todos_sv_bc[idx_bc_rand]
        b_ids_bc_sub = tokens_bc[idx_bc_rand]
        b_idx_bc_sub = todas_acoes_bc[idx_bc_rand]

        # 4. Atualização Conjunta em Minilotes (Minilote = 16)
        vla.train()
        torch.cuda.empty_cache()
        minilote = 16
        indices_rl = torch.randperm(num_rl)
        otimizador.zero_grad()

        total_pg_loss = 0.0
        total_bc_loss = 0.0
        entropia_media = 0.0

        # Loop de minilotes RL
        for mb in range(0, num_rl, minilote):
            mb_idx = indices_rl[mb:mb + minilote]
            mb_sv = b_sv_rl[mb_idx]
            mb_ids = b_ids_rl[mb_idx]
            mb_a = b_idx_rl[mb_idx]
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

                pg_loss = -(log_probs * mb_adv).mean()
                loss_rl = (pg_loss - 0.005 * entropia) * (len(mb_idx) / max(1, num_rl))

            loss_rl.backward()
            total_pg_loss += pg_loss.item() * len(mb_idx)
            entropia_media += entropia.item() * len(mb_idx)

        # Loop de minilotes BC (Ancoragem Causal Offline)
        indices_bc = torch.randperm(num_bc_amostra)
        for mb in range(0, num_bc_amostra, minilote):
            mb_idx = indices_bc[mb:mb + minilote]
            mb_sv = b_sv_bc_sub[mb_idx]
            mb_ids = b_ids_bc_sub[mb_idx]
            mb_alvo = b_idx_bc_sub[mb_idx]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = vla.state_encoder(mb_sv)
                t_embeds = vla.qwen_model.get_input_embeddings()(mb_ids)
                inputs_embeds = torch.cat([s_embeds, t_embeds], dim=1)

                outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                last_hidden = outputs.last_hidden_state[:, -1, :]
                logits = vla.cabeca_acao_36(last_hidden)

                LOGIT_CLIP = 3.0
                logits = torch.tanh(logits / LOGIT_CLIP) * LOGIT_CLIP

                bc_loss = loss_ce(logits, mb_alvo)
                loss_bc_step = (lambda_bc * bc_loss) * (len(mb_idx) / max(1, num_bc_amostra))

            loss_bc_step.backward()
            total_bc_loss += bc_loss.item() * len(mb_idx)

        torch.nn.utils.clip_grad_norm_(treinaveis, max_norm=1.0)
        otimizador.step()
        scheduler.step()

        pg_loss_final = total_pg_loss / max(1, num_rl)
        bc_loss_final = total_bc_loss / max(1, num_bc_amostra)
        ent_final     = entropia_media / max(1, num_rl)

        print(
            f"  Iteração {it:3d}/{iteracoes} (lr={lr_atual:.1e}) | "
            f"Recompensa: {recompensa_media:+6.2f} | "
            f"Submeta 1: {taxa_sub1:5.1f}% | "
            f"Sucesso Total: {taxa_tot:5.1f}% | "
            f"PG Loss: {pg_loss_final:+.4f} | "
            f"BC Loss: {bc_loss_final:.4f} | "
            f"Entropia: {ent_final:.3f}",
            flush=True
        )

        # Salva o checkpoint atualizado a cada iteração
        os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
        tensores_treinaveis = {
            k: v for k, v in vla.state_dict().items()
            if any(t in k for t in ["lora_", "state_encoder", "cabeca_acao_36"])
        }
        ckpt_dict = {
            "treinaveis": tensores_treinaveis,
            "iteracao": it,
            "taxa_submeta1": taxa_sub1,
            "taxa_total": taxa_tot,
            "recompensa": recompensa_media,
            "num_acoes": 36
        }
        torch.save(ckpt_dict, ckpt_saida)

        # Salva o melhor checkpoint histórico
        score_atual = taxa_tot * 2.0 + taxa_sub1 + recompensa_media * 0.1
        if score_atual > melhor_score:
            melhor_score = score_atual
            torch.save(ckpt_dict, ckpt_melhor)

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Treinamento PPO-BC Híbrido ({iteracoes} iterações) concluído em {duracao:.1f}s.")
    print(f"[OK] Checkpoint final salvo em: {ckpt_saida}")
    print(f"[OK] Melhor checkpoint salvo em: {ckpt_melhor}")

    # Dispara automaticamente o benchmark TopView 2D ao final
    print("\n--- Disparando Benchmark TopView Oficial (24 Episódios) ---")
    try:
        from fase5.avaliar_fase5_topview import avaliar_fase5
        avaliar_fase5(
            modelo_ckpt=ckpt_saida,
            num_lotes=3,
            passos_max=100,
            seed=42,
            amostrar=False
        )
    except Exception as e:
        print(f"[Benchmark Auto] Erro ao disparar benchmark: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default="fase5/dados/dataset_wasd_tatico_36.pt")
    ap.add_argument("--base",       default="checkpoints_vla/vla_fase5_ppo_bc.pt")
    ap.add_argument("--saida",      default="checkpoints_vla/vla_fase5_ppo_bc.pt")
    ap.add_argument("--iteracoes",  type=int,   default=100)
    ap.add_argument("--passos",     type=int,   default=50)
    ap.add_argument("--lr",         type=float, default=3e-5)
    ap.add_argument("--lambda-bc",  type=float, default=1.0)
    args = ap.parse_args()

    treinar_ppo_bc_hibrido(
        dataset_path=args.dataset,
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        iteracoes=args.iteracoes,
        passos_ep=args.passos,
        lr=args.lr,
        lambda_bc=args.lambda_bc
    )
