# coding=utf-8
"""
fase5/treinar_ppo_bc_hibrido.py — Treinamento PPO Multimodal Verdadeiro com Ancoragem BC Fatorada,
Shaping de Potencial Geométrico Não-Privilegiado e Currículo Progressivo (Fase 5.5).

Melhorias Algorítmicas e Arquiteturais:
  1. REWARD / HORIZONTE DE CRÉDITO:
     - R_total = R_visual + λ * [Φ(s') - Φ(s)] + R_terminal
     - com Φ(s) = -distância até a submeta atual (ΔΦ = dist_anterior - dist_atual)
     - λ padrão = 0.10 (proporcional ao passo físico de ~0.21m por tick, sem ofuscar a visão)
     - Decomposição explícita: RewardVisual, RewardPotential, RewardTerminal, RewardTotal
     - Cobertura de horizonte GAE: γ=0.98, λ_gae=0.95 para 100 passos
  2. CURRÍCULO PROGRESSIVO DA TAREFA:
     - ETAPA A: Pilar 1 apenas (curto: 4-6.5m, frontal: ±35°)
     - ETAPA B: Pilar 1 -> Pilar 2 (médio: 5.5-8m, moderado: ±60°/±75°)
     - ETAPA C: Tarefa completa com dispersão total (6.5-10m, ±75°/±110°)
     - Modos: 'auto' (avanço adaptativo por métricas de Submeta 1 e Sucesso) ou fixo ('A', 'B', 'C')
  3. QUALIDADE DO DATASET:
     - Integração com dataset_wasd_tatico_36_v2.pt (36 ações cobertas, sem colapso de strafe)
     - Ancoragem causal fatorada (Modo 6 classes + Yaw 9 classes)
  4. QUANTIDADE DE PPO / PIPELINE:
     - Orçamentos estruturados: warmup (λ_bc: 0.85->0.50), adaptação (0.50->0.25), refinamento (0.25->0.05), completo (0.85->0.15)
     - Checkpoints intermediários e melhor modelo preservados
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
    RAIO_CHEGADA_SUBMETA,
    BONUS_SUBMETA,
    BONUS_FINAL
)
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from modelo.lora_vla import aplicar_lora
from infra.gpu_utils import compactar_backbone
from infra.run_vla_agent import load_vla_agent
from fase5.acoes_taticas import (
    fatorar_indice_36,
    NUM_MODOS,
    NUM_YAW
)
from fase5.recompensa_visual import RastreadorVisualEpisodio
from fase5.curriculo_fase5 import CurriculoFase5

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def calcular_gae(
    R: np.ndarray,
    VIVO: np.ndarray,
    VAL: np.ndarray,
    gamma: float = 0.98,
    lmbda: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula Vantagem Generalizada (GAE) e Alvos de Retorno do Critic.
    R, VIVO, VAL: matrizes [T, N]
    Retorna:
      ADV: [T, N] (Vantagem GAE não-normalizada)
      TARGET_G: [T, N] (Retorno alvo para o Value Head: ADV + VAL)
    """
    T, N = R.shape
    ADV = np.zeros_like(R, dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t + 1 < T:
            v_prox = VAL[t + 1]
            vivo_prox = VIVO[t + 1]
        else:
            v_prox = np.zeros(N, dtype=np.float32)
            vivo_prox = np.zeros(N, dtype=np.float32)

        delta = R[t] + gamma * v_prox * vivo_prox - VAL[t]
        gae = delta + gamma * lmbda * vivo_prox * gae
        ADV[t] = gae * VIVO[t]

    TARGET_G = ADV + VAL
    return ADV, TARGET_G


def rollout_rl_wasd_ppo(
    pol: PoliticaRaciocinioLoop,
    tarefas: list,
    blocos: list = None,
    passos: int = 100,
    shaping_geometrico: bool = True,
    lambda_shaping: float = 0.10
) -> tuple[tuple, list]:
    """
    Coleta trajetórias completas armazenando log_probs da política antiga, predições do Critic V(s)
    e decomposição detalhada de recompensas (visual, potencial, terminal, total).
    """
    n = len(tarefas)
    rastreador = RastreadorVisualEpisodio(n)

    r = post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
    if blocos:
        post("/lote/colocar_bloco", {"blocos": blocos})

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
    avistou_alguma_vez = [False] * n

    d_ini = [
        math.hypot(
            t["estagios"][0]["alvo_abs"][0] - est[i]["x"],
            t["estagios"][0]["alvo_abs"][1] - est[i]["z"]
        ) for i, t in enumerate(tarefas)
    ]
    dant = list(d_ini)
    d_final = list(d_ini)

    rec_vis_acc = [0.0] * n
    rec_pot_acc = [0.0] * n
    rec_term_acc = [0.0] * n

    SV_L, IDS_L, VEMB_L = [], [], []
    MODO_L, YAW_L, LOGP_L, VAL_L = [], [], [], []
    R_L, VIVO_L = [], []

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

        # O modelo recebe apenas entradas padrão (pixels, vetor de estado relativo, prompt)
        # NUNCA coordenadas absolutas privilegiadas do alvo
        acoes = pol.agir(est, alvos_ativos_abs, obs, prompts=prompts_ativos, estagios=estagio_atual)
        for i in range(n):
            if not vivo[i]:
                acoes[i] = {"hold": [], "mouse": [0, 0], "duration_ms": 50}

        u = pol.ultimo
        sv_passo   = u["sv"]         # [N, 32]
        ids_passo  = u["ids"]        # [N, 48]
        vemb_passo = u["v_emb"]      # [N, 32, 896]
        modo_passo = u["idx_modo"]   # [N]
        yaw_passo  = u["idx_yaw"]    # [N]
        logp_passo = u["logp_old"]   # [N]
        val_passo  = u["val"]        # [N]
        u8_frames  = u["u8"]         # [N, H, W, 3]

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
            t_i = tarefas[i]
            est_idx = min(estagio_atual[i], len(t_i["estagios"]) - 1)
            alvo_atual = t_i["estagios"][est_idx]["alvo_abs"]
            cor_alvo = t_i["estagios"][est_idx]["cor"]

            dx = alvo_atual[0] - e["x"]
            dz = alvo_atual[1] - e["z"]
            d_atual = math.hypot(dx, dz)
            d_final[i] = d_atual

            frame_i = u8_frames[i] if u8_frames is not None and i < len(u8_frames) else None

            # Cálculo de Recompensa: R_total = R_visual + λ * [Φ(s') - Φ(s)] + R_terminal
            r_passo, info = rastreador.calcular_recompensa_passo(
                env_id=i,
                estado=e,
                frame_u8=frame_i,
                cor_alvo=cor_alvo,
                acao_exec=acoes[i],
                estagio_atual=estagio_atual[i],
                shaping_geometrico=shaping_geometrico,
                lambda_potencial=lambda_shaping,
                dist_atual=d_atual,
                dist_anterior=dant[i]
            )

            if info.get("visivel"):
                avistou_alguma_vez[i] = True

            rec_vis_acc[i] += info.get("rec_visual", 0.0)
            rec_pot_acc[i] += info.get("rec_potencial", 0.0)
            rec_term_acc[i] += info.get("rec_terminal", 0.0)

            rec[i] += r_passo
            dant[i] = d_atual

            if e.get("in_water") or e.get("in_lava"):
                vivo[i] = False
                continue

            # Verificação de Chegada em Submetas
            num_estagios_tarefa = len(t_i["estagios"])
            if d_atual <= RAIO_CHEGADA_SUBMETA:
                if estagio_atual[i] == 0:
                    if num_estagios_tarefa == 1:
                        # Tarefa de estágio único (ETAPA A) concluída com sucesso no Pilar 1!
                        rec[i] += BONUS_FINAL
                        rec_term_acc[i] += BONUS_FINAL
                        submeta_atingida[i] = True
                        concluiu_em[i] = p
                        vivo[i] = False
                    else:
                        # Tarefa de 2 estágios (ETAPA B/C): cruzou Pilar 1, avança para Pilar 2
                        rec[i] += BONUS_SUBMETA
                        rec_term_acc[i] += BONUS_SUBMETA
                        submeta_atingida[i] = True
                        estagio_atual[i] = 1
                        rastreador.cooldown_frenagem[i] = 4
                        rastreador.reset_ambiente(i)
                        novo_alvo = t_i["estagios"][1]["alvo_abs"]
                        # Reseta potencial para o novo alvo evitando descontinuidade
                        dant[i] = math.hypot(novo_alvo[0] - e["x"], novo_alvo[1] - e["z"])

                elif estagio_atual[i] == 1:
                    # Alcançou Pilar 2 (conclusão total)
                    rec[i] += BONUS_FINAL
                    rec_term_acc[i] += BONUS_FINAL
                    concluiu_em[i] = p
                    vivo[i] = False

        SV_L.append(sv_passo)
        IDS_L.append(ids_passo)
        VEMB_L.append(vemb_passo)
        MODO_L.append(modo_passo)
        YAW_L.append(yaw_passo)
        LOGP_L.append(logp_passo)
        VAL_L.append(val_passo)
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
            "estagio_final": estagio_atual[i],
            "avistou": avistou_alguma_vez[i],
            "dist_ini": d_ini[i],
            "dist_fim": d_final[i],
            "rec_vis": rec_vis_acc[i],
            "rec_pot": rec_pot_acc[i],
            "rec_term": rec_term_acc[i],
            "rec_total": rec_vis_acc[i] + rec_pot_acc[i] + rec_term_acc[i]
        })

    SV_T   = np.array(SV_L)    # [T, N, 32]
    IDS_T  = np.array(IDS_L)   # [T, N, 48]
    VEMB_T = np.array(VEMB_L)  # [T, N, 32, 896]
    MODO_T = np.array(MODO_L)  # [T, N]
    YAW_T  = np.array(YAW_L)   # [T, N]
    LOGP_T = np.array(LOGP_L)  # [T, N]
    VAL_T  = np.array(VAL_L)   # [T, N]
    R_T    = np.array(R_L)     # [T, N]
    VIVO_T = np.array(VIVO_L)  # [T, N]

    return (SV_T, IDS_T, VEMB_T, MODO_T, YAW_T, LOGP_T, VAL_T, R_T, VIVO_T), met


def treinar_ppo_bc_hibrido(
    dataset_path: str = "fase5/dados/dataset_wasd_tatico_36_v2.pt",
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_ppo_bc.pt",
    ckpt_saida:   str = "checkpoints_vla/vla_fase5_ppo_bc.pt",
    iteracoes:    int = 100,
    passos_ep:    int = 100,
    lr:         float = 3e-5,
    gamma:      float = 0.98,
    gae_lambda: float = 0.95,
    ppo_epochs:   int = 3,
    clip_eps:   float = 0.2,
    lambda_shaping: float = 0.10,
    curriculo_estagio: str = "auto",
    criterio_a: float = 0.35,
    criterio_b: float = 0.20,
    fase_treino: str = "completo",
    salvar_cada: int = 20,
    seed:         int = 42
):
    print("=" * 80)
    print(" [FASE 5.5] PPO MULTIMODAL VERDADEIRO + CRITIC GAE + SHAPING POTENCIAL + CURRÍCULO")
    print(f"    Dataset Base      : {dataset_path}")
    print(f"    Checkpoint Base   : {ckpt_entrada}")
    print(f"    Checkpoint Fim    : {ckpt_saida}")
    print(f"    Iterações         : {iteracoes} | Passos/Ep: {passos_ep} | PPO Epochs: {ppo_epochs} | Clip: {clip_eps} | LR: {lr}")
    print(f"    GAE: γ={gamma}, λ={gae_lambda} | Shaping: λ_pot={lambda_shaping} (Φ=-dist)")
    print(f"    Currículo         : Modo={curriculo_estagio} (Critério A={criterio_a*100:.0f}%, B={criterio_b*100:.0f}%)")
    print(f"    Fase de Treino    : {fase_treino}")
    print("=" * 80)

    # 1. Inicializa Gerenciador de Currículo
    curriculo = CurriculoFase5(
        modo_estagio=curriculo_estagio,
        criterio_a=criterio_a,
        criterio_b=criterio_b
    )

    # 2. Carrega modelo VLA e Tokenizer
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)

    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    pol = PoliticaRaciocinioLoop(None, amostrar=True, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)

    # Restaura pesos do checkpoint base
    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            state_filtrado = {k: v for k, v in ckpt_data["treinaveis"].items()}
            vla.load_state_dict(state_filtrado, strict=False)
            print(f"[VLA] Pesos restaurados com sucesso de '{ckpt_entrada}' ({len(state_filtrado)} tensores).")

    vla.to(dev)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(treinaveis, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=iteracoes, eta_min=lr * 0.1)
    loss_ce = nn.CrossEntropyLoss()

    # 3. Carrega Dataset Offline para Ancoragem Causal Fatorada
    if not os.path.exists(dataset_path):
        caminho_alt = "fase5/dados/dataset_wasd_tatico_36.pt"
        print(f"[Aviso] Dataset {dataset_path} não encontrado, usando {caminho_alt}")
        dataset_path = caminho_alt

    dados_offline = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset BC] {len(dados_offline)} amostras carregadas para ancoragem fatorada.")

    todos_sv_bc = torch.stack([d["sv"] for d in dados_offline]).to(dev)
    todas_acoes_36 = [int(d["acao_otima"]) for d in dados_offline]

    todos_modos_bc = torch.tensor([fatorar_indice_36(a)[0] for a in todas_acoes_36], dtype=torch.long, device=dev)
    todos_yaws_bc  = torch.tensor([fatorar_indice_36(a)[1] for a in todas_acoes_36], dtype=torch.long, device=dev)

    prompts_bc = [d.get("prompt", "Objetivo: vá até o bloco amarelo [Etapa 1/2]") for d in dados_offline]
    enc_bc     = tokenizer(prompts_bc, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    tokens_bc  = enc_bc["input_ids"].to(dev)
    num_bc_total = len(dados_offline)

    N = get("/lote/info")["envs"]
    print(f"[Simulador] Conectado a {N} ambientes paralelos.")
    print(f"[Otimizador] {len(treinaveis)} tensores ativos para otimização conjunta PPO + Value + BC.")
    print("--- Iniciando Iterações PPO-BC Híbridas com Currículo e Shaping ---")

    t_ini = time.time()
    melhor_score = -999.0
    ckpt_melhor = ckpt_saida.replace(".pt", "_melhor.pt")

    for it in range(1, iteracoes + 1):
        lr_atual = otimizador.param_groups[0]["lr"]

        # Escalonamento curricular de BC conforme a fase de treino
        progresso = (it - 1) / max(1, iteracoes - 1)
        if fase_treino == "warmup":
            lambda_bc_atual = float(0.50 + (0.85 - 0.50) * (1.0 - progresso))
        elif fase_treino == "adaptacao":
            lambda_bc_atual = float(0.25 + (0.50 - 0.25) * (1.0 - progresso))
        elif fase_treino == "refinamento":
            lambda_bc_atual = float(0.05 + (0.25 - 0.05) * (1.0 - progresso))
        else:  # completo (cosine annealing 0.85 -> 0.15)
            lambda_bc_atual = float(0.15 + (0.85 - 0.15) * 0.5 * (1.0 + math.cos(math.pi * progresso)))

        # 1. Coleta Rollout RL com Tarefas geradas pelo Currículo
        vla.eval()
        pol.amostrar = True
        tarefas, blocos_tarefas = curriculo.gerar_tarefas(N, seed=seed + it * 19)

        (SV_T, IDS_T, VEMB_T, MODO_T, YAW_T, LOGP_T, VAL_T, R_T, VIVO_T), met = rollout_rl_wasd_ppo(
            pol,
            tarefas,
            blocos=blocos_tarefas,
            passos=passos_ep,
            shaping_geometrico=True,
            lambda_shaping=lambda_shaping
        )

        taxa_sub1 = sum(m["submeta_ok"] for m in met) / len(met) * 100.0
        taxa_tot  = sum(m["concluiu"] for m in met) / len(met) * 100.0
        taxa_desc = sum(m["avistou"] for m in met) / len(met) * 100.0

        rec_tot_media = float(sum(m["rec_total"] for m in met) / max(1, len(met)))
        rec_vis_media = float(sum(m["rec_vis"] for m in met) / max(1, len(met)))
        rec_pot_media = float(sum(m["rec_pot"] for m in met) / max(1, len(met)))
        rec_term_media = float(sum(m["rec_term"] for m in met) / max(1, len(met)))

        dist_ini_media = float(sum(m["dist_ini"] for m in met) / max(1, len(met)))
        dist_fim_media = float(sum(m["dist_fim"] for m in met) / max(1, len(met)))

        # Atualiza métricas no gerenciador de currículo
        avancou_curriculo, msg_curriculo = curriculo.atualizar_desempenho(
            taxa_sub1=taxa_sub1 / 100.0,
            taxa_sucesso=taxa_tot / 100.0,
            recompensa=rec_tot_media
        )
        if avancou_curriculo:
            print(f"\n{msg_curriculo}\n")

        # 2. Calcula Vantagem Generalizada (GAE) e Target de Retorno
        ADV_T, TARGET_G_T = calcular_gae(R_T, VIVO_T, VAL_T, gamma=gamma, lmbda=gae_lambda)

        mascara = VIVO_T.reshape(-1) > 0
        T_len, N_len = VIVO_T.shape
        num_v_tokens = VEMB_T.shape[2]
        hidden_dim = VEMB_T.shape[3]

        b_sv_rl       = torch.tensor(SV_T.reshape(T_len * N_len, -1)[mascara], dtype=torch.float32, device=dev)
        b_ids_rl      = torch.tensor(IDS_T.reshape(T_len * N_len, -1)[mascara], dtype=torch.long, device=dev)
        b_vemb_rl     = torch.tensor(VEMB_T.reshape(T_len * N_len, num_v_tokens, hidden_dim)[mascara], dtype=torch.bfloat16, device=dev)
        b_modo_rl     = torch.tensor(MODO_T.reshape(-1)[mascara], dtype=torch.long, device=dev)
        b_yaw_rl      = torch.tensor(YAW_T.reshape(-1)[mascara], dtype=torch.long, device=dev)
        b_logp_old_rl = torch.tensor(LOGP_T.reshape(-1)[mascara], dtype=torch.float32, device=dev)

        adv_ativo = ADV_T.reshape(-1)[mascara]
        adv_mean = float(adv_ativo.mean()) if len(adv_ativo) > 0 else 0.0
        adv_std = float(adv_ativo.std()) if len(adv_ativo) > 1 else 1.0
        adv_norm = (adv_ativo - adv_mean) / (max(adv_std, 1e-4) + 1e-8)
        b_adv_rl = torch.tensor(adv_norm, dtype=torch.float32, device=dev)

        b_target_g_rl = torch.tensor(TARGET_G_T.reshape(-1)[mascara], dtype=torch.float32, device=dev)

        num_rl = len(b_modo_rl)
        num_bc_amostra = min(int(num_rl * 2.0), num_bc_total)
        idx_bc_rand = torch.randperm(num_bc_total, device=dev)[:num_bc_amostra]

        b_sv_bc_sub   = todos_sv_bc[idx_bc_rand]
        b_ids_bc_sub  = tokens_bc[idx_bc_rand]
        b_modo_bc_sub = todos_modos_bc[idx_bc_rand]
        b_yaw_bc_sub  = todos_yaws_bc[idx_bc_rand]

        # 3. Otimização Conjunta com PPO Clipping Multi-Epoch
        vla.train()
        torch.cuda.empty_cache()
        minilote = 16

        total_ppo_loss = 0.0
        total_val_loss = 0.0
        total_bc_loss  = 0.0
        total_clip_frac = 0.0
        total_ent_modo = 0.0
        total_ent_yaw = 0.0
        total_ent_total = 0.0
        ent_totais_epoch0 = torch.zeros(num_rl, dtype=torch.float32)
        num_updates = 0

        for epoch in range(ppo_epochs):
            indices_rl = torch.randperm(num_rl)
            indices_bc = torch.randperm(num_bc_amostra)
            bc_ptr = 0

            for mb in range(0, num_rl, minilote):
                mb_idx = indices_rl[mb:mb + minilote]
                mb_sv = b_sv_rl[mb_idx]
                mb_ids = b_ids_rl[mb_idx]
                mb_vemb = b_vemb_rl[mb_idx]
                mb_m = b_modo_rl[mb_idx]
                mb_y = b_yaw_rl[mb_idx]
                mb_logp_old = b_logp_old_rl[mb_idx]
                mb_adv = b_adv_rl[mb_idx]
                mb_target_g = b_target_g_rl[mb_idx]

                otimizador.zero_grad()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s_embeds = vla.state_encoder(mb_sv)
                    t_embeds = vla.qwen_model.get_input_embeddings()(mb_ids)
                    inputs_embeds = torch.cat([mb_vemb, s_embeds, t_embeds], dim=1)

                    outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                    last_hidden = outputs.last_hidden_state[:, -1, :]

                    lg_modo = vla.cabeca_modo(last_hidden)
                    lg_yaw  = vla.cabeca_yaw(last_hidden)
                    v_pred  = vla.cabeca_valor(last_hidden).squeeze(-1)

                    LOGIT_CLIP = 3.0
                    lg_modo = torch.tanh(lg_modo / LOGIT_CLIP) * LOGIT_CLIP
                    lg_yaw  = torch.tanh(lg_yaw / LOGIT_CLIP) * LOGIT_CLIP

                    dist_modo = torch.distributions.Categorical(logits=lg_modo.to(torch.float32))
                    dist_yaw  = torch.distributions.Categorical(logits=lg_yaw.to(torch.float32))

                    logp_modo = dist_modo.log_prob(mb_m)
                    logp_yaw  = dist_yaw.log_prob(mb_y)
                    logp_total = logp_modo + logp_yaw

                    ent_m_vec = dist_modo.entropy()
                    ent_y_vec = dist_yaw.entropy()
                    ent_tot_vec = ent_m_vec + ent_y_vec

                    ent_modo = ent_m_vec.mean()
                    ent_yaw  = ent_y_vec.mean()
                    ent_total = ent_tot_vec.mean()

                    if epoch == 0:
                        ent_totais_epoch0[mb_idx] = ent_tot_vec.detach().cpu().to(torch.float32)

                    # PPO Ratio e Surrogate Clipping Loss
                    log_ratio = torch.clamp(logp_total - mb_logp_old, -10.0, 10.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()

                    # Value Function Loss (Critic MSE)
                    val_loss = 0.5 * nn.functional.mse_loss(v_pred.to(torch.float32), mb_target_g)

                    loss_rl = ppo_loss + 0.5 * val_loss - 0.005 * ent_total

                loss_rl.backward()

                # Minilote BC Supervisionado Fatorado
                if bc_ptr < num_bc_amostra:
                    mb_bc_idx = indices_bc[bc_ptr:bc_ptr + minilote]
                    bc_ptr += minilote
                    mb_sv_bc = b_sv_bc_sub[mb_bc_idx]
                    mb_ids_bc = b_ids_bc_sub[mb_bc_idx]
                    mb_m_bc = b_modo_bc_sub[mb_bc_idx]
                    mb_y_bc = b_yaw_bc_sub[mb_bc_idx]

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s_embeds_bc = vla.state_encoder(mb_sv_bc)
                        t_embeds_bc = vla.qwen_model.get_input_embeddings()(mb_ids_bc)
                        inputs_embeds_bc = torch.cat([s_embeds_bc, t_embeds_bc], dim=1)

                        outputs_bc = vla.qwen_model(inputs_embeds=inputs_embeds_bc)
                        last_hidden_bc = outputs_bc.last_hidden_state[:, -1, :]

                        lg_modo_bc = torch.tanh(vla.cabeca_modo(last_hidden_bc) / LOGIT_CLIP) * LOGIT_CLIP
                        lg_yaw_bc  = torch.tanh(vla.cabeca_yaw(last_hidden_bc) / LOGIT_CLIP) * LOGIT_CLIP

                        bc_loss_m = loss_ce(lg_modo_bc.to(torch.float32), mb_m_bc)
                        bc_loss_y = loss_ce(lg_yaw_bc.to(torch.float32), mb_y_bc)
                        bc_loss = bc_loss_m + bc_loss_y
                        loss_bc_step = lambda_bc_atual * bc_loss

                    loss_bc_step.backward()
                    total_bc_loss += bc_loss.item()

                torch.nn.utils.clip_grad_norm_(treinaveis, max_norm=1.0)
                otimizador.step()

                total_ppo_loss += ppo_loss.item()
                total_val_loss += val_loss.item()
                total_clip_frac += clip_frac.item()
                total_ent_modo += ent_modo.item()
                total_ent_yaw += ent_yaw.item()
                total_ent_total += ent_total.item()
                num_updates += 1

        scheduler.step()

        ppo_loss_final  = total_ppo_loss / max(1, num_updates)
        val_loss_final  = total_val_loss / max(1, num_updates)
        bc_loss_final   = total_bc_loss / max(1, num_updates)
        clip_frac_final = (total_clip_frac / max(1, num_updates)) * 100.0
        ent_modo_final  = total_ent_modo / max(1, num_updates)
        ent_yaw_final   = total_ent_yaw / max(1, num_updates)
        ent_total_final = total_ent_total / max(1, num_updates)

        all_ent_np = ent_totais_epoch0.numpy() if isinstance(ent_totais_epoch0, torch.Tensor) else np.array([])
        if len(all_ent_np) > 0 and len(all_ent_np) == len(adv_ativo):
            q75 = float(np.percentile(all_ent_np, 75))
            high_ent_pct = float((all_ent_np >= q75).mean() * 100.0)
            high_ent_mask = all_ent_np >= q75
            low_ent_mask = ~high_ent_mask

            adv_high = adv_ativo[high_ent_mask]
            adv_low  = adv_ativo[low_ent_mask]

            adv_high_mean = float(adv_high.mean()) if len(adv_high) > 0 else 0.0
            adv_high_std  = float(adv_high.std()) if len(adv_high) > 0 else 0.0
            adv_high_abs  = float(np.abs(adv_high).mean()) if len(adv_high) > 0 else 0.0

            adv_low_mean  = float(adv_low.mean()) if len(adv_low) > 0 else 0.0
            adv_low_std   = float(adv_low.std()) if len(adv_low) > 0 else 0.0
            adv_low_abs   = float(np.abs(adv_low).mean()) if len(adv_low) > 0 else 0.0
        else:
            q75 = 0.0
            high_ent_pct = 0.0
            adv_high_mean = adv_high_std = adv_high_abs = 0.0
            adv_low_mean = adv_low_std = adv_low_abs = 0.0

        cur_est = curriculo.estagio_atual
        print(
            f"  Iteração {it:2d}/{iteracoes} [ETAPA {cur_est}] (lr={lr_atual:.1e}, λ_bc={lambda_bc_atual:.2f}) | "
            f"Rec: Tot={rec_tot_media:+5.2f} (Vis={rec_vis_media:+5.2f}, Pot={rec_pot_media:+5.2f}, Term={rec_term_media:+5.2f}) | "
            f"Sub1: {taxa_sub1:5.1f}% | Suc: {taxa_tot:5.1f}% | Desc: {taxa_desc:5.1f}% | Dist: {dist_ini_media:.1f}m->{dist_fim_media:.1f}m | "
            f"Adv: mean={adv_mean:+6.3f}, std={adv_std:.3f} | "
            f"Adv[HighH]: mean={adv_high_mean:+6.3f}, |adv|={adv_high_abs:.3f} | "
            f"Adv[LowH]: mean={adv_low_mean:+6.3f}, |adv|={adv_low_abs:.3f} | "
            f"Ent: m={ent_modo_final:.2f}, y={ent_yaw_final:.2f}, tot={ent_total_final:.2f} (q75={q75:.2f}) | "
            f"PPO: {ppo_loss_final:+.4f} | Val: {val_loss_final:.4f} | BC: {bc_loss_final:.4f} | Clip: {clip_frac_final:4.1f}%"
        )

        # Salva checkpoint com suporte às cabeças fatoradas e critic
        os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
        tensores_treinaveis = {
            k: v for k, v in vla.state_dict().items()
            if any(t in k for t in ["lora_", "state_encoder", "cabeca_modo", "cabeca_yaw", "cabeca_valor", "cabeca_acao_36"])
        }
        ckpt_dict = {
            "treinaveis": tensores_treinaveis,
            "iteracao": it,
            "estagio_curriculo": cur_est,
            "taxa_submeta1": taxa_sub1,
            "taxa_total": taxa_tot,
            "taxa_descoberta": taxa_desc,
            "recompensa": rec_tot_media,
            "recompensa_vis": rec_vis_media,
            "recompensa_pot": rec_pot_media,
            "recompensa_term": rec_term_media,
            "dist_ini_media": dist_ini_media,
            "dist_fim_media": dist_fim_media,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "adv_high_mean": adv_high_mean,
            "adv_high_std": adv_high_std,
            "adv_high_abs": adv_high_abs,
            "adv_low_mean": adv_low_mean,
            "adv_low_std": adv_low_std,
            "adv_low_abs": adv_low_abs,
            "ent_mode": ent_modo_final,
            "ent_yaw": ent_yaw_final,
            "ent_total": ent_total_final,
            "high_ent_pct": high_ent_pct,
            "q75_ent": q75,
            "ppo_loss": ppo_loss_final,
            "val_loss": val_loss_final,
            "bc_loss": bc_loss_final,
            "clip_frac": clip_frac_final,
            "fatorada": True
        }
        torch.save(ckpt_dict, ckpt_saida)

        # Checkpoints intermediários periódicos
        if salvar_cada > 0 and it % salvar_cada == 0:
            ckpt_periodico = ckpt_saida.replace(".pt", f"_it{it}.pt")
            torch.save(ckpt_dict, ckpt_periodico)
            print(f"    [CHECKPOINT] Snapshot periódico salvo: {ckpt_periodico}")

        # Avalia melhor modelo pelo score composto
        score_atual = taxa_tot * 3.0 + taxa_sub1 * 1.5 + rec_tot_media * 0.1
        if score_atual > melhor_score:
            melhor_score = score_atual
            torch.save(ckpt_dict, ckpt_melhor)

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Treinamento PPO-BC Híbrido com Currículo e Shaping ({iteracoes} iterações) concluído em {duracao:.1f}s.")
    print(f"[OK] Checkpoint final salvo em: {ckpt_saida}")
    print(f"[OK] Melhor checkpoint salvo em: {ckpt_melhor}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treinamento PPO-BC Híbrido com Shaping e Currículo (Fase 5.5)")
    ap.add_argument("--dataset",           default="fase5/dados/dataset_wasd_tatico_36_v2.pt", help="Caminho do dataset ancorado")
    ap.add_argument("--base",              default="checkpoints_vla/vla_fase5_ppo_bc.pt", help="Checkpoint base de entrada")
    ap.add_argument("--saida",             default="checkpoints_vla/vla_fase5_ppo_bc.pt", help="Checkpoint de saída")
    ap.add_argument("--iteracoes",         type=int,   default=100, help="Número de iterações PPO")
    ap.add_argument("--passos",            type=int,   default=100, help="Número máximo de passos por episódio")
    ap.add_argument("--lr",                type=float, default=3e-5, help="Taxa de aprendizado")
    ap.add_argument("--gamma",             type=float, default=0.98, help="Fator de desconto temporal γ")
    ap.add_argument("--gae-lambda",        type=float, default=0.95, help="Parâmetro λ do GAE")
    ap.add_argument("--ppo-epochs",        type=int,   default=3,    help="Épocas PPO por iteração")
    ap.add_argument("--clip-eps",          type=float, default=0.2,  help="Epsilon de clipping PPO")
    ap.add_argument("--lambda-shaping",    type=float, default=0.10, help="Peso λ do shaping de potencial físico Φ=-dist")
    ap.add_argument("--curriculo-estagio", type=str,   default="auto", choices=["auto", "A", "B", "C"], help="Estágio do currículo")
    ap.add_argument("--criterio-a",        type=float, default=0.35, help="Taxa Submeta 1 para avançar de A para B")
    ap.add_argument("--criterio-b",        type=float, default=0.20, help="Taxa Sucesso para avançar de B para C")
    ap.add_argument("--fase-treino",       type=str,   default="completo", choices=["completo", "warmup", "adaptacao", "refinamento"])
    ap.add_argument("--salvar-cada",       type=int,   default=20,   help="Intervalo de iterações para checkpoints intermediários")
    ap.add_argument("--seed",              type=int,   default=42,   help="Semente aleatória")
    args = ap.parse_args()

    treinar_ppo_bc_hibrido(
        dataset_path=args.dataset,
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        iteracoes=args.iteracoes,
        passos_ep=args.passos,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        lambda_shaping=args.lambda_shaping,
        curriculo_estagio=args.curriculo_estagio,
        criterio_a=args.criterio_a,
        criterio_b=args.criterio_b,
        fase_treino=args.fase_treino,
        salvar_cada=args.salvar_cada,
        seed=args.seed
    )
