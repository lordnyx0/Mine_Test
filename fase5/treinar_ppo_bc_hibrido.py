# coding=utf-8
"""
fase5/treinar_ppo_bc_hibrido.py — Treinamento PPO-BC Híbrido Multimodal (Fase 5.5).

Pipeline com Percepção Visual 4D/3D Calibrada + Shaping Potencial Físico + Currículo Progressivo:
  - Espaço de Ações: Fatorado Canônico (6 Modos x 9 Bins de Yaw).
  - Recompensa Visual: Busca ativa e centralização condicionada ao avanço motor (W / W+A / W+D).
  - Penalidade de Fixação Estacionária: Desincentiva permanecer parado olhando para o alvo.
  - Shaping Não-Privilegiado: R_total = R_visual + λ_pot * [Φ(s') - Φ(s)] + R_terminal.
  - Ancoragem BC Moderada: λ_bc inicia em 0.40 com decaimento suave até 0.10.
  - Suporte a Mascaramento Dinâmico no Treino: Desabilita atrator degenerado 'alinhar' quando centralizado.
  - Telemetria Completa de Modos: % Sprint, Strafe, Pulo, Recuar, Alinhar, W, Giro, StatFoc.
  - Currículo com Streak Consecutivo: Requer estabilidade de 3 iterações consecutivas para avançar.
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
from politica.cerebro import PoliticaCerebroVLA
from modelo.lora_vla import aplicar_lora
from infra.gpu_utils import compactar_backbone
from infra.run_vla_agent import load_vla_agent
from fase5.acoes_taticas import (
    fatorar_indice_36,
    MODOS,
    NUM_MODOS,
    NUM_YAW
)
from fase5.recompensa_visual import RastreadorVisualEpisodio, detectar_alvo_no_frame
from fase5.curriculo_fase5 import CurriculoFase5

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    vivos: np.ndarray,
    gamma: float = 0.98,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Computa Vantagens GAE e Retornos Normalizados."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = np.zeros(N, dtype=np.float32)
            next_non_terminal = np.zeros(N, dtype=np.float32)
        else:
            next_value = values[t + 1]
            next_non_terminal = vivos[t + 1].astype(np.float32)

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae * vivos[t].astype(np.float32)

    returns = advantages + values
    return advantages, returns


def coletar_rollout_curriculo(
    pol: PoliticaRaciocinioLoop,
    curriculo: CurriculoFase5,
    passos: int = 100,
    shaping_geometrico: bool = True,
    lambda_shaping: float = 0.15,
    penalidade_fixacao: float = 0.08,
    mascarar_se_centralizado: bool = False,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, ...], list[dict], dict]:
    """Executa um episódio de rollout nos 8 ambientes com suporte a currículo progressivo."""
    n = 8
    tarefas, blocos = curriculo.gerar_tarefas(n, seed=seed)

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
    rastreador = RastreadorVisualEpisodio(n, penalidade_fixacao=penalidade_fixacao)

    estagio_atual = [0] * n
    submeta_atingida = [False] * n
    concluiu_em = [None] * n
    vivo = [True] * n
    avistou_alguma_vez = [False] * n

    d_ini = [math.hypot(t["estagios"][0]["alvo_abs"][0] - est[i]["x"], t["estagios"][0]["alvo_abs"][1] - est[i]["z"]) for i, t in enumerate(tarefas)]
    d_min = list(d_ini)
    d_final = list(d_ini)
    dant = list(d_ini)

    d_min_p = [[999.0 for _ in range(len(t["estagios"]))] for t in tarefas]
    d_ini_p = [[None for _ in range(len(t["estagios"]))] for t in tarefas]
    for i in range(n):
        d_ini_p[i][0] = d_ini[i]
        d_min_p[i][0] = d_ini[i]

    passos_w = [0] * n
    passos_giro = [0] * n
    passos_stat_foc = [0] * n
    contagem_modos_ep = {m: 0 for m in MODOS}
    total_passos_acoes = 0

    rec_vis_acc = [0.0] * n
    rec_pot_acc = [0.0] * n
    rec_term_acc = [0.0] * n
    rec_stat_pen_acc = [0.0] * n
    rec_centermove_acc = [0.0] * n

    passos_no_estagio = [0] * n
    max_estagios_lote = max(len(t["estagios"]) for t in tarefas)
    limite_passos_total = passos * max_estagios_lote

    SV_L, IDS_L, VEMB_L = [], [], []
    MODO_L, YAW_L, LOGP_L, VAL_L, R_L, VIVO_L = [], [], [], [], [], []
    u8_frames = None

    for p in range(limite_passos_total):
        prompts = [t["estagios"][estagio_atual[i]]["prompt_estagio"] for i, t in enumerate(tarefas)]
        alvos_abs = [t["estagios"][estagio_atual[i]]["alvo_abs"] for i, t in enumerate(tarefas)]

        # Mascaramento dinâmico no rollout para exploração ativa
        mascara_modo = None
        if mascarar_se_centralizado and p > 0 and u8_frames is not None:
            mascara_list = []
            for i in range(n):
                cor_alvo = tarefas[i]["estagios"][estagio_atual[i]]["cor"]
                frame_i = u8_frames[i, 0] if u8_frames.ndim == 5 else (u8_frames[i] if u8_frames.ndim == 4 else None)
                det = detectar_alvo_no_frame(frame_i, cor_alvo)
                if det["visivel"] and abs(det["centro_x"]) < 0.30:
                    mascara_list.append([True, False, False, False, False, False])
                else:
                    mascara_list.append([False, False, False, False, False, False])
            mascara_modo = torch.tensor(mascara_list, dtype=torch.bool, device=pol.device)

        acoes = pol.agir(est, alvos_abs, obs, prompts=prompts, estagios=estagio_atual, mascara_modo=mascara_modo)

        u = pol.ultimo
        sv_passo   = u["sv"]
        ids_passo  = u["ids"]
        vemb_passo = u["v_emb"]
        modo_passo = u["idx_modo"]
        yaw_passo  = u["idx_yaw"]
        logp_passo = u["logp_old"]
        val_passo  = u["val"]
        u8_frames  = u["u8"]
        vivo_passo = list(vivo)

        r = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = r["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        rec = [0.0] * n

        for i in range(n):
            if not vivo[i]:
                continue

            passos_no_estagio[i] += 1
            if passos_no_estagio[i] > passos:
                vivo[i] = False
                continue

            total_passos_acoes += 1
            m_idx = int(modo_passo[i])
            contagem_modos_ep[MODOS[m_idx]] += 1

            e = est[i]
            t_i = tarefas[i]
            est_idx = estagio_atual[i]
            alvo_atual = t_i["estagios"][est_idx]["alvo_abs"]
            cor_alvo = t_i["estagios"][est_idx]["cor"]

            dx = alvo_atual[0] - e["x"]
            dz = alvo_atual[1] - e["z"]
            d_atual = math.hypot(dx, dz)
            d_final[i] = d_atual
            if d_atual < d_min[i]:
                d_min[i] = d_atual
            for k_pilar, est_k in enumerate(t_i["estagios"]):
                d_k = math.hypot(est_k["alvo_abs"][0] - e["x"], est_k["alvo_abs"][1] - e["z"])
                if d_k < d_min_p[i][k_pilar]:
                    d_min_p[i][k_pilar] = d_k

            if u8_frames is not None and i < len(u8_frames):
                if u8_frames.ndim == 5:
                    frame_i = u8_frames[i, 0]
                elif u8_frames.ndim == 4:
                    frame_i = u8_frames[i]
                else:
                    frame_i = None
            else:
                frame_i = None

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

            if "W" in acoes[i].get("hold", []):
                passos_w[i] += 1

            dx_mouse = acoes[i].get("mouse", [0, 0])[0]
            if abs(dx_mouse) > 5:
                passos_giro[i] += 1

            if info.get("estacionario_focado"):
                passos_stat_foc[i] += 1

            rec_vis_acc[i] += info.get("rec_visual", 0.0)
            rec_pot_acc[i] += info.get("rec_potencial", 0.0)
            rec_term_acc[i] += info.get("rec_terminal", 0.0)
            rec_stat_pen_acc[i] += info.get("rec_stat_pen", 0.0)
            rec_centermove_acc[i] += info.get("rec_centermove", 0.0)

            rec[i] += r_passo
            dant[i] = d_atual

            if e.get("in_water") or e.get("in_lava"):
                vivo[i] = False
                continue

            num_estagios_tarefa = len(t_i["estagios"])
            if d_atual <= RAIO_CHEGADA_SUBMETA:
                prox_estagio = estagio_atual[i] + 1
                if prox_estagio < num_estagios_tarefa:
                    # Submeta intermediária concluída (P1 em 2 pilares, ou P1/P2 em 3 pilares)
                    rec[i] += BONUS_SUBMETA
                    rec_term_acc[i] += BONUS_SUBMETA
                    submeta_atingida[i] = True
                    estagio_atual[i] = prox_estagio
                    passos_no_estagio[i] = 0  # RESET DE PASSOS: Ganha orçamento dedicado para o próximo pilar!
                    if hasattr(pol, 'ativar_varredura'):
                        pol.ativar_varredura(i, passos_varredura=3)
                    rastreador.cooldown_frenagem[i] = 4
                    rastreador.reset_ambiente(i)
                    novo_alvo = t_i["estagios"][prox_estagio]["alvo_abs"]
                    dist_novo_alvo = math.hypot(novo_alvo[0] - e["x"], novo_alvo[1] - e["z"])
                    dant[i] = dist_novo_alvo
                    d_ini_p[i][prox_estagio] = dist_novo_alvo
                    d_min_p[i][prox_estagio] = dist_novo_alvo
                else:
                    # Meta Final de todos os pilares alcançada!
                    rec[i] += BONUS_FINAL
                    rec_term_acc[i] += BONUS_FINAL
                    submeta_atingida[i] = True
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
        ep_passos = concluiu_em[i] if concluiu_em[i] is not None else (p + 1)
        num_est = len(tarefas[i]["estagios"])
        sub1_ok = (estagio_atual[i] >= 1) or (concluiu_em[i] is not None)
        sub2_ok = (estagio_atual[i] >= 2) or (concluiu_em[i] is not None and num_est >= 2)
        sub3_ok = (concluiu_em[i] is not None and num_est >= 3)
        met.append({
            "submeta_ok": submeta_atingida[i],
            "sub1": sub1_ok,
            "sub2": sub2_ok,
            "sub3": sub3_ok,
            "concluiu": concluiu_em[i] is not None,
            "passos": ep_passos,
            "estagio_final": estagio_atual[i],
            "avistou": avistou_alguma_vez[i],
            "dist_ini": d_ini[i],
            "dist_min": d_min[i],
            "dist_fim": d_final[i],
            "dist_min_p": d_min_p[i],
            "dist_ini_p": d_ini_p[i],
            "pct_w": (passos_w[i] / max(1, ep_passos)) * 100.0,
            "pct_giro": (passos_giro[i] / max(1, ep_passos)) * 100.0,
            "pct_stat_foc": (passos_stat_foc[i] / max(1, ep_passos)) * 100.0,
            "rec_vis": rec_vis_acc[i],
            "rec_pot": rec_pot_acc[i],
            "rec_term": rec_term_acc[i],
            "rec_stat_pen": rec_stat_pen_acc[i],
            "rec_centermove": rec_centermove_acc[i],
            "rec_total": rec_vis_acc[i] + rec_pot_acc[i] + rec_term_acc[i]
        })

    modos_pct = {
        m: (contagem_modos_ep[m] / max(1, total_passos_acoes)) * 100.0
        for m in MODOS
    }

    SV_T   = np.nan_to_num(np.array(SV_L), nan=0.0)
    IDS_T  = np.array(IDS_L)
    VEMB_T = np.nan_to_num(np.array(VEMB_L), nan=0.0)
    MODO_T = np.array(MODO_L)
    YAW_T  = np.array(YAW_L)
    LOGP_T = np.nan_to_num(np.array(LOGP_L), nan=-5.0)
    VAL_T  = np.nan_to_num(np.array(VAL_L), nan=0.0)
    R_T    = np.nan_to_num(np.array(R_L), nan=0.0)
    VIVO_T = np.array(VIVO_L)

    return (SV_T, IDS_T, VEMB_T, MODO_T, YAW_T, LOGP_T, VAL_T, R_T, VIVO_T), met, modos_pct


def treinar_ppo_bc_hibrido(
    dataset_path: str = "fase5/dados/dataset_wasd_tatico_36_v3.pt",
    usar_cerebro: bool = True,
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_ppo_bc.pt",
    ckpt_saida:   str = "checkpoints_vla/vla_fase5_ppo_bc.pt",
    iteracoes:    int = 50,
    passos_ep:    int = 100,
    lr:         float = 3e-5,
    gamma:      float = 0.98,
    gae_lambda: float = 0.95,
    ppo_epochs:   int = 2,
    clip_eps:   float = 0.2,
    mini_batch_size: int = 16,
    lambda_shaping: float = 0.15,
    penalidade_fixacao: float = 0.08,
    lambda_bc_ini: float = 0.40,
    lambda_bc_fim: float = 0.10,
    mascarar_se_centralizado: bool = False,
    curriculo_estagio: str = "auto",
    estagio_inicial: Optional[str] = None,
    criterio_a: float = 0.35,
    criterio_b: float = 0.20,
    consecutivas_curriculo: int = 3,
    fase_treino: str = "completo",
    salvar_cada: int = 5,
    seed:         int = 42
):
    print("=" * 80, flush=True)
    print(" [FASE 5.5] PPO MULTIMODAL VERDADEIRO + CRITIC GAE + SHAPING POTENCIAL + CURRÍCULO", flush=True)
    print(f"    Dataset Base      : {dataset_path}", flush=True)
    print(f"    Checkpoint Base   : {ckpt_entrada}", flush=True)
    print(f"    Checkpoint Fim    : {ckpt_saida}", flush=True)
    print(f"    Iterações         : {iteracoes} | Passos/Ep: {passos_ep} | PPO Epochs: {ppo_epochs} | Mini-Batch: {mini_batch_size}", flush=True)
    print(f"    GAE: γ={gamma}, λ={gae_lambda} | Shaping: λ_pot={lambda_shaping} (Φ=-dist)", flush=True)
    print(f"    Penalidade Fixação: {penalidade_fixacao:.2f} | Centralização: Condicionada a W/A/D", flush=True)
    print(f"    Ablação Dinâmica  : {mascarar_se_centralizado} (Proíbe 'alinhar' quando centralizado)", flush=True)
    print(f"    Âncora BC         : λ_bc={lambda_bc_ini:.2f} -> {lambda_bc_fim:.2f} (Annealing Linear)", flush=True)
    crit_a_display = criterio_a * 100.0 if criterio_a <= 1.0 else criterio_a
    crit_b_display = criterio_b * 100.0 if criterio_b <= 1.0 else criterio_b
    print(f"    Currículo         : Modo={curriculo_estagio} (Critério A={crit_a_display:.0f}%, B={crit_b_display:.0f}%, Streak={consecutivas_curriculo}x)", flush=True)
    print("=" * 80, flush=True)

    curriculo = CurriculoFase5(
        modo_estagio=curriculo_estagio,
        estagio_inicial=estagio_inicial,
        criterio_a=criterio_a,
        criterio_b=criterio_b,
        consecutivas_necessarias=consecutivas_curriculo
    )

    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)

    vla.vision_encoder.eval()
    for p in vla.vision_encoder.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    pol_vla = PoliticaRaciocinioLoop(None, amostrar=True, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)
    if usar_cerebro:
        pol = PoliticaCerebroVLA(pol_vla)
        print("[CEREBRO] Cérebro Supervisor (~1-2 Hz) acoplado ao VLA reflexo (4 Hz).", flush=True)
    else:
        pol = pol_vla

    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            state_filtrado = {k: v for k, v in ckpt_data["treinaveis"].items()}
            vla.load_state_dict(state_filtrado, strict=False)
            print(f"[VLA] Pesos restaurados com sucesso de '{ckpt_entrada}' ({len(state_filtrado)} tensores).", flush=True)

    vla.to(dev)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(treinaveis, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=iteracoes, eta_min=lr * 0.1)
    loss_ce = nn.CrossEntropyLoss()

    if not os.path.exists(dataset_path):
        caminho_alt = "fase5/dados/dataset_wasd_tatico_36.pt"
        print(f"[Aviso] Dataset {dataset_path} não encontrado, usando {caminho_alt}", flush=True)
        dataset_path = caminho_alt

    dados_offline = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset BC] {len(dados_offline)} amostras carregadas para ancoragem fatorada.", flush=True)

    todos_sv_bc = torch.stack([d["sv"] for d in dados_offline]).to(dev)
    todas_acoes_36 = [int(d["acao_otima"]) for d in dados_offline]

    todos_modos_bc = torch.tensor([fatorar_indice_36(a)[0] for a in todas_acoes_36], dtype=torch.long, device=dev)
    todos_yaws_bc  = torch.tensor([fatorar_indice_36(a)[1] for a in todas_acoes_36], dtype=torch.long, device=dev)

    prompts_bc = [d.get("prompt", "Objetivo: vá até o bloco amarelo [Etapa 1/2]") for d in dados_offline]
    enc_bc     = tokenizer(prompts_bc, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    tokens_bc  = enc_bc["input_ids"].to(dev)

    gv_dummy_bc = torch.zeros((len(dados_offline), 4), dtype=torch.float32, device=dev)

    print("\n--- INICIANDO ITERAÇÕES PPO-BC MULTIMODAL ---", flush=True)
    melhor_score = (-1.0, -1.0, -999.0)  # (taxa_total, taxa_sub1, r_medio)

    for it in range(1, iteracoes + 1):
        t0_it = time.time()
        vla.eval()
        vla.vision_encoder.eval()
        torch.cuda.empty_cache()
        estagio_ativo_nome = curriculo.estagio_atual

        (SV_T, IDS_T, VEMB_T, MODO_T, YAW_T, LOGP_T, VAL_T, R_T, VIVO_T), met, modos_pct = coletar_rollout_curriculo(
            pol=pol,
            curriculo=curriculo,
            passos=passos_ep,
            shaping_geometrico=True,
            lambda_shaping=lambda_shaping,
            penalidade_fixacao=penalidade_fixacao,
            mascarar_se_centralizado=mascarar_se_centralizado,
            seed=seed + it * 17
        )

        taxa_sub1 = float(np.mean([m.get("sub1", m["submeta_ok"]) for m in met])) * 100.0
        taxa_sub2 = float(np.mean([m.get("sub2", False) for m in met])) * 100.0
        taxa_sub3 = float(np.mean([m.get("sub3", False) for m in met])) * 100.0
        taxa_total = float(np.mean([m["concluiu"] for m in met])) * 100.0
        taxa_desc = float(np.mean([m["avistou"] for m in met])) * 100.0
        taxa_w = float(np.mean([m["pct_w"] for m in met]))
        taxa_giro = float(np.mean([m["pct_giro"] for m in met]))
        taxa_statfoc = float(np.mean([m["pct_stat_foc"] for m in met]))

        d_ini_med = float(np.mean([m["dist_ini"] for m in met]))
        d_min_med = float(np.mean([m["dist_min"] for m in met]))
        d_fim_med = float(np.mean([m["dist_fim"] for m in met]))

        r_vis_med = float(np.mean([m["rec_vis"] for m in met]))
        r_pot_med = float(np.mean([m["rec_pot"] for m in met]))
        r_term_med = float(np.mean([m["rec_term"] for m in met]))
        r_medio = float(np.mean([m["rec_total"] for m in met]))

        avancou_curriculo, msg_curriculo = curriculo.atualizar_desempenho(
            taxa_sub1=taxa_sub1,
            taxa_sucesso=taxa_total,
            recompensa=r_medio
        )

        advantages, returns = compute_gae(R_T, VAL_T, VIVO_T, gamma=gamma, gae_lambda=gae_lambda)

        flat_vivos = VIVO_T.reshape(-1).astype(bool)
        if not flat_vivos.any():
            continue

        flat_adv = advantages.reshape(-1)[flat_vivos]
        adv_mean = float(np.mean(flat_adv))
        adv_std  = float(np.std(flat_adv) + 1e-8)
        norm_adv = np.clip((flat_adv - adv_mean) / adv_std, -4.0, 4.0)

        flat_sv   = SV_T.reshape(-1, SV_T.shape[-1])[flat_vivos]
        flat_ids  = IDS_T.reshape(-1, IDS_T.shape[-1])[flat_vivos]
        flat_vemb = VEMB_T.reshape(-1, VEMB_T.shape[-2], VEMB_T.shape[-1])[flat_vivos]
        flat_modo = MODO_T.reshape(-1)[flat_vivos]
        flat_yaw  = YAW_T.reshape(-1)[flat_vivos]
        flat_logp = LOGP_T.reshape(-1)[flat_vivos]
        flat_ret  = returns.reshape(-1)[flat_vivos]

        T_eff = len(flat_sv)
        b_sv   = torch.nan_to_num(torch.tensor(flat_sv, dtype=torch.float32, device=dev), 0.0)
        b_ids  = torch.tensor(flat_ids, dtype=torch.long, device=dev)
        b_vemb = torch.nan_to_num(torch.tensor(flat_vemb, dtype=torch.bfloat16, device=dev), 0.0)
        b_modo = torch.tensor(flat_modo, dtype=torch.long, device=dev)
        b_yaw  = torch.tensor(flat_yaw, dtype=torch.long, device=dev)
        b_logp = torch.nan_to_num(torch.tensor(flat_logp, dtype=torch.float32, device=dev), -5.0)
        b_adv  = torch.nan_to_num(torch.tensor(norm_adv, dtype=torch.float32, device=dev), 0.0)
        b_ret  = torch.nan_to_num(torch.tensor(flat_ret, dtype=torch.float32, device=dev), 0.0)
        b_gv   = torch.zeros((T_eff, 4), dtype=torch.float32, device=dev)

        vla.train()
        vla.vision_encoder.eval()

        progresso = (it - 1) / max(1, iteracoes - 1)
        peso_bc = lambda_bc_fim + (lambda_bc_ini - lambda_bc_fim) * (1.0 - progresso)

        clip_counts = []
        ent_modos, ent_yaws, ent_tots = [], [], []
        ppo_losses, val_losses, bc_losses = [], [], []
        passo_otimizador_executado = False

        for ep in range(ppo_epochs):
            indices = np.arange(T_eff)
            np.random.shuffle(indices)

            for start in range(0, T_eff, mini_batch_size):
                end = min(start + mini_batch_size, T_eff)
                mb_idx = indices[start:end]
                mb_len = len(mb_idx)

                otimizador.zero_grad()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg_modo, lg_yaw, values = pol.forward_pensamento(
                        pixel_tensor=None,
                        state_tensor=b_sv[mb_idx],
                        goal_tensor=b_gv[mb_idx],
                        input_ids=b_ids[mb_idx],
                        precomputed_v_emb=b_vemb[mb_idx]
                    )

                    lg_modo = torch.nan_to_num(lg_modo.float(), 0.0)
                    lg_yaw  = torch.nan_to_num(lg_yaw.float(), 0.0)

                    dist_m = torch.distributions.Categorical(logits=lg_modo)
                    dist_y = torch.distributions.Categorical(logits=lg_yaw)

                    new_logp = dist_m.log_prob(b_modo[mb_idx]) + dist_y.log_prob(b_yaw[mb_idx])
                    log_diff = torch.clamp(new_logp - b_logp[mb_idx], -10.0, 2.0)
                    ratio = torch.exp(log_diff)

                    surr1 = ratio * b_adv[mb_idx]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv[mb_idx]
                    loss_ppo = -torch.min(surr1, surr2).mean()

                    # Value Head Loss com garantia estrita de shapes equivalentes: [B] == [B]
                    val_pred = torch.nan_to_num(values.reshape(-1).float(), 0.0)
                    target_val = b_ret[mb_idx].reshape(-1)
                    loss_val = torch.clamp(nn.MSELoss()(val_pred, target_val), 0.0, 50.0)

                    em = dist_m.entropy().mean()
                    ey = dist_y.entropy().mean()
                    ent_modos.append(em.item())
                    ent_yaws.append(ey.item())
                    ent_tots.append((em + ey).item())
                    loss_ent = -0.01 * (em + ey)

                    loss_ppo_total = loss_ppo + 0.25 * loss_val + loss_ent

                if not torch.isnan(loss_ppo_total) and not torch.isinf(loss_ppo_total):
                    loss_ppo_total.backward()
                    ppo_losses.append(loss_ppo.item())
                    val_losses.append(loss_val.item())

                if peso_bc > 0:
                    idx_bc = torch.randint(0, len(dados_offline), (min(mb_len, 16),), device=dev)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg_m_bc, lg_y_bc, _ = pol.forward_pensamento(
                            pixel_tensor=None,
                            state_tensor=todos_sv_bc[idx_bc],
                            goal_tensor=gv_dummy_bc[idx_bc],
                            input_ids=tokens_bc[idx_bc],
                            precomputed_v_emb=None
                        )
                        lg_m_bc = torch.nan_to_num(lg_m_bc.float(), 0.0)
                        lg_y_bc = torch.nan_to_num(lg_y_bc.float(), 0.0)
                        loss_bc = 0.5 * (loss_ce(lg_m_bc, todos_modos_bc[idx_bc]) +
                                         loss_ce(lg_y_bc, todos_yaws_bc[idx_bc]))
                        loss_bc_weighted = peso_bc * loss_bc

                    if not torch.isnan(loss_bc_weighted) and not torch.isinf(loss_bc_weighted):
                        loss_bc_weighted.backward()
                        bc_losses.append(loss_bc.item())

                grad_ok = True
                for p in treinaveis:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            grad_ok = False
                            break

                if grad_ok:
                    nn.utils.clip_grad_norm_(treinaveis, max_norm=0.5)
                    otimizador.step()
                    passo_otimizador_executado = True
                else:
                    otimizador.zero_grad()

                with torch.no_grad():
                    clipped = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))
                    clip_counts.append(clipped.float().mean().item())

        if passo_otimizador_executado:
            scheduler.step()

        lr_atual = otimizador.param_groups[0]["lr"]
        taxa_clip = float(np.mean(clip_counts)) * 100.0 if clip_counts else 0.0
        ent_m = float(np.mean(ent_modos)) if ent_modos else 0.0
        ent_y = float(np.mean(ent_yaws)) if ent_yaws else 0.0
        ent_tot = float(np.mean(ent_tots)) if ent_tots else 0.0

        ppo_loss_val = float(np.mean(ppo_losses)) if ppo_losses else 0.0
        val_loss_val = float(np.mean(val_losses)) if val_losses else 0.0
        bc_loss_val  = float(np.mean(bc_losses)) if bc_losses else 0.0
        dt_it = time.time() - t0_it

        pct_strafe_tot = modos_pct.get("strafe_esq", 0.0) + modos_pct.get("strafe_dir", 0.0)
        cur_stat = curriculo.obter_status()
        num_pilares_ep = max(len(m.get("dist_min_p", [])) for m in met) if met else 1
        dist_pilares_str = " | ".join([f"P{j+1}={np.mean([m['dist_min_p'][j] for m in met if len(m.get('dist_min_p', []))>j and m['dist_min_p'][j]<900] or [0.0]):.1f}m" for j in range(num_pilares_ep)])

        if num_pilares_ep == 1:
            taxas_str = f"Sub1: {taxa_sub1:5.1f}% | Suc: {taxa_total:5.1f}%"
        elif num_pilares_ep == 2:
            taxas_str = f"Sub1: {taxa_sub1:5.1f}% | Sub2: {taxa_sub2:5.1f}% (Suc)"
        else:
            taxas_str = f"Sub1: {taxa_sub1:5.1f}% | Sub2: {taxa_sub2:5.1f}% | Sub3: {taxa_sub3:5.1f}% (Suc)"

        print(
            f"  Iteracao {it:02d}/{iteracoes:02d} [{estagio_ativo_nome}] ({dt_it:.0f}s, lr={lr_atual:.1e}, lambda_bc={peso_bc:.2f}) | "
            f"Rec: Tot={r_medio:+5.2f} (Vis={r_vis_med:+5.2f}, Pot={r_pot_med:+5.2f}, Term={r_term_med:+5.2f}) | "
            f"{taxas_str} | Desc: {taxa_desc:5.1f}% | "
            f"Dist: {dist_pilares_str} (min={d_min_med:.1f}m) | "
            f"W: {taxa_w:4.1f}% | Giro: {taxa_giro:4.1f}% | StatFoc: {taxa_statfoc:4.1f}%\n"
            f"    -> Modos: Sprint={modos_pct.get('sprint', 0.0):4.1f}% | Strafe={pct_strafe_tot:4.1f}% (Esq={modos_pct.get('strafe_esq', 0.0):4.1f}%, Dir={modos_pct.get('strafe_dir', 0.0):4.1f}%) | "
            f"Pulo={modos_pct.get('pulo', 0.0):4.1f}% | Recuar={modos_pct.get('recuar', 0.0):4.1f}% | Alinhar={modos_pct.get('alinhar', 0.0):4.1f}% | "
            f"PPO: {ppo_loss_val:+6.4f} | Val: {val_loss_val:6.4f} | BC: {bc_loss_val:6.4f} | Clip: {taxa_clip:4.1f}%\n"
            f"    -> Curriculo: [{cur_stat['estagio']}] Sub1={taxa_sub1:4.1f}% | Suc={taxa_total:4.1f}% | streak={cur_stat['streak']}/{cur_stat['consecutivas_necessarias']} | precisa={cur_stat['precisa_str']}",
            flush=True
        )
        if avancou_curriculo:
            print(f"    {msg_curriculo}", flush=True)

        os.makedirs(os.path.dirname(ckpt_saida) or ".", exist_ok=True)
        dados_salvar = {
            "iteracao": it,
            "estagio_curriculo": estagio_ativo_nome,
            "taxa_submeta1": taxa_sub1,
            "taxa_total": taxa_total,
            "taxa_descoberta": taxa_desc,
            "taxa_w": taxa_w,
            "taxa_giro": taxa_giro,
            "taxa_statfoc": taxa_statfoc,
            "dist_ini": d_ini_med,
            "dist_min": d_min_med,
            "dist_fim": d_fim_med,
            "modos_pct": modos_pct,
            "recompensa": r_medio,
            "recompensa_vis": r_vis_med,
            "recompensa_pot": r_pot_med,
            "recompensa_term": r_term_med,
            "val_loss": val_loss_val,
            "bc_loss": bc_loss_val,
            "ppo_loss": ppo_loss_val,
            "curriculo_status": cur_stat,
            "treinaveis": {k: v.cpu() for k, v in vla.state_dict().items() if any(p in k for p in ["lora_", "cabeca_modo", "cabeca_yaw", "cabeca_valor", "state_encoder"])}
        }
        torch.save(dados_salvar, ckpt_saida)

        if salvar_cada > 0 and (it % salvar_cada == 0):
            caminho_snap = ckpt_saida.replace(".pt", f"_it{it}.pt")
            torch.save(dados_salvar, caminho_snap)
            print(f"    [CHECKPOINT] Snapshot periódico salvo: {caminho_snap}", flush=True)

        # Critério lexicográfico de melhor modelo: 1. Sucesso -> 2. Submeta 1 -> 3. Recompensa
        score_atual = (taxa_total, taxa_sub1, r_medio)
        if score_atual > melhor_score:
            melhor_score = score_atual
            caminho_melhor = ckpt_saida.replace(".pt", "_melhor.pt")
            torch.save(dados_salvar, caminho_melhor)
            print(f"    [CHECKPOINT] Novo recorde! Modelo salvo em: {caminho_melhor} (Suc={taxa_total:.1f}%, Sub1={taxa_sub1:.1f}%, Rec={r_medio:+.2f})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f" TREINAMENTO FINALIZADO COM SUCESSO! Checkpoint final salvo em: {ckpt_saida}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treinamento PPO-BC Híbrido com Shaping e Currículo (Fase 5.5)")
    ap.add_argument("--dataset",           default="fase5/dados/dataset_wasd_tatico_36_v2.pt", help="Caminho do dataset ancorado")
    ap.add_argument("--base",              default="checkpoints_vla/vla_fase5_ppo_bc.pt", help="Checkpoint base de entrada")
    ap.add_argument("--saida",             default="checkpoints_vla/vla_fase5_ppo_bc.pt", help="Checkpoint de saída")
    ap.add_argument("--iteracoes",         type=int,   default=25,   help="Número de iterações PPO")
    ap.add_argument("--passos",            type=int,   default=50,   help="Número máximo de passos por episódio")
    ap.add_argument("--lr",                type=float, default=3e-5, help="Taxa de aprendizado")
    ap.add_argument("--gamma",             type=float, default=0.98, help="Fator de desconto temporal γ")
    ap.add_argument("--gae-lambda",        type=float, default=0.95, help="Parâmetro λ do GAE")
    ap.add_argument("--ppo-epochs",        type=int,   default=2,    help="Épocas PPO por iteração")
    ap.add_argument("--clip-eps",          type=float, default=0.2,  help="Epsilon de clipping PPO")
    ap.add_argument("--mini-batch-size",   type=int,   default=16,   help="Tamanho do mini-batch PPO")
    ap.add_argument("--lambda-shaping",    type=float, default=0.15, help="Peso λ do shaping de potencial físico Φ=-dist")
    ap.add_argument("--penalidade-fixacao",type=float, default=0.08, help="Penalidade por fixação estacionária com alvo focado")
    ap.add_argument("--mascarar-se-centralizado", action="store_true", default=False, help="Proíbe modo 'alinhar' no rollout se alvo centralizado")
    ap.add_argument("--lambda-bc-ini",     type=float, default=0.40, help="Peso inicial do BC (warm-up moderado)")
    ap.add_argument("--lambda-bc-fim",     type=float, default=0.10, help="Peso final do BC")
    ap.add_argument("--curriculo-estagio", type=str,   default="auto", choices=["auto", "A", "B", "C"], help="Estágio do currículo")
    ap.add_argument("--estagio-inicial",   type=str,   default=None, help="Estagio inicial para progressao adaptativa (ex: B)")
    ap.add_argument("--criterio-a",        type=float, default=0.35, help="Taxa Submeta 1 para avançar de A para B")
    ap.add_argument("--criterio-b",        type=float, default=0.20, help="Taxa Sucesso para avançar de B para C")
    ap.add_argument("--consecutivas-curriculo", type=int, default=3, help="Iterações consecutivas necessárias para avançar de estágio")
    ap.add_argument("--fase-treino",       type=str,   default="completo", choices=["completo", "warmup", "adaptacao", "refinamento"])
    ap.add_argument("--salvar-cada",       type=int,   default=5,    help="Intervalo de iterações para checkpoints intermediários")
    ap.add_argument("--usar-cerebro", action="store_true", default=True, help="Ativa o Cérebro Supervisor no treino")
    ap.add_argument("--sem-cerebro", dest="usar_cerebro", action="store_false", help="Desativa o Cérebro Supervisor")
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
        mini_batch_size=args.mini_batch_size,
        lambda_shaping=args.lambda_shaping,
        penalidade_fixacao=args.penalidade_fixacao,
        mascarar_se_centralizado=args.mascarar_se_centralizado,
        lambda_bc_ini=args.lambda_bc_ini,
        lambda_bc_fim=args.lambda_bc_fim,
        curriculo_estagio=args.curriculo_estagio,
        criterio_a=args.criterio_a,
        criterio_b=args.criterio_b,
        consecutivas_curriculo=args.consecutivas_curriculo,
        estagio_inicial=args.estagio_inicial,
        fase_treino=args.fase_treino,
        salvar_cada=args.salvar_cada,
        usar_cerebro=args.usar_cerebro,
        seed=args.seed
    )
