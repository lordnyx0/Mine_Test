# coding=utf-8
"""
fase5/experimento_treino_ablacao_controlada.py — Experimento de Treinamento com Ablação Dinâmica Controlada.

Objetivo do Experimento:
- Testar a hipótese do atrator degenerado: Quando o alvo está visível e centralizado (|cx| < 0.30),
  o modo 'alinhar' é desabilitado dinamicamente na amostragem do rollout.
- Medir se o PPO + BC passam a aprender naturalmente a preferir 'sprint' e locomoção direta
  ao receber o feedback de aproximação física e alcance de submeta.
"""
from __future__ import annotations
import os
import sys
import time
import math
import random
import json
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
    decodificar_acao_fatorada,
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


def coletar_rollout_com_ablacao_dinamica(
    pol: PoliticaRaciocinioLoop,
    curriculo: CurriculoFase5,
    passos: int = 50,
    shaping_geometrico: bool = True,
    lambda_shaping: float = 0.15,
    penalidade_fixacao: float = 0.08,
    mascarar_se_centralizado: bool = True,
    seed: int = 42
):
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

    passos_w = [0] * n
    passos_giro = [0] * n
    passos_stat_foc = [0] * n
    contagem_modos_ep = {m: 0 for m in MODOS}
    total_passos_acoes = 0

    rec_vis_acc = [0.0] * n
    rec_pot_acc = [0.0] * n
    rec_term_acc = [0.0] * n

    SV_L, IDS_L, VEMB_L = [], [], []
    MODO_L, YAW_L, LOGP_L, VAL_L, R_L, VIVO_L = [], [], [], [], [], []

    for p in range(passos):
        prompts = [t["prompt"] for t in tarefas]
        alvos_abs = [t["estagios"][estagio_atual[i]]["alvo_abs"] for i, t in enumerate(tarefas)]

        px, sv, gv, u8 = pol._entradas(est, alvos_abs, obs)
        ids = pol.obter_ids(prompts)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg_modo, lg_yaw, values = pol.forward_pensamento(px, sv, gv, ids)

        lg_modo_amostragem = lg_modo.clone().float()

        # Detecção visual prévia para aplicação da máscara de ablação
        for i in range(n):
            if not vivo[i]:
                continue
            cor_alvo = tarefas[i]["estagios"][estagio_atual[i]]["cor"]
            frame_i = u8[i, 0] if u8 is not None and u8.ndim == 5 else (u8[i] if u8 is not None else None)
            det = detectar_alvo_no_frame(frame_i, cor_alvo)

            if mascarar_se_centralizado and det["visivel"] and abs(det["centro_x"]) < 0.30:
                # Disallow 'alinhar' (mode 0) when target is visible and centered
                lg_modo_amostragem[i, 0] = -1e9

        p_modo = torch.softmax(lg_modo_amostragem / pol.temperatura, dim=-1)
        p_yaw  = torch.softmax(lg_yaw.float() / pol.temperatura, dim=-1)

        dist_m = torch.distributions.Categorical(probs=p_modo)
        dist_y = torch.distributions.Categorical(probs=p_yaw)
        a_modo = dist_m.sample()
        a_yaw  = dist_y.sample()

        logp_amostrado = dist_m.log_prob(a_modo) + dist_y.log_prob(a_yaw)

        acoes = []
        for i in range(n):
            m_idx = int(a_modo[i])
            y_idx = int(a_yaw[i])
            nome_modo = MODOS[m_idx]
            contagem_modos_ep[nome_modo] += 1
            total_passos_acoes += 1

            acao_dict = decodificar_acao_fatorada(m_idx, y_idx)
            raw_dx = acao_dict["mouse"][0]
            prev_dx = pol.ultimo_dx.get(i, 0.0)
            smooth_dx = int(0.65 * raw_dx + 0.35 * prev_dx)
            pol.ultimo_dx[i] = smooth_dx
            acao_dict["mouse"] = [smooth_dx, 0]
            acoes.append(acao_dict)

            if "W" in acao_dict.get("hold", []):
                passos_w[i] += 1
            if abs(smooth_dx) > 5:
                passos_giro[i] += 1

        r = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = r["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        rec = [0.0] * n

        for i in range(n):
            if not vivo[i]:
                continue

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

            frame_i = u8[i, 0] if u8 is not None and u8.ndim == 5 else (u8[i] if u8 is not None else None)

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
            if info.get("estacionario_focado"):
                passos_stat_foc[i] += 1

            rec_vis_acc[i] += info.get("rec_visual", 0.0)
            rec_pot_acc[i] += info.get("rec_potencial", 0.0)
            rec_term_acc[i] += info.get("rec_terminal", 0.0)

            rec[i] += r_passo
            dant[i] = d_atual

            if e.get("in_water") or e.get("in_lava"):
                vivo[i] = False
                continue

            if d_atual <= RAIO_CHEGADA_SUBMETA:
                rec[i] += BONUS_FINAL
                rec_term_acc[i] += BONUS_FINAL
                submeta_atingida[i] = True
                concluiu_em[i] = p
                vivo[i] = False

        SV_L.append(sv.cpu().numpy())
        IDS_L.append(ids.cpu().numpy())
        VEMB_L.append(np.zeros((n, 32, 1024), dtype=np.float16)) # placeholder
        MODO_L.append(a_modo.cpu().numpy())
        YAW_L.append(a_yaw.cpu().numpy())
        LOGP_L.append(logp_amostrado.cpu().numpy())
        VAL_L.append(values.squeeze(-1).float().cpu().numpy())
        R_L.append(rec)
        VIVO_L.append(list(vivo))

        if not any(vivo):
            break

    met = []
    for i in range(n):
        ep_passos = concluiu_em[i] if concluiu_em[i] is not None else (p + 1)
        met.append({
            "submeta_ok": submeta_atingida[i],
            "concluiu": concluiu_em[i] is not None,
            "passos": ep_passos,
            "avistou": avistou_alguma_vez[i],
            "dist_ini": d_ini[i],
            "dist_min": d_min[i],
            "dist_fim": d_final[i],
            "pct_w": (passos_w[i] / max(1, ep_passos)) * 100.0,
            "pct_giro": (passos_giro[i] / max(1, ep_passos)) * 100.0,
            "pct_stat_foc": (passos_stat_foc[i] / max(1, ep_passos)) * 100.0,
            "rec_vis": rec_vis_acc[i],
            "rec_pot": rec_pot_acc[i],
            "rec_term": rec_term_acc[i],
            "rec_total": rec_vis_acc[i] + rec_pot_acc[i] + rec_term_acc[i]
        })

    modos_pct = {
        m: (contagem_modos_ep[m] / max(1, total_passos_acoes)) * 100.0
        for m in MODOS
    }

    SV_T   = np.array(SV_L)
    IDS_T  = np.array(IDS_L)
    VEMB_T = np.array(VEMB_L)
    MODO_T = np.array(MODO_L)
    YAW_T  = np.array(YAW_L)
    LOGP_T = np.array(LOGP_L)
    VAL_T  = np.array(VAL_L)
    R_T    = np.array(R_L)
    VIVO_T = np.array(VIVO_L)

    return (SV_T, IDS_T, VEMB_T, MODO_T, YAW_T, LOGP_T, VAL_T, R_T, VIVO_T), met, modos_pct


def rodar_experimento_5_iteracoes():
    print("=" * 80, flush=True)
    print(" [FASE 5.5] EXPERIMENTO CONTROLADO: TREINO COM ABLAÇÃO DINÂMICA DE 'ALINHAR'", flush=True)
    print("    Regra: Se alvo visível e |cx| < 0.30 -> Modo 'alinhar' proibido no rollout", flush=True)
    print("    Objetivo: Medir se o PPO+BC migra os gradientes para Sprint e atinge Submeta 1", flush=True)
    print("=" * 80, flush=True)

    curriculo = CurriculoFase5(modo_estagio="A")
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

    pol = PoliticaRaciocinioLoop(None, amostrar=True, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)

    ckpt_base = "checkpoints_vla/vla_fase5_ppo_bc.pt"
    if os.path.exists(ckpt_base):
        ckpt_data = torch.load(ckpt_base, map_location=dev)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Checkpoint '{ckpt_base}' carregado.", flush=True)

    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(treinaveis, lr=3e-5, weight_decay=1e-4)
    loss_ce = nn.CrossEntropyLoss()

    dataset_path = "fase5/dados/dataset_wasd_tatico_36_v2.pt"
    dados_offline = torch.load(dataset_path, weights_only=False)
    todos_sv_bc = torch.stack([d["sv"] for d in dados_offline]).to(dev)
    todas_acoes_36 = [int(d["acao_otima"]) for d in dados_offline]
    todos_modos_bc = torch.tensor([fatorar_indice_36(a)[0] for a in todas_acoes_36], dtype=torch.long, device=dev)
    todos_yaws_bc  = torch.tensor([fatorar_indice_36(a)[1] for a in todas_acoes_36], dtype=torch.long, device=dev)
    prompts_bc = [d.get("prompt", "Objetivo: vá até o bloco amarelo [Etapa 1/2]") for d in dados_offline]
    enc_bc     = tokenizer(prompts_bc, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    tokens_bc  = enc_bc["input_ids"].to(dev)
    gv_dummy_bc = torch.zeros((len(dados_offline), 4), dtype=torch.float32, device=dev)

    for it in range(1, 6):
        t0 = time.time()
        vla.eval()
        torch.cuda.empty_cache()

        (SV_T, IDS_T, VEMB_T, MODO_T, YAW_T, LOGP_T, VAL_T, R_T, VIVO_T), met, modos_pct = coletar_rollout_com_ablacao_dinamica(
            pol=pol,
            curriculo=curriculo,
            passos=50,
            shaping_geometrico=True,
            lambda_shaping=0.15,
            penalidade_fixacao=0.08,
            mascarar_se_centralizado=True,
            seed=100 + it * 13
        )

        taxa_sub1 = float(np.mean([m["submeta_ok"] for m in met])) * 100.0
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

        advantages, returns = compute_gae(R_T, VAL_T, VIVO_T, gamma=0.98, gae_lambda=0.95)
        flat_vivos = VIVO_T.reshape(-1).astype(bool)

        if flat_vivos.any():
            flat_adv = advantages.reshape(-1)[flat_vivos]
            norm_adv = np.clip((flat_adv - np.mean(flat_adv)) / (np.std(flat_adv) + 1e-8), -4.0, 4.0)

            flat_sv   = SV_T.reshape(-1, SV_T.shape[-1])[flat_vivos]
            flat_ids  = IDS_T.reshape(-1, IDS_T.shape[-1])[flat_vivos]
            flat_modo = MODO_T.reshape(-1)[flat_vivos]
            flat_yaw  = YAW_T.reshape(-1)[flat_vivos]
            flat_logp = LOGP_T.reshape(-1)[flat_vivos]
            flat_ret  = returns.reshape(-1)[flat_vivos]

            T_eff = len(flat_sv)
            b_sv   = torch.tensor(flat_sv, dtype=torch.float32, device=dev)
            b_ids  = torch.tensor(flat_ids, dtype=torch.long, device=dev)
            b_modo = torch.tensor(flat_modo, dtype=torch.long, device=dev)
            b_yaw  = torch.tensor(flat_yaw, dtype=torch.long, device=dev)
            b_logp = torch.tensor(flat_logp, dtype=torch.float32, device=dev)
            b_adv  = torch.tensor(norm_adv, dtype=torch.float32, device=dev)
            b_ret  = torch.tensor(flat_ret, dtype=torch.float32, device=dev)
            b_gv   = torch.zeros((T_eff, 4), dtype=torch.float32, device=dev)

            vla.train()
            vla.vision_encoder.eval()

            for ep in range(2):
                indices = np.arange(T_eff)
                np.random.shuffle(indices)

                for start in range(0, T_eff, 16):
                    end = min(start + 16, T_eff)
                    mb_idx = indices[start:end]
                    mb_len = len(mb_idx)

                    otimizador.zero_grad()

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg_modo, lg_yaw, values = pol.forward_pensamento(
                            pixel_tensor=None,
                            state_tensor=b_sv[mb_idx],
                            goal_tensor=b_gv[mb_idx],
                            input_ids=b_ids[mb_idx],
                            precomputed_v_emb=None
                        )
                        lg_modo = lg_modo.float()
                        lg_yaw  = lg_yaw.float()

                        dist_m = torch.distributions.Categorical(logits=lg_modo)
                        dist_y = torch.distributions.Categorical(logits=lg_yaw)

                        new_logp = dist_m.log_prob(b_modo[mb_idx]) + dist_y.log_prob(b_yaw[mb_idx])
                        ratio = torch.exp(torch.clamp(new_logp - b_logp[mb_idx], -10.0, 2.0))

                        surr1 = ratio * b_adv[mb_idx]
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv[mb_idx]
                        loss_ppo = -torch.min(surr1, surr2).mean()

                        val_pred = values.squeeze(-1).float()
                        loss_val = torch.clamp(nn.MSELoss()(val_pred, b_ret[mb_idx]), 0.0, 50.0)

                        loss_total = loss_ppo + 0.25 * loss_val

                    loss_total.backward()

                    # BC
                    idx_bc = torch.randint(0, len(dados_offline), (min(mb_len, 16),), device=dev)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg_m_bc, lg_y_bc, _ = pol.forward_pensamento(
                            pixel_tensor=None,
                            state_tensor=todos_sv_bc[idx_bc],
                            goal_tensor=gv_dummy_bc[idx_bc],
                            input_ids=tokens_bc[idx_bc],
                            precomputed_v_emb=None
                        )
                        loss_bc = 0.5 * (loss_ce(lg_m_bc.float(), todos_modos_bc[idx_bc]) +
                                         loss_ce(lg_y_bc.float(), todos_yaws_bc[idx_bc]))
                        loss_bc_w = 0.40 * loss_bc

                    loss_bc_w.backward()

                    nn.utils.clip_grad_norm_(treinaveis, max_norm=0.5)
                    otimizador.step()

        dt = time.time() - t0
        pct_strafe = modos_pct.get("strafe_esq", 0.0) + modos_pct.get("strafe_dir", 0.0)
        print(
            f"  Iteração {it:02d}/05 [Ablação Dinâmica] ({dt:.0f}s) | Rec: Tot={r_medio:+5.2f} (Vis={r_vis_med:+5.2f}, Pot={r_pot_med:+5.2f}, Term={r_term_med:+5.2f}) | "
            f"Sub1: {taxa_sub1:5.1f}% | Dist: {d_ini_med:.1f}m->{d_fim_med:.1f}m (min={d_min_med:.1f}m) | W: {taxa_w:4.1f}% | Giro: {taxa_giro:4.1f}% | StatFoc: {taxa_statfoc:4.1f}%\n"
            f"    -> Modos Escolhidos: Sprint={modos_pct.get('sprint', 0.0):4.1f}% | Strafe={pct_strafe:4.1f}% | Pulo={modos_pct.get('pulo', 0.0):4.1f}% | Recuar={modos_pct.get('recuar', 0.0):4.1f}% | Alinhar={modos_pct.get('alinhar', 0.0):4.1f}%",
            flush=True
        )


if __name__ == "__main__":
    rodar_experimento_5_iteracoes()
