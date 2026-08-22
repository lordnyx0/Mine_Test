# coding=utf-8
"""
fase5/auditoria_visao_acao_completa.py — Auditoria Experimental Completa da Cadeia Visão -> Ação -> Física.

Executa as 7 auditorias empíricas sem alterar o treinamento:
  1. Cadeia Visão -> Ação e Causalidade Visual (intervenção esquerda / centro / direita e prompt).
  2. Cabeça de Yaw (correlação centro_x vs yaw, entropia, frequência por bin).
  3. Mapeamento de Ações Táticas (tabela modo x yaw, sinal de mouse, magnitude).
  4. Controle Físico no Simulador (medição real de Delta_yaw, deslocamento, velocidade por ação).
  5. Análise de Aproximação Episódica (distâncias min/max, tempos de fixação, oscilação, causa da não-chegada).
  6. Auditoria de Recompensa vs Comportamento (diagnóstico de gaze fixation / recompensa sem avanço).
  7. Auditoria do Dataset v2 (relação erro angular vs ação tática ensinada).
"""
from __future__ import annotations

import os
import sys
import math
import time
import json
import base64
import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import CORES_MAP, RAIO_CHEGADA_SUBMETA
from fase5.acoes_taticas import (
    decodificar_acao_fatorada,
    fatorar_indice_36,
    unificar_indices,
    MODOS,
    YAW_BINS_9,
    NUM_MODOS,
    NUM_YAW
)
from fase5.recompensa_visual import detectar_alvo_no_frame, _obter_mascara_cor, RastreadorVisualEpisodio
from fase5.curriculo_fase5 import CurriculoFase5
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone


def auditar_1_causalidade_visao_acao(pol: PoliticaRaciocinioLoop, vla, dev, ckpt_path: str):
    print("\n" + "=" * 80)
    print(" [AUDITORIA 1] CADEIA VISÃO -> AÇÃO & TESTE DE CAUSALIDADE VISUAL")
    print("=" * 80)

    # Cria 3 frames sintéticos com pilar amarelo: Esquerda, Centro, Direita
    H, W = 224, 224
    frame_esq = np.zeros((H, W, 3), dtype=np.uint8)
    frame_esq[40:190, 20:50] = [245, 215, 20]  # x ~ 35 (centro_x ~ -0.69)

    frame_cen = np.zeros((H, W, 3), dtype=np.uint8)
    frame_cen[40:190, 97:127] = [245, 215, 20] # x ~ 112 (centro_x ~ 0.00)

    frame_dir = np.zeros((H, W, 3), dtype=np.uint8)
    frame_dir[40:190, 174:204] = [245, 215, 20] # x ~ 189 (centro_x ~ +0.69)

    # Frame com pilar roxo (para teste de prompt cruzado)
    frame_roxo_cen = np.zeros((H, W, 3), dtype=np.uint8)
    frame_roxo_cen[40:190, 97:127] = [155, 38, 182]

    # Prepara vetor de estado proprioceptivo neutro (estagio 0)
    sv = torch.zeros((1, 32), dtype=torch.float32, device=dev)
    sv[0, 2] = 0.35 # dist ~ 5.2m
    sv[0, 3] = 1.0  # cos(0)
    sv[0, 4] = 0.0  # sin(0)
    sv[0, 16] = 0.0 # estagio 0

    gv = torch.zeros((1, 4), dtype=torch.float32, device=dev)

    casos = [
        ("Pilar Amarelo à ESQUERDA (cx=-0.69)", frame_esq, "Objetivo: vá até o bloco amarelo [Etapa 1/1]"),
        ("Pilar Amarelo no CENTRO (cx=0.00)",   frame_cen, "Objetivo: vá até o bloco amarelo [Etapa 1/1]"),
        ("Pilar Amarelo à DIREITA (cx=+0.69)",  frame_dir, "Objetivo: vá até o bloco amarelo [Etapa 1/1]"),
        ("Pilar Roxo no CENTRO (prompt amarelo)", frame_roxo_cen, "Objetivo: vá até o bloco amarelo [Etapa 1/1]"),
        ("Pilar Roxo no CENTRO (prompt roxo)",    frame_roxo_cen, "Objetivo: vá até o bloco roxo [Etapa 1/1]"),
    ]

    print(f"\nCheckpoint em teste: {ckpt_path}")
    print(f"{'Cenário':<42} | {'Top Modo':<12} | {'P(Modo)':<8} | {'Top Yaw (dx)':<14} | {'P(Yaw)':<8} | {'Ação Decodificada'}")
    print("-" * 115)

    vla.eval()
    for nome, frame, prompt in casos:
        u8_stack = np.stack([frame, frame, frame], axis=0)
        u8_batch = np.expand_dims(u8_stack, axis=0)
        px = pol.normalizar(u8_batch)

        ids = pol.obter_ids([prompt])

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg_modo, lg_yaw, val = pol.forward_pensamento(px, sv, gv, ids)

        p_modo = F.softmax(lg_modo.float(), dim=-1)[0].cpu().numpy()
        p_yaw  = F.softmax(lg_yaw.float(), dim=-1)[0].cpu().numpy()

        top_m = int(np.argmax(p_modo))
        top_y = int(np.argmax(p_yaw))
        dx_val = YAW_BINS_9[top_y]

        acao = decodificar_acao_fatorada(top_m, top_y)
        acao_str = f"hold={acao['hold']}, mouse={acao['mouse']}"

        print(f"{nome:<42} | {MODOS[top_m]:<12} | {p_modo[top_m]*100:5.1f}%   | {dx_val:+4d} ({top_y:d})        | {p_yaw[top_y]*100:5.1f}%   | {acao_str}")


def auditar_2_correlacao_centro_x_yaw(pol: PoliticaRaciocinioLoop, vla, dev, N_samples: int = 150):
    print("\n" + "=" * 80)
    print(" [AUDITORIA 2] AUDITORIA DA CABEÇA DE YAW (CORRELAÇÃO CENTRO_X VS YAW)")
    print("=" * 80)

    H, W = 224, 224
    sv = torch.zeros((1, 32), dtype=torch.float32, device=dev)
    sv[0, 2] = 0.35
    sv[0, 3] = 1.0
    gv = torch.zeros((1, 4), dtype=torch.float32, device=dev)
    ids = pol.obter_ids(["Objetivo: vá até o bloco amarelo [Etapa 1/1]"])

    faixas = {
        "< -0.6":   {"cx_vals": [], "yaws": [], "modos": []},
        "-0.6..-0.3": {"cx_vals": [], "yaws": [], "modos": []},
        "-0.3..0.3":  {"cx_vals": [], "yaws": [], "modos": []},
        "0.3..0.6":   {"cx_vals": [], "yaws": [], "modos": []},
        "> 0.6":    {"cx_vals": [], "yaws": [], "modos": []}
    }

    for x_center in np.linspace(15, 209, 65):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        x0 = int(max(0, x_center - 12))
        x1 = int(min(W, x_center + 12))
        frame[35:185, x0:x1] = [245, 215, 20]

        det = detectar_alvo_no_frame(frame, "amarelo")
        cx = det["centro_x"]

        if cx < -0.6:
            fk = "< -0.6"
        elif cx < -0.3:
            fk = "-0.6..-0.3"
        elif cx <= 0.3:
            fk = "-0.3..0.3"
        elif cx <= 0.6:
            fk = "0.3..0.6"
        else:
            fk = "> 0.6"

        u8_stack = np.stack([frame, frame, frame], axis=0)
        u8_batch = np.expand_dims(u8_stack, axis=0)
        px = pol.normalizar(u8_batch)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg_modo, lg_yaw, val = pol.forward_pensamento(px, sv, gv, ids)

        top_m = int(lg_modo.argmax(-1)[0].item())
        top_y = int(lg_yaw.argmax(-1)[0].item())
        dx_val = YAW_BINS_9[top_y]

        faixas[fk]["cx_vals"].append(cx)
        faixas[fk]["yaws"].append(dx_val)
        faixas[fk]["modos"].append(top_m)

    print(f"{'Faixa centro_x':<15} | {'N Amostras':<10} | {'dx Medio':<10} | {'Modo Dominante':<15} | {'Distribuicao de Bins Yaw (dx)'}")
    print("-" * 90)
    for fk, d in faixas.items():
        n = len(d["cx_vals"])
        if n == 0:
            continue
        dx_med = float(np.mean(d["yaws"]))
        m_counts = np.bincount(d["modos"], minlength=6)
        top_m_idx = int(np.argmax(m_counts))
        top_m_nome = MODOS[top_m_idx]

        yaw_counts = {b: d["yaws"].count(b) for b in sorted(set(d["yaws"]))}
        yaw_str = ", ".join([f"{b:+d}: {c}" for b, c in yaw_counts.items()])
        print(f"{fk:<15} | {n:<10} | {dx_med:+8.1f}   | {top_m_nome:<15} | {yaw_str}")


def auditar_3_tabela_acoes_taticas():
    print("\n" + "=" * 80)
    print(" [AUDITORIA 3] TABELA DO ESPAÇO DE AÇÕES TÁTICAS (54 COMBINAÇÕES CANÔNICAS)")
    print("=" * 80)

    print(f"{'Modo (0..5)':<14} | {'Yaw Bin (0..8)':<16} | {'dx Mouse':<10} | {'Teclas Hold':<15} | {'Duracao (ms)':<12} | {'Giro Fisico (graus/passo)'}")
    print("-" * 95)
    for m in range(NUM_MODOS):
        for y in range(NUM_YAW):
            acao = decodificar_acao_fatorada(m, y)
            dx = acao["mouse"][0]
            graus_fisicos = dx * 0.003 * (180.0 / math.pi)
            hold_str = str(acao["hold"])
            if y in [0, 4, 8]:  # Amostra extremos e centro
                print(f"{MODOS[m]:<14} | Bin {y} ({YAW_BINS_9[y]:+4d} unidades) | {dx:+8d}   | {hold_str:<15} | {acao['duration_ms']:<12} | {graus_fisicos:+7.2f} deg")


def auditar_4_controle_fisico_simulador():
    print("\n" + "=" * 80)
    print(" [AUDITORIA 4] CONTROLE FÍSICO REAL NO SIMULADOR OFFLINE")
    print("=" * 80)

    post("/lote/reset", {"posicoes": [[143.5, 240.5]] * 8})

    acoes_teste = [
        ("Giro Parado Esquerda Rapido (dx=-120)", {"hold": [], "mouse": [-120, 0], "duration_ms": 250}),
        ("Giro Parado Esquerda Curto (dx=-25)",   {"hold": [], "mouse": [-25, 0], "duration_ms": 250}),
        ("Frente Neutra (W, dx=0)",               {"hold": ["W"], "mouse": [0, 0], "duration_ms": 250}),
        ("Frente com Curva Suave (W, dx=+25)",   {"hold": ["W"], "mouse": [25, 0], "duration_ms": 250}),
        ("Frente com Curva Forte (W, dx=+120)",   {"hold": ["W"], "mouse": [120, 0], "duration_ms": 250}),
        ("Strafe Esquerda (W+A, dx=0)",           {"hold": ["W", "A"], "mouse": [0, 0], "duration_ms": 250}),
        ("Strafe Direita (W+D, dx=0)",            {"hold": ["W", "D"], "mouse": [0, 0], "duration_ms": 250}),
        ("Recuo Desengate (S, dx=0)",             {"hold": ["S"], "mouse": [0, 0], "duration_ms": 250}),
    ]

    r0 = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * 8, "frames": False})
    obs0 = r0["obs"]

    r1 = post("/lote/passo", {"acoes": [a[1] for a in acoes_teste], "frames": False})
    obs1 = r1["obs"]

    print(f"{'Acao Executada':<40} | {'Yaw Ini':<8} | {'Yaw Fim':<8} | {'Delta Yaw':<10} | {'Deslocamento Delta(X,Z)':<22} | {'Velocidade'}")
    print("-" * 115)
    for i, (nome, a) in enumerate(acoes_teste):
        e0 = obs0[i]["estado"]
        e1 = obs1[i]["estado"]

        yaw0 = e0["yaw"]
        yaw1 = e1["yaw"]
        dyaw = (yaw1 - yaw0 + 180.0) % 360.0 - 180.0

        dx = e1["x"] - e0["x"]
        dz = e1["z"] - e0["z"]
        dist = math.hypot(dx, dz)
        vel = dist / (a["duration_ms"] / 1000.0)

        print(f"{nome:<40} | {yaw0:6.1f} deg | {yaw1:6.1f} deg | {dyaw:+7.2f} deg | dx={dx:+5.2f}, dz={dz:+5.2f} (d={dist:.2f}m) | {vel:4.2f} m/s")


def auditar_5_e_6_aproximacao_e_reward(pol: PoliticaRaciocinioLoop, vla, dev, num_episodios: int = 16):
    print("\n" + "=" * 80)
    print(" [AUDITORIA 5 & 6] ANÁLISE DE APROXIMAÇÃO AO ALVO & COMPORTAMENTO DE REWARD (GAZE FIXATION)")
    print("=" * 80)

    curriculo = CurriculoFase5(modo_estagio="A")
    rastreador = RastreadorVisualEpisodio(8)

    episodios_info = []

    for lote in range(num_episodios // 8):
        tarefas, blocos = curriculo.gerar_tarefas(8, seed=200 + lote * 37)

        r = post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        if blocos:
            post("/lote/colocar_bloco", {"blocos": blocos})

        alvos_web = []
        for t in tarefas:
            a0 = t["estagios"][0]["alvo_abs"]
            alvos_web.append({"x": a0[0], "z": a0[1], "dist": 8.0, "graus": 0})
        post("/lote/alvos", {"alvos": alvos_web})

        obs = r["obs"][:8]
        est = [o["estado"] for o in obs]
        pol.reiniciar(obs)
        rastreador.reset_todos()

        d_ini = [math.hypot(t["estagios"][0]["alvo_abs"][0] - est[i]["x"], t["estagios"][0]["alvo_abs"][1] - est[i]["z"]) for i, t in enumerate(tarefas)]
        d_min = list(d_ini)
        passo_d_min = [0] * 8
        passos_visivel = [0] * 8
        passos_centralizado = [0] * 8
        passos_W = [0] * 8
        passos_girando = [0] * 8
        cx_acum = [[] for _ in range(8)]
        yaw_acum = [0.0] * 8
        ultimo_yaw = [e["yaw"] for e in est]

        rec_vis_acc = [0.0] * 8
        rec_pot_acc = [0.0] * 8
        rec_tot_acc = [0.0] * 8
        chegou = [False] * 8

        MAX_PASSOS = 60
        for p in range(MAX_PASSOS):
            prompts = [t["prompt"] for t in tarefas]
            alvos_abs = [t["estagios"][0]["alvo_abs"] for t in tarefas]

            acoes = pol.agir(est, alvos_abs, obs, prompts=prompts, estagios=[0]*8)
            u = pol.ultimo
            u8_raw = u["u8"]

            rr = post("/lote/passo", {"acoes": acoes, "frames": True})
            obs = rr["obs"][:8]
            est = [o["estado"] for o in obs]
            pol.observar(obs)

            for i in range(8):
                e = est[i]
                t = tarefas[i]
                alvo = t["estagios"][0]["alvo_abs"]
                cor = t["estagios"][0]["cor"]

                d_atual = math.hypot(alvo[0] - e["x"], alvo[1] - e["z"])
                if d_atual < d_min[i]:
                    d_min[i] = d_atual
                    passo_d_min[i] = p

                if d_atual <= RAIO_CHEGADA_SUBMETA:
                    chegou[i] = True

                frame_i = u8_raw[i, 0] if u8_raw is not None and u8_raw.ndim == 5 else None

                r_p, info = rastreador.calcular_recompensa_passo(
                    env_id=i, estado=e, frame_u8=frame_i, cor_alvo=cor,
                    acao_exec=acoes[i], estagio_atual=0, shaping_geometrico=True,
                    lambda_potencial=0.10, dist_atual=d_atual, dist_anterior=d_ini[i]
                )

                rec_vis_acc[i] += info.get("rec_visual", 0.0)
                rec_pot_acc[i] += info.get("rec_potencial", 0.0)
                rec_tot_acc[i] += r_p

                if info.get("visivel"):
                    passos_visivel[i] += 1
                    cx_acum[i].append(info.get("centro_x", 0.0))
                    if abs(info.get("centro_x", 1.0)) < 0.25:
                        passos_centralizado[i] += 1

                if "W" in acoes[i].get("hold", []):
                    passos_W[i] += 1

                dx_mouse = acoes[i].get("mouse", [0, 0])[0]
                if abs(dx_mouse) > 5:
                    passos_girando[i] += 1

                dyaw_step = abs((e["yaw"] - ultimo_yaw[i] + 180.0) % 360.0 - 180.0)
                yaw_acum[i] += dyaw_step
                ultimo_yaw[i] = e["yaw"]

        for i in range(8):
            episodios_info.append({
                "cor": tarefas[i]["estagios"][0]["cor"],
                "d_ini": d_ini[i],
                "d_min": d_min[i],
                "d_fim": d_atual,
                "passo_d_min": passo_d_min[i],
                "pct_visivel": (passos_visivel[i] / MAX_PASSOS) * 100.0,
                "pct_centralizado": (passos_centralizado[i] / MAX_PASSOS) * 100.0,
                "pct_W": (passos_W[i] / MAX_PASSOS) * 100.0,
                "pct_giro": (passos_girando[i] / MAX_PASSOS) * 100.0,
                "cx_med": float(np.mean(cx_acum[i])) if cx_acum[i] else 0.0,
                "yaw_acum": yaw_acum[i],
                "rec_vis": rec_vis_acc[i],
                "rec_pot": rec_pot_acc[i],
                "rec_tot": rec_tot_acc[i],
                "chegou": chegou[i]
            })

    print(f"{'Ep':<3} | {'Cor':<8} | {'D_Ini':<6} | {'D_Min':<6} | {'D_Fim':<6} | {'%Vis':<6} | {'%Cent':<6} | {'%W':<5} | {'%Giro':<6} | {'Rec_Vis':<8} | {'Rec_Pot':<8} | {'Rec_Tot':<8} | {'Diagnostico de Comportamento'}")
    print("-" * 125)
    for idx, ep in enumerate(episodios_info):
        if ep["chegou"]:
            diag = "SUCESSO (Cruzou Raio 1.5m)"
        elif ep["pct_visivel"] > 70.0 and ep["pct_W"] < 25.0:
            diag = "GAZE FIXATION (Olha para o alvo mas quase nao aperta W)"
        elif ep["pct_visivel"] > 70.0 and ep["pct_W"] >= 50.0 and ep["d_min"] > 3.0:
            diag = "DRIFT / TRAVA (Aperta W mas nao fecha angulo)"
        elif ep["pct_visivel"] < 30.0:
            diag = "PERDA VISUAL (Alvo fora do campo de visao)"
        else:
            diag = f"APROXIMACAO PARCIAL (d_min={ep['d_min']:.1f}m)"

        print(f"{idx+1:2d}  | {ep['cor']:<8} | {ep['d_ini']:4.1f}m | {ep['d_min']:4.1f}m | {ep['d_fim']:4.1f}m | {ep['pct_visivel']:4.0f}% | {ep['pct_centralizado']:4.0f}% | {ep['pct_W']:3.0f}% | {ep['pct_giro']:4.0f}% | {ep['rec_vis']:+7.2f} | {ep['rec_pot']:+7.2f} | {ep['rec_tot']:+7.2f} | {diag}")


def auditar_7_dataset_v2():
    print("\n" + "=" * 80)
    print(" [AUDITORIA 7] AUDITORIA DO DATASET DE ANCORAGEM v2 CONTRA AÇÕES REAIS")
    print("=" * 80)

    dataset_path = "fase5/dados/dataset_wasd_tatico_36_v2.pt"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} nao encontrado!")
        return

    dados = torch.load(dataset_path, weights_only=False)
    total = len(dados)
    print(f"Total de amostras analisadas: {total}")

    faixas_ang = {
        "Alvo Quase Alinhado (|erro| <= 10 deg)": {"total": 0, "modos": [0]*6, "w_count": 0},
        "Alvo com Desvio Moderado (10 deg < |erro| <= 35 deg)": {"total": 0, "modos": [0]*6, "w_count": 0},
        "Alvo com Grande Desvio (35 deg < |erro| <= 90 deg)": {"total": 0, "modos": [0]*6, "w_count": 0},
        "Alvo de Costas / Extremo (|erro| > 90 deg)": {"total": 0, "modos": [0]*6, "w_count": 0},
    }

    for d in dados:
        erro = abs(float(d.get("erro_yaw_graus", 0.0)))
        acao = int(d["acao_otima"])
        m, y = fatorar_indice_36(acao)

        if erro <= 10.0:
            fk = "Alvo Quase Alinhado (|erro| <= 10 deg)"
        elif erro <= 35.0:
            fk = "Alvo com Desvio Moderado (10 deg < |erro| <= 35 deg)"
        elif erro <= 90.0:
            fk = "Alvo com Grande Desvio (35 deg < |erro| <= 90 deg)"
        else:
            fk = "Alvo de Costas / Extremo (|erro| > 90 deg)"

        faixas_ang[fk]["total"] += 1
        faixas_ang[fk]["modos"][m] += 1
        if m in [1, 2, 3, 4]:  # Modos que usam W
            faixas_ang[fk]["w_count"] += 1

    print(f"{'Faixa Angular':<45} | {'Amostras':<9} | {'% W':<6} | {'Modo 0 (Alinhar)':<16} | {'Modo 1 (Sprint)':<15} | {'Modo 3/4 (Strafe)'}")
    print("-" * 115)
    for fk, d in faixas_ang.items():
        tot = max(1, d["total"])
        pct_w = (d["w_count"] / tot) * 100.0
        pct_m0 = (d["modos"][0] / tot) * 100.0
        pct_m1 = (d["modos"][1] / tot) * 100.0
        pct_strafe = ((d["modos"][3] + d["modos"][4]) / tot) * 100.0
        print(f"{fk:<45} | {tot:<9} | {pct_w:5.1f}% | {d['modos'][0]:5d} ({pct_m0:4.1f}%)   | {d['modos'][1]:5d} ({pct_m1:4.1f}%)  | {d['modos'][3]+d['modos'][4]:5d} ({pct_strafe:4.1f}%)")


def executar_todas_auditorias():
    print("=" * 80)
    print(" INICIANDO INVESTIGAÇÃO DO GARGALO DA FASE 5.5")
    print("=" * 80)

    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)

    ckpt_path = "checkpoints_vla/vla_fase5_ppo_bc.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=dev)
        if "treinaveis" in ckpt:
            vla.load_state_dict(ckpt["treinaveis"], strict=False)

    auditar_1_causalidade_visao_acao(pol, vla, dev, ckpt_path)
    auditar_2_correlacao_centro_x_yaw(pol, vla, dev)
    auditar_3_tabela_acoes_taticas()
    auditar_4_controle_fisico_simulador()
    auditar_5_e_6_aproximacao_e_reward(pol, vla, dev, num_episodios=16)
    auditar_7_dataset_v2()

    print("\n" + "=" * 80)
    print(" TODAS AS AUDITORIAS FORAM CONCLUÍDAS COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    executar_todas_auditorias()
