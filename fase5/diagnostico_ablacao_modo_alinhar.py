# coding=utf-8
"""
fase5/diagnostico_ablacao_modo_alinhar.py — Diagnóstico de Ablação do Modo Alinhar.

Investiga se o agente:
1. Começa a se locomover (W / Sprint / Strafe) quando o modo 'alinhar' é mascarado/desabilitado com alvo centralizado.
2. Qual é a 2ª opção preferida da política quando 'alinhar' não está disponível.
3. Ranking completo de probabilidades dos 6 modos sob diferentes condições visuais.
"""
from __future__ import annotations
import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.gpu_utils import compactar_backbone
from infra.run_vla_agent import load_vla_agent
from fase5.acoes_taticas import (
    decodificar_acao_fatorada,
    fatorar_indice_36,
    unificar_indices,
    MODOS,
    YAW_BINS_9
)
from fase5.recompensa_visual import RastreadorVisualEpisodio, detectar_alvo_no_frame

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def rodar_teste_ablacao(
    modo_mascara: str = "nenhum", # 'nenhum', 'mascarar_se_centralizado', 'mascarar_sempre'
    passos: int = 25,
    seed: int = 100
):
    """
    Executa um rollout de 25 passos com 8 ambientes em paralelo sob a regra de ablação.
    """
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)

    ckpt_path = "checkpoints_vla/vla_fase5_ppo_bc.pt"
    if os.path.exists(ckpt_path):
        ckpt_data = torch.load(ckpt_path, map_location=dev)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[Diagnóstico] Pesos carregados de '{ckpt_path}'", flush=True)

    pol = PoliticaRaciocinioLoop(None, amostrar=True, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)

    n = 8
    from fase5.curriculo_fase5 import CurriculoFase5
    curriculo = CurriculoFase5(modo_estagio="A")
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

    d_ini = [math.hypot(t["estagios"][0]["alvo_abs"][0] - est[i]["x"], t["estagios"][0]["alvo_abs"][1] - est[i]["z"]) for i, t in enumerate(tarefas)]
    d_min = list(d_ini)
    d_final = list(d_ini)

    contagem_modos = {m: 0 for m in MODOS}
    passos_w_total = 0
    passos_giro_total = 0
    passos_totais = 0
    submetas_atingidas = 0

    print(f"\n================================================================================", flush=True)
    print(f" TESTE DE ABLAÇÃO: Modo Máscara = '{modo_mascara}' ({passos} passos, 8 robôs)", flush=True)
    print(f"================================================================================", flush=True)

    for p in range(passos):
        prompts = [t["prompt"] for t in tarefas]
        alvos_abs = [t["estagios"][0]["alvo_abs"] for t in tarefas]

        px, sv, gv, u8 = pol._entradas(est, alvos_abs, obs)
        ids = pol.obter_ids(prompts)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg_modo, lg_yaw, values = pol.forward_pensamento(px, sv, gv, ids)

        lg_modo_mod = lg_modo.clone().float()

        for i in range(n):
            cor_alvo = tarefas[i]["estagios"][0]["cor"]
            frame_i = u8[i, 0] if u8 is not None and u8.ndim == 5 else (u8[i] if u8 is not None else None)
            det = detectar_alvo_no_frame(frame_i, cor_alvo)

            mascarar = False
            if modo_mascara == "mascarar_sempre":
                mascarar = True
            elif modo_mascara == "mascarar_se_centralizado":
                if det["visivel"] and abs(det["centro_x"]) < 0.30:
                    mascarar = True

            if mascarar:
                # Mascara modo 'alinhar' (índice 0)
                lg_modo_mod[i, 0] = -1e9

        p_modo = torch.softmax(lg_modo_mod / pol.temperatura, dim=-1)
        p_yaw  = torch.softmax(lg_yaw.float() / pol.temperatura, dim=-1)

        dist_m = torch.distributions.Categorical(probs=p_modo)
        dist_y = torch.distributions.Categorical(probs=p_yaw)
        a_modo = dist_m.sample().cpu().numpy()
        a_yaw  = dist_y.sample().cpu().numpy()

        acoes = []
        for i in range(n):
            m_idx = int(a_modo[i])
            y_idx = int(a_yaw[i])
            nome_modo = MODOS[m_idx]
            contagem_modos[nome_modo] += 1
            passos_totais += 1

            acao_dict = decodificar_acao_fatorada(m_idx, y_idx)
            raw_dx = acao_dict["mouse"][0]
            prev_dx = pol.ultimo_dx.get(i, 0.0)
            smooth_dx = int(0.65 * raw_dx + 0.35 * prev_dx)
            pol.ultimo_dx[i] = smooth_dx
            acao_dict["mouse"] = [smooth_dx, 0]
            acoes.append(acao_dict)

            if "W" in acao_dict.get("hold", []):
                passos_w_total += 1
            if abs(smooth_dx) > 5:
                passos_giro_total += 1

        r = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = r["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        for i in range(n):
            alvo = tarefas[i]["estagios"][0]["alvo_abs"]
            d_atual = math.hypot(alvo[0] - est[i]["x"], alvo[1] - est[i]["z"])
            d_final[i] = d_atual
            if d_atual < d_min[i]:
                d_min[i] = d_atual
            if d_atual <= 1.2:
                submetas_atingidas += 1

    pct_w = (passos_w_total / max(1, passos_totais)) * 100.0
    pct_giro = (passos_giro_total / max(1, passos_totais)) * 100.0
    d_ini_m = float(np.mean(d_ini))
    d_min_m = float(np.mean(d_min))
    d_fim_m = float(np.mean(d_final))
    delta_d = d_ini_m - d_min_m

    print(f"\n--- RESULTADOS DO TESTE ('{modo_mascara}') ---", flush=True)
    print(f"Distância Média   : {d_ini_m:.2f}m -> {d_fim_m:.2f}m (Mínima: {d_min_m:.2f}m | Ganho: {delta_d:+.2f}m)", flush=True)
    print(f"Uso de W          : {pct_w:5.1f}%", flush=True)
    print(f"Uso de Giro       : {pct_giro:5.1f}%", flush=True)
    print(f"Submeta 1 Tocada  : {submetas_atingidas} vezes", flush=True)
    print("Distribuição dos Modos Escolhidos:", flush=True)
    for nome, count in contagem_modos.items():
        pct = (count / max(1, passos_totais)) * 100.0
        print(f"  - {nome:<12}: {pct:5.1f}% ({count}/{passos_totais})", flush=True)

    return {
        "modo_mascara": modo_mascara,
        "pct_w": pct_w,
        "pct_giro": pct_giro,
        "d_ini": d_ini_m,
        "d_min": d_min_m,
        "d_fim": d_fim_m,
        "delta_d": delta_d,
        "modos": contagem_modos,
        "passos_totais": passos_totais,
        "submetas": submetas_atingidas
    }


def executar_auditoria_completa_ablacao():
    print("=" * 80, flush=True)
    print(" [FASE 5.5] DIAGNÓSTICO PROFUNDO: ABLAÇÃO DO MODO ALINHAR & PREFERÊNCIAS MOTORAS", flush=True)
    print("=" * 80, flush=True)

    # 1. Sem máscara (comportamento padrão atual)
    res_padrao = rodar_teste_ablacao(modo_mascara="nenhum", passos=25, seed=42)

    # 2. Máscara seletiva (proíbe alinhar se alvo centralizado)
    res_seletivo = rodar_teste_ablacao(modo_mascara="mascarar_se_centralizado", passos=25, seed=42)

    # 3. Máscara total (proíbe alinhar sempre)
    res_total = rodar_teste_ablacao(modo_mascara="mascarar_sempre", passos=25, seed=42)

    print("\n" + "=" * 80, flush=True)
    print(" RESUMO COMPARATIVO DA AUDITORIA DE ABLAÇÃO", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Condição':<28} | {'W%':<6} | {'Giro%':<6} | {'Dist Ini->Min':<16} | {'Aproximação':<12} | {'Top 2 Modos'}", flush=True)
    print("-" * 100, flush=True)
    for r in [res_padrao, res_seletivo, res_total]:
        modos_ord = sorted(r["modos"].items(), key=lambda x: x[1], reverse=True)
        N = r["passos_totais"]
        t1 = f"{modos_ord[0][0]} ({modos_ord[0][1] / N * 100:.0f}%)"
        t2 = f"{modos_ord[1][0]} ({modos_ord[1][1] / N * 100:.0f}%)"
        top_str = f"{t1}, {t2}"
        print(f"{r['modo_mascara']:<28} | {r['pct_w']:5.1f}% | {r['pct_giro']:5.1f}% | {r['d_ini']:.1f}m -> {r['d_min']:.1f}m | {r['delta_d']:+5.2f}m       | {top_str}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    executar_auditoria_completa_ablacao()
