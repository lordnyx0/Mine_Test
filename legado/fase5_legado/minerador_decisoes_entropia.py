# coding=utf-8
"""
fase5/minerador_decisoes_entropia.py — Minerador e Causal Curator de Decisões de Alta Entropia.

Executa rollouts com exploração estocástica no simulador de Minecraft, calculando a entropia
de Shannon H(s_t) em cada passo. Aplica o Causal Curation Filter para:
  1. Identificar picos de alta entropia (H_norm >= 0.40).
  2. Filtrar ruídos de "robô perdido" (velocidade nula, colisão com paredes, deriva divergente).
  3. Reter apenas bifurcações construtivas de decisão (spawn, transição de submetas, contorno).
  4. Rotular a ação causal ótima (a*) e salvar o dataset de decisões calibrado.

Saída:
  - fase5/dados/dataset_decisoes_alta_entropia.pt
  - fase5/dados/metricas_mineracao_entropia.json
"""
from __future__ import annotations

import os
import sys
import math
import json
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import montar_tarefas_logicas, RAIO_CHEGADA_SUBMETA
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora
from fase5.gerar_demonstracoes_esparsas import YAW_BINS, GRAUS_POR_UNIDADE, calcular_bin_yaw, calcular_acao_18

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def minerar_decisoes(
    modelo_ckpt: str = "checkpoints_vla/vla_fase5_coldstart.pt",
    num_lotes: int = 5,
    passos_max: int = 100,
    temperatura: float = 0.9,
    limiar_entropia_norm: float = 0.35,
    seed: int = 200,
    saida_pt: str = "fase5/dados/dataset_decisoes_alta_entropia.pt"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" [FASE 5] MINERADOR DE DECISÕES DE ALTA ENTROPIA & FILTRO CAUSAL")
    print(f"    Checkpoint           : {modelo_ckpt}")
    print(f"    Lotes                : {num_lotes} (8 robôs/lote = {num_lotes * 8} episódios)")
    print(f"    Temperatura Amostral : {temperatura}")
    print(f"    Limiar Entropia Norm : {limiar_entropia_norm:.2f} (H >= {limiar_entropia_norm * math.log(18):.2f} nats)")
    print(f"    Saída Dataset        : {saida_pt}")
    print("=" * 80)

    # 1. Carrega modelo VLA e política
    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    if not any("lora_" in n for n, _ in vla.named_parameters()):
        vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    from politica.cerebro import PoliticaCerebroVLA

    pol_vla = PoliticaRaciocinioLoop(None, amostrar=True, device=device, vla=vla, loops_pensamento=3)
    pol = PoliticaCerebroVLA(pol_vla)
    pol.amostrar = True
    pol.temperatura = temperatura

    if os.path.exists(modelo_ckpt):
        ckpt_data = torch.load(modelo_ckpt, map_location=device)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos restaurados de '{modelo_ckpt}' ({len(ckpt_data['treinaveis'])} tensores).", flush=True)

    vla.to(device)
    vla.eval()

    info = get("/lote/info")
    N = info["envs"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    amostras_brutas_alta_entropia = 0
    amostras_descartadas_perdido = 0
    amostras_retidas_limpas = []
    
    total_passos_analisados = 0
    submetas1_atingidas = 0
    submetas2_atingidas = 0

    t_inicio = time.time()

    for lote_idx in range(num_lotes):
        print(f"\n--- Coletando Lote {lote_idx + 1}/{num_lotes} ---", flush=True)
        tarefas = montar_tarefas_logicas(N, seed=seed + lote_idx, proporcao_seq=1.0, nivel_curriculo=2)
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * N, "frames": True})
        obs = r["obs"][:N]
        est = [o["estado"] for o in obs]
        pol.reiniciar(obs)

        estagio_atual = [0] * N
        vivo = [True] * N

        # Buffer para registrar todo o histórico do episódio para filtragem a posteriori
        historico_episodios = [[] for _ in range(N)]

        for p in range(passos_max):
            prompts_ativos = [
                tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)].get("prompt_estagio", tarefas[i]["prompt"])
                for i in range(N)
            ]
            alvos_ativos_abs = [tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)]["alvo_abs"] for i in range(N)]

            # Inspeciona as ativações e logits para extrair entropia por agente
            px, sv, gv, u8 = pol_vla._entradas(est, alvos_ativos_abs, obs)
            for i, st in enumerate(estagio_atual):
                sv[i, 16] = float(st)

            ids = pol_vla.obter_ids(prompts_ativos)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits_18 = pol_vla.forward_pensamento(px, sv, gv, ids)
                LOGIT_CLIP = 3.0
                logits_18 = torch.tanh(logits_18 / LOGIT_CLIP) * LOGIT_CLIP
                probs = torch.softmax(logits_18 / temperatura, dim=-1)
                acoes_idx = torch.multinomial(probs, 1).squeeze(-1)

            # Entropia de Shannon por agente
            log_probs = torch.log(probs + 1e-8)
            entropias = -(probs * log_probs).sum(dim=-1) # [N]
            entropias_norm = entropias / math.log(18.0) # [0.0 a 1.0]

            idx_np = acoes_idx.detach().cpu().numpy()
            ent_np = entropias.detach().cpu().numpy()
            ent_norm_np = entropias_norm.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            sv_np = sv.detach().cpu().numpy()

            acoes = []
            for k in idx_np:
                if k < 9:
                    dx = int(pol_vla.YB[int(k)])
                    acoes.append({"hold": ["W"], "mouse": [dx, 0], "duration_ms": 250})
                else:
                    dx = int(pol_vla.YB[int(k - 9)])
                    acoes.append({"hold": ["W", "SPACE"], "mouse": [dx, 0], "duration_ms": 250})

            for i in range(N):
                if not vivo[i]:
                    acoes[i] = {"hold": [], "mouse": [0, 0], "duration_ms": 50}

            rr = post("/lote/passo", {"acoes": acoes, "frames": True})
            obs_prox = rr["obs"][:N]
            est_prox = [o["estado"] for o in obs_prox]
            pol.observar(obs_prox)

            for i in range(N):
                if not vivo[i]:
                    continue

                e_atual = est[i]
                e_prox = est_prox[i]
                tar = tarefas[i]
                st = estagio_atual[i]
                alvo_abs = tar["estagios"][min(st, len(tar["estagios"]) - 1)]["alvo_abs"]

                d_atual = math.hypot(alvo_abs[0] - e_atual["x"], alvo_abs[1] - e_atual["z"])
                d_prox = math.hypot(alvo_abs[0] - e_prox["x"], alvo_abs[1] - e_prox["z"])
                vel_passo = math.hypot(e_prox["x"] - e_atual["x"], e_prox["z"] - e_atual["z"])

                # Calcula a ação ótima causal teórica (a*) para este estado
                # Ângulo ideal rumo ao alvo
                dx_alvo = alvo_abs[0] - e_atual["x"]
                dz_alvo = alvo_abs[1] - e_atual["z"]
                yaw_rad = math.radians(e_atual["yaw"])
                # No Minecraft: frente = (-sin(yaw), -cos(yaw))
                ang_alvo = math.atan2(-dx_alvo, -dz_alvo)
                diff_rad = ang_alvo - yaw_rad
                diff_rad = (diff_rad + math.pi) % (2 * math.pi) - math.pi
                erro_yaw_graus = math.degrees(diff_rad)

                bin_otimo = calcular_bin_yaw(erro_yaw_graus)
                acao_otima = calcular_acao_18(bin_otimo, deve_pular=False)

                # Salva passo no histórico do episódio
                historico_episodios[i].append({
                    "passo": p,
                    "estagio": st,
                    "prompt": prompts_ativos[i],
                    "alvo_abs": alvo_abs,
                    "sv": sv_np[i].copy(),
                    "probs": probs_np[i].copy(),
                    "acao_executada": int(idx_np[i]),
                    "acao_otima": int(acao_otima),
                    "entropia": float(ent_np[i]),
                    "entropia_norm": float(ent_norm_np[i]),
                    "dist_atual": float(d_atual),
                    "dist_prox": float(d_prox),
                    "delta_dist": float(d_prox - d_atual),
                    "vel_passo": float(vel_passo),
                    "x": float(e_atual["x"]),
                    "z": float(e_atual["z"]),
                    "yaw": float(e_atual["yaw"])
                })

                total_passos_analisados += 1

                # Verifica transição de submeta
                if st == 0 and d_prox <= RAIO_CHEGADA_SUBMETA:
                    estagio_atual[i] = 1
                    submetas1_atingidas += 1
                    if len(tar["estagios"]) == 1:
                        vivo[i] = False
                elif st == 1 and len(tar["estagios"]) > 1:
                    if d_prox <= RAIO_CHEGADA_SUBMETA:
                        estagio_atual[i] = 2
                        submetas2_atingidas += 1
                        vivo[i] = False

            est = est_prox
            obs = obs_prox
            if not any(vivo):
                break

        # =========================================================================
        # CAUSAL CURATION FILTER (Filtragem a posteriori por episódio)
        # =========================================================================
        for i in range(N):
            hist = historico_episodios[i]
            if not hist:
                continue

            dist_min_ep = min(h["dist_atual"] for h in hist)
            fez_progresso_ep = dist_min_ep <= 3.5

            for t_idx, trans in enumerate(hist):
                ent_n = trans["entropia_norm"]

                # Critério 1: Pico de Entropia
                if ent_n < limiar_entropia_norm:
                    continue

                amostras_brutas_alta_entropia += 1

                # Critério 2: Filtro de Robô Preso / Colisão
                if trans["vel_passo"] < 0.04 and t_idx > 2:
                    amostras_descartadas_perdido += 1
                    continue

                # Critério 3: Filtro de Janela Causal Futura (nos próximos 4 passos, a distância não pode explodir)
                janela_futura = hist[t_idx : min(t_idx + 5, len(hist))]
                if len(janela_futura) > 1:
                    delta_janela = janela_futura[-1]["dist_atual"] - trans["dist_atual"]
                    # Se nos próximos 4 passos o robô se afastou mais de 1.2m do alvo, estava perdido
                    if delta_janela > 1.2:
                        amostras_descartadas_perdido += 1
                        continue

                # Critério 4: Relevância Estrutural (Spawn, Proximidade ou Transição)
                is_spawn = (t_idx <= 4)
                is_submeta_transicao = (trans["estagio"] == 1 and t_idx <= 20)
                is_near_target = (trans["dist_atual"] <= 4.0)

                # Se a entropia foi alta mas o robô não estava perto do alvo, nem no spawn, nem na transição, e o ep falhou feio
                if not (is_spawn or is_submeta_transicao or is_near_target or fez_progresso_ep):
                    amostras_descartadas_perdido += 1
                    continue

                # AMOSTRA VÁLIDA APROVADA PELO FILTRO CAUSAL!
                amostras_retidas_limpas.append({
                    "sv": torch.tensor(trans["sv"], dtype=torch.float32),
                    "prompt": trans["prompt"],
                    "acao_otima": trans["acao_otima"],
                    "acao_executada": trans["acao_executada"],
                    "entropia": trans["entropia"],
                    "entropia_norm": trans["entropia_norm"],
                    "dist_alvo": trans["dist_atual"],
                    "estagio": trans["estagio"],
                    "probs": trans["probs"],
                    "tipo_fork": "spawn" if is_spawn else ("transicao_submeta" if is_submeta_transicao else "ajuste_fino")
                })

        print(f"  Lote {lote_idx + 1}: Retidas {len(amostras_retidas_limpas)} decisões limpas de {amostras_brutas_alta_entropia} picos brutos de entropia.", flush=True)

    duracao = time.time() - t_inicio
    taxa_retencao = (len(amostras_retidas_limpas) / max(1, amostras_brutas_alta_entropia)) * 100.0

    print("\n" + "=" * 80)
    print(" [FASE 5] RESULTADOS DA MINERAÇÃO DE ALTA ENTROPIA")
    print(f"    Passos Totais Analisados    : {total_passos_analisados}")
    print(f"    Picos Brutos de Entropia    : {amostras_brutas_alta_entropia}")
    print(f"    Descartados (Robô Perdido)  : {amostras_descartadas_perdido}")
    print(f"    Decisões Limpas Retidas     : {len(amostras_retidas_limpas)} ({taxa_retencao:.1f}% de retenção)")
    print(f"    Submeta 1 Atingidas no Roll : {submetas1_atingidas}")
    print(f"    Submeta 2 Atingidas no Roll : {submetas2_atingidas}")
    print(f"    Tempo Total de Mineração    : {duracao:.1f}s")
    print("=" * 80)

    # Salva dataset no formato PyTorch
    os.makedirs(os.path.dirname(saida_pt), exist_ok=True)
    torch.save(amostras_retidas_limpas, saida_pt)
    print(f"[OK] Dataset de Decisões de Alta Entropia salvo em: {saida_pt}", flush=True)

    # Salva relatório JSON
    json_path = "fase5/dados/metricas_mineracao_entropia.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modelo_base": modelo_ckpt,
            "passos_totais": total_passos_analisados,
            "picos_brutos_entropia": amostras_brutas_alta_entropia,
            "descartados_ruido_perdido": amostras_descartadas_perdido,
            "decisoes_limpas_retidas": len(amostras_retidas_limpas),
            "taxa_retencao": taxa_retencao,
            "submetas1_atingidas": submetas1_atingidas,
            "submetas2_atingidas": submetas2_atingidas,
            "duracao_segundos": duracao,
            "distribuicao_tipos": {
                "spawn": sum(1 for a in amostras_retidas_limpas if a["tipo_fork"] == "spawn"),
                "transicao_submeta": sum(1 for a in amostras_retidas_limpas if a["tipo_fork"] == "transicao_submeta"),
                "ajuste_fino": sum(1 for a in amostras_retidas_limpas if a["tipo_fork"] == "ajuste_fino")
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Métricas salvas em: {json_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase5_coldstart.pt")
    ap.add_argument("--lotes", type=int, default=5)
    ap.add_argument("--passos", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--limiar", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--saida", default="fase5/dados/dataset_decisoes_alta_entropia.pt")
    args = ap.parse_args()

    minerar_decisoes(
        modelo_ckpt=args.ckpt,
        num_lotes=args.lotes,
        passos_max=args.passos,
        temperatura=args.temp,
        limiar_entropia_norm=args.limiar,
        seed=args.seed,
        saida_pt=args.saida
    )
