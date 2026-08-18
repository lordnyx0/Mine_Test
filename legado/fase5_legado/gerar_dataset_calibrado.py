# coding=utf-8
"""
fase5/gerar_dataset_calibrado.py — Gerador de Dataset Calibrado de Bifurcacoes.

Diferenca critica em relacao ao minerador_decisoes_entropia.py:

  O minerador coleta qualquer pico de entropia e rotula a acao_otima.
  Este script usa o modelo ANCORADO (que sabe navegar mas erra bifurcacoes)
  e aplica um filtro adicional:

  FILTRO DE CALIBRACAO: Descarta amostras em que o modelo ja escolheu
  a acao correta (acao_executada == acao_otima). Manter esses casos seria
  treinar em dados sem sinal de correcao. Queremos APENAS os casos de erro
  de decisao com alta entropia, onde o modelo estava incerto E errou.

  Resultado: dataset limpo de erros de bifurcacao com rotulo oraculo,
  pronto para fine-tuning de calibracao direcional.

Saida:
  - fase5/dados/dataset_calibrado.pt
  - fase5/dados/dataset_calibrado_metricas.json
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
from typing import List, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import montar_tarefas_logicas, RAIO_CHEGADA_SUBMETA
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora
from fase5.gerar_demonstracoes_esparsas import calcular_bin_yaw, calcular_acao_18

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def gerar_dataset_calibrado(
    modelo_ckpt: str = "checkpoints_vla/vla_fase5_sparse_anchored.pt",
    num_lotes: int = 8,
    passos_max: int = 100,
    temperatura: float = 0.85,
    limiar_entropia_norm: float = 0.33,
    seed: int = 777,
    saida_pt: str = "fase5/dados/dataset_calibrado.pt"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" [FASE 5] GERADOR DE DATASET CALIBRADO DE BIFURCACOES")
    print(f"    Checkpoint           : {modelo_ckpt}")
    print(f"    Lotes                : {num_lotes} (8 robos/lote = {num_lotes * 8} episodios)")
    print(f"    Temperatura Amostral : {temperatura}")
    print(f"    Limiar Entropia Norm : {limiar_entropia_norm:.2f}")
    print(f"    Saida Dataset        : {saida_pt}")
    print("=" * 80)

    # Carrega modelo ANCORADO
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
    else:
        print(f"[AVISO] Checkpoint '{modelo_ckpt}' nao encontrado. Rodando com pesos base.", flush=True)

    vla.to(device)
    vla.eval()

    info = get("/lote/info")
    N = info["envs"]
    print(f"[ENV] Conectado ao simulador com {N} ambientes paralelos.", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    passos_totais       = 0
    picos_brutos        = 0
    descartados_correto = 0   # modelo ja acertou -> sem sinal de calibracao
    descartados_perdido = 0   # robo preso / divergindo
    descartados_janela  = 0   # janela futura diverge > 1.5m
    retidos: List[Dict] = []
    submetas1_ok = 0
    submetas2_ok = 0
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
        historico = [[] for _ in range(N)]

        for p in range(passos_max):
            prompts_ativos = [
                tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)].get(
                    "prompt_estagio", tarefas[i]["prompt"])
                for i in range(N)
            ]
            alvos_ativos = [
                tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)]["alvo_abs"]
                for i in range(N)
            ]

            # Forward pass com acesso a logits e entropia
            px, sv, gv, u8 = pol_vla._entradas(est, alvos_ativos, obs)
            for i, st in enumerate(estagio_atual):
                sv[i, 16] = float(st)

            ids = pol_vla.obter_ids(prompts_ativos)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits_18 = pol_vla.forward_pensamento(px, sv, gv, ids)
                LOGIT_CLIP = 3.0
                logits_18 = torch.tanh(logits_18 / LOGIT_CLIP) * LOGIT_CLIP
                probs = torch.softmax(logits_18 / temperatura, dim=-1)
                acoes_idx = torch.multinomial(probs, 1).squeeze(-1)

            log_p      = torch.log(probs + 1e-8)
            entropias  = -(probs * log_p).sum(dim=-1)
            ent_norm   = entropias / math.log(18.0)

            idx_np   = acoes_idx.detach().cpu().numpy()
            ent_np   = entropias.detach().cpu().numpy()
            enorm_np = ent_norm.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            sv_np    = sv.detach().cpu().numpy()

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

                e_cur = est[i]
                e_nxt = est_prox[i]
                tar   = tarefas[i]
                st    = estagio_atual[i]
                alvo  = tar["estagios"][min(st, len(tar["estagios"]) - 1)]["alvo_abs"]

                d_cur = math.hypot(alvo[0] - e_cur["x"], alvo[1] - e_cur["z"])
                d_nxt = math.hypot(alvo[0] - e_nxt["x"], alvo[1] - e_nxt["z"])
                vel   = math.hypot(e_nxt["x"] - e_cur["x"], e_nxt["z"] - e_cur["z"])

                # Acao otima (oraculo): angulo geometrico real ate o alvo
                dx_alvo    = alvo[0] - e_cur["x"]
                dz_alvo    = alvo[1] - e_cur["z"]
                yaw_rad    = math.radians(e_cur["yaw"])
                ang_alvo   = math.atan2(-dx_alvo, -dz_alvo)
                diff_rad   = (ang_alvo - yaw_rad + math.pi) % (2 * math.pi) - math.pi
                erro_graus = math.degrees(diff_rad)

                bin_otimo  = calcular_bin_yaw(erro_graus)
                acao_otima = calcular_acao_18(bin_otimo, deve_pular=False)

                historico[i].append({
                    "passo":          p,
                    "estagio":        st,
                    "prompt":         prompts_ativos[i],
                    "alvo_abs":       alvo,
                    "sv":             sv_np[i].copy(),
                    "probs":          probs_np[i].copy(),
                    "acao_executada": int(idx_np[i]),
                    "acao_otima":     int(acao_otima),
                    "entropia":       float(ent_np[i]),
                    "entropia_norm":  float(enorm_np[i]),
                    "dist_atual":     float(d_cur),
                    "dist_prox":      float(d_nxt),
                    "vel_passo":      float(vel),
                    "x":              float(e_cur["x"]),
                    "z":              float(e_cur["z"]),
                    "yaw":            float(e_cur["yaw"]),
                    "erro_yaw_graus": float(erro_graus),
                })

                passos_totais += 1

                # Transicao de submeta
                if st == 0 and d_nxt <= RAIO_CHEGADA_SUBMETA:
                    estagio_atual[i] = 1
                    submetas1_ok += 1
                    if len(tar["estagios"]) == 1:
                        vivo[i] = False
                elif st == 1 and len(tar["estagios"]) > 1 and d_nxt <= RAIO_CHEGADA_SUBMETA:
                    estagio_atual[i] = 2
                    submetas2_ok += 1
                    vivo[i] = False

            est = est_prox
            obs = obs_prox
            if not any(vivo):
                break

        # ======================================================================
        # FILTRO DUPLO: CALIBRACAO + CAUSAL
        # ======================================================================
        lote_retido = 0
        for i in range(N):
            hist = historico[i]
            if not hist:
                continue

            dist_min_ep   = min(h["dist_atual"] for h in hist)
            fez_progresso = dist_min_ep <= 3.5

            for t_idx, trans in enumerate(hist):
                ent_n = trans["entropia_norm"]

                # Criterio 0: Pico de Entropia
                if ent_n < limiar_entropia_norm:
                    continue
                picos_brutos += 1

                # FILTRO DE CALIBRACAO: modelo ja acertou -> sem sinal de correcao
                if trans["acao_executada"] == trans["acao_otima"]:
                    descartados_correto += 1
                    continue

                # Criterio 1: Robo preso / velocidade nula
                if trans["vel_passo"] < 0.04 and t_idx > 2:
                    descartados_perdido += 1
                    continue

                # Criterio 2: Janela causal futura (proximos 5 passos)
                janela = hist[t_idx: min(t_idx + 5, len(hist))]
                if len(janela) > 1:
                    delta_janela = janela[-1]["dist_atual"] - trans["dist_atual"]
                    if delta_janela > 1.5:
                        descartados_janela += 1
                        continue

                # Criterio 3: Relevancia estrutural
                is_spawn       = (t_idx <= 4)
                is_transicao   = (trans["estagio"] == 1 and t_idx <= 20)
                is_near_target = (trans["dist_atual"] <= 5.0)

                if not (is_spawn or is_transicao or is_near_target or fez_progresso):
                    descartados_perdido += 1
                    continue

                # AMOSTRA VALIDA: ERRO DE BIFURCACAO COM ROTULO ORACULO
                tipo = ("spawn" if is_spawn
                        else "transicao_submeta" if is_transicao
                        else "erro_bifurcacao")
                retidos.append({
                    "sv":             torch.tensor(trans["sv"], dtype=torch.float32),
                    "prompt":         trans["prompt"],
                    "acao_otima":     trans["acao_otima"],
                    "acao_executada": trans["acao_executada"],
                    "entropia":       trans["entropia"],
                    "entropia_norm":  trans["entropia_norm"],
                    "dist_alvo":      trans["dist_atual"],
                    "estagio":        trans["estagio"],
                    "erro_yaw_graus": trans["erro_yaw_graus"],
                    "probs":          trans["probs"],
                    "tipo":           tipo,
                })
                lote_retido += 1

        print(
            f"  Lote {lote_idx + 1}: {lote_retido} erros de bifurcacao retidos "
            f"(acumulado: {len(retidos)})",
            flush=True
        )

    # Relatorio final
    duracao = time.time() - t_inicio
    taxa_retencao = (len(retidos) / max(1, picos_brutos)) * 100.0

    print("\n" + "=" * 80)
    print(" [FASE 5] RESULTADOS DA COLETA CALIBRADA")
    print(f"    Passos Totais Analisados     : {passos_totais}")
    print(f"    Picos Brutos de Entropia     : {picos_brutos}")
    print(f"    Descartados (Ja Corretos)    : {descartados_correto}  <- filtro de calibracao")
    print(f"    Descartados (Perdido/Preso)  : {descartados_perdido}")
    print(f"    Descartados (Janela Futura)  : {descartados_janela}")
    print(f"    Erros de Bifurcacao Retidos  : {len(retidos)} ({taxa_retencao:.1f}% de retencao)")
    print(f"    Submeta 1 Atingidas          : {submetas1_ok}")
    print(f"    Submeta 2 Atingidas          : {submetas2_ok}")
    print(f"    Duracao Total                : {duracao:.1f}s")
    print("=" * 80)

    if not retidos:
        print("[AVISO] Nenhuma amostra retida. Tente --lotes maior ou --limiar menor.", flush=True)
        return

    tipos = {}
    for a in retidos:
        tipos[a["tipo"]] = tipos.get(a["tipo"], 0) + 1

    print("\n  Distribuicao por tipo:")
    for t, cnt in sorted(tipos.items()):
        print(f"    {t:25s}: {cnt:4d} ({100*cnt/len(retidos):.1f}%)")

    erros = [abs(a["erro_yaw_graus"]) for a in retidos]
    print(f"\n  Erro angular medio       : {np.mean(erros):.1f} graus")
    print(f"  Erro angular mediana     : {np.median(erros):.1f} graus")
    pct45 = sum(1 for e in erros if e > 45)
    print(f"  Erro angular > 45 graus  : {pct45} ({100*pct45/len(erros):.1f}%)")

    os.makedirs(os.path.dirname(saida_pt), exist_ok=True)
    torch.save(retidos, saida_pt)
    print(f"\n[OK] Dataset calibrado salvo em: {saida_pt}  ({len(retidos)} amostras)", flush=True)

    json_path = saida_pt.replace(".pt", "_metricas.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
            "modelo_base":          modelo_ckpt,
            "passos_totais":        passos_totais,
            "picos_brutos":         picos_brutos,
            "descartados_corretos": descartados_correto,
            "descartados_perdido":  descartados_perdido,
            "descartados_janela":   descartados_janela,
            "erros_bifurcacao":     len(retidos),
            "taxa_retencao":        taxa_retencao,
            "submetas1":            submetas1_ok,
            "submetas2":            submetas2_ok,
            "duracao_s":            duracao,
            "tipos":                tipos,
            "erro_angular_medio":   float(np.mean(erros)),
            "erro_angular_mediana": float(np.median(erros)),
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metricas salvas em: {json_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   default="checkpoints_vla/vla_fase5_sparse_anchored.pt")
    ap.add_argument("--lotes",  type=int,   default=8)
    ap.add_argument("--passos", type=int,   default=100)
    ap.add_argument("--temp",   type=float, default=0.85)
    ap.add_argument("--limiar", type=float, default=0.33)
    ap.add_argument("--seed",   type=int,   default=777)
    ap.add_argument("--saida",  default="fase5/dados/dataset_calibrado.pt")
    args = ap.parse_args()

    gerar_dataset_calibrado(
        modelo_ckpt=args.ckpt,
        num_lotes=args.lotes,
        passos_max=args.passos,
        temperatura=args.temp,
        limiar_entropia_norm=args.limiar,
        seed=args.seed,
        saida_pt=args.saida,
    )
