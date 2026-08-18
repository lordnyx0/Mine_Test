# coding=utf-8
"""
FASE 3 — Avaliação de Navegação Servo-Visual (Sem Coordenadas) com Ablação.

Compara:
  - aleatorio: piso estocástico (ignora a entrada).
  - so_W: controle cego reto.
  - piloto_bloco: teto BFS (lê voxels via Objetivos.bloco).
  - modelo: VLA com visão ativa.
  - modelo_cego: VLA com pixels zerados (ablação obrigatória).
"""
import os
import sys
import time
import math
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiente.arena_plana import rodar_lote, get, post
from ambiente.fase3 import (montar_tarefas_visuais, AleatorioVisual, SempreFrente,
                            PilotoBloco, PASSOS_MAX_F3, RAIO_CHEGADA_F3)


def rodar_lote_f3(pol, lote, passos=PASSOS_MAX_F3, com_frames=True):
    n = len(lote)
    alvos_abs = [t["alvo_abs"] for t in lote]
    prompts = [t["prompt"] for t in lote]

    r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in lote]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    if hasattr(pol, "reiniciar"):
        pol.reiniciar(obs)

    blocos = [{"env": i, "x": math.floor(t["alvo_abs"][0]), "y": t["alvo_y"],
               "z": math.floor(t["alvo_abs"][1]), "id": t.get("bloco_id", 49)} for i, t in enumerate(lote)]
    post("/lote/colocar_bloco", {"blocos": blocos})

    d0 = [math.hypot(alvos_abs[i][0] - est[i]["x"], alvos_abs[i][1] - est[i]["z"]) for i in range(n)]
    dant = list(d0)
    dmin = list(d0)
    chegou_em = [None] * n

    for t in range(passos):
        if hasattr(pol, "prompts_atuais"):
            acoes = pol.agir(est, alvos_abs, obs, prompts=prompts)
        else:
            acoes = pol.agir(est, alvos_abs, obs)
        rr = post("/lote/passo", {"acoes": acoes, "frames": com_frames})
        obs = rr["obs"][:n]
        est = [o["estado"] for o in obs]
        if hasattr(pol, "observar"):
            pol.observar(obs)

        for i in range(n):
            d = math.hypot(alvos_abs[i][0] - est[i]["x"], alvos_abs[i][1] - est[i]["z"])
            dant[i] = d
            dmin[i] = min(dmin[i], d)
            if d <= RAIO_CHEGADA_F3 and chegou_em[i] is None:
                chegou_em[i] = t + 1

    return [{"d0": d0[i], "dmin": dmin[i], "dfinal": dant[i],
             "chegou": chegou_em[i] is not None, "passos": chegou_em[i],
             "cor": lote[i]["alvo_nome"]}
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodios", type=int, default=48)
    ap.add_argument("--politicas", default="aleatorio,so_W,piloto_bloco,modelo,modelo_cego")
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase3.pt")
    ap.add_argument("--amostrar", action="store_true", default=True)
    ap.add_argument("--passos", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N = get("/lote/info")["envs"]
    tarefas = montar_tarefas_visuais(args.episodios, seed=args.seed)
    print(f"[fase3] {len(tarefas)} tarefas visuais multi-cores (Roxo, Amarelo, Azul) geradas", flush=True)

    res_por = {}
    neural = None
    for nome in args.politicas.split(","):
        t0 = time.time()
        if nome == "aleatorio":
            pol, frames = AleatorioVisual(seed=args.seed), False
        elif nome == "so_W":
            pol, frames = SempreFrente(), False
        elif nome == "piloto_bloco":
            pol, frames = PilotoBloco(), False
        elif nome in ("modelo", "modelo_cego"):
            from politica.politica_fase3 import PoliticaFase3
            if neural is None:
                neural = PoliticaFase3(args.ckpt, amostrar=args.amostrar)
            neural.cego = (nome == "modelo_cego")
            pol, frames = neural, True
        elif nome == "cerebro_vla":
            from politica.politica_fase3 import PoliticaFase3
            from politica.cerebro import PoliticaCerebroVLA
            if neural is None:
                neural = PoliticaFase3(args.ckpt, amostrar=args.amostrar)
            neural.cego = False
            pol, frames = PoliticaCerebroVLA(neural), True
        else:
            print(f"  politica desconhecida: {nome}")
            continue

        res = []
        for b0 in range(0, len(tarefas), N):
            lote = tarefas[b0:b0 + N]
            r = rodar_lote_f3(pol, lote, passos=args.passos, com_frames=frames)
            res += r

        taxa = sum(x["chegou"] for x in res) / len(res)
        d_final_medio = sum(x["dfinal"] for x in res) / len(res)
        res_por[nome] = (taxa, d_final_medio, res)
        print("  %-14s chegada %5.1f%% (%d de %d) | d_final: %5.2fm | (%.0fs)"
              % (nome.upper(), 100 * taxa, sum(x["chegou"] for x in res), len(res),
                 d_final_medio, time.time() - t0), flush=True)

    print("\n" + "=" * 60)
    print("RESUMO DA AVALIAÇÃO FASE 3 (ALVOS VISUAIS MULTI-CORES):")
    for k, (tx, df, r_list) in res_por.items():
        print(f"  {k:<16}: chegada {100*tx:5.1f}% | d_final: {df:5.2f}m")
        # Desagregação por cor
        for c in ("roxo", "amarelo", "azul"):
            sub = [x for x in r_list if x.get("cor") == c]
            if sub:
                tx_c = sum(x["chegou"] for x in sub) / len(sub)
                print(f"     -> cor {c:<8}: {100*tx_c:4.1f}% ({sum(x['chegou'] for x in sub)}/{len(sub)})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
