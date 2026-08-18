# coding=utf-8
"""
Avaliacao da FASE 2 — terreno real, quebrada por GRAU DE OBSTRUCAO.

O agregado esconde o que importa. Uma politica que empata com o baseline cego em
alvo de reta livre e perde nos obstruidos ainda marca bem na media — e a media
foi o que me enganou duas vezes na Fase 1.

Por isso tudo aqui e reportado por faixa de `desvio`:

    reta       o baseline cego e OTIMO; a rede so pode empatar
    obstruido  ha algo no meio; SO AQUI a visao tem o que contribuir

**A evidencia de uso da visao e ganhar do `geo_pulo` na faixa "obstruido".**

    python avaliar_fase2.py --episodios 96 --politicas geo_pulo,piloto
"""
import os
import sys
import json
import time
import random
import argparse
import statistics as st
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiente.arena_plana import rodar_lote, get, RAIO_CHEGADA, DIST_MIN, DIST_MAX
from ambiente.fase2 import (montar_tarefas, faixa_de, GeometricoComPulo, AleatorioComPulo,
                            PilotoBFS, PASSOS_MAX_F2, FAIXAS, BANCO, carregar_banco_split)

LARGADAS = "dataset/largadas_fase2.json"
ORDEM = ("reta", "leve", "obstruido")


def avaliar(politica, tarefas, N, com_frames=False):
    res = []
    for b0 in range(0, len(tarefas), N):
        lote = tarefas[b0:b0 + N]
        r = rodar_lote(politica, [t["largada"] for t in lote],
                       [t["rel"] for t in lote],
                       passos=PASSOS_MAX_F2, com_frames=com_frames)
        for x, t in zip(r, lote):
            x["desvio"] = t["desvio"]
            x["faixa"] = faixa_de(t["desvio"])
        res += r
    return res


def relatar(nome, res):
    n = len(res)
    taxa = sum(r["chegou"] for r in res) / n
    print("\n  %-10s geral %5.1f%%   (%d de %d)" % (nome.upper(), 100 * taxa,
                                                    sum(r["chegou"] for r in res), n))
    print("      %-11s %8s %9s %8s %6s" % ("faixa", "chegada", "d_final", "desvio", "n"))
    por_faixa = {}
    for f in ORDEM:
        sub = [r for r in res if r["faixa"] == f]
        if not sub:
            continue
        t = sum(r["chegou"] for r in sub) / len(sub)
        por_faixa[f] = t
        print("      %-11s %7.0f%% %9.2f %8.2f %6d"
              % (f, 100 * t, st.mean(r["dfinal"] for r in sub),
                 st.mean(r["desvio"] for r in sub), len(sub)))
    return taxa, por_faixa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodios", type=int, default=80)
    ap.add_argument("--politicas", default="geo_pulo,piloto")
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase2.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desvio-min", type=float, default=0.0,
                    help="so alvos com desvio >= isto (obstruidos)")
    ap.add_argument("--raio", type=int, default=40)
    ap.add_argument("--held-out", action="store_true", default=False,
                    help="avalia estritamente no split reservado (15%%) do banco pre-gerado")
    ap.add_argument("--amostrar", action="store_true", default=False,
                    help="avaliacao em modo estocastico (amostragem) em vez de argmax")
    args = ap.parse_args()

    N = get("/lote/info")["envs"]

    if args.held_out and os.path.exists(BANCO):
        _, held = carregar_banco_split(BANCO)
        if args.desvio_min > 0:
            held = [t for t in held if t.get("desvio", 0) >= args.desvio_min]
        rng = random.Random(args.seed)
        if len(held) > args.episodios:
            held = rng.sample(held, args.episodios)
        tarefas = held
        print("[fase2] avaliando sobre HELD-OUT (%d tarefas reservadas nunca vistas no treino)"
              % len(tarefas), flush=True)
    else:
        largadas = [tuple(v) for v in json.load(open(LARGADAS, encoding="utf-8"))]
        print("[fase2] %d largadas em terreno real | alvo %.0f-%.0f blocos | "
              "raio %.1f | %d passos" % (len(largadas), DIST_MIN, DIST_MAX,
                                         RAIO_CHEGADA, PASSOS_MAX_F2), flush=True)
        tarefas = montar_tarefas(largadas, args.episodios, args.seed,
                                 raio=args.raio, desvio_min=args.desvio_min)
        print("[fase2] %d tarefas alcancaveis geradas ao vivo" % len(tarefas), flush=True)

    res_por = {}
    neural = None
    for nome in args.politicas.split(","):
        t0 = time.time()
        if nome == "geo_pulo":
            pol, frames = GeometricoComPulo(), False
        elif nome == "aleatorio_pulo":
            pol, frames = AleatorioComPulo(seed=args.seed), False
        elif nome == "piloto":
            pol, frames = PilotoBFS(raio=args.raio), False
        elif nome in ("modelo", "modelo_cego"):
            from politica.politica_fase2 import PoliticaFase2
            if neural is None:
                neural = PoliticaFase2(args.ckpt, amostrar=args.amostrar)
            neural.cego = (nome == "modelo_cego")
            neural.amostrar = args.amostrar
            pol, frames = neural, True
        else:
            print("  politica desconhecida: %s" % nome)
            continue
        res = avaliar(pol, tarefas, N, com_frames=frames)
        res_por[nome] = relatar(nome, res)
        print("      (%.0fs)" % (time.time() - t0), flush=True)

    if "geo_pulo" in res_por and "piloto" in res_por:
        print("\n" + "=" * 62)
        gp, pl = res_por["geo_pulo"][1], res_por["piloto"][1]
        print("  ESPACO PARA APRENDER (piloto - geo_pulo), por faixa:")
        for f in ORDEM:
            if f in gp and f in pl:
                print("      %-11s %+5.0f pontos   (%.0f%% -> %.0f%%)"
                      % (f, 100 * (pl[f] - gp[f]), 100 * gp[f], 100 * pl[f]))
        obst = pl.get("obstruido", 0) - gp.get("obstruido", 0)
        if obst < 0.10:
            print("\n  -> Espaco pequeno mesmo em 'obstruido'. A tarefa NAO exige")
            print("     visao o bastante; endurecer antes de treinar.")
        else:
            print("\n  -> Ha espaco real em 'obstruido'. E ali que a visao precisa")
            print("     aparecer, e e esse numero que o modelo tem que atacar.")
        print("=" * 62)

    if "modelo" in res_por:
        print("\n  modelo por faixa vs baselines:")
        for f in ORDEM:
            linha = ["%s %.0f%%" % (k, 100 * v[1].get(f, 0)) for k, v in res_por.items()
                     if f in v[1]]
            print("      %-11s %s" % (f, " | ".join(linha)))
    print()


if __name__ == "__main__":
    main()
