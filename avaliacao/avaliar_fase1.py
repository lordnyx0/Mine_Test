# coding=utf-8
"""
Avaliacao da FASE 1 — alvo local (<=8 blocos), terreno plano, so W + giro.

Reporta as cinco metricas pedidas e a QUEBRA POR SETOR. A quebra nao e enfeite:
uma politica que so funciona com o alvo a frente marca bem na media e nao serve.
Por isso os alvos sao amostrados com cobertura igual dos 8 setores.

Politicas:
  so_W         — anda reto, nunca gira. Mede quanto do sucesso vem so de o alvo
                 estar perto. E o piso.
  geometrico   — trigonometria pura, sem visao e sem rede. E o TETO desta fase,
                 e a pergunta do experimento e se a rede chega perto dele.
  modelo       — a politica treinada (--ckpt)
  modelo_cego  — a MESMA politica com os pixels zerados. Controle obrigatorio:
                 terreno plano com alvo entregue em coordenadas e resoluvel por
                 trigonometria sem olhar nada. Se `modelo` e `modelo_cego`
                 empatam, a visao nao contribuiu e o veredito NAO e sucesso.

    python avaliar_fase1.py --episodios 128 --politicas so_W,geometrico
"""
import os
import sys
import json
import math
import time
import random
import argparse
import statistics as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiente.arena_plana import (post, get, amostrar_alvo, rodar_lote, SETORES,
                                  ControladorGeometrico, SoAndar, PasseioAleatorio,
                                  DistribuicaoFixa, RAIO_CHEGADA,
                                  DIST_MIN, DIST_MAX, PASSOS_MAX)

ARENAS = "dataset/arenas_planas.json"


def carregar_arenas(caminho=ARENAS):
    with open(caminho, encoding="utf-8") as f:
        return [tuple(v) for v in json.load(f)]


def montar_tarefas(arenas, n_ep, seed=0):
    """Uma tarefa = (arena, alvo relativo). Setores em rodizio para cobertura
    exatamente uniforme, nao apenas uniforme em esperanca."""
    rng = random.Random(seed)
    tarefas = []
    for k in range(n_ep):
        arena = arenas[k % len(arenas)]
        dx, dz, setor = amostrar_alvo(rng, setor=SETORES[k % 8])
        tarefas.append((arena, (dx, dz, setor)))
    return tarefas


def avaliar(politica, tarefas, N, com_frames=False):
    res = []
    for b0 in range(0, len(tarefas), N):
        lote = tarefas[b0:b0 + N]
        res += rodar_lote(politica, [t[0] for t in lote], [t[1] for t in lote],
                          com_frames=com_frames)
    return res


def relatar(nome, res):
    n = len(res)
    ok = [r for r in res if r["chegou"]]
    taxa = len(ok) / n
    print("\n  %s" % nome.upper())
    print("    taxa de chegada .... %5.1f%%  (%d de %d)" % (100 * taxa, len(ok), n))
    print("    distancia inicial .. %5.2f" % st.mean(r["d0"] for r in res))
    print("    distancia final .... %5.2f" % st.mean(r["dfinal"] for r in res))
    print("    distancia minima ... %5.2f" % st.mean(r["dmin"] for r in res))
    print("    passos ate chegar .. %s"
          % ("%.1f" % st.mean(r["passos"] for r in ok) if ok else "-"))
    print("    recompensa media ... %+5.2f" % st.mean(r["recompensa"] for r in res))

    print("    por setor:")
    print("      %-9s %8s %8s %8s" % ("setor", "chegada", "d_final", "n"))
    for s in SETORES:
        sub = [r for r in res if r["setor"] == s]
        if not sub:
            continue
        t = sum(r["chegou"] for r in sub) / len(sub)
        print("      %-9s %7.0f%% %8.2f %8d"
              % (s, 100 * t, st.mean(r["dfinal"] for r in sub), len(sub)))
    piores = min((sum(r["chegou"] for r in [x for x in res if x["setor"] == s])
                  / max(1, len([x for x in res if x["setor"] == s])), s)
                 for s in SETORES)
    print("    pior setor ......... %s (%.0f%%)" % (piores[1], 100 * piores[0]))
    return taxa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodios", type=int, default=128)
    ap.add_argument("--politicas", default="so_W,aleatorio,geometrico")
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase1.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arenas", default="todas", choices=["todas","treino","teste"],
                    help="teste = as 25%% reservadas, nunca vistas no treino")
    args = ap.parse_args()

    todas = carregar_arenas()
    corte = max(1, int(0.75 * len(todas)))     # mesmo corte de treinar_fase1.py
    arenas = {"todas": todas, "treino": todas[:corte], "teste": todas[corte:]}[args.arenas]
    N = get("/lote/info")["envs"]
    tarefas = montar_tarefas(arenas, args.episodios, args.seed)
    print("[fase1] %d episodios | %d arenas planas | alvo %.0f-%.0f blocos | "
          "raio de chegada %.1f | maximo %d passos"
          % (len(tarefas), len(arenas), DIST_MIN, DIST_MAX, RAIO_CHEGADA, PASSOS_MAX),
          flush=True)
    print("[fase1] setores em rodizio: cada um recebe %d episodios"
          % (len(tarefas) // 8), flush=True)

    resultados = {}
    neural = None      # carregado UMA vez: travar_gpu() aborta na segunda
    for nome in args.politicas.split(","):
        t0 = time.time()
        if nome == "so_W":
            pol, frames = SoAndar(), False
        elif nome == "aleatorio":
            pol, frames = PasseioAleatorio(), False
        elif nome == "dist_fixa":
            pol, frames = DistribuicaoFixa(), False
        elif nome == "geometrico":
            pol, frames = ControladorGeometrico(), False
        elif nome in ("modelo", "modelo_cego", "modelo_amostrado"):
            from politica.politica_fase1 import PoliticaNeural
            if neural is None:
                neural = PoliticaNeural(args.ckpt)
            # O treino amostra da softmax; a avaliacao usa argmax. Se a politica
            # for estocastica e util, os dois numeros divergem — e e o argmax
            # que esta errado como leitura, nao a politica.
            neural.amostrar = (nome == "modelo_amostrado")
            # Mesmos pesos, so muda se os pixels sao zerados — o que torna a
            # ablacao uma comparacao limpa e nao dois carregamentos distintos.
            neural.cego = (nome == "modelo_cego")
            pol, frames = neural, True
        else:
            print("  politica desconhecida: %s" % nome)
            continue
        res = avaliar(pol, tarefas, N, com_frames=frames)
        resultados[nome] = relatar(nome, res)
        print("    (%.0fs)" % (time.time() - t0), flush=True)

    if "geometrico" in resultados and "modelo" in resultados:
        print("\n" + "=" * 60)
        print("  modelo %.0f%%  vs  geometrico %.0f%%  ->  %.0f%% do baseline"
              % (100 * resultados["modelo"], 100 * resultados["geometrico"],
                 100 * resultados["modelo"] / max(1e-9, resultados["geometrico"])))
        if "modelo_cego" in resultados:
            d = resultados["modelo"] - resultados["modelo_cego"]
            print("  contribuicao da VISAO: %+.1f pontos (modelo %.0f%% vs cego %.0f%%)"
                  % (100 * d, 100 * resultados["modelo"], 100 * resultados["modelo_cego"]))
            if abs(d) < 0.05:
                print("  -> a visao NAO contribuiu. Mesmo que a taxa seja alta, a")
                print("     Fase 1 nao testou a ponte multimodal.")
        print("=" * 60)
    print()


if __name__ == "__main__":
    main()
