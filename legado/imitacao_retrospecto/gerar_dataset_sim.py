# coding=utf-8
"""
Gera dataset dentro do SIMULADOR, com o Piloto (planejador BFS) como professor.

Por que este professor: em teste pareado com 180 posicoes identicas ele fez
41.4 blocos por episodio contra 19.2 de andar reto (+22.3, IC95% [+18.9,+25.6],
venceu em 78% das posicoes). E o primeiro professor que supera de forma
inequivoca uma politica trivial — os anteriores empatavam ou perdiam.

DOIS OBJETIVOS de proposito. Com uma tarefa so, a acao correta acaba sendo
quase sempre a mesma e o alvo vira constante — foi assim que a via visual
colapsou (posto efetivo 1.4 de 1024). Com "explorar" e "coletar madeira"
gerando acoes DIFERENTES na MESMA cena, instrucao e imagem passam a ser ambas
necessarias para acertar.

    python gerar_dataset_sim.py --episodios 60
"""
import os
import io
import json
import time
import base64
import random
import argparse
import urllib.request

from estado_sim import EstadoEpisodio, N_FRAMES

BASE = "http://127.0.0.1:3002"
SAIDA = "dataset/sim_piloto.jsonl"

OBJETIVOS = {
    "explorar": ("Objetivo: explorar. Ande o maximo que puder e afaste-se o maximo "
                 "possivel do ponto onde voce nasceu. Nao fique parado no mesmo lugar."),
    "bloco": ("Objetivo: coletar madeira. Encontre uma arvore e va ate o tronco."),
}


def post(caminho, corpo, timeout=300):
    d = json.dumps(corpo).encode()
    r = urllib.request.Request(BASE + caminho, data=d, method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodios", type=int, default=60, help="episodios por objetivo")
    ap.add_argument("--passos", type=int, default=70)
    ap.add_argument("--saida", default=SAIDA)
    # Raio 16 rende IGUAL a 40 (40.8 vs 40.8, ambos +25 sobre andar cego), mas
    # decide com informacao proxima do que o aluno enxerga. Com raio 40 a acao
    # do professor dependia de terreno invisivel e virava ruido para o aluno:
    # uma MLP com acesso direto a rotas+estado so alcancava 35% de recall.
    ap.add_argument("--raio", type=int, default=16)
    # compromisso=1 (replanejar todo passo) deixa o professor MARKOVIANO no
    # espaco de observacao do aluno. Com 12 passos de compromisso, a acao
    # dependia de um alvo escolhido ate 3s antes — estado interno invisivel ao
    # aluno, que tornava o rotulo impossivel de inferir. Medido: markoviano
    # empata em desempenho (35.6 vs 38.2, IC [-6.2,+1.2]).
    ap.add_argument("--compromisso", type=int, default=1)
    args = ap.parse_args()

    random.seed(0)
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    if os.path.exists(args.saida):
        os.remove(args.saida)

    info = json.loads(urllib.request.urlopen(BASE + "/lote/info", timeout=30).read())
    N = info["envs"]
    print(f"[sim] {N} ambientes | {info['frame']}", flush=True)

    escritos = 0
    t0 = time.time()

    with open(args.saida, "a", encoding="utf-8") as f:
        for objetivo in ("explorar", "bloco"):
            rodadas = max(1, args.episodios // N)
            for rod in range(rodadas):
                r = post("/lote/reset", {})
                estados = [EstadoEpisodio() for _ in range(N)]
                for i, o in enumerate(r["obs"]):
                    estados[i].reiniciar(o["estado"])
                    estados[i].registrar(o["estado"], base64.b64decode(o["frame_b64"]))

                for t in range(args.passos):
                    extra = {"raio": args.raio}
                    if objetivo == "bloco":
                        extra["blocos"] = ["log", "log2"]
                    acoes = post("/lote/piloto", {"objetivo": objetivo, "extra": extra,
                                                 "compromisso": args.compromisso})["acoes"]

                    # Observacao ANTES de agir — e sobre ela que a decisao foi tomada
                    amostras = []
                    for i in range(N):
                        pilha = estados[i].pilha_frames()
                        if len(pilha) < N_FRAMES:
                            continue
                        # Posicao e id de episodio: sem eles nao da para rotular
                        # em RETROSPECTO (onde o agente de fato chegou k passos
                        # depois vira o objetivo do passo t). O state_vec so tem
                        # deslocamento para TRAS, e o objetivo precisa do futuro.
                        st = estados[i]._st
                        amostras.append({
                            "i": i,
                            "instrucao": OBJETIVOS[objetivo],
                            "objetivo": objetivo,
                            "action": {k: v for k, v in acoes[i].items() if not k.startswith("_")},
                            "state_vec": [round(v, 5) for v in estados[i].vetor()],
                            "frames_b64": [base64.b64encode(b).decode("ascii") for b in pilha],
                            "ep": "%s-r%d-e%d" % (objetivo, rod, i),
                            "t": t,
                            "pos": [round(st["x"], 3), round(st["z"], 3), round(st["yaw"], 2)],
                        })

                    resp = post("/lote/passo", {"acoes": acoes})
                    for i, o in enumerate(resp["obs"]):
                        estados[i].passo(o["estado"], base64.b64decode(o["frame_b64"]))

                    # rotas medidas DEPOIS do reset de estado, mas referentes a cena vista
                    for a in amostras:
                        i = a.pop("i")
                        a["rotas"] = resp["obs"][i].get("rotas")
                        a["reward"] = 1.0
                        a["task"] = "sim_piloto"
                        f.write(json.dumps(a, ensure_ascii=False) + "\n")
                        escritos += 1

                dt = time.time() - t0
                print(f"  [{objetivo}] rodada {rod+1}/{rodadas} | {escritos} amostras "
                      f"| {dt/60:.1f} min | {escritos/max(dt,1):.0f} amostras/s", flush=True)

    dt = time.time() - t0
    print(f"\n[OK] {escritos} amostras em {args.saida} ({dt/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
