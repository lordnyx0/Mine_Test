# coding=utf-8
"""
FASE 2 — terreno real, com obstaculos, e `SPACE`.

A pergunta muda. A Fase 1 perguntava se o laco de RL fecha; ela respondeu que
sim, mas a tarefa era resolvivel por `atan2` sem olhar um pixel, entao nao
testou a ponte multimodal. Aqui a pergunta e:

    A politica usa a IMAGEM para contornar o que as coordenadas nao descrevem?

O que muda:

  terreno    plano filtrado (13 arenas)  ->  terreno REAL, sem filtro (milhares)
  acoes      W + giro                    ->  W + giro + SPACE
  piso       geometrico, 86%             ->  geometrico COM pulo (cego, bate na
                                             parede)
  teto       —                           ->  piloto BFS (le voxels)

**A inversao de papel do baseline geometrico e o ponto central.** Na Fase 1 ele
era o teto e a pergunta era se a rede chegava perto. Aqui ele e o PISO: ele e
cego por construcao, entao falha exatamente onde a visao seria necessaria. A
distancia entre ele e o piloto e o tamanho do que ha para ganhar.

## A medida que decide: `desvio`

    desvio = custo do caminho da busca / distancia em linha reta

  ~1.0   reta livre. O geometrico e OTIMO aqui; a rede so pode empatar.
  >1.3   ha algo no meio. So AQUI existe algo para a visao contribuir.

Reportar o agregado esconde isso: uma rede que empata em desvio 1.0 e perde em
desvio alto pode marcar bem na media. Por isso tudo aqui e quebrado por faixa
de desvio — e a evidencia de uso da visao e ganhar sobre o geometrico
ESPECIFICAMENTE nos alvos obstruidos.
"""
import os
import sys
import math
import json
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiente.arena_plana import (post, get, amostrar_alvo, alvo_para_mundo, erro_angular,
                                  bin_para_erro, seco, SETORES, YAW_BINS,
                                  DIST_MIN, DIST_MAX, RAIO_CHEGADA)

PASSOS_MAX_F2 = int(os.environ.get("PASSOS_MAX_F2", "20"))   # contornar custa passos
# Reta livre da ~0.86, nao 1.0: `custo` conta CELULAS e o passo diagonal cobre
# 1.41 blocos por celula. As faixas sao calibradas nesse valor observado.
FAIXAS = ((0.0, 1.15, "reta"), (1.15, 1.45, "leve"), (1.45, 99.0, "obstruido"))


def faixa_de(desvio):
    if desvio is None:
        return "?"
    for lo, hi, nome in FAIXAS:
        if lo <= desvio < hi:
            return nome
    return "?"


# ── Largadas ──────────────────────────────────────────────────────────────────
def procurar_largadas(n_alvo, max_tentativas=40, verbose=True):
    """Posicoes de largada em terreno REAL: so exige estar seco e de pe.

    Sem filtro de planura. Era ele que reprovava 97% do mundo e deixava so 13
    arenas — e era ele que removia justamente os obstaculos que a Fase 2 quer.
    """
    N = get("/lote/info")["envs"]
    boas = []
    for _ in range(max_tentativas):
        if len(boas) >= n_alvo:
            break
        post("/lote/reset", {})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0],
                                            "duration_ms": 50}] * N,
                                 "frames": False, "diag": True})
        for o in r["obs"][:N]:
            e, d = o["estado"], o.get("diag", {})
            if seco(e, d.get("agua_perto")) and e.get("on_ground"):
                boas.append((round(e["x"], 2), round(e["z"], 2)))
        if verbose and len(boas) % 40 < N:
            print("    %d largadas" % len(boas), flush=True)
    return boas[:n_alvo]


# ── Tarefas com obstrucao medida ──────────────────────────────────────────────
def montar_tarefas(largadas, n_ep, seed=0, raio=40, verbose=True,
                   desvio_min=0.0, max_tent=200, alvos_por_reset=24):
    """Sorteia alvos, VERIFICA alcancabilidade e mede o desvio de cada um.

    Alvo inalcancavel (dentro de pedra, do outro lado de um lago) e ruido: nem
    o teto chega, entao ele so dilui a metrica. Descartar exige o planejador,
    que e justamente quem sabe o que e passavel.

    `desvio_min` FILTRA por obstrucao. Amostragem cega em terreno real quase
    nunca produz o caso dificil — a 14-30 blocos so 4 de 96 alvos ficaram
    obstruidos, e uma faixa com n=4 nao mede nada.

    UM RESET, MUITOS ALVOS. O `/lote/reset` respawna 8 ambientes e carrega
    chunks do DISCO — e o custo dominante, e a busca em si e barata. A versao
    anterior gastava um reset por 8 alvos e descartava 95%: CPU a 6%, tudo
    esperando I/O. Com a posicao fixa, dezenas de alvos podem ser avaliados
    sem respawnar.
    """
    rng = random.Random(seed)
    N = get("/lote/info")["envs"]
    tarefas = []
    tentativas = 0
    while len(tarefas) < n_ep and tentativas < max_tent:
        tentativas += 1
        lote = [largadas[rng.randrange(len(largadas))] for _ in range(N)]
        r = post("/lote/reset", {"posicoes": [[x, z] for x, z in lote]})
        est = [o["estado"] for o in r["obs"][:N]]

        for _ in range(alvos_por_reset):
            if len(tarefas) >= n_ep:
                break
            rel, alvos = [], []
            for i in range(N):
                fr, la, setor = amostrar_alvo(rng, setor=SETORES[(len(tarefas) + i) % 8])
                dx, dz = alvo_para_mundo(est[i], fr, la)
                rel.append((fr, la, setor))
                alvos.append({"x": est[i]["x"] + dx, "z": est[i]["z"] + dz})

            info = post("/lote/alcancavel", {"alvos": alvos, "raio": raio})["alvos"]
            for i in range(N):
                v = info[i]
                if (v and v["alcancavel"] and v["desvio"] is not None
                        and v["desvio"] >= desvio_min):
                    tarefas.append({"largada": lote[i], "rel": rel[i],
                                    "desvio": v["desvio"], "reta": v["reta"]})

        if verbose and tentativas % 5 == 0:
            print("    %d tarefas em %d resets (%.1f por reset)"
                  % (len(tarefas), tentativas, len(tarefas) / tentativas), flush=True)

    if verbose:
        from collections import Counter
        c = Counter(faixa_de(t["desvio"]) for t in tarefas)
        print("    faixas de desvio: %s" % dict(c), flush=True)
    return tarefas[:n_ep]


# ── Politicas de referencia ───────────────────────────────────────────────────
class GeometricoComPulo:
    """O PISO DETERMINÍSTICO da Fase 2. Aponta para o alvo, anda, e pula quando trava.

    Cego por construcao: nao sabe que ha uma parede, so que parou de andar. A
    regra de pulo e a mesma do Piloto (`precisaSubir || travado >= 1`), mas sem
    o `precisaSubir` — que exigiria ler voxels.
    """
    nome = "geo_pulo"

    def __init__(self):
        self.travado = None
        self.ant = None

    def reiniciar(self, obs):
        n = len(obs)
        self.travado = [0] * n
        self.ant = [(o["estado"]["x"], o["estado"]["z"]) for o in obs]

    def agir(self, ests, alvos_abs, obs):
        acoes = []
        for i, (est, alvo) in enumerate(zip(ests, alvos_abs)):
            andou = math.hypot(est["x"] - self.ant[i][0], est["z"] - self.ant[i][1])
            self.travado[i] = self.travado[i] + 1 if andou < 0.15 else 0
            self.ant[i] = (est["x"], est["z"])
            e = erro_angular(est, alvo)
            hold = ["W", "SPACE"] if self.travado[i] >= 1 else ["W"]
            acoes.append({"hold": hold, "mouse": [int(bin_para_erro(e)), 0],
                          "duration_ms": 250})
        return acoes


class AleatorioComPulo:
    """O PISO ESTOCÁSTICO da Fase 2. Vaga e pula aleatoriamente sem ler a entrada.

    Prescrito por docs/metodo.md §1: o piso precisa ser do mesmo tipo da política.
    Uma política estocástica que ignora a entrada vaga e pula no acaso.
    """
    nome = "aleatorio_pulo"

    def __init__(self, prob_pulo=0.3, seed=0):
        self.prob_pulo = prob_pulo
        self.rng = random.Random(seed)

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        acoes = []
        for _ in ests:
            hold = ["W", "SPACE"] if self.rng.random() < self.prob_pulo else ["W"]
            dx = self.rng.choice(YAW_BINS)
            acoes.append({"hold": hold, "mouse": [int(dx), 0], "duration_ms": 250})
        return acoes


class PilotoBFS:
    """O TETO da Fase 2. Planejador com acesso aos voxels — ele SABE onde esta a
    parede. E o que a politica visual teria que descobrir olhando."""
    nome = "piloto"

    def __init__(self, raio=40):
        # raio=16 era o default antigo: com alvos a 14-30 blocos, metade fica
        # fora do horizonte de busca e o teto mede 36% em vez de 60% (ver
        # docs/metodo.md §11). 40 cobre DIST_MAX.
        self.raio = raio

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        extras = [{"alvo": {"x": a[0], "y": e.get("y", 64), "z": a[1]},
                   "raio": self.raio} for e, a in zip(ests, alvos_abs)]
        return post("/lote/piloto", {"objetivo": "ponto",
                                     "extras": extras})["acoes"][:len(ests)]


BANCO = "dataset/banco_fase2.json"


def carregar_banco_split(caminho=BANCO, frac_treino=0.85):
    """Carrega o banco de tarefas e devolve (treino, held_out) separados por tarefa (docs/metodo.md §4)."""
    banco = json.load(open(caminho, encoding="utf-8"))
    corte = int(frac_treino * len(banco))
    return banco[:corte], banco[corte:]


def gerar_banco(largadas, n, frac_obstruido=0.5, seed=0, raio=40, verbose=True):
    """Pre-gera tarefas com mistura CONTROLADA de obstrucao.

    Sorteio cego produz ~5% de alvos obstruidos a 14-30 blocos: gerar 80 leva
    4 min. Num treino de centenas de iteracoes isso viraria metade do tempo, e
    o sinal de aprendizado sobre contornar obstaculo ficaria diluido 20:1.

    Metade obstruida e metade livre: a politica precisa das duas — se so vir
    obstaculo, aprende a contornar sempre, inclusive onde a reta era otima.
    """
    n_obs = int(n * frac_obstruido)
    if verbose:
        print("[banco] %d obstruidas (desvio>=1.2) + %d livres" % (n_obs, n - n_obs),
              flush=True)
    obs = montar_tarefas(largadas, n_obs, seed=seed, raio=raio,
                         desvio_min=1.2, max_tent=4000, verbose=verbose)
    livres = montar_tarefas(largadas, n - n_obs, seed=seed + 1, raio=raio,
                            max_tent=4000, verbose=verbose)
    banco = obs + livres
    random.Random(seed).shuffle(banco)
    return banco


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--largadas", type=int, default=200)
    ap.add_argument("--banco", type=int, default=0,
                    help="gera banco de N tarefas em vez de procurar largadas")
    ap.add_argument("--saida", default="dataset/largadas_fase2.json")
    args = ap.parse_args()

    if args.banco:
        lg = [tuple(v) for v in json.load(open("dataset/largadas_fase2.json",
                                               encoding="utf-8"))]
        b = gerar_banco(lg, args.banco)
        json.dump(b, open(BANCO, "w", encoding="utf-8"))
        from collections import Counter
        print("[banco] %d tarefas em %s | %s"
              % (len(b), BANCO, dict(Counter(faixa_de(t["desvio"]) for t in b))))
        raise SystemExit

    print("[fase2] procurando largadas em terreno real (sem filtro de planura)...")
    lg = procurar_largadas(args.largadas)
    print("[fase2] %d largadas" % len(lg))
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    json.dump(lg, open(args.saida, "w", encoding="utf-8"))
    print("[fase2] salvo em %s" % args.saida)
