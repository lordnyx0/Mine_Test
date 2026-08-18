# coding=utf-8
"""
FASE 1 — controle visuomotor local em terreno plano.

A pergunta, e so ela: dado um alvo a <=8 blocos, a politica aprende a girar na
direcao certa e andar ate la, usando SO W + giro, com feedback do ambiente?

Nao ha professor. Nao ha BFS. Nao ha imitacao de acao.

Por que terreno plano nao existe e mesmo assim da para ter arena: o mundo vem
dos arquivos de regiao de um save real, e nao ha gerador de superplano. Em vez
de construir um, FILTRAMOS posicoes de largada pela planura, usando a sonda de
relevo do servidor. Arena plana e sem obstaculo, com a distribuicao visual
realista, e zero codigo de geracao de mundo.

    python arena_plana.py --arenas 64      # so procura e reporta
"""
import os
import sys
import math
import json
import random
import argparse
import urllib.request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE = "http://127.0.0.1:3002"
GRAUS_POR_UNIDADE = 0.003 * 180 / math.pi
YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)

RAIO_CHEGADA = 1.5      # apertado: o alvo esta a <=8 blocos, nao a 100
# Configuraveis: a Fase 1 usa 3-8 (controle local puro), mas nessa faixa NAO
# CABE obstaculo — so 13 de 96 alvos ficaram obstruidos, e o planejador chega a
# ser PIOR que apontar-e-andar porque a distancia e curta demais para planejar
# compensar. Obstaculo precisa de espaco para o desvio existir.
DIST_MIN = float(os.environ.get("DIST_MIN", "3.0"))
DIST_MAX = float(os.environ.get("DIST_MAX", "8.0"))
# 14, nao 40. Com 40 passos num raio de 3-8 blocos, PASSEIO ALEATORIO encontra
# o alvo — e foi exatamente isso que a politica aprendeu a fazer, marcando 90%
# de recompensa sem nunca reagir a direcao do alvo. Caminho direto leva ~8
# passos; 14 da margem para uma correcao e nao para vagar.
PASSOS_MAX = int(os.environ.get("PASSOS_MAX", "14"))

# 8 setores, amostrados por igual. Sem isto a distribuicao vira degenerada e
# uma politica que so sabe ir para frente marca bem na media.
SETORES = ("frente", "NE", "direita", "SE", "tras", "SW", "esquerda", "NW")


def post(caminho, corpo, timeout=300):
    d = json.dumps(corpo).encode()
    r = urllib.request.Request(BASE + caminho, data=d, method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read())


def get(caminho, timeout=60):
    with urllib.request.urlopen(BASE + caminho, timeout=timeout) as x:
        return json.loads(x.read())


# ── Arena ─────────────────────────────────────────────────────────────────────
def e_plano(perfis, tolerancia=1):
    """Plano = nenhuma direcao sobe ou desce mais que `tolerancia` blocos.

    `null` no perfil significa coluna nao carregada — trata como reprovado,
    porque ali a busca (e a fisica) nao sabem o que existe.

    ATENCAO: planura sozinha NAO basta. Agua nao e bloco solido, entao o perfil
    atravessa a agua e mede o LEITO — e fundo de lago raso e perfeitamente
    plano. A primeira versao deste filtro aprovou 4 de 8 arenas dentro d'agua,
    duas delas a y=40 (oceano). Sempre checar `seco()` junto.
    """
    for p in perfis:
        if not p:
            return False
        for v in p:
            if v is None or abs(v) > tolerancia:
                return False
    return True


def seco(estado, agua_perto):
    """Nenhuma agua no bot nem no raio da sonda (6 blocos)."""
    return not estado.get("in_water") and not estado.get("in_lava") \
        and agua_perto is None


def perfis_em_volta(env_obs):
    return env_obs.get("diag", {}).get("perfis") or []


def procurar_arenas(n_alvo, raio_teste=12, max_tentativas=400, verbose=True):
    """Sorteia respawns e devolve as posicoes cujo entorno e plano."""
    info = get("/lote/info")
    N = info["envs"]
    boas, testadas = [], 0

    # 8 direcoes cardinais/diagonais, para checar planura em volta inteira
    dirs8 = [[math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)]
             for i in range(8)]

    for tentativa in range(max_tentativas):
        if len(boas) >= n_alvo:
            break
        r = post("/lote/reset", {})
        est = [o["estado"] for o in r["obs"]]
        # Uma sonda por direcao: o servidor devolve um perfil por chamada, entao
        # oito chamadas com acao nula (o bot nao se move sem tecla).
        perfis_por_env = [[] for _ in range(N)]
        agua_por_env = [None] * N
        for d in dirs8:
            rr = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0],
                                                 "duration_ms": 50}] * N,
                                      "frames": False, "diag": True,
                                      "dirs": [d] * N})
            for i, o in enumerate(rr["obs"][:N]):
                perfis_por_env[i].append(o.get("diag", {}).get("perfil"))
                a = o.get("diag", {}).get("agua_perto")
                if a is not None:
                    agua_por_env[i] = a
                est[i] = o["estado"]

        for i in range(N):
            testadas += 1
            if e_plano(perfis_por_env[i]) and seco(est[i], agua_por_env[i]):
                boas.append((round(est[i]["x"], 2), round(est[i]["z"], 2)))
        if verbose and (tentativa + 1) % 5 == 0:
            print("    %d arenas planas em %d posicoes testadas" % (len(boas), testadas),
                  flush=True)

    return boas[:n_alvo], testadas


# ── Alvo ──────────────────────────────────────────────────────────────────────
def amostrar_alvo(rng, setor=None):
    """Alvo em coordenadas EGOCENTRICAS: (frente, lado, setor).

    O setor e relativo ao BOT, nao ao mundo. A versao anterior sorteava um
    angulo absoluto e chamava de "frente" o que apontava para +x do mundo; como
    o respawn randomiza o yaw (`entity.yaw = Math.random()*2pi`), cada rotulo
    continha uma mistura uniforme de direcoes relativas. A quebra por setor
    media ruido, e o baseline geometrico parecia uniforme por isso.

    A conversao para mundo depende do yaw e e feita em `alvo_para_mundo`.
    """
    k = rng.randrange(8) if setor is None else SETORES.index(setor)
    ang = 2 * math.pi * (k + rng.random() - 0.5) / 8      # centro do setor +- meio setor
    dist = DIST_MIN + rng.random() * (DIST_MAX - DIST_MIN)
    return dist * math.cos(ang), dist * math.sin(ang), SETORES[k]


def alvo_para_mundo(est, frente, lado):
    """(frente, lado) egocentricos -> deslocamento (dx, dz) no mundo.

    Mesma convencao de erro_angular: fx,fz e o vetor "para frente" derivado do
    yaw, e `lado` positivo e a esquerda.
    """
    yaw = math.radians(est["yaw"])
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    return frente * fx + lado * (-fz), frente * fz + lado * fx


def bin_para_erro(graus):
    """Converte erro angular em bin de mouse, na convencao ja validada:
    o sinal e invertido porque mouse positivo AFASTA do alvo (medido em malha
    fechada em planejador.js)."""
    desejado = -graus / GRAUS_POR_UNIDADE
    return min(YAW_BINS, key=lambda v: abs(v - desejado))


def erro_angular(est, alvo_abs):
    """Graus entre a direcao encarada e a direcao do alvo. Positivo = alvo a esquerda."""
    yaw = math.radians(est["yaw"])
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    rx, rz = alvo_abs[0] - est["x"], alvo_abs[1] - est["z"]
    frente = rx * fx + rz * fz
    lado = rx * (-fz) + rz * fx
    return math.atan2(lado, frente) * 180 / math.pi


# ── Baseline geometrico ───────────────────────────────────────────────────────
class ControladorGeometrico:
    """O minimo a ser batido. Sem visao, sem rede: trigonometria pura.

    A pergunta do experimento e se a politica neural chega perto DISTO.
    """
    nome = "geometrico"

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        acoes = []
        for est, alvo in zip(ests, alvos_abs):
            e = erro_angular(est, alvo)
            acoes.append({"hold": ["W"], "mouse": [int(bin_para_erro(e)), 0],
                          "duration_ms": 250})
        return acoes


class SoAndar:
    """Controle degenerado: anda reto, nunca gira. Mede quanto do sucesso vem
    de simplesmente ter alvo perto, e nao de saber virar."""
    nome = "so_W"

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        return [{"hold": ["W"], "mouse": [0, 0], "duration_ms": 250}
                for _ in ests]


class PasseioAleatorio:
    """Anda sempre, girando um bin ao acaso. O baseline que FALTAVA.

    `so_W` e deterministico e marca 1,6%, entao parecia um piso seguro. Mas uma
    politica estocastica que ignora a entrada vaga pelo terreno, e com orcamento
    folgado isso ENCONTRA o alvo. Sem este baseline, "90% de chegada" pareceu
    aprendizado quando era ruido — custou 4h de treino descobrir.
    """
    nome = "aleatorio"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        return [{"hold": ["W"], "mouse": [self.rng.choice(YAW_BINS), 0],
                 "duration_ms": 250} for _ in ests]


class DistribuicaoFixa:
    """Amostra de uma distribuicao FIXA sobre os bins, ignorando toda entrada.

    E o baseline que casa com a politica de verdade. `PasseioAleatorio` sorteia
    uniforme e por isso GIRA quase sempre, rodando no lugar — piso baixo demais.
    A politica treinada emite ~50% no bin zero, que e um passeio ENVIESADO: ele
    avanca enquanto vagueia, e isso e muito mais eficaz.

    Se o modelo empatar com este baseline, ele nao aprendeu nada: a taxa vem da
    forma da distribuicao, nao de reagir ao alvo.
    """
    nome = "dist_fixa"

    # Medida na varredura de angulo do checkpoint da iteracao 140.
    PADRAO = (5, 7, 7, 7, 50, 6, 7, 7, 5)

    def __init__(self, pesos=None, seed=0):
        self.rng = random.Random(seed)
        self.pesos = list(pesos or self.PADRAO)

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        return [{"hold": ["W"],
                 "mouse": [self.rng.choices(YAW_BINS, weights=self.pesos)[0], 0],
                 "duration_ms": 250} for _ in ests]


# ── Episodio ──────────────────────────────────────────────────────────────────
def rodar_lote(politica, arenas, alvos_rel, passos=PASSOS_MAX, com_frames=False):
    """Roda um lote de episodios em paralelo. Uma entrada por ambiente.

    Devolve um dict de metricas por episodio — as cinco que a Fase 1 pede, mais
    o setor, para a quebra por direcao.
    """
    n = len(arenas)
    r = post("/lote/reset", {"posicoes": [[x, z] for x, z in arenas]})
    est = [o["estado"] for o in r["obs"][:n]]
    obs = r["obs"][:n]

    # O alvo e egocentrico NO INSTANTE DA LARGADA e so entao vira mundo: e o
    # que faz "frente" significar a frente do bot, com yaw de respawn aleatorio.
    alvo_abs = []
    for i in range(n):
        dx, dz = alvo_para_mundo(est[i], alvos_rel[i][0], alvos_rel[i][1])
        alvo_abs.append((est[i]["x"] + dx, est[i]["z"] + dz))
    d0 = [math.hypot(alvo_abs[i][0] - est[i]["x"], alvo_abs[i][1] - est[i]["z"])
          for i in range(n)]
    dmin = list(d0)
    dant = list(d0)
    chegou_em = [None] * n
    recompensa = [0.0] * n
    erro0 = [erro_angular(est[i], alvo_abs[i]) for i in range(n)]

    if hasattr(politica, "reiniciar"):
        politica.reiniciar(obs)

    for t in range(passos):
        acoes = politica.agir(est, alvo_abs, obs)
        r = post("/lote/passo", {"acoes": acoes, "frames": com_frames})
        obs = r["obs"][:n]
        est = [o["estado"] for o in obs]
        # SEM ISTO a pilha de frames e o state_vec da politica congelam no
        # instante inicial: ela recebe objetivo atualizado com imagem e estado
        # de 40 passos atras. Foi o que fez a politica cair de 90% no treino
        # para 11% na avaliacao. O laco de treino sempre chamou observar();
        # este nao chamava — duas verdades, o bug que estado_sim.py ja alertava.
        if hasattr(politica, "observar"):
            politica.observar(obs)
        for i in range(n):
            d = math.hypot(alvo_abs[i][0] - est[i]["x"], alvo_abs[i][1] - est[i]["z"])
            # Recompensa densa: reducao de distancia. Aproximar e positivo,
            # afastar negativo, parado ~zero.
            if chegou_em[i] is None:
                recompensa[i] += dant[i] - d
            dant[i] = d
            dmin[i] = min(dmin[i], d)
            if d <= RAIO_CHEGADA and chegou_em[i] is None:
                chegou_em[i] = t + 1
                recompensa[i] += 5.0            # bonus de sucesso

    if os.environ.get("DEBUG_ROLLOUT"):
        print("    [lote] " + " ".join(
            "%s:%.1f->%.1f%s" % (alvos_rel[i][2][:3], d0[i], dant[i],
                                 "OK" if chegou_em[i] is not None else "")
            for i in range(n)), flush=True)
    return [{"setor": alvos_rel[i][2], "d0": d0[i], "dmin": dmin[i],
             "dfinal": dant[i], "chegou": chegou_em[i] is not None,
             "passos": chegou_em[i], "recompensa": recompensa[i],
             "erro0": erro0[i]}
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arenas", type=int, default=64)
    ap.add_argument("--tentativas", type=int, default=400)
    ap.add_argument("--saida", default="dataset/arenas_planas.json")
    args = ap.parse_args()

    print("[arena] procurando terreno plano (tolerancia 1 bloco, raio 12)...", flush=True)
    # Acumula com o que ja foi encontrado: o rendimento e 3%, entao cada
    # corrida cara demais para ser jogada fora.
    antigas = []
    if os.path.exists(args.saida):
        try: antigas = [tuple(v) for v in json.load(open(args.saida, encoding="utf-8"))]
        except Exception: antigas = []
    boas, testadas = procurar_arenas(args.arenas, max_tentativas=args.tentativas)
    vistas = set(antigas)
    for b in boas:
        if b not in vistas:
            antigas.append(b); vistas.add(b)
    boas = antigas
    print("[arena] %d planas de %d testadas (%.0f%%)"
          % (len(boas), testadas, 100 * len(boas) / max(1, testadas)), flush=True)
    if not boas:
        print("[ERRO] nenhuma arena plana encontrada. Afrouxe a tolerancia.")
        return
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(boas, f)
    print("[arena] salvo em %s" % args.saida, flush=True)


if __name__ == "__main__":
    main()
