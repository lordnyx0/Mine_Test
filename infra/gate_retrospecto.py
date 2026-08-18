# coding=utf-8
"""
GATE: rotular em retrospecto torna a acao inferivel?

O teto medido da clonagem do Piloto e ~40% de recall por classe no giro, e ele
NAO vem da representacao (testar_pooling.py mostrou 39.7% lendo do backbone;
a MLP com rotas privilegiadas deu ~39%). A hipotese e que o teto vem da
PERGUNTA: "que tecla o professor apertou" nao tem resposta unica.

Trocando a pergunta para "que tecla leva ate B", onde B e o ponto onde o
agente DE FATO chegou k passos depois, o rotulo deixa de ser ambiguo por
construcao. Este script mede se isso e verdade, sem treinar o VLA.

Tres condicoes, mesmas features privilegiadas (rotas + estado), mesmo MLP:

  A) sem objetivo          — reproduz o teto conhecido
  B) objetivo em retrospecto — + para onde o agente foi em k passos
  C) objetivo EMBARALHADO  — + o objetivo de outra amostra qualquer

(C) e o controle que separa "informacao real" de "mais capacidade": se (B) e
(C) empatam, o ganho de (B) e ruido.

Varre varios k porque para k pequeno o objetivo quase codifica a propria acao
(vazamento trivial) — a tendencia ao longo de k e que informa.

    python gate_retrospecto.py --dataset dataset/sim_hindsight.jsonl
"""
import os
import sys
import json
import math
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)
PASSO_S = 0.25


def bin_proximo(v):
    return min(range(len(YAW_BINS)), key=lambda i: abs(YAW_BINS[i] - v))


def carregar(caminho):
    """Agrupa por episodio e ordena por passo — sem isso nao ha 'futuro'."""
    eps = {}
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get("pos") and d.get("rotas") and len(d["rotas"]) == 12:
                eps.setdefault(d["ep"], []).append(d)
    for k in eps:
        eps[k].sort(key=lambda d: d["t"])
    return eps


def objetivo_relativo(agora, futuro):
    """Deslocamento ate o ponto futuro, no referencial EGOCENTRICO de agora.

    Mesma convencao de estado_sim.vetor(): fx,fz e o vetor 'para frente'
    derivado do yaw, entao 'frente' e positivo a frente e 'lado' e positivo a
    esquerda. Sem isso o objetivo estaria em coordenadas do mundo, que nao
    transferem entre episodios nem entre jogos.
    """
    x0, z0, yaw_g = agora
    x1, z1, _ = futuro
    yaw = math.radians(yaw_g)
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    ddx, ddz = x1 - x0, z1 - z0
    frente = ddx * fx + ddz * fz
    lado = ddx * (-fz) + ddz * fx
    dist = math.hypot(ddx, ddz)
    ang = math.atan2(lado, frente)
    return [frente / 30.0, lado / 30.0, dist / 30.0, ang / math.pi]


def montar(eps, k):
    """X_base, X_obj, y, id_episodio para um horizonte de k passos."""
    Xb, Xo, Y, E = [], [], [], []
    for nome, seq in eps.items():
        for i in range(len(seq) - k):
            d = seq[i]
            Xb.append(list(d["rotas"]) + list(d["state_vec"]))
            Xo.append(objetivo_relativo(d["pos"], seq[i + k]["pos"]))
            Y.append(bin_proximo(d["action"]["mouse"][0]))
            E.append(nome)
    return (np.array(Xb, dtype=np.float32), np.array(Xo, dtype=np.float32),
            np.array(Y), np.array(E))


def treinar(X, y, eps_id, device, epocas=120, seed=0):
    """MLP pequena, holdout POR EPISODIO. Retorna (acc, recall por classe)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    unicos = np.unique(eps_id)
    rng.shuffle(unicos)
    teste = set(unicos[:max(1, int(0.2 * len(unicos)))])
    m_te = np.array([e in teste for e in eps_id])

    Xtr, Xte = X[~m_te], X[m_te]
    ytr, yte = y[~m_te], y[m_te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    Xtr, Xte = t((Xtr - mu) / sd), t((Xte - mu) / sd)
    ytr_t, yte_t = t(ytr, torch.long), t(yte, torch.long)

    cont = np.bincount(ytr, minlength=len(YAW_BINS))
    pesos = t([min(8.0, cont.sum() / (len(YAW_BINS) * max(1, c))) for c in cont])

    rede = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(),
                         nn.Linear(256, 256), nn.GELU(),
                         nn.Linear(256, len(YAW_BINS))).to(device)
    opt = torch.optim.AdamW(rede.parameters(), lr=1e-3, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss(weight=pesos)

    for _ in range(epocas):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            ce(rede(Xtr[j]), ytr_t[j]).backward()
            opt.step()

    with torch.no_grad():
        pred = rede(Xte).argmax(-1)
    acc = float((pred == yte_t).float().mean())
    rec, rec1 = [], []
    for c in range(len(YAW_BINS)):
        m = yte_t == c
        if int(m.sum()) >= 5:
            rec.append(float((pred[m] == c).float().mean()))
            # Tolerancia de +-1 bin: bins vizinhos sao giros quase iguais
            # (-17 vs 0 vs +17 unidades = 3 graus de diferenca). Acerto exato
            # e metrica dura demais para responder "isso pilotaria?".
            rec1.append(float(((pred[m] - c).abs() <= 1).float().mean()))
    n = max(1, len(rec))
    return acc, sum(rec) / n, sum(rec1) / n, int(m_te.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/sim_hindsight.jsonl")
    ap.add_argument("--ks", default="4,8,12,20")
    args = ap.parse_args()

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eps = carregar(args.dataset)
    print("[dados] %d episodios | %d amostras | %s"
          % (len(eps), sum(len(v) for v in eps.values()), args.dataset))

    print("\nrecall por classe do giro — holdout POR EPISODIO")
    print("(entre parenteses: tolerancia de +-1 bin)")
    print("=" * 78)
    print("%-5s %-7s %16s %18s %18s" %
          ("k", "horiz.", "A) sem obj.", "B) retrospecto", "C) embaralhado"))
    print("-" * 78)

    for k in [int(x) for x in args.ks.split(",")]:
        Xb, Xo, Y, E = montar(eps, k)
        rng = np.random.RandomState(1)
        Xo_emb = Xo[rng.permutation(len(Xo))]

        _, ra, ra1, n_te = treinar(Xb, Y, E, device)
        _, rb, rb1, _ = treinar(np.hstack([Xb, Xo]), Y, E, device)
        _, rc, rc1, _ = treinar(np.hstack([Xb, Xo_emb]), Y, E, device)

        f = lambda r, r1: "%5.1f%% (%4.1f%%)" % (100 * r, 100 * r1)
        print("%-5d %-7s %16s %18s %18s"
              % (k, "%.1fs" % (k * PASSO_S), f(ra, ra1), f(rb, rb1), f(rc, rc1)))

    print("=" * 78)
    print("(holdout ~%d amostras | acaso = 11.1%% exato, 33.3%% com +-1)" % n_te)
    print("\nB - A e o ganho do objetivo. B - C separa informacao de capacidade.")
    print("k pequeno tem vazamento: o objetivo quase codifica a propria acao.")
    print("O que interessa e o ganho SOBREVIVER a k grande.\n")


if __name__ == "__main__":
    main()
