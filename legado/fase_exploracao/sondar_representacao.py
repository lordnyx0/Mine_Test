# coding=utf-8
"""
O modelo esta formando alguma representacao util?

Tres testes, do mais basico ao mais informativo:

  1. COLAPSO — frames muito diferentes produzem estados internos diferentes?
     Se a similaridade media entre eles for ~1.0, a representacao colapsou:
     tudo vira o mesmo vetor e a visao foi efetivamente descartada.

  2. SENSIBILIDADE — quanto a saida muda quando so a IMAGEM muda, comparado
     a quando so o VETOR DE ESTADO muda. Diz de onde a politica esta de fato
     tirando a decisao.

  3. SONDA LINEAR — treina um classificador linear para ler, do estado interno,
     um fato que o agente precisa saber: "tem obstaculo a <=3 blocos a frente?"
     A verdade vem do /debug do servidor. Se um classificador linear acerta bem
     acima do acaso, a informacao ESTA representada e o problema e a politica.
     Se acerta no nivel do acaso, a representacao e cega e nenhum algoritmo de
     RL vai resolver desviar de obstaculo.

    python sondar_representacao.py --amostras 150
"""
import os
import sys
import json
import math
import time
import random
import argparse
import urllib.request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_utils import limitar_recursos, limitar_vram, travar_gpu, compactar_backbone
limitar_recursos()

import numpy as np
import torch

from run_vla_agent import load_vla_agent
from bot_vision_capture import BotVisionCapture, send_action
from exploration_env import ExplorationEnv, GoalEncoder
from treinar_exploracao import preparar_pixels, BUTTONS, USABLE

BASE = "http://127.0.0.1:3001"


def get(path, timeout=4.0):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def distancia_obstaculo():
    """Blocos ate o primeiro nao-ar a frente, na altura dos olhos (1..8, 9=livre)."""
    try:
        frente = get("/debug")["blocks_ahead_at_eye_level"]
    except Exception:
        return None
    for i, b in enumerate(frente, start=1):
        if not b.endswith("air"):
            return i
    return 9


class Captador:
    """Captura o estado interno via hooks, sem alterar o modelo."""

    def __init__(self, vla):
        self.hidden = None      # entrada das action heads (o vetor de decisao)
        self.visual = None      # saida do projetor (so visao)
        vla.action_heads.register_forward_pre_hook(self._h_heads)
        vla.projector.register_forward_hook(self._h_proj)

    def _h_heads(self, mod, args):
        self.hidden = args[0].detach().float().cpu().numpy()[0]

    def _h_proj(self, mod, args, out):
        # [B*K, 32, H] -> media sobre tokens e frames
        self.visual = out.detach().float().mean(dim=(0, 1)).cpu().numpy()


def coletar(vla, device, env, goal_ids, cap, n):
    """Anda pelo mundo guardando (hidden, visual, rotulo de obstaculo)."""
    H, V, Y, IMGS = [], [], [], []
    captador = Captador(vla)
    img, sv = env.reset(respawnar=True)

    for i in range(n):
        d_obs = distancia_obstaculo()
        px = preparar_pixels(vla.vision_processor, [img], device)
        svt = torch.tensor([sv], dtype=torch.float32, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            vla(pixel_values=px, state_vec=svt, input_ids=goal_ids)

        if d_obs is not None:
            H.append(captador.hidden.copy())
            V.append(captador.visual.copy())
            Y.append(1 if d_obs <= 3 else 0)
            if len(IMGS) < 40:
                IMGS.append((img[0], sv))

        # politica exploratoria simples, so para variar as cenas
        hold = [BUTTONS[k] for k in USABLE if random.random() < 0.3]
        act = {"hold": hold, "mouse": [random.randint(-40, 40), 0], "duration_ms": 250}
        img, sv, _r, respawnou, info = env.step(act)
        if respawnou or info.get("descontinuidade"):
            img, sv = env.observar()

        if (i + 1) % 30 == 0:
            print("    coletadas %d/%d amostras (obstaculo perto: %.0f%%)"
                  % (len(H), n, 100 * np.mean(Y)), flush=True)

    return np.array(H), np.array(V), np.array(Y), IMGS


def cos_medio(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    return float(S[iu].mean()), float(S[iu].std())


def posto_efetivo(X):
    """Quantas direcoes realmente carregam variancia (entropia dos autovalores)."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s**2 / (s**2).sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def sonda_linear(X, y, nome, epocas=400):
    """Regressao logistica com holdout. Retorna (acc, acc_acaso)."""
    if len(set(y.tolist())) < 2:
        print(f"    [{nome}] rotulo constante, sonda impossivel")
        return None, None
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    corte = int(0.7 * len(X))
    mu, sd = X[:corte].mean(0), X[:corte].std(0) + 1e-6
    Xtr, Xte = (X[:corte] - mu) / sd, (X[corte:] - mu) / sd
    ytr, yte = y[:corte], y[corte:]

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epocas):
        opt.zero_grad()
        l = lossf(lin(Xtr_t).squeeze(-1), ytr_t)
        l.backward(); opt.step()

    with torch.no_grad():
        pred = (lin(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1) > 0).numpy()
    acc = float((pred == yte).mean())
    acaso = float(max(yte.mean(), 1 - yte.mean()))
    return acc, acaso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostras", type=int, default=150)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    travar_gpu()

    # O teto de VRAM so pode entrar DEPOIS da compactacao: o modelo nasce em
    # fp32 (8.7GB) e so entao vira bf16 (4.4GB). Aplicar o teto antes faz o
    # proprio carregamento estourar.
    vla, device = load_vla_agent(args.ckpt)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    limitar_vram(0.62)
    cap = BotVisionCapture()
    env = ExplorationEnv(cap, segundos_sem_progresso=999, verbose=False)
    goal_ids = GoalEncoder(device=device).ids("explorar")

    print("\n[1/3] Coletando estados internos com rotulos reais do mundo...", flush=True)
    H, V, Y, IMGS = coletar(vla, device, env, goal_ids, cap, args.amostras)
    print(f"    {len(H)} amostras | dim hidden={H.shape[1]} visual={V.shape[1]} "
          f"| obstaculo perto em {100*Y.mean():.0f}%", flush=True)

    print("\n[2/3] COLAPSO da representacao")
    for nome, X in (("hidden (decisao)", H), ("visual (projetor)", V)):
        m, s = cos_medio(X)
        pe = posto_efetivo(X)
        veredito = ("COLAPSADA — tudo vira o mesmo vetor" if m > 0.99
                    else "pouca variacao" if m > 0.95 else "variada")
        print(f"    {nome:20s} cos medio={m:.4f} (dp {s:.4f}) | posto efetivo={pe:.1f} "
              f"de {X.shape[1]} -> {veredito}")

    print("\n[3/3] SONDA LINEAR: 'tem obstaculo a <=3 blocos a frente?'")
    for nome, X in (("hidden (decisao)", H), ("visual (projetor)", V)):
        acc, acaso = sonda_linear(X, Y, nome)
        if acc is None:
            continue
        ganho = acc - acaso
        veredito = ("REPRESENTA bem" if ganho > 0.15 else
                    "representa fraco" if ganho > 0.05 else "NAO representa (nivel do acaso)")
        print(f"    {nome:20s} acuracia={acc:.3f} | acaso={acaso:.3f} "
              f"| ganho={ganho:+.3f} -> {veredito}")

    print("\n[extra] SENSIBILIDADE: a saida depende da imagem ou do state_vec?")
    if len(IMGS) >= 8:
        base_img, base_sv = IMGS[0]
        def saida(imgs, sv):
            px = preparar_pixels(vla.vision_processor, [imgs], device)
            svt = torch.tensor([sv], dtype=torch.float32, device=device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                     enabled=torch.cuda.is_available()):
                o = vla(pixel_values=px, state_vec=svt, input_ids=goal_ids)
            return o["buttons"][0].float().cpu().numpy()

        ref = saida([base_img] * 3, base_sv)
        d_img = [np.abs(saida([im] * 3, base_sv) - ref).mean() for im, _ in IMGS[1:8]]
        d_sv = []
        for _, sv2 in IMGS[1:8]:
            d_sv.append(np.abs(saida([base_img] * 3, sv2) - ref).mean())
        print(f"    trocando so a IMAGEM      -> variacao media das teclas: {np.mean(d_img):.4f}")
        print(f"    trocando so o STATE_VEC   -> variacao media das teclas: {np.mean(d_sv):.4f}")
        if np.mean(d_img) < 0.01:
            print("    -> a politica esta praticamente IGNORANDO a imagem")

    print()


if __name__ == "__main__":
    main()
