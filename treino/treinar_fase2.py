# coding=utf-8
"""
FASE 2 — RL em terreno real, com obstaculos, giro + PULO.

Mesmo REINFORCE da Fase 1. Tres coisas mudam:

  1. acao conjunta (giro, pulo), com log-prob somada
  2. tarefas vem de um BANCO pre-gerado com mistura controlada de obstrucao
     (sorteio ao vivo produz ~5% de alvos obstruidos e viraria metade do tempo)
  3. a recompensa densa passa a premiar contornar, nao so apontar

O que a Fase 2 pergunta, e a Fase 1 nao pode responder:

    A politica usa a IMAGEM para contornar o que as coordenadas nao descrevem?

Piso e teto medidos ANTES, na faixa que exige visao (`avaliar_fase2.py`):

    geo_pulo (cego)        33%
    piloto BFS raio 40     60%      <- 27 pontos de espaco

    python treinar_fase2.py --iteracoes 150
"""
import os
import sys
import json
import math
import time
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.gpu_utils import (limitar_recursos, limitar_vram, travar_gpu,
                             compactar_backbone, memoria_gpu)
limitar_recursos()

import numpy as np
import torch
import torch.nn as nn

from infra.run_vla_agent import load_vla_agent
from ambiente.arena_plana import post, get, alvo_para_mundo, RAIO_CHEGADA
from ambiente.fase2 import BANCO, PASSOS_MAX_F2, faixa_de
from politica.politica_fase2 import PoliticaFase2
from treino.treinar_fase1 import mascara_do_passo, retornos, GAMMA, BONUS_CHEGADA

CKPT_SAIDA = "checkpoints_vla/vla_fase2.pt"

# Custo por pulo. Sem ele a politica pula em 99% dos passos e a taxa de chegada
# CAI de 50% para 32% — pular nao e neutro: `prismarine-physics` da
# `airborneAcceleration = 0.02` contra ~0.10 no chao, entao no ar o bot tem 5x
# menos autoridade de direcao. Pulando sempre, ele perde justamente a habilidade
# que a tarefa exige.
#
# 0.05 ja degenerou 3x (docs/atual.md, docs/metodo.md §2b/§15): e um preco
# SOLTO, nao um potencial, entao compete contra o retorno futuro descontado
# por gamma^k. Com GAMMA=0.95 o horizonte de credito e so ~20 passos
# (1/(1-gamma)) contra PASSOS_MAX_F2=50 — o episodio e 2.5x mais longo que o
# horizonte, entao nenhum preco pequeno "segura" o pulo por si so. 0.15 e o
# valor documentado como proximo passo em docs/atual.md; GAMMA sobe junto
# (ver --gamma abaixo) para o horizonte cobrir o episodio inteiro.
CUSTO_PULO = float(os.environ.get("CUSTO_PULO", "0.15"))


def rollout(pol, tarefas, passos=None):
    """Um lote de episodios. `tarefas` sao dicts do banco."""
    passos = passos or PASSOS_MAX_F2
    n = len(tarefas)
    r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in tarefas]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    alvo_abs = []
    for i, t in enumerate(tarefas):
        dx, dz = alvo_para_mundo(est[i], t["rel"][0], t["rel"][1])
        alvo_abs.append((est[i]["x"] + dx, est[i]["z"] + dz))
    # Publica os alvos para o visualizador. Uma chamada por EPISODIO, nao por
    # passo: sem isso o /ver mostra um boneco andando e nao ha como julgar, a
    # olho, se ele navega ou so se move.
    try:
        post("/lote/alvos", {"alvos": [{"x": a[0], "z": a[1]} for a in alvo_abs]})
    except Exception:
        pass

    d0 = [math.hypot(alvo_abs[i][0] - est[i]["x"], alvo_abs[i][1] - est[i]["z"])
          for i in range(n)]
    dant, dmin = list(d0), list(d0)
    vivo, chegou_em = [True] * n, [None] * n

    U8, SV, GV, IDX, PULO, R, VIVO = [], [], [], [], [], [], []

    for t in range(passos):
        acoes = pol.agir(est, alvo_abs, obs)
        u = pol.ultimo
        rr = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = rr["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        rec = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if not vivo[i]:
                continue
            d = math.hypot(alvo_abs[i][0] - est[i]["x"], alvo_abs[i][1] - est[i]["z"])
            custo_pulo_efetivo = CUSTO_PULO * float(u["pulo"][i])
            # Penalidade se pular ja estando no ar (docs/metodo.md §15: no ar a aceleracao cai para 0.02)
            if u["pulo"][i] == 1 and not est[i].get("on_ground", True):
                custo_pulo_efetivo += 0.15
            rec[i] = dant[i] - d - custo_pulo_efetivo
            dant[i] = d
            dmin[i] = min(dmin[i], d)
            if d <= RAIO_CHEGADA:
                rec[i] += BONUS_CHEGADA
                chegou_em[i] = t + 1
                vivo[i] = False

        U8.append(u["u8"]); SV.append(u["sv"]); GV.append(u["gv"])
        IDX.append(u["idx"]); PULO.append(u["pulo"]); R.append(rec)
        VIVO.append(np.array(mascara_do_passo(vivo, chegou_em, t), dtype=np.float32))
        if not any(vivo):
            break

    met = [{"faixa": faixa_de(tarefas[i]["desvio"]), "desvio": tarefas[i]["desvio"],
            "d0": d0[i], "dfinal": dant[i], "dmin": dmin[i],
            "chegou": chegou_em[i] is not None} for i in range(n)]
    return (np.stack(U8), np.stack(SV), np.stack(GV), np.stack(IDX),
            np.stack(PULO), np.stack(R), np.stack(VIVO)), met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteracoes", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--minilote", type=int, default=12)
    ap.add_argument("--entropia", type=float, default=0.06)
    ap.add_argument("--entropia-final", type=float, default=0.025)
    ap.add_argument("--ckpt-entrada", default=None)
    ap.add_argument("--retomar", default=None)
    ap.add_argument("--vram", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    # 1/(1-gamma) = horizonte de credito, em passos. 0.95 (o da Fase 1) da
    # ~20, contra PASSOS_MAX_F2=50 — o custo do pulo (perda de autoridade de
    # giro no ar) se manifesta depois do horizonte e o desconto o amassa
    # antes de chegar (docs/metodo.md §15). 0.98 -> ~50, cobre o episodio.
    ap.add_argument("--gamma", type=float, default=0.98)
    args = ap.parse_args()

    assert PASSOS_MAX_F2 >= 40, (
        "PASSOS_MAX_F2=%d parece o regime da Fase 1 (default 20 quando a env "
        "var nao e exportada) — exporte PASSOS_MAX_F2=50 antes de treinar a "
        "Fase 2 (docs/metodo.md §3/§18)." % PASSOS_MAX_F2)

    # Trava contra retomar de checkpoint contaminado (docs/metodo.md §20)
    CONTAMINADOS = ["pulo_degenerado", "entropia_alta", "gstd_baixo_zero"]
    if args.retomar and any(c in args.retomar for c in CONTAMINADOS):
        raise ValueError(
            "Checkpoint '%s' e reconhecidamente contaminado (docs/metodo.md §20). "
            "Use --ckpt-entrada checkpoints_vla/vla_fase1.pt para iniciar limpo." % args.retomar)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    print("[fase2] seed=%d | GAMMA=%.3f (horizonte ~%.0f passos) | CUSTO_PULO=%.3f"
          % (args.seed, args.gamma, 1 / (1 - args.gamma), CUSTO_PULO), flush=True)

    banco = json.load(open(BANCO, encoding="utf-8"))
    from collections import Counter
    print("[fase2] banco com %d tarefas | %s"
          % (len(banco), dict(Counter(faixa_de(t["desvio"]) for t in banco))), flush=True)
    # Split por TAREFA: as de avaliacao nunca entram no treino.
    corte = int(0.85 * len(banco))
    treino = banco[:corte]
    print("[fase2] %d de treino, %d reservadas" % (len(treino), len(banco) - corte),
          flush=True)

    it0, hist_ant = 0, []
    if args.retomar:
        ck = torch.load(args.retomar, map_location="cpu")
        it0 = int(ck.get("iteracao", 0)); hist_ant = list(ck.get("hist", []))
        args.ckpt_entrada = args.retomar
        print("[retomar] iteracao %d" % it0, flush=True)

    N = get("/lote/info")["envs"]
    travar_gpu()
    vla, device = load_vla_agent(args.ckpt_entrada)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    print("[VRAM] %s | %s" % (limitar_vram(args.vram), memoria_gpu()), flush=True)

    pol = PoliticaFase2(None, amostrar=True, device=device, vla=vla)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    print("[treino] %.2fM parametros | acao = giro x pulo"
          % (sum(p.numel() for p in treinaveis) / 1e6), flush=True)
    opt = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=1e-2)
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    hist = list(hist_ant)
    t0 = time.time()
    ent_y_media, ent_j_media = 0.50, 0.50

    for it in range(it0 + 1, args.iteracoes + 1):
        tarefas = [treino[rng.randrange(len(treino))] for _ in range(N)]

        vla.eval(); pol.amostrar = True
        (U8, SV, GV, IDX, PULO, R, VIVO), met = rollout(pol, tarefas)

        G = retornos(R, VIVO, gamma=args.gamma)
        m = VIVO.reshape(-1) > 0
        g = G.reshape(-1)[m]
        g_std = float(g.std())
        # Piso no desvio do lote antes de normalizar (docs/metodo.md §25)
        PISO_G_STD = 3.0
        adv = (g - g.mean()) / (max(g_std, PISO_G_STD) + 1e-6)
        u8 = U8.reshape(-1, *U8.shape[2:])[m]
        sv = SV.reshape(-1, SV.shape[-1])[m]
        gv = GV.reshape(-1, GV.shape[-1])[m]
        ay = IDX.reshape(-1)[m]
        aj = PULO.reshape(-1)[m]

        adv_pulou = float(adv[aj == 1].mean()) if (aj == 1).any() else float("nan")
        adv_npulou = float(adv[aj == 0].mean()) if (aj == 0).any() else float("nan")
        del U8
        torch.cuda.empty_cache()

        frac = (it - 1) / max(1, args.iteracoes - 1)
        base_beta = args.entropia + frac * (args.entropia_final - args.entropia)

        # Target entropy adaptativo por dimensao (impede colapso de exploracao):
        TARGET_ENT_Y = 0.35
        TARGET_ENT_J = 0.40
        beta_y = base_beta * (1.0 + max(0.0, 4.0 * (TARGET_ENT_Y - ent_y_media) / TARGET_ENT_Y))
        beta_j = base_beta * (1.0 + max(0.0, 4.0 * (TARGET_ENT_J - ent_j_media) / TARGET_ENT_J))

        vla.train(); vla.vision_encoder.eval(); vla.qwen_model.eval()
        ordem = np.random.permutation(len(ay))
        ent_soma, ent_y_soma, ent_j_soma, nb = 0.0, 0.0, 0.0, 0
        for b0 in range(0, len(ordem), args.minilote):
            sel = ordem[b0:b0 + args.minilote]
            px = pol.normalizar(u8[sel])
            svt = torch.tensor(sv[sel], dtype=torch.float32, device=device)
            gvt = torch.tensor(gv[sel], dtype=torch.float32, device=device)
            yt = torch.tensor(ay[sel], dtype=torch.long, device=device)
            jt = torch.tensor(aj[sel], dtype=torch.long, device=device)
            advt = torch.tensor(adv[sel], dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                lp, ent, ey, ej = pol.log_prob(px, svt, gvt, yt, jt)
            perda = -(lp.float() * advt).mean() - beta_y * ey.float() - beta_j * ej.float()
            escala.scale(perda).backward()
            escala.unscale_(opt)
            # Clip de gradiente a 0.5 protege a rota visual (resampler e projector)
            torch.nn.utils.clip_grad_norm_(treinaveis, 0.5)
            escala.step(opt); escala.update()
            ent_soma += float(ent.item())
            ent_y_soma += float(ey.item())
            ent_j_soma += float(ej.item())
            nb += 1
            if nb % 40 == 0:
                torch.cuda.empty_cache()

        taxa = sum(x["chegou"] for x in met) / len(met)
        obst = [x for x in met if x["faixa"] == "obstruido"]
        t_obst = (sum(x["chegou"] for x in obst) / len(obst)) if obst else float("nan")
        pulos = float(PULO.reshape(-1)[m].mean())
        ent_media = ent_soma / max(1, nb)
        ent_y_media = ent_y_soma / max(1, nb)
        ent_j_media = ent_j_soma / max(1, nb)

        hist.append({"it": it, "taxa": taxa, "obstruido": t_obst,
                     "recompensa": float(R.sum(axis=0).mean()), "pulo": pulos,
                     "g_std": g_std, "adv_pulou": adv_pulou, "adv_npulou": adv_npulou,
                     "entropia": ent_media, "ent_yaw": ent_y_media, "ent_pulo": ent_j_media})
        print("it %3d/%d | chegada %3.0f%% | obstruido %3.0f%% | recompensa %+6.2f | "
              "pulo %2.0f%% | ent %.2f (y:%.2f j:%.2f) | g_std %.3f | adv(pulou/nao) %+.2f/%+.2f | %4.0fs"
              % (it, args.iteracoes, 100 * taxa, 100 * t_obst,
                 float(R.sum(axis=0).mean()), 100 * pulos,
                 ent_media, ent_y_media, ent_j_media, g_std, adv_pulou, adv_npulou,
                 time.time() - t0), flush=True)

        if it % 5 == 0 or it == args.iteracoes:
            torch.save({"treinaveis": {n_: p.detach().cpu()
                                        for n_, p in vla.named_parameters()
                                        if p.requires_grad},
                        "iteracao": it, "hist": hist}, CKPT_SAIDA)

    if len(hist) >= 6:
        ini = np.nanmean([h["obstruido"] for h in hist[:5]])
        fim = np.nanmean([h["obstruido"] for h in hist[-5:]])
        print("\n[tendencia] obstruido %.0f%% -> %.0f%%  (piso 33%%, teto 60%%)"
              % (100 * ini, 100 * fim))
    print("\n[OK] %s" % CKPT_SAIDA, flush=True)


if __name__ == "__main__":
    main()
