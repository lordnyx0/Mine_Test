# coding=utf-8
"""
FASE 1 — treino por RECOMPENSA DO AMBIENTE. Nenhum professor, nenhuma imitacao.

REINFORCE com vantagem normalizada. A politica so escolhe o bin de giro; `W` e
sempre apertado. A recompensa e densa:

    r_t = distancia_anterior - distancia_atual        (+5 ao chegar)

Aproximar e positivo, afastar negativo, parado ~zero. E o sinal que substitui o
rotulo: nao importa qual tecla "o professor teria apertado", so se o agente
encostou no alvo.

Por que dois passos (rollout sem grafo, depois recalculo com grafo): guardar o
grafo de 40 passos x 8 ambientes atraves de SigLIP + um 0.6B estoura os 12 GB.
Guardamos os pixels em uint8 (144 MB) e recalculamos os log-probs em minilotes.

    python treinar_fase1.py --iteracoes 10        # sonda curta: a recompensa se move?
    python treinar_fase1.py --iteracoes 120       # corrida
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
from ambiente.arena_plana import (post, get, amostrar_alvo, SETORES, RAIO_CHEGADA,
                                  PASSOS_MAX, erro_angular, alvo_para_mundo)
from avaliacao.avaliar_fase1 import carregar_arenas
from politica.politica_fase1 import PoliticaNeural

CKPT_SAIDA = "checkpoints_vla/vla_fase1.pt"
GAMMA = 0.95
BONUS_CHEGADA = 5.0


def rollout(pol, arenas, alvos_rel, passos=PASSOS_MAX):
    """Um lote de episodios amostrando acoes. Devolve as transicoes e as metricas."""
    n = len(arenas)
    r = post("/lote/reset", {"posicoes": [[x, z] for x, z in arenas]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    # Egocentrico na largada, como na avaliacao. O respawn randomiza o yaw,
    # entao offset em coordenadas de mundo ja daria angulo relativo uniforme —
    # mas so o egocentrico da cobertura CONTROLADA por setor, e mantem treino e
    # avaliacao com a mesma definicao de alvo.
    alvo_abs = []
    for i in range(n):
        dx, dz = alvo_para_mundo(est[i], alvos_rel[i][0], alvos_rel[i][1])
        alvo_abs.append((est[i]["x"] + dx, est[i]["z"] + dz))
    d0 = [math.hypot(alvo_abs[i][0] - est[i]["x"], alvo_abs[i][1] - est[i]["z"])
          for i in range(n)]
    dant = list(d0)
    dmin = list(d0)
    vivo = [True] * n
    chegou_em = [None] * n

    U8, SV, GV, IDX, R, VIVO = [], [], [], [], [], []

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
            rec[i] = dant[i] - d
            dant[i] = d
            dmin[i] = min(dmin[i], d)
            if d <= RAIO_CHEGADA:
                rec[i] += BONUS_CHEGADA
                chegou_em[i] = t + 1
                vivo[i] = False        # episodio encerrado; passos seguintes nao contam

        U8.append(u["u8"]); SV.append(u["sv"]); GV.append(u["gv"])
        IDX.append(u["idx"]); R.append(rec); VIVO.append(np.array(mascara_do_passo(vivo, chegou_em, t), dtype=np.float32))

        if not any(vivo):
            break

    met = [{"setor": alvos_rel[i][2], "d0": d0[i], "dmin": dmin[i], "dfinal": dant[i],
            "chegou": chegou_em[i] is not None, "passos": chegou_em[i]}
           for i in range(n)]
    if os.environ.get("DEBUG_ROLLOUT"):
        import collections as _c
        cnt = _c.Counter(int(v) for arr in IDX for v in arr)
        print("    [dbg] bins emitidos: " + " ".join(
            "%d:%d" % (k, cnt[k]) for k in range(9)), flush=True)
        print("    [dbg] passos=%d  " % (t + 1) + " ".join(
            "%s:d0=%.1f->%.1f%s" % (m["setor"][:3], m["d0"], m["dfinal"],
                                    "OK" if m["chegou"] else "")
            for m in met), flush=True)
    return (np.stack(U8), np.stack(SV), np.stack(GV), np.stack(IDX),
            np.stack(R), np.stack(VIVO)), met


def mascara_do_passo(vivo, chegou_em, t):
    """Mascara do passo que ACABOU de ser executado: conta se o episodio estava
    vivo ANTES dele. Sem isso, o passo da chegada seria descartado — justamente
    o unico que carrega o bonus."""
    return [1.0 if (v or c == t + 1) else 0.0 for v, c in zip(vivo, chegou_em)]


def retornos(R, VIVO, gamma=GAMMA):
    """Retorno descontado por episodio, para tras."""
    T, n = R.shape
    G = np.zeros_like(R)
    acc = np.zeros(n, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        acc = R[t] + gamma * acc * VIVO[t]
        G[t] = acc
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteracoes", type=int, default=120)
    # lr 3e-5, nao 1e-4: com 1e-4 os logits cresceram sem limite e a
    # entropia foi a ZERO na 5a iteracao — politica deterministica, exploracao
    # morta, e a corrida passou a oscilar entre 50% e 0%.
    ap.add_argument("--lr", type=float, default=3e-5)
    # minilote 2, nao 4: os modulos treinaveis (resampler, projetor,
    # goal_encoder) ficam ANTES do backbone, entao o backward atravessa as 56
    # execucoes de camada do LoopSplit guardando ativacoes. Com 4 deu OOM.
    ap.add_argument("--minilote", type=int, default=4)
    # 0.01 nao segurou nada. Com 9 bins a entropia maxima e ln(9)=2.20;
    # queremos que fique acima de ~0.8 durante o treino todo.
    ap.add_argument("--entropia", type=float, default=0.05)
    ap.add_argument("--entropia-final", type=float, default=0.01)
    ap.add_argument("--temperatura", type=float, default=1.0)
    ap.add_argument("--ckpt-entrada", default=None)
    ap.add_argument("--cabecas-novas", action="store_true", default=True)
    ap.add_argument("--manter-cabecas", dest="cabecas_novas", action="store_false")
    # Retomar de verdade: sem isto o recozimento da entropia reinicia em 0.05 e
    # desfaz o afinamento ja conquistado, e o historico se perde.
    ap.add_argument("--retomar", default=None,
                    help="checkpoint de onde continuar (mantem cabecas e o "
                         "ponto do recozimento)")
    ap.add_argument("--vram", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    arenas_todas = carregar_arenas()
    # Arenas de TREINO e de AVALIACAO separadas: com poucas arenas, ganho da
    # visao pode ser memorizacao de cena. A diferenca entre as duas denuncia.
    corte = max(1, int(0.75 * len(arenas_todas)))
    arenas_tr = arenas_todas[:corte]
    print("[fase1] %d arenas de treino, %d reservadas para avaliacao"
          % (len(arenas_tr), len(arenas_todas) - corte), flush=True)

    it0, hist_ant = 0, []
    if args.retomar:
        ck = torch.load(args.retomar, map_location="cpu")
        it0 = int(ck.get("iteracao", 0))
        hist_ant = list(ck.get("hist", []))
        args.ckpt_entrada = args.retomar
        args.cabecas_novas = False
        print("[retomar] de %s na iteracao %d (%d no historico)"
              % (args.retomar, it0, len(hist_ant)), flush=True)

    N = get("/lote/info")["envs"]
    travar_gpu()
    vla, device = load_vla_agent(args.ckpt_entrada)

    if args.cabecas_novas:
        # A cabeca de giro herdada vem de um treino por imitacao que fracassou
        # (colapsou para "nunca virar" na execucao). Comecar dali envenena o RL.
        # A inicializacao original e melhor de proposito: pesos zerados e vies
        # 2.0 no bin zero, ou seja "comeca andando reto e APRENDE a virar".
        from modelo.vla_model import IndependentActionHeads
        ah = vla.action_heads
        for head, bins in ((ah.yaw_head, ah.YAW_BINS), (ah.pitch_head, ah.PITCH_BINS)):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
            head.bias.data[bins.index(0)] = 2.0
        print("[init] cabeca de giro reinicializada (reto, com vies no bin 0)", flush=True)

    compactar_backbone(vla)
    torch.cuda.empty_cache()
    print("[VRAM] %s | %s" % (limitar_vram(args.vram), memoria_gpu()), flush=True)

    pol = PoliticaNeural(None, amostrar=True, temperatura=args.temperatura,
                         device=device, vla=vla)

    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    print("[treino] %.2fM parametros (backbone e visao congelados) | so o giro e aprendido"
          % (sum(p.numel() for p in treinaveis) / 1e6), flush=True)
    opt = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=1e-2)
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    hist = list(hist_ant)
    t0 = time.time()
    for it in range(it0 + 1, args.iteracoes + 1):
        arenas = [arenas_tr[rng.randrange(len(arenas_tr))] for _ in range(N)]
        alvos = [amostrar_alvo(rng, setor=SETORES[(it * N + i) % 8]) for i in range(N)]

        vla.eval()
        pol.amostrar = True
        (U8, SV, GV, IDX, R, VIVO), met = rollout(pol, arenas, alvos)

        G = retornos(R, VIVO)
        m = VIVO.reshape(-1) > 0
        g = G.reshape(-1)[m]
        adv = (g - g.mean()) / (g.std() + 1e-6)

        u8 = U8.reshape(-1, *U8.shape[2:])[m]
        sv = SV.reshape(-1, SV.shape[-1])[m]
        gv = GV.reshape(-1, GV.shape[-1])[m]
        idx = IDX.reshape(-1)[m]

        # Recozimento da entropia: alta no comeco para explorar, baixa no fim
        # para afiar. Com 0.05 fixo a politica fica em entropia ~1.8 de um
        # maximo de 2.20 — quase uniforme, o que limita o teto; com 0.01 fixo
        # ela colapsa para zero na 5a iteracao. O recozimento pega os dois.
        # frac usa a iteracao ABSOLUTA, entao retomar continua a curva de
        # recozimento em vez de recomeca-la.
        frac = (it - 1) / max(1, args.iteracoes - 1)
        beta = args.entropia + frac * (args.entropia_final - args.entropia)

        # Recalculo COM grafo, em minilotes. O cache do rollout (forward de
        # lote 8) precisa sair antes, senao o pico do backward nao cabe.
        del U8
        torch.cuda.empty_cache()
        vla.train(); vla.vision_encoder.eval(); vla.qwen_model.eval()
        ordem = np.random.permutation(len(idx))
        perda_soma, ent_soma, nb = 0.0, 0.0, 0
        for b0 in range(0, len(ordem), args.minilote):
            sel = ordem[b0:b0 + args.minilote]
            px = pol.normalizar(u8[sel])
            svt = torch.tensor(sv[sel], dtype=torch.float32, device=device)
            gvt = torch.tensor(gv[sel], dtype=torch.float32, device=device)
            at = torch.tensor(idx[sel], dtype=torch.long, device=device)
            advt = torch.tensor(adv[sel], dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                lg = pol.logits(px, svt, gvt)
            logp = torch.log_softmax(lg.float(), dim=-1)
            lp = logp.gather(1, at.unsqueeze(1)).squeeze(1)
            ent = -(logp.exp() * logp).sum(-1).mean()
            # Entropia segura contra o colapso para o bin zero, que e o prior
            # com que a cabeca nasce (bias 2.0 no bin 0).
            perda = -(lp * advt).mean() - beta * ent

            escala.scale(perda).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
            escala.step(opt); escala.update()
            perda_soma += float(perda.item()); ent_soma += float(ent.item()); nb += 1
            if nb % 40 == 0:
                torch.cuda.empty_cache()

        taxa = sum(x["chegou"] for x in met) / len(met)
        rmed = float(R.sum(axis=0).mean())
        hist.append({"it": it, "taxa": taxa, "recompensa": rmed,
                     "dfinal": float(np.mean([x["dfinal"] for x in met]))})
        print("it %3d/%d | chegada %3.0f%% | recompensa %+6.2f | d_final %5.2f | "
              "entropia %.2f | %4.0fs"
              % (it, args.iteracoes, 100 * taxa, rmed,
                 np.mean([x["dfinal"] for x in met]), ent_soma / max(1, nb),
                 time.time() - t0), flush=True)
        if it % 10 == 0:
            ult = hist[-10:]
            print("    [media das ultimas 10] chegada %.0f%% | recompensa %+.2f | beta %.3f"
                  % (100 * np.mean([h["taxa"] for h in ult]),
                     np.mean([h["recompensa"] for h in ult]), beta), flush=True)

        if it % 5 == 0 or it == args.iteracoes:
            # Salva TODOS os parametros treinaveis por nome, nao uma lista de
            # submodulos escolhida a mao. A lista anterior deixava de fora
            # `frame_time_embed` (4096 params, somado aos tokens visuais), que
            # voltava ALEATORIO no recarregamento e embaralhava a via visual:
            # o treino reportava 78-100% e o checkpoint recarregado dava 5%.
            torch.save({"treinaveis": {n: p.detach().cpu()
                                       for n, p in vla.named_parameters()
                                       if p.requires_grad},
                        "resampler": vla.resampler.state_dict(),
                        "projector": vla.projector.state_dict(),
                        "state_encoder": vla.state_encoder.state_dict(),
                        "goal_encoder": vla.goal_encoder.state_dict(),
                        "action_heads": vla.action_heads.state_dict(),
                        "iteracao": it, "hist": hist}, CKPT_SAIDA)

    # Tendencia: a recompensa se moveu?
    if len(hist) >= 6:
        ini = np.mean([h["recompensa"] for h in hist[:3]])
        fim = np.mean([h["recompensa"] for h in hist[-3:]])
        print("\n[tendencia] recompensa %.2f -> %.2f  (%+.2f)" % (ini, fim, fim - ini))
        ti = np.mean([h["taxa"] for h in hist[:3]])
        tf = np.mean([h["taxa"] for h in hist[-3:]])
        print("[tendencia] chegada %.0f%% -> %.0f%%  (%+.0f pontos)"
              % (100 * ti, 100 * tf, 100 * (tf - ti)))
    print("\n[OK] %s" % CKPT_SAIDA, flush=True)


if __name__ == "__main__":
    main()
