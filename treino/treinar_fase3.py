# coding=utf-8
"""
FASE 3 — Treino de RL Servo-Visual Puro (Sem Coordenadas).

Aprende a navegar até um bloco visual utilizando apenas a imagem da câmera
e uma instrução textual, eliminando o atalho trigonométrico (atan2).
"""
import os
import sys
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

from infra.run_vla_agent import load_vla_agent
from ambiente.arena_plana import post, get
from ambiente.fase3 import PASSOS_MAX_F3, RAIO_CHEGADA_F3, montar_tarefas_visuais
from politica.politica_fase3 import PoliticaFase3
from treino.treinar_fase1 import mascara_do_passo, retornos, GAMMA, BONUS_CHEGADA

CKPT_SAIDA = "checkpoints_vla/vla_fase3.pt"


def rollout_f3(pol, tarefas, passos=None):
    passos = passos or PASSOS_MAX_F3
    n = len(tarefas)
    alvos_abs = [t["alvo_abs"] for t in tarefas]
    prompts = [t["prompt"] for t in tarefas]

    r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in tarefas]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    # Re-insere os pilares coloridos (Roxo, Amarelo, Azul) nos ambientes após o reset
    blocos = [{"env": t["env"], "x": math.floor(t["alvo_abs"][0]), "y": t["alvo_y"],
               "z": math.floor(t["alvo_abs"][1]), "id": t.get("bloco_id", 49)} for t in tarefas]
    post("/lote/colocar_bloco", {"blocos": blocos})

    d0 = [math.hypot(alvos_abs[i][0] - est[i]["x"], alvos_abs[i][1] - est[i]["z"]) for i in range(n)]
    dant = list(d0)
    dmin = list(d0)
    vivo, chegou_em = [True] * n, [None] * n
    U8, SV, GV, IDS, IDX, R, VIVO = [], [], [], [], [], [], []

    for t in range(passos):
        acoes = pol.agir(est, [None] * n, obs, prompts=prompts)
        u = pol.ultimo
        rr = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = rr["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        rec = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if not vivo[i]:
                continue
            d = math.hypot(alvos_abs[i][0] - est[i]["x"], alvos_abs[i][1] - est[i]["z"])
            rec[i] = dant[i] - d
            dant[i] = d
            dmin[i] = min(dmin[i], d)
            if d <= RAIO_CHEGADA_F3:
                rec[i] += BONUS_CHEGADA
                chegou_em[i] = t + 1
                vivo[i] = False

        U8.append(u["u8"]); SV.append(u["sv"]); GV.append(u["gv"]); IDS.append(u["ids"])
        IDX.append(u["idx"]); R.append(rec)
        VIVO.append(np.array(mascara_do_passo(vivo, chegou_em, t), dtype=np.float32))
        if not any(vivo):
            break

    met = [{"chegou": chegou_em[i] is not None, "d0": d0[i], "dfinal": dant[i], "dmin": dmin[i]} for i in range(n)]
    return (np.stack(U8), np.stack(SV), np.stack(GV), np.stack(IDS), np.stack(IDX),
            np.stack(R), np.stack(VIVO)), met


class BufferSucesso:
    """Armazena as melhores trajetórias de chegada para manter reforço positivo constante."""
    def __init__(self, cap=20):
        self.cap = cap
        self.buffer = []

    def adicionar(self, u8, sv, gv, ids, ay, adv):
        self.buffer.append({
            "u8": u8, "sv": sv, "gv": gv, "ids": ids, "ay": ay, "adv": adv
        })
        if len(self.buffer) > self.cap:
            self.buffer.pop(0)

    def amostrar(self, k):
        if not self.buffer:
            return None
        # Sorteia aleatoriamente amostras do buffer
        b = random.choice(self.buffer)
        n = len(b["ay"])
        if n == 0:
            return None
        idx = np.random.choice(n, min(k, n), replace=False)
        return {k_: b[k_][idx] for k_ in ("u8", "sv", "gv", "ids", "ay", "adv")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteracoes", type=int, default=120)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--clip-ppo", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--minilote", type=int, default=12)
    ap.add_argument("--entropia", type=float, default=0.06)
    ap.add_argument("--entropia-final", type=float, default=0.025)
    ap.add_argument("--ckpt-entrada", default="checkpoints_vla/vla_fase3.pt")
    ap.add_argument("--vram", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.98)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    N = get("/lote/info")["envs"]
    travar_gpu()
    vla, device = load_vla_agent(args.ckpt_entrada)
    compactar_backbone(vla)
    torch.cuda.empty_cache()

    pol = PoliticaFase3(None, amostrar=True, device=device, vla=vla)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=1e-2)
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    buffer_sucesso = BufferSucesso(cap=20)
    hist = []
    t0 = time.time()

    for it in range(1, args.iteracoes + 1):
        # Currículo progressivo: distância cresce de 12m até 24m até a iteração 40
        c_frac = min(1.0, it / 40.0)
        tarefas = montar_tarefas_visuais(N, seed=args.seed + it, verbose=False, curriculo_frac=c_frac)
        vla.eval(); pol.amostrar = True
        (U8, SV, GV, IDS, IDX, R, VIVO), met = rollout_f3(pol, tarefas)

        G = retornos(R, VIVO, gamma=args.gamma)
        m = VIVO.reshape(-1) > 0
        g = G.reshape(-1)[m]
        g_std = float(g.std())
        PISO_G_STD = 3.0
        adv = (g - g.mean()) / (max(g_std, PISO_G_STD) + 1e-6)

        u8 = U8.reshape(-1, *U8.shape[2:])[m]
        sv = SV.reshape(-1, SV.shape[-1])[m]
        gv = GV.reshape(-1, GV.shape[-1])[m]
        ids = IDS.reshape(-1, IDS.shape[-1])[m]
        ay = IDX.reshape(-1)[m]

        # Salva trajetórias de sucesso no buffer de prioridade
        for i_env, m_info in enumerate(met):
            if m_info["chegou"]:
                m_env = VIVO[:, i_env] > 0
                if any(m_env):
                    buffer_sucesso.adicionar(
                        U8[:, i_env][m_env], SV[:, i_env][m_env],
                        GV[:, i_env][m_env], IDS[:, i_env][m_env],
                        IDX[:, i_env][m_env], adv[:sum(m_env)]
                    )

        frac = (it - 1) / max(1, args.iteracoes - 1)
        beta = args.entropia + frac * (args.entropia_final - args.entropia)

        vla.train(); vla.vision_encoder.eval(); vla.qwen_model.eval()

        # Calcula old_logp sob a política anterior para o ratio PPO
        old_logp = np.zeros(len(ay), dtype=np.float32)
        with torch.no_grad():
            for b0 in range(0, len(ay), args.minilote):
                sel = slice(b0, b0 + args.minilote)
                px = pol.normalizar(u8[sel])
                svt = torch.tensor(sv[sel], dtype=torch.float32, device=device)
                gvt = torch.tensor(gv[sel], dtype=torch.float32, device=device)
                idst = torch.tensor(ids[sel], dtype=torch.long, device=device)
                yt = torch.tensor(ay[sel], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    lp_tmp, _ = pol.log_prob(px, svt, gvt, yt, ids=idst)
                old_logp[sel] = lp_tmp.float().cpu().numpy()

        ent_soma, nb = 0.0, 0
        # Múltiplas mini-épocas com clipping PPO
        for ep in range(args.epocas):
            ordem = np.random.permutation(len(ay))
            for b0 in range(0, len(ordem), args.minilote):
                sel = ordem[b0:b0 + args.minilote]
                px = pol.normalizar(u8[sel])
                svt = torch.tensor(sv[sel], dtype=torch.float32, device=device)
                gvt = torch.tensor(gv[sel], dtype=torch.float32, device=device)
                idst = torch.tensor(ids[sel], dtype=torch.long, device=device)
                yt = torch.tensor(ay[sel], dtype=torch.long, device=device)
                advt = torch.tensor(adv[sel], dtype=torch.float32, device=device)
                old_lpt = torch.tensor(old_logp[sel], dtype=torch.float32, device=device)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=torch.cuda.is_available()):
                    lp, ent = pol.log_prob(px, svt, gvt, yt, ids=idst)

                    # PPO Ratio Clipping
                    ratio = torch.exp(lp.float() - old_lpt)
                    surr1 = ratio * advt
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_ppo, 1.0 + args.clip_ppo) * advt
                    perda_pol = -torch.min(surr1, surr2).mean()
                    perda = perda_pol - beta * ent.float()

                escala.scale(perda).backward()
                escala.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(treinaveis, 0.5)
                escala.step(opt); escala.update()
                ent_soma += float(ent.item()); nb += 1

        taxa = sum(x["chegou"] for x in met) / len(met)
        hist.append({"it": it, "taxa": taxa, "entropia": ent_soma / max(1, nb)})
        print("it %3d/%d | chegada %3.0f%% | ent %.2f | g_std %.3f | PPO 2ep | %4.0fs"
              % (it, args.iteracoes, 100 * taxa, ent_soma / max(1, nb), g_std,
                 time.time() - t0), flush=True)

        if it % 5 == 0 or it == args.iteracoes:
            torch.save({"treinaveis": {n_: p.detach().cpu()
                                        for n_, p in vla.named_parameters()
                                        if p.requires_grad},
                        "iteracao": it, "hist": hist}, CKPT_SAIDA)

    print("\n[OK] Fase 3 treino otimizado concluido -> %s" % CKPT_SAIDA, flush=True)


if __name__ == "__main__":
    main()
