# coding=utf-8
"""
FASE 3 AVANÇADA — Treinamento com LoRA no Qwen3Loop e Perda de Modelo de Mundo.

Destrava a capacidade cognitiva das camadas intermediárias de loop:
  - LoRA (r=16, alpha=32) nas projeções de atenção do Qwen (expande de 10.6M para ~35M treináveis).
  - Perda Auxiliar de Modelo de Mundo (World Model Loss): previsão do próximo estado latente.
  - Warm-Start a partir dos adaptadores treinados em `checkpoints_vla/vla_fase3.pt`.
"""
import os
import sys
import time
import math
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from ambiente.arena_plana import post, get
from treino.treinar_fase1 import mascara_do_passo, retornos
from ambiente.fase3 import (montar_tarefas_visuais, PASSOS_MAX_F3,
                            RAIO_CHEGADA_F3, CORES_ALVO)
from politica.politica_fase3 import PoliticaFase3
from modelo.lora_vla import aplicar_lora
from modelo.world_model_loss import PreditorDinamicaLatente, perda_modelo_mundo
from infra.gpu_utils import travar_gpu, compactar_backbone
from infra.run_vla_agent import load_vla_agent

BONUS_CHEGADA = 10.0
CKPT_SAIDA = "checkpoints_vla/vla_fase3_lora.pt"


def rollout_f3(pol, tarefas, passos=None):
    passos = passos or PASSOS_MAX_F3
    n = len(tarefas)
    alvos_abs = [t["alvo_abs"] for t in tarefas]
    prompts = [t["prompt"] for t in tarefas]

    r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in tarefas]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteracoes", type=int, default=100)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--clip-ppo", type=float, default=0.20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--peso-world", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--minilote", type=int, default=12)
    ap.add_argument("--entropia", type=float, default=0.06)
    ap.add_argument("--entropia-final", type=float, default=0.02)
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

    # Aplica LoRA no backbone Qwen3Loop para destravar a capacidade de raciocínio
    vla.qwen_model = aplicar_lora(vla.qwen_model, r=args.lora_r, alpha=args.lora_alpha)
    vla.to(device)

    # Preditor de Dinâmica Latente para perda de Modelo de Mundo
    hidden_dim = getattr(vla, "hidden_size", 1024)
    world_predictor = PreditorDinamicaLatente(dim_oculta=hidden_dim, dim_acao=9, dim_latente=hidden_dim).to(device)

    pol = PoliticaFase3(None, amostrar=True, device=device, vla=vla)
    treinaveis = [p for p in vla.parameters() if p.requires_grad] + list(world_predictor.parameters())
    num_treinaveis = sum(p.numel() for p in treinaveis)
    print(f"[LoRA + World Model] Total de parâmetros treináveis: {num_treinaveis/1e6:.2f}M (expandido de 10.6M)")

    opt = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=1e-2)
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    hist = []
    t0 = time.time()

    for it in range(1, args.iteracoes + 1):
        c_frac = min(1.0, it / 35.0)
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

        frac = (it - 1) / max(1, args.iteracoes - 1)
        beta = args.entropia + frac * (args.entropia_final - args.entropia)

        vla.train(); vla.vision_encoder.eval(); world_predictor.train()

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

        ent_soma, loss_world_soma, nb = 0.0, 0.0, 0
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

                    ratio = torch.exp(lp.float() - old_lpt)
                    surr1 = ratio * advt
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_ppo, 1.0 + args.clip_ppo) * advt
                    perda_pol = -torch.min(surr1, surr2).mean()

                    # Perda de Modelo de Mundo
                    # Cria representação one-hot das ações para predição latente
                    a_onehot = torch.zeros((len(sel), 9), device=device, dtype=torch.float32)
                    a_onehot.scatter_(1, yt.unsqueeze(1), 1.0)
                    px_flat = px.flatten(0, 1) if px.dim() == 5 else px
                    with torch.no_grad():
                        v_feats = pol.vla.vision_encoder(pixel_values=px_flat).last_hidden_state
                    v_res = pol.vla.resampler(v_feats)
                    v_emb = pol.vla.projector(v_res)
                    if px.dim() == 5:
                        v_emb = v_emb.view(px.size(0), px.size(1) * 32, -1)
                    z_latente = v_emb.mean(dim=1)
                    z_prev = world_predictor(z_latente, a_onehot)
                    l_world = perda_modelo_mundo(z_prev, z_latente.detach())

                    perda = perda_pol - beta * ent.float() + args.peso_world * l_world

                escala.scale(perda).backward()
                escala.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(treinaveis, 0.5)
                escala.step(opt); escala.update()
                ent_soma += float(ent.item())
                loss_world_soma += float(l_world.item())
                nb += 1

        taxa = sum(x["chegou"] for x in met) / len(met)
        hist.append({"it": it, "taxa": taxa, "entropia": ent_soma / max(1, nb),
                     "world_loss": loss_world_soma / max(1, nb)})
        print("it %3d/%d | chegada %3.0f%% | ent %.2f | world_loss %.4f | g_std %.3f | %4.0fs"
              % (it, args.iteracoes, 100 * taxa, ent_soma / max(1, nb),
                 loss_world_soma / max(1, nb), g_std, time.time() - t0), flush=True)

        if it % 5 == 0 or it == args.iteracoes:
            torch.save({
                "treinaveis": {n_: p.detach().cpu()
                               for n_, p in vla.named_parameters()
                               if p.requires_grad},
                "world_predictor": world_predictor.state_dict(),
                "iteracao": it, "hist": hist
            }, CKPT_SAIDA)

    print("\n[OK] Treinamento com LoRA e World Model concluído -> %s" % CKPT_SAIDA, flush=True)


if __name__ == "__main__":
    main()
