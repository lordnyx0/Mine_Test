# coding=utf-8
"""
Politica condicionada a OBJETIVO, treinada por rotulagem em retrospecto.

Por que isto e diferente da clonagem que falhou: "que tecla o professor
apertou" nao tem resposta unica — medido, 19% de recall por classe com split
por episodio, contra 11% de acaso. "que tecla leva ate B", onde B e o ponto
onde o agente DE FATO chegou k passos depois, tem: 30% no mesmo teste, com o
controle de objetivo embaralhado empatando com 19%, e o ganho sobrevivendo a
k=20 (5s), entao nao e vazamento.

O rotulo deixa de ser ambiguo por construcao: toda trajetoria, boa ou ruim,
e uma demonstracao PERFEITA de como chegar onde ela chegou. Nao ha professor.

Alvo a bater: `reto` (aponta e anda) chega em 55% a ~100 blocos. Teto:
o planejador, 80-90%. Sao 25+ pontos de faixa dinamica — ao contrario da
metrica antiga, que saturava.

O backbone e a visao ficam CONGELADOS. Treinam resampler, projetor,
state_encoder, goal_encoder e as action heads.

    python treinar_objetivo.py --epocas 4
"""
import os
import io
import sys
import json
import time
import math
import base64
import random
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_utils import (limitar_recursos, limitar_vram, travar_gpu,
                       compactar_backbone, memoria_gpu)
limitar_recursos()

import numpy as np
import torch
import torch.nn as nn

from run_vla_agent import load_vla_agent
from train_vla import get_tokenizer
from treinar_base_real import (predecodificar, lote_pixels, alvo_botoes,
                               bin_proximo, BUTTONS)
from infra.gate_retrospecto import objetivo_relativo

DATASET = "dataset/sim_hindsight.jsonl"
CKPT_DIR = "checkpoints_vla"
CKPT_SAIDA = os.path.join(CKPT_DIR, "vla_objetivo.pt")   # nome NOVO: nao toca a base
K_ROTA = 12
K_MIN, K_MAX = 4, 20      # horizonte do objetivo, em passos (1s a 5s)


def carregar_com_futuro(caminho, limite=0):
    """Carrega agrupando por episodio e anexa as posicoes futuras de cada amostra.

    Sem o agrupamento nao existe 'futuro': o arquivo e escrito intercalando os
    8 ambientes a cada passo, entao linhas vizinhas sao de episodios diferentes.
    """
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
            if d.get("pos") and d.get("frames_b64") and d.get("rotas"):
                eps.setdefault(d["ep"], []).append(d)
    for k in eps:
        eps[k].sort(key=lambda d: d["t"])

    dados = []
    for seq in eps.values():
        for i, d in enumerate(seq):
            # Guarda so as posicoes; os frames ja estao na propria amostra.
            futuros = [seq[j]["pos"] for j in range(i + K_MIN, min(i + K_MAX + 1, len(seq)))]
            if not futuros:
                continue          # fim do episodio: sem futuro, sem objetivo
            d = dict(d)
            d["_futuros"] = futuros
            dados.append(d)
    if limite:
        dados = dados[:limite]
    return dados, len(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--epocas", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--peso-rota", type=float, default=2.0)
    ap.add_argument("--vram", type=float, default=0.72)
    ap.add_argument("--limite", type=int, default=0)
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    travar_gpu()

    dados, n_eps = carregar_com_futuro(args.dataset, args.limite)
    if not dados:
        print("[ERRO] dataset vazio ou sem 'pos'. Rode gerar_dataset_sim.py.")
        return
    print("[dados] %d amostras utilizaveis de %d episodios | horizonte %d-%d passos"
          % (len(dados), n_eps, K_MIN, K_MAX), flush=True)

    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    print("[VRAM] teto %s | %s" % (limitar_vram(args.vram), memoria_gpu()), flush=True)

    tok = get_tokenizer()
    cache_px = predecodificar(dados)
    ordem = list(range(len(dados)))
    YB = vla.action_heads.YAW_BINS
    PB = vla.action_heads.PITCH_BINS

    cont = [0] * len(YB)
    for d in dados:
        cont[bin_proximo(d["action"]["mouse"][0], YB)] += 1
    total = sum(cont)
    pesos = [min(8.0, total / (len(YB) * max(1, c))) for c in cont]
    print("[classes] giro: " + " ".join(
        "%+d:%.1f%%(w%.1f)" % (YB[i], 100 * cont[i] / total, pesos[i])
        for i in range(len(YB))), flush=True)
    peso_yaw = torch.tensor(pesos, dtype=torch.float32, device=device)

    params = [p for p in vla.parameters() if p.requires_grad]
    print("[treino] %.2fM parametros treinaveis (backbone e visao congelados)"
          % (sum(p.numel() for p in params) / 1e6), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    ce_yaw = nn.CrossEntropyLoss(weight=peso_yaw)
    mse = nn.MSELoss()
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    os.makedirs(CKPT_DIR, exist_ok=True)

    for epoca in range(1, args.epocas + 1):
        vla.train(); vla.vision_encoder.eval(); vla.qwen_model.eval()
        random.shuffle(ordem)
        soma, soma_rota, n = 0.0, 0.0, 0
        acerto_cls = [0] * len(YB)
        total_cls = [0] * len(YB)
        t0 = time.time()

        for i in range(0, len(ordem) - args.batch + 1, args.batch):
            idx = ordem[i:i + args.batch]
            lote = [dados[j] for j in idx]

            px = lote_pixels(cache_px, idx, device)
            sv = torch.tensor([d["state_vec"] for d in lote],
                              dtype=torch.float32, device=device)
            ids = tok([d["instrucao"] for d in lote], return_tensors="pt",
                      padding=True, truncation=True, max_length=24)["input_ids"].to(device)

            # O horizonte e sorteado A CADA VEZ: a mesma cena aparece com
            # objetivos a 1s e a 5s, e a politica precisa obedecer ao objetivo
            # em vez de decorar uma acao por cena.
            gv = []
            for d in lote:
                fut = random.choice(d["_futuros"])
                gv.append(objetivo_relativo(d["pos"], fut))
            gv = torch.tensor(gv, dtype=torch.float32, device=device)

            tb = torch.tensor([alvo_botoes(d["action"].get("hold", [])) for d in lote],
                              dtype=torch.float32, device=device)
            ty = torch.tensor([bin_proximo(d["action"]["mouse"][0], YB) for d in lote],
                              dtype=torch.long, device=device)
            tp = torch.tensor([bin_proximo(d["action"]["mouse"][1], PB) for d in lote],
                              dtype=torch.long, device=device)
            alvo_r = torch.tensor([d["rotas"] for d in lote],
                                  dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                out = vla(pixel_values=px, state_vec=sv, input_ids=ids, goal_vec=gv)

            perda_rota = mse(out["rotas"].float(), alvo_r)
            perda = (bce(out["buttons_logits"].float(), tb)
                     + ce_yaw(out["yaw_logits"].float(), ty)
                     + 0.3 * ce(out["pitch_logits"].float(), tp)
                     + args.peso_rota * perda_rota)

            escala.scale(perda).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            escala.step(opt)
            escala.update()

            soma += float(perda.item()); soma_rota += float(perda_rota.item()); n += 1
            if n % 200 == 0:
                torch.cuda.empty_cache()
            pred = out["yaw_logits"].float().argmax(-1)
            for j in range(len(lote)):
                c = int(ty[j]); total_cls[c] += 1
                if int(pred[j]) == c: acerto_cls[c] += 1
            if n % 300 == 0:
                print("    epoca %d | %d/%d lotes | perda %.4f | %.0fs"
                      % (epoca, n, len(ordem) // args.batch, soma / n,
                         time.time() - t0), flush=True)

        # Salva TODA epoca, com nome proprio: uma queda de energia as 3h nao
        # pode custar a noite inteira, e a base do usuario nao e tocada.
        torch.save({
            "resampler":     vla.resampler.state_dict(),
            "projector":     vla.projector.state_dict(),
            "state_encoder": vla.state_encoder.state_dict(),
            "goal_encoder":  vla.goal_encoder.state_dict(),
            "action_heads":  vla.action_heads.state_dict(),
            "epoca": epoca, "k_min": K_MIN, "k_max": K_MAX,
        }, CKPT_SAIDA)

        recalls = [acerto_cls[i] / total_cls[i]
                   for i in range(len(YB)) if total_cls[i] >= 5]
        print("epoca %d/%d | perda %.4f (rota %.4f) | recall/classe %.0f%% | %.0fs -> %s"
              % (epoca, args.epocas, soma / max(1, n), soma_rota / max(1, n),
                 100 * sum(recalls) / max(1, len(recalls)), time.time() - t0,
                 CKPT_SAIDA), flush=True)

    print("\n[OK] %s" % CKPT_SAIDA, flush=True)


if __name__ == "__main__":
    main()
