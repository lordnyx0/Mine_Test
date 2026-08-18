# coding=utf-8
"""
Pré-treino da base sobre o dataset REAL (dataset/locomocao_real.jsonl).

Substitui o caminho do train_vla.py, que serve ao formato antigo — imagem
unica desenhada com PIL e sem alvo de rota. Aqui o formato de entrada e
IDENTICO ao de execucao, que era a outra metade do problema: pilha de 3
frames, state_vec de 32 dims e instrucao tokenizada.

Duas familias de perda:
  ACAO   — botoes (BCE) + yaw/pitch em bins (CE). Determinada pela instrucao.
  ROTA   — 12 setores de navegabilidade (MSE). Determinada pela IMAGEM, e o
           unico termo que obriga a via visual a permanecer informativa.

    python treinar_base_real.py --epocas 8
"""
import os
import io
import sys
import glob
import json
import time
import base64
import random
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_utils import (limitar_recursos, limitar_vram, travar_gpu,
                       compactar_backbone, memoria_gpu)
limitar_recursos()

import torch
import torch.nn as nn
from PIL import Image

from run_vla_agent import load_vla_agent
from train_vla import get_tokenizer

DATASET = "dataset/locomocao_real.jsonl"
CKPT_DIR = "checkpoints_vla"
CKPT_SAIDA = os.path.join(CKPT_DIR, "vla_locomotion.pt")
MAX_CHECKPOINTS = 5

BUTTONS = ["W", "S", "A", "D", "SPACE", "LCLICK", "RCLICK", "SHIFT"]


def carregar(caminho):
    linhas = []
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                linhas.append(json.loads(l))
            except Exception:
                pass
    return linhas


def descomprimir(b64s):
    return [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in b64s]


# SigLIP: redimensiona para 224 bicubico e normaliza com mean=std=0.5,
# ou seja pixel = uint8/127.5 - 1. Simples o bastante para fazer na GPU.
TAM = 224


def predecodificar(dados, n_frames=3):
    """
    Decodifica TODOS os frames uma vez para uint8 [N, K, 224, 224, 3].

    Com 2 threads de CPU (limite imposto para nao travar a maquina), decodificar
    6 JPEGs por lote dentro do laco deixava a GPU em 52% e a epoca em ~35 min.
    Pre-decodificar custa ~1min uma vez e 1.9GB de RAM, e tira PIL e o
    processador do caminho quente.
    """
    import numpy as np
    n = len(dados)
    buf = np.empty((n, n_frames, TAM, TAM, 3), dtype=np.uint8)
    t0 = time.time()
    for i, d in enumerate(dados):
        for k, b64 in enumerate(d["frames_b64"][:n_frames]):
            im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            if im.size != (TAM, TAM):
                im = im.resize((TAM, TAM), Image.BICUBIC)
            buf[i, k] = np.asarray(im, dtype=np.uint8)
        if (i + 1) % 1000 == 0:
            print(f"  [cache] {i+1}/{n} amostras decodificadas "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  [cache] {n} amostras em {time.time()-t0:.0f}s "
          f"({buf.nbytes/2**30:.1f}GB)", flush=True)
    return buf


def lote_pixels(cache, indices, device):
    """uint8 [B,K,H,W,3] -> float [B,K,3,H,W] normalizado, tudo na GPU."""
    x = torch.from_numpy(cache[indices]).to(device, non_blocking=True)
    x = x.permute(0, 1, 4, 2, 3).float()
    return x.div_(127.5).sub_(1.0)


def alvo_botoes(hold):
    h = [k.upper() for k in hold]
    return [1.0 if b in h else 0.0 for b in BUTTONS]


def bin_proximo(v, bins):
    return min(range(len(bins)), key=lambda i: abs(bins[i] - float(v)))


def podar(limite=MAX_CHECKPOINTS):
    arqs = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime)
    while len(arqs) > limite:
        try:
            os.remove(arqs.pop(0))
        except OSError:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--epocas", type=int, default=8)
    # batch 2: cada amostra sao 3 frames, entao batch 4 = 12 imagens pelo SigLIP
    # mais o backward sobre 132 tokens. Batch 4 estourou os 8.6GB do teto.
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--peso-rota", type=float, default=2.0)
    ap.add_argument("--vram", type=float, default=0.72)
    args = ap.parse_args()

    random.seed(0); torch.manual_seed(0)
    travar_gpu()

    dados = carregar(args.dataset)
    if not dados:
        print(f"[ERRO] '{args.dataset}' vazio. Rode gerar_dataset_real.py antes.")
        return
    com_rota = sum(1 for d in dados if d.get("rotas"))
    print(f"[dados] {len(dados)} amostras reais | {com_rota} com alvo de rota", flush=True)

    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    print(f"[VRAM] teto {limitar_vram(args.vram)} | {memoria_gpu()}", flush=True)

    tok = get_tokenizer()
    proc = vla.vision_processor
    cache_px = predecodificar(dados)
    ordem = list(range(len(dados)))
    YB = vla.action_heads.YAW_BINS
    PB = vla.action_heads.PITCH_BINS
    K_ROTA = vla.action_heads.num_rotas

    # ── Peso por classe no giro ───────────────────────────────────────────────
    # 66.7% do dataset e "nao virar". Sem reponderar, o otimo e prever sempre
    # o bin zero: foi o que aconteceu — 70% de acerto no treino contra 66.7%
    # da classe majoritaria, ou seja, aprendizado ~nulo sobre QUANDO virar,
    # que e justamente a unica decisao que importa (W e constante em 100%).
    cont = [0] * len(YB)
    for d in dados:
        cont[bin_proximo(d["action"]["mouse"][0], YB)] += 1
    total = sum(cont)
    pesos = [min(8.0, total / (len(YB) * max(1, c))) for c in cont]
    print("[classes] giro: " + " ".join(
        f"{YB[i]:+d}:{100*cont[i]/total:.1f}%(w{pesos[i]:.1f})" for i in range(len(YB))),
        flush=True)
    peso_yaw = torch.tensor(pesos, dtype=torch.float32, device=device)

    params = [p for p in vla.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    ce_yaw = nn.CrossEntropyLoss(weight=peso_yaw)
    mse = nn.MSELoss()
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    os.makedirs(CKPT_DIR, exist_ok=True)

    for epoca in range(1, args.epocas + 1):
        vla.train(); vla.vision_encoder.eval(); vla.qwen_model.eval()
        random.shuffle(ordem)
        soma, soma_rota, n = 0.0, 0.0, 0
        acertos_yaw, total_yaw = 0, 0
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

            tb = torch.tensor([alvo_botoes(d["action"].get("hold", [])) for d in lote],
                              dtype=torch.float32, device=device)
            ty = torch.tensor([bin_proximo(d["action"]["mouse"][0], YB) for d in lote],
                              dtype=torch.long, device=device)
            tp = torch.tensor([bin_proximo(d["action"]["mouse"][1], PB) for d in lote],
                              dtype=torch.long, device=device)

            idx_rota = [j for j, d in enumerate(lote)
                        if d.get("rotas") and len(d["rotas"]) == K_ROTA]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                out = vla(pixel_values=px, state_vec=sv, input_ids=ids)

            perda = (bce(out["buttons_logits"].float(), tb)
                     + ce_yaw(out["yaw_logits"].float(), ty)
                     + 0.3 * ce(out["pitch_logits"].float(), tp))

            perda_rota = torch.zeros((), device=device)
            if idx_rota:
                sel = torch.tensor(idx_rota, dtype=torch.long, device=device)
                alvo_r = torch.tensor([lote[j]["rotas"] for j in idx_rota],
                                      dtype=torch.float32, device=device)
                perda_rota = mse(out["rotas"].float()[sel], alvo_r)
                perda = perda + args.peso_rota * perda_rota

            escala.scale(perda).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            escala.step(opt)
            escala.update()

            soma += float(perda.item()); soma_rota += float(perda_rota.item()); n += 1
            if n % 200 == 0:
                torch.cuda.empty_cache()   # evita fragmentacao ao longo da epoca
            pred = out["yaw_logits"].float().argmax(-1)
            acertos_yaw += int((pred == ty).sum())
            total_yaw += len(lote)
            for j in range(len(lote)):
                c = int(ty[j]); total_cls[c] += 1
                if int(pred[j]) == c: acerto_cls[c] += 1

        torch.save({
            "resampler":     vla.resampler.state_dict(),
            "projector":     vla.projector.state_dict(),
            "state_encoder": vla.state_encoder.state_dict(),
            "action_heads":  vla.action_heads.state_dict(),
            "epoca": epoca,
        }, CKPT_SAIDA)
        podar()

        # Recall MEDIO POR CLASSE: e esta a metrica honesta aqui. A acuracia
        # simples fica em 67% so por prever sempre "nao virar".
        recalls = [acerto_cls[i] / total_cls[i] for i in range(len(YB)) if total_cls[i] >= 5]
        recall_medio = sum(recalls) / max(1, len(recalls))
        print("epoca %d/%d | perda %.4f (rota %.4f) | yaw acc %.0f%% | recall/classe %.0f%% | %.0fs"
              % (epoca, args.epocas, soma / max(1, n), soma_rota / max(1, n),
                 100 * acertos_yaw / max(1, total_yaw), 100 * recall_medio,
                 time.time() - t0), flush=True)
        print("   recall por bin: " + " ".join(
            f"{YB[i]:+d}:{100*acerto_cls[i]/max(1,total_cls[i]):.0f}%"
            for i in range(len(YB)) if total_cls[i] >= 5), flush=True)

    print(f"\n[OK] base real salva em {CKPT_SAIDA}", flush=True)


if __name__ == "__main__":
    main()
