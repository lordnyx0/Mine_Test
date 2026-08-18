# coding=utf-8
"""
Avalia a politica treinada DENTRO do simulador, contra dois pontos de
referencia, nas MESMAS posicoes iniciais.

  so W     — piso trivial. Qualquer politica que nao o supere nao aprendeu nada.
  Piloto   — o professor (planejador BFS). E o teto da imitacao.

O protocolo e pareado de proposito: a variancia entre pontos de partida e
enorme (desvio ~20 blocos), e comparacoes nao pareadas com n pequeno ja me
produziram dois falsos positivos nesta sessao.

    python avaliar_no_sim.py --posicoes 180
"""
import os
import sys
import json
import math
import time
import base64
import argparse
import statistics as st
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.gpu_utils import (limitar_recursos, limitar_vram, travar_gpu,
                             compactar_backbone, memoria_gpu)
limitar_recursos()

import torch
from PIL import Image
import io as _io

from infra.run_vla_agent import load_vla_agent
from infra.train_vla import get_tokenizer
from modelo.estado_sim import EstadoEpisodio, N_FRAMES

BASE = "http://127.0.0.1:3002"
BUTTONS = ["W", "S", "A", "D", "SPACE", "LCLICK", "RCLICK", "SHIFT"]
USABLE = [0, 1, 2, 3, 4, 7]

INSTRUCOES = {
    "explorar": ("Objetivo: explorar. Ande o maximo que puder e afaste-se o maximo "
                 "possivel do ponto onde voce nasceu. Nao fique parado no mesmo lugar."),
    "bloco": "Objetivo: coletar madeira. Encontre uma arvore e va ate o tronco.",
}


def post(caminho, corpo, timeout=300):
    d = json.dumps(corpo).encode()
    r = urllib.request.Request(BASE + caminho, data=d, method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read())


def acao_do_modelo(vla, device, tok, ids_cache, pilhas, estados, objetivo):
    """Uma inferencia em LOTE para todos os ambientes."""
    proc = vla.vision_processor
    planas = [Image.open(_io.BytesIO(b)).convert("RGB") for p in pilhas for b in p]
    px = proc(images=planas, return_tensors="pt")["pixel_values"]
    px = px.view(len(pilhas), N_FRAMES, *px.shape[1:]).to(device)
    sv = torch.tensor(estados, dtype=torch.float32, device=device)

    if objetivo not in ids_cache:
        ids_cache[objetivo] = tok([INSTRUCOES[objetivo]], return_tensors="pt",
                                  truncation=True, max_length=24)["input_ids"].to(device)
    ids = ids_cache[objetivo].expand(len(pilhas), -1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=torch.cuda.is_available()):
        out = vla(pixel_values=px, state_vec=sv, input_ids=ids)

    p = out["buttons"].float().cpu().numpy()
    yb = vla.action_heads.YAW_BINS
    pb = vla.action_heads.PITCH_BINS
    iy = out["yaw_logits"].float().argmax(-1).cpu().numpy()
    ip = out["pitch_logits"].float().argmax(-1).cpu().numpy()

    acoes = []
    for i in range(len(pilhas)):
        hold = [BUTTONS[k] for k in USABLE if p[i][k] > 0.5]
        acoes.append({"hold": hold, "mouse": [int(yb[iy[i]]), int(pb[ip[i]])],
                      "duration_ms": 250})
    return acoes


def episodio(politica, posicoes, passos, ctx=None, objetivo="explorar"):
    """Roda um lote de posicoes. Retorna a maior distancia da origem de cada uma."""
    r = post("/lote/reset", {"posicoes": [[x, z] for x, z in posicoes]})
    n = len(posicoes)
    estados = [EstadoEpisodio() for _ in range(n)]
    for i, o in enumerate(r["obs"][:n]):
        estados[i].reiniciar(o["estado"])
        estados[i].registrar(o["estado"], base64.b64decode(o["frame_b64"]))
    org = [(o["estado"]["x"], o["estado"]["z"]) for o in r["obs"][:n]]
    melhor = [0.0] * n

    for t in range(passos):
        if politica == "modelo":
            pilhas = [estados[i].pilha_frames() for i in range(n)]
            if any(len(p) < N_FRAMES for p in pilhas):
                acoes = [{"hold": ["W"], "mouse": [0, 0], "duration_ms": 250}] * n
            else:
                acoes = acao_do_modelo(ctx["vla"], ctx["device"], ctx["tok"],
                                       ctx["ids"], pilhas,
                                       [estados[i].vetor() for i in range(n)], objetivo)
        elif politica == "piloto":
            extra = {"blocos": ["log", "log2"]} if objetivo == "bloco" else {}
            acoes = post("/lote/piloto", {"objetivo": objetivo, "extra": extra})["acoes"][:n]
        else:
            acoes = [{"hold": ["W"], "mouse": [0, 0], "duration_ms": 250}] * n

        resp = post("/lote/passo", {"acoes": acoes})
        for i, o in enumerate(resp["obs"][:n]):
            estados[i].passo(o["estado"], base64.b64decode(o["frame_b64"]))
            melhor[i] = max(melhor[i], math.hypot(o["estado"]["x"] - org[i][0],
                                                  o["estado"]["z"] - org[i][1]))
    return melhor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posicoes", type=int, default=180)
    ap.add_argument("--passos", type=int, default=70)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--objetivo", default="explorar", choices=["explorar", "bloco"])
    args = ap.parse_args()

    travar_gpu()
    vla, device = load_vla_agent(args.ckpt)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    limitar_vram(0.62)
    vla.eval()
    ctx = {"vla": vla, "device": device, "tok": get_tokenizer(), "ids": {}}

    info = json.loads(urllib.request.urlopen(BASE + "/lote/info", timeout=30).read())
    N = info["envs"]
    print(f"[sim] {N} ambientes | objetivo: {args.objetivo} | {memoria_gpu()}", flush=True)

    # sorteia as posicoes uma vez; todas as politicas veem as mesmas
    pos = []
    while len(pos) < args.posicoes:
        pos += [(o["estado"]["x"], o["estado"]["z"]) for o in post("/lote/reset", {})["obs"]]
    pos = pos[:args.posicoes]
    print(f"[aval] {len(pos)} posicoes pareadas", flush=True)

    resultados = {}
    for nome in ("soW", "modelo", "piloto"):
        t0 = time.time()
        vals = []
        for b0 in range(0, len(pos), N):
            vals += episodio(nome, pos[b0:b0 + N], args.passos, ctx, args.objetivo)
        resultados[nome] = vals
        print(f"  {nome:8s} media {st.mean(vals):5.1f} | mediana {st.median(vals):5.1f} "
              f"| {time.time()-t0:.0f}s", flush=True)

    print()
    print("=" * 62)
    base = resultados["soW"]
    for nome in ("modelo", "piloto"):
        dif = [a - b for a, b in zip(resultados[nome], base)]
        m = st.mean(dif); ic = 1.96 * st.stdev(dif) / math.sqrt(len(dif))
        veredito = "MELHOR" if m - ic > 0 else ("PIOR" if m + ic < 0 else "empate")
        print(f"  {nome:8s} vs soW: {m:+6.1f}  IC95% [{m-ic:+6.1f},{m+ic:+6.1f}]  {veredito}")

    dif = [a - b for a, b in zip(resultados["modelo"], resultados["piloto"])]
    m = st.mean(dif); ic = 1.96 * st.stdev(dif) / math.sqrt(len(dif))
    frac = st.mean(resultados["modelo"]) / max(1e-6, st.mean(resultados["piloto"]))
    print(f"  modelo vs professor: {m:+6.1f}  IC95% [{m-ic:+6.1f},{m+ic:+6.1f}]"
          f"  ({100*frac:.0f}% do professor)")
    print("=" * 62, flush=True)


if __name__ == "__main__":
    main()
