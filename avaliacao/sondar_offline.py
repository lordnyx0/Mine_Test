# coding=utf-8
"""
A via visual ainda esta colapsada? — versao OFFLINE.

O sondar_representacao.py original precisa do servidor vivo em 3001, que por
sua vez precisa do Minecraft aberto. Este roda sobre o proprio dataset, entao
mede exatamente a distribuicao em que o modelo foi treinado, sem depender de
nada externo e de forma reproduzivel.

Quatro medidas sobre o checkpoint atual:

  1. COLAPSO       — cos medio e posto efetivo de `visual` (saida do projetor)
                     e de `hidden` (entrada das action heads). Posto ~1 = a
                     visao virou constante e foi efetivamente descartada.

  2. ROTA HELD-OUT — a MSE que o route_head atinge em amostras que NAO entraram
                     no ajuste da sonda, contra o preditor cego (media por
                     setor). O 0.0548 do log de treino e numero de treino.

  3. SONDA LINEAR  — regressao ridge de `hidden` e `visual` para as 12 rotas,
                     com holdout. Responde se a geometria esta LINEARMENTE
                     legivel na representacao, independente do que a politica
                     faz com ela.

  4. SENSIBILIDADE — quanto as teclas mudam trocando so a imagem, contra
                     trocando so o state_vec.

    python sondar_offline.py --amostras 400
"""
import os
import io
import sys
import json
import base64
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.gpu_utils import limitar_recursos, limitar_vram, travar_gpu, compactar_backbone
limitar_recursos()

import numpy as np
import torch
from PIL import Image

from infra.run_vla_agent import load_vla_agent
from infra.train_vla import get_tokenizer
from avaliacao.avaliar_no_sim import INSTRUCOES

DATASET = "dataset/sim_piloto.jsonl"
K_ROTA = 12


# ── captura dos estados internos ──────────────────────────────────────────────
class Captador:
    """Le os tensores internos por hook, sem tocar no modelo.

    O captador do script original fazia mean(dim=(0,1)) na saida do projetor,
    o que achata o BATCH junto com os tokens. Funcionava porque ele rodava com
    B=1. Aqui o batch e maior, entao a media e so sobre frames e tokens.
    """

    def __init__(self, vla, n_frames):
        self.n_frames = n_frames
        self.hidden = None
        self.visual = None
        vla.action_heads.register_forward_pre_hook(self._h_heads)
        vla.projector.register_forward_hook(self._h_proj)

    def _h_heads(self, mod, args):
        self.hidden = args[0].detach().float().cpu().numpy()          # [B, H]

    def ligar_backbone(self, vla):
        """Guarda TODAS as posicoes de saida do backbone, nao so a ultima.

        Sem isto nao da para distinguir duas causas para o `hidden` carregar
        menos rota que o `visual`: (a) o backbone congelado destroi a
        informacao, ou (b) ler uma posicao so e ruidoso, enquanto o `visual` e
        media de 96 tokens. A media sobre as posicoes separa as duas.
        """
        vla.qwen_model.register_forward_hook(self._h_backbone)

    def _h_backbone(self, mod, args, out):
        h = out.last_hidden_state.detach().float()                    # [B, T, H]
        self.bb_media = h.mean(dim=1).cpu().numpy()
        self.bb_visual = h[:, :96, :].mean(dim=1).cpu().numpy()       # so as posicoes de imagem

    def _h_proj(self, mod, args, out):
        # [B*K, 32, H] -> [B, H], media sobre os K frames e os 32 tokens
        o = out.detach().float()
        BK, T, H = o.shape
        B = BK // self.n_frames
        self.visual = o.view(B, self.n_frames, T, H).mean(dim=(1, 2)).cpu().numpy()


# ── metricas de colapso ───────────────────────────────────────────────────────
def cos_medio(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    return float(S[iu].mean()), float(S[iu].std())


def posto_efetivo(X):
    """Entropia exponencial do espectro: quantas direcoes carregam variancia."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def ridge_probe(X, Y, alpha=1.0, frac_treino=0.7, seed=0):
    """Regressao linear regularizada X -> Y com holdout. Retorna MSE de teste."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]
    c = int(frac_treino * len(X))
    Xtr, Xte, Ytr, Yte = X[:c], X[c:], Y[:c], Y[c:]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    Xte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])

    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return float(((Xte @ W - Yte) ** 2).mean()), Yte, idx[c:]


# ── dados ─────────────────────────────────────────────────────────────────────
def carregar_amostras(caminho, n, seed=0):
    """Amostra espacada ao longo do arquivo: amostras vizinhas sao do mesmo
    episodio e quase identicas, o que inflaria artificialmente o cos medio."""
    linhas = []
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l:
                linhas.append(l)
    passo = max(1, len(linhas) // n)
    escolhidas = linhas[::passo][:n]
    out = []
    for l in escolhidas:
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("rotas") and len(d["rotas"]) == K_ROTA and d.get("frames_b64"):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--amostras", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vram", type=float, default=0.62)
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    travar_gpu()

    dados = carregar_amostras(args.dataset, args.amostras)
    n_frames = len(dados[0]["frames_b64"])
    print("[dados] %d amostras de %s | %d frames por amostra"
          % (len(dados), args.dataset, n_frames), flush=True)

    vla, device = load_vla_agent(args.ckpt)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    limitar_vram(args.vram)
    vla.eval()

    proc = vla.vision_processor
    tok = get_tokenizer()
    ids1 = tok([INSTRUCOES["explorar"]], return_tensors="pt",
               truncation=True, max_length=24)["input_ids"].to(device)
    cap = Captador(vla, n_frames)
    cap.ligar_backbone(vla)

    def rodar(lote_imgs, lote_sv):
        """lote_imgs: [B][K] de PIL. Retorna (hidden, visual, rotas_previstas, botoes)."""
        planas = [im for pilha in lote_imgs for im in pilha]
        px = proc(images=planas, return_tensors="pt")["pixel_values"]
        px = px.view(len(lote_imgs), n_frames, *px.shape[1:]).to(device)
        sv = torch.tensor(lote_sv, dtype=torch.float32, device=device)
        ids = ids1.expand(len(lote_imgs), -1)
        with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = vla(pixel_values=px, state_vec=sv, input_ids=ids)
        return (cap.hidden.copy(), cap.visual.copy(),
                out["rotas"].float().cpu().numpy(),
                out["buttons"].float().cpu().numpy(),
                cap.bb_media.copy(), cap.bb_visual.copy())

    print("\n[1/4] passando o dataset pelo modelo...", flush=True)
    H, V, Rp, Rt, BM, BV = [], [], [], [], [], []
    imgs_guardadas = []
    for i in range(0, len(dados), args.batch):
        lote = dados[i:i + args.batch]
        lote_imgs = [[Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                      for b in d["frames_b64"]] for d in lote]
        lote_sv = [d["state_vec"] for d in lote]
        h, v, rp, _, bm, bv = rodar(lote_imgs, lote_sv)
        H.append(h); V.append(v); Rp.append(rp); BM.append(bm); BV.append(bv)
        Rt.extend(d["rotas"] for d in lote)
        if len(imgs_guardadas) < 8:
            for j, d in enumerate(lote):
                if len(imgs_guardadas) < 8:
                    imgs_guardadas.append((lote_imgs[j], d["state_vec"]))
        if (i + args.batch) % 100 < args.batch:
            print("    %d/%d" % (min(i + args.batch, len(dados)), len(dados)), flush=True)

    H = np.concatenate(H); V = np.concatenate(V); Rp = np.concatenate(Rp)
    BM = np.concatenate(BM); BV = np.concatenate(BV)
    Rt = np.array(Rt, dtype=np.float64)
    print("    hidden %s | visual %s | rotas %s" % (H.shape, V.shape, Rp.shape))

    print("\n[2/4] COLAPSO da representacao")
    print("    %-20s %14s %10s %s" % ("tensor", "cos medio", "posto ef.", "veredito"))
    for nome, X in (("hidden (decisao)", H), ("visual (projetor)", V)):
        m, s = cos_medio(X)
        pe = posto_efetivo(X)
        ver = ("COLAPSADA" if m > 0.99 else "pouca variacao" if m > 0.95 else "variada")
        print("    %-20s %8.4f (%.4f) %8.1f de %d  -> %s"
              % (nome, m, s, pe, X.shape[1], ver))

    print("\n[3/4] ROTA: o route_head bate o preditor cego FORA do ajuste?")
    media_setor = Rt.mean(0, keepdims=True)
    mse_cego = float(((media_setor - Rt) ** 2).mean())
    mse_head = float(((Rp - Rt) ** 2).mean())
    print("    preditor cego (media por setor)   MSE = %.4f" % mse_cego)
    print("    route_head do modelo              MSE = %.4f  (%.1f%% abaixo do cego)"
          % (mse_head, 100 * (mse_cego - mse_head) / mse_cego))

    print("\n[4/4] SONDA LINEAR: as 12 rotas sao legiveis da representacao?")
    print("    (ordem = o caminho que a informacao percorre ate a decisao)")
    print("    %-34s %12s %12s" % ("fonte", "MSE teste", "vs cego"))
    fontes = (
        ("1. projetor (antes do backbone)", V),
        ("2. backbone, posicoes de imagem", BV),
        ("3. backbone, media de tudo",      BM),
        ("4. hidden = ultima posicao  <-usado", H),
    )
    for nome, X in fontes:
        mse_p, Yte, _ = ridge_probe(X.astype(np.float64), Rt)
        cego_te = float(((Yte.mean(0, keepdims=True) - Yte) ** 2).mean())
        print("    %-34s %12.4f %11.1f%%" % (nome, mse_p, 100 * (cego_te - mse_p) / cego_te))

    print("\n[extra] SENSIBILIDADE: a saida depende da imagem ou do state_vec?")
    base_imgs, base_sv = imgs_guardadas[0]
    ref = rodar([base_imgs], [base_sv])[3][0]
    d_img = [np.abs(rodar([im], [base_sv])[3][0] - ref).mean()
             for im, _ in imgs_guardadas[1:]]
    d_sv = [np.abs(rodar([base_imgs], [sv])[3][0] - ref).mean()
            for _, sv in imgs_guardadas[1:]]
    print("    trocando so a IMAGEM     -> variacao media das teclas: %.4f" % np.mean(d_img))
    print("    trocando so o STATE_VEC  -> variacao media das teclas: %.4f" % np.mean(d_sv))
    if np.mean(d_img) < 0.01:
        print("    -> a politica esta praticamente IGNORANDO a imagem")
    print()


if __name__ == "__main__":
    main()
