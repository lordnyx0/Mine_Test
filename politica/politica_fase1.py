# coding=utf-8
"""
Politica da FASE 1: entrada multimodal -> UM bin de giro. `W` e sempre apertado.

Por que so o giro: a especificacao da fase e "W + yaw". Isso tira as cabecas de
botao da equacao inteiramente — e foram elas que quebraram o treino anterior
(emitiam `A` em 93% e `SHIFT` em 62% contra alvos de 0%). Aqui nao ha como esse
modo de falha voltar.

O objetivo entra como vetor relativo EGOCENTRICO, normalizado por 8 blocos (o
alcance da fase) e nao por 30 como no treino antigo: com alvos de 3 a 8 blocos,
dividir por 30 espremeria tudo em [0.1, 0.27] e desperdicaria a faixa util da
entrada.
"""
import os
import io
import sys
import math
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

ALCANCE = 8.0          # normalizador do objetivo: o raio da Fase 1


def objetivo_rel(est, alvo_abs):
    """(frente, lado, dist, angulo) no referencial do bot, tudo em [-1, 1]."""
    yaw = math.radians(est["yaw"])
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    rx, rz = alvo_abs[0] - est["x"], alvo_abs[1] - est["z"]
    frente = rx * fx + rz * fz
    lado = rx * (-fz) + rz * fx
    d = math.hypot(rx, rz)
    return [frente / ALCANCE, lado / ALCANCE, d / ALCANCE,
            math.atan2(lado, frente) / math.pi]


class PoliticaNeural:
    """Envolve o VLA. Mantem a pilha de frames por ambiente, como no treino.

    `cego=True` zera os pixels — o controle obrigatorio que separa "a rede usou
    a imagem" de "a rede fez trigonometria com o vetor de objetivo".
    """

    def __init__(self, ckpt, cego=False, amostrar=False, temperatura=1.0,
                 device=None, vla=None):
        self.cego = cego
        self.amostrar = amostrar
        self.temperatura = temperatura
        if vla is not None:
            self.vla, self.device = vla, device
        else:
            from infra.gpu_utils import limitar_recursos, limitar_vram, travar_gpu, compactar_backbone
            limitar_recursos()
            from infra.run_vla_agent import load_vla_agent
            travar_gpu()
            self.vla, self.device = load_vla_agent(ckpt)
            compactar_backbone(self.vla)
            torch.cuda.empty_cache()
            limitar_vram(0.62)
            self.vla.eval()
        from infra.train_vla import get_tokenizer
        tok = get_tokenizer()
        self.ids1 = tok(["Objetivo: va ate o ponto indicado."], return_tensors="pt",
                        truncation=True, max_length=24)["input_ids"].to(self.device)
        self.proc = self.vla.vision_processor
        self.YB = self.vla.action_heads.YAW_BINS
        self.est_ep = None
        self.ultimo = None      # (pixels, state, goal, indice_do_bin, logprob)

    # ── ciclo de episodio ────────────────────────────────────────────────────
    def reiniciar(self, obs):
        from modelo.estado_sim import EstadoEpisodio
        self.est_ep = [EstadoEpisodio() for _ in obs]
        for i, o in enumerate(obs):
            self.est_ep[i].reiniciar(o["estado"])
            self.est_ep[i].registrar(o["estado"], base64.b64decode(o["frame_b64"]))

    def pixels_uint8(self):
        """[n, K, 224, 224, 3] uint8. Guardar assim, e nao em float, e o que faz
        um rollout de 40 passos x 8 ambientes caber na memoria: 144 MB contra
        1,4 GB. Equivalente ao AutoProcessor do SigLIP — medido, diferenca media
        de 0,00001 — porque ambos sao resize bicubico para 224 e x/127.5-1."""
        pilhas = [e.pilha_frames() for e in self.est_ep]
        n, K = len(pilhas), len(pilhas[0])
        buf = np.empty((n, K, 224, 224, 3), dtype=np.uint8)
        for i, p in enumerate(pilhas):
            for k, b in enumerate(p):
                im = Image.open(io.BytesIO(b)).convert("RGB")
                if im.size != (224, 224):
                    im = im.resize((224, 224), Image.BICUBIC)
                buf[i, k] = np.asarray(im, dtype=np.uint8)
        return buf

    def normalizar(self, u8):
        """uint8 [B,K,H,W,3] -> float [B,K,3,H,W] em [-1,1], na GPU."""
        x = torch.as_tensor(u8).to(self.device, non_blocking=True)
        x = x.permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
        return torch.zeros_like(x) if self.cego else x

    def _entradas(self, ests, alvos_abs, obs):
        n = len(ests)
        u8 = self.pixels_uint8()
        px = self.normalizar(u8)
        sv = torch.tensor([e.vetor() for e in self.est_ep],
                          dtype=torch.float32, device=self.device)
        gv = torch.tensor([objetivo_rel(ests[i], alvos_abs[i]) for i in range(n)],
                          dtype=torch.float32, device=self.device)
        return px, sv, gv, u8

    def logits(self, px, sv, gv):
        out = self.vla(pixel_values=px, state_vec=sv,
                       input_ids=self.ids1.expand(px.size(0), -1), goal_vec=gv)
        return out["yaw_logits"].float()

    def agir(self, ests, alvos_abs, obs):
        px, sv, gv, u8 = self._entradas(ests, alvos_abs, obs)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            lg = self.logits(px, sv, gv)
        if self.amostrar:
            # Amostrar (e nao argmax) e o que da exploracao ao RL. Com argmax a
            # politica determinista nunca descobre um bin que ainda nao prefere.
            probs = torch.softmax(lg / self.temperatura, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
        else:
            idx = lg.argmax(-1)
        # Guardado para o RL: o gradiente NAO passa por aqui. O rollout roda sem
        # grafo (senao 40 passos x 8 ambientes de um 0.6B estouram a VRAM) e os
        # log-probs sao recalculados depois, em minilotes.
        self.ultimo = {"u8": u8, "sv": sv.detach().cpu().numpy(),
                       "gv": gv.detach().cpu().numpy(),
                       "idx": idx.detach().cpu().numpy()}
        return [{"hold": ["W"], "mouse": [int(self.YB[int(k)]), 0],
                 "duration_ms": 250} for k in idx]

    def observar(self, obs):
        """Atualiza a pilha de frames apos o passo do ambiente."""
        for i, o in enumerate(obs):
            self.est_ep[i].passo(o["estado"], base64.b64decode(o["frame_b64"]))
