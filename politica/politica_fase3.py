# coding=utf-8
"""
Politica da FASE 3: Servo-Visual Puro (Sem Coordenadas).

A entrada multimodal é alimentada pelo SigLIP + instrução em texto, com o
`goal_vec` de coordenadas ZERADO. A rede é forçada a guiar o bot
exclusivamente pelas características visuais dos frames.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from politica.politica_fase1 import PoliticaNeural


class PoliticaFase3(PoliticaNeural):
    """Política servo-visual multi-cores sem coordenadas."""

    def __init__(self, ckpt=None, prompt="Objetivo: va ate o bloco roxo.", **kw):
        super().__init__(ckpt, **kw)
        from infra.train_vla import get_tokenizer
        self.tok = get_tokenizer()
        self.default_prompt = prompt
        self.prompts_atuais = [prompt]
        self._cache_ids = {}

    def obter_ids(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        res = []
        for p in prompts:
            if p not in self._cache_ids:
                ids = self.tok([p], return_tensors="pt",
                               padding="max_length", max_length=16,
                               truncation=True)["input_ids"].to(self.device)
                self._cache_ids[p] = ids
            res.append(self._cache_ids[p])
        return torch.cat(res, dim=0)

    def _entradas(self, ests, alvos_abs, obs):
        n = len(ests)
        u8 = self.pixels_uint8()
        px = self.normalizar(u8)
        sv = torch.tensor([e.vetor() for e in self.est_ep],
                          dtype=torch.float32, device=self.device)
        # O vetor de objetivo espacial é ZERADO: a navegação depende puramente dos pixels e prompt
        gv = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        return px, sv, gv, u8

    def logits(self, px, sv, gv, ids=None):
        if ids is None:
            if len(self.prompts_atuais) == px.size(0):
                ids = self.obter_ids(self.prompts_atuais)
            else:
                ids = self.obter_ids([self.default_prompt] * px.size(0))
        out = self.vla(pixel_values=px, state_vec=sv,
                       input_ids=ids, goal_vec=gv)
        raw_yaw = out["yaw_logits"].float()
        return torch.tanh(raw_yaw / 3.0) * 3.0

    def agir(self, ests, alvos_abs, obs, prompts=None):
        if prompts is not None:
            self.prompts_atuais = list(prompts)
        px, sv, gv, u8 = self._entradas(ests, alvos_abs, obs)
        ids = self.obter_ids(self.prompts_atuais if len(self.prompts_atuais) == len(ests)
                             else [self.default_prompt] * len(ests))

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            lg_yaw = self.logits(px, sv, gv, ids=ids)

        if self.amostrar:
            p_yaw = torch.softmax(lg_yaw / self.temperatura, dim=-1)
            i_yaw = torch.multinomial(p_yaw, 1).squeeze(-1)
        else:
            i_yaw = lg_yaw.argmax(-1)

        self.ultimo = {"u8": u8, "sv": sv.detach().cpu().numpy(),
                       "gv": gv.detach().cpu().numpy(),
                       "idx": i_yaw.detach().cpu().numpy(),
                       "ids": ids.detach().cpu().numpy()}

        acoes = []
        for k in i_yaw:
            acoes.append({"hold": ["W"], "mouse": [int(self.YB[int(k)]), 0],
                          "duration_ms": 250})
        return acoes

    def log_prob(self, px, sv, gv, a_yaw, ids=None):
        """log p(giro) e entropia normalizada para treino de RL."""
        lg_yaw = self.logits(px, sv, gv, ids=ids)
        logp_y = torch.log_softmax(lg_yaw, dim=-1)
        lp_y = logp_y.gather(1, a_yaw.unsqueeze(1)).squeeze(1)
        ent_y = -(logp_y.exp() * logp_y).sum(-1).mean()
        ent_y_norm = ent_y / torch.log(torch.tensor(float(len(self.YB)), device=self.device))
        return lp_y, ent_y_norm
