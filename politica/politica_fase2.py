# coding=utf-8
"""
Politica da FASE 2: giro + PULO.

A Fase 1 emitia so o bin de giro, com `W` fixo. Aqui a acao e conjunta:

    a = (bin de giro, pular ou nao)

O pulo vem da cabeca de botoes, indice 4 (`jump` em `button_keys`, `SPACE` em
BUTTONS). E uma Bernoulli, entao a log-probabilidade conjunta e a soma:

    log p(a) = log p(giro) + log p(pulo)

Por que so essas duas: `SPACE` e o que torna degrau de 1 bloco transponivel — a
fisica usa `stepHeight 0.6`, entao sem pulo qualquer degrau e parede. `A`/`D`/`S`
e `SHIFT` ficam para depois, quando houver problema de controle lateral e a
latencia de 4 Hz for revista.
"""
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from politica.politica_fase1 import PoliticaNeural

IDX_PULO = 4          # 'jump' em IndependentActionHeads.button_keys
ALCANCE_F2 = float(os.environ.get("DIST_MAX", "30.0"))   # raio da Fase 2


def objetivo_rel_f2(est, alvo_abs, alcance=ALCANCE_F2):
    """(frente, lado, dist, angulo) no referencial do bot, em [-1, 1] escalado para Fase 2."""
    yaw = math.radians(est["yaw"])
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    rx, rz = alvo_abs[0] - est["x"], alvo_abs[1] - est["z"]
    frente = rx * fx + rz * fz
    lado = rx * (-fz) + rz * fx
    d = math.hypot(rx, rz)
    return [frente / alcance, lado / alcance, min(1.0, d / alcance),
            math.atan2(lado, frente) / math.pi]


class PoliticaFase2(PoliticaNeural):
    """Herda toda a montagem de entrada da Fase 1, com objetivo escalado para Fase 2 (ALCANCE=30)."""

    def _entradas(self, ests, alvos_abs, obs):
        n = len(ests)
        u8 = self.pixels_uint8()
        px = self.normalizar(u8)
        sv = torch.tensor([e.vetor() for e in self.est_ep],
                          dtype=torch.float32, device=self.device)
        # Normalizado pelo alcance da Fase 2 (14-30 blocos) em vez de 8.0,
        # mantendo a entrada do GoalEncoder no domínio [-1, 1] correto.
        gv = torch.tensor([objetivo_rel_f2(ests[i], alvos_abs[i], ALCANCE_F2) for i in range(n)],
                          dtype=torch.float32, device=self.device)
        return px, sv, gv, u8

    def saidas(self, px, sv, gv):
        out = self.vla(pixel_values=px, state_vec=sv,
                       input_ids=self.ids1.expand(px.size(0), -1), goal_vec=gv)
        # Logit bounding com tanh suave: impede que qualquer logit passe de 3.0 / 2.0,
        # garantindo que a entropia nunca colapse e que a probabilidade não sature em 100%.
        raw_yaw = out["yaw_logits"].float()
        raw_pulo = out["buttons_logits"].float()[:, IDX_PULO]
        lg_yaw = torch.tanh(raw_yaw / 3.0) * 3.0
        lg_pulo = torch.tanh(raw_pulo / 2.0) * 2.0
        return lg_yaw, lg_pulo

    def agir(self, ests, alvos_abs, obs):
        px, sv, gv, u8 = self._entradas(ests, alvos_abs, obs)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            lg_yaw, lg_pulo = self.saidas(px, sv, gv)

        if self.amostrar:
            p_yaw = torch.softmax(lg_yaw / self.temperatura, dim=-1)
            i_yaw = torch.multinomial(p_yaw, 1).squeeze(-1)
            i_pulo = torch.bernoulli(torch.sigmoid(lg_pulo)).long()
        else:
            i_yaw = lg_yaw.argmax(-1)
            i_pulo = (lg_pulo > 0).long()

        self.ultimo = {"u8": u8, "sv": sv.detach().cpu().numpy(),
                       "gv": gv.detach().cpu().numpy(),
                       "idx": i_yaw.detach().cpu().numpy(),
                       "pulo": i_pulo.detach().cpu().numpy()}

        acoes = []
        for k, j in zip(i_yaw, i_pulo):
            hold = ["W", "SPACE"] if int(j) else ["W"]
            acoes.append({"hold": hold, "mouse": [int(self.YB[int(k)]), 0],
                          "duration_ms": 250})
        return acoes

    def log_prob(self, px, sv, gv, a_yaw, a_pulo):
        """log p(giro) + log p(pulo), e a entropia normalizada de cada um."""
        lg_yaw, lg_pulo = self.saidas(px, sv, gv)
        logp_y = torch.log_softmax(lg_yaw, dim=-1)
        lp_y = logp_y.gather(1, a_yaw.unsqueeze(1)).squeeze(1)
        ent_y = -(logp_y.exp() * logp_y).sum(-1).mean()

        # Bernoulli: log p = -softplus(-x) se pulou, -softplus(x) se nao.
        lp_j = -torch.nn.functional.softplus(
            torch.where(a_pulo.bool(), -lg_pulo, lg_pulo))
        p = torch.sigmoid(lg_pulo).clamp(1e-6, 1 - 1e-6)
        ent_j = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

        # Entropia NORMALIZADA pelo maximo de cada dimensao
        ent_y_norm = ent_y / math.log(len(self.YB))
        ent_j_norm = ent_j / math.log(2.0)
        ent = 0.5 * (ent_y_norm + ent_j_norm)
        return lp_y + lp_j, ent, ent_y_norm.detach(), ent_j_norm.detach()
