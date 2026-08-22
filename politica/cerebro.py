# coding=utf-8
"""
scratch/cerebro_estabilizado.py — Cérebro Supervisor com Estabilização de Mira Proporcional (Laser-Straight Sprint)
"""
from __future__ import annotations
import math
import time
from typing import Optional, List, Dict, Any, Tuple
import torch


def erro_angular_est(est: dict, alvo_abs: list[float]) -> Tuple[float, float]:
    """Retorna (distancia, erro_angular_graus).

    0° = alvo perfeitamente centralizado à frente (-Z no Minecraft).
    +90° = alvo à esquerda.
    -90° = alvo à direita.
    ±180° = alvo nas costas.
    """
    yaw = math.radians(est.get("yaw", 0.0))
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    rx, rz = alvo_abs[0] - est["x"], alvo_abs[1] - est["z"]
    dist = math.hypot(rx, rz)
    frente = rx * fx + rz * fz
    lado = rx * (-fz) + rz * fx
    erro_deg = math.atan2(lado, frente) * 180.0 / math.pi
    return dist, erro_deg


class PoliticaCerebroVLA:
    """Política hierárquica completa: Cérebro supervisor (~1-2 Hz) + VLA reflexo (4 Hz)."""
    def __init__(self, vla_pol):
        self.vla = vla_pol
        self.nome = "cerebro_vla"
        self.travado = {}
        self.ultima_pos = {}
        self.em_desengate = {}
        self.em_varredura = {}
        self.chegou = {}

    @property
    def device(self):
        return getattr(self.vla, "device", "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def cego(self):
        return getattr(self.vla, "cego", False)

    @cego.setter
    def cego(self, val):
        self.vla.cego = val

    @property
    def prompts_atuais(self):
        return getattr(self.vla, "prompts_atuais", [])

    @prompts_atuais.setter
    def prompts_atuais(self, val):
        self.vla.prompts_atuais = val

    @property
    def ultimo(self):
        return getattr(self.vla, "ultimo", {})

    @property
    def amostrar(self):
        return getattr(self.vla, "amostrar", False)

    @amostrar.setter
    def amostrar(self, val):
        self.vla.amostrar = val

    @property
    def temperatura(self):
        return getattr(self.vla, "temperatura", 0.8)

    @temperatura.setter
    def temperatura(self, val):
        self.vla.temperatura = val

    def normalizar(self, u8):
        return self.vla.normalizar(u8)

    def log_prob(self, px, sv, gv, a_idx, ids=None):
        return self.vla.log_prob(px, sv, gv, a_idx, ids=ids)

    def forward_pensamento(self, *args, **kwargs):
        return self.vla.forward_pensamento(*args, **kwargs)

    def reiniciar(self, obs):
        self.vla.reiniciar(obs)
        n = len(obs)
        self.travado = {i: 0 for i in range(n)}
        self.ultima_pos = {i: (o["estado"]["x"], o["estado"]["z"]) for i, o in enumerate(obs)}
        self.em_desengate = {i: 0 for i in range(n)}
        self.em_varredura = {i: 0 for i in range(n)}
        self.chegou = {i: False for i in range(n)}

    def observar(self, obs):
        self.vla.observar(obs)

    def ativar_varredura(self, env_idx: int, passos_varredura: int = 3):
        self.em_varredura[env_idx] = passos_varredura

    def agir(
        self,
        ests: list[dict],
        alvos_abs: list[list[float]],
        obs: list[dict],
        prompts: Optional[list[str]] = None,
        estagios: Optional[list[int]] = None,
        mascara_modo: Optional[torch.Tensor] = None
    ) -> list[dict]:
        n = len(ests)

        # 1. Executa o VLA reflexo com as máscaras contextuais
        acoes_vla = self.vla.agir(
            ests, alvos_abs, obs,
            prompts=prompts,
            estagios=estagios,
            mascara_modo=mascara_modo
        )

        acoes_finais = []

        for i in range(n):
            e = ests[i]
            pos_atual = (e["x"], e["z"])
            pos_ant = self.ultima_pos.get(i, pos_atual)
            dist_andou = math.hypot(pos_atual[0] - pos_ant[0], pos_atual[1] - pos_ant[1])
            self.ultima_pos[i] = pos_atual

            dist_alvo = 999.0
            erro_ang = 0.0
            if alvos_abs and i < len(alvos_abs) and alvos_abs[i]:
                dist_alvo, erro_ang = erro_angular_est(e, alvos_abs[i])

            # A. Chegada terminal no raio da submeta (< 1.2m)
            if dist_alvo <= 1.2:
                self.chegou[i] = True
                acoes_finais.append({"hold": ["W"], "mouse": [0, 0], "duration_ms": 50})
                continue

            # B. Varredura Visual Pós-Submeta Ativa
            if self.em_varredura.get(i, 0) > 0:
                self.em_varredura[i] -= 1
                acoes_finais.append({"hold": [], "mouse": [58, 0], "duration_ms": 50})
                continue

            # C. Manobra Evasiva de Desengate de Colisão Autêntica
            if self.em_desengate.get(i, 0) > 0:
                self.em_desengate[i] -= 1
                acoes_finais.append({"hold": ["W", "SPACE", "D"], "mouse": [60, 0], "duration_ms": 50})
                continue

            # D. Detector Calibrado de Colisão Real (1.5cm em 50ms)
            tem_intencao_avanco = any(k in acoes_vla[i].get("hold", []) for k in ["W", "S", "A", "D"])
            if dist_andou < 0.015 and tem_intencao_avanco and dist_alvo > 1.5:
                self.travado[i] = self.travado.get(i, 0) + 1
            else:
                self.travado[i] = 0

            if self.travado[i] >= 6:
                self.em_desengate[i] = 3
                self.travado[i] = 0
                acoes_finais.append({"hold": ["SPACE", "A"], "mouse": [-60, 0], "duration_ms": 50})
                continue

            # E. Supervisão Tática do Cérebro: Sprint Direto Estabilizado (Sem Espirais)
            acao = dict(acoes_vla[i])
            holds = list(acao.get("hold", []))

            # Diretiva 1: Alvo no cone frontal (|erro_ang| <= 30°) -> Sprint direto com mira estabilizada!
            if abs(erro_ang) <= 30.0 and dist_alvo > 1.2:
                # Remove recuo espúrio para garantir avanço contínuo
                if "S" in holds:
                    holds.remove("S")
                if "W" not in holds:
                    holds.append("W")
                acao["hold"] = list(set(holds))
                ajuste_mouse_x = int(max(-12, min(12, -erro_ang * 1.5)))
                acao["mouse"] = [ajuste_mouse_x, 0]

            # Diretiva 2: Alvo fora do campo de visão (|erro_ang| > 30°) -> Reorientação angular ativa no eixo em direção ao alvo!
            elif abs(erro_ang) > 30.0 and dist_alvo > 1.2:
                if "W" in holds:
                    holds.remove("W")
                acao["hold"] = list(set(holds))
                dir_giro = -1.0 if erro_ang > 0 else 1.0
                giro_x = int(dir_giro * min(45, max(18, abs(erro_ang) * 0.8)))
                acao["mouse"] = [giro_x, 0]

            # Diretiva 3: Assistência de Degrau
            if e.get("is_collided_horizontally") and "W" in acao.get("hold", []):
                acao["hold"] = list(set(acao.get("hold", []) + ["SPACE"]))

            acao["duration_ms"] = 50
            acoes_finais.append(acao)

        return acoes_finais
