# coding=utf-8
"""
fase5/recompensa_visual.py — Sistema de Recompensa Não-Privilegiada Baseada em Percepção Visual.

Elimina o oráculo geométrico invisível:
  1. A recompensa de alinhamento é estritamente vinculada à presença e centralização
     do pilar-alvo no frame RGB da câmera do robô.
  2. Recompensa de "Busca Ativa" é concedida por Descoberta Visual (Delta Visão: 0 -> 1),
     e não por mera ausência de locomoção.
  3. Penalidade de "Corrida Cega" se o agente corre para frente sem que o alvo esteja visível.
  4. Suporta shaping de potencial geométrico não-privilegiado R_total = R_visual + λ * [Φ(s') - Φ(s)] + R_terminal.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Definição de máscaras RGB para os blocos de pilares do Minecraft
# Formato do frame suportado: [H, W, 3], [3, H, W] ou pilha temporal 4D [K, H, W, 3]
def _obter_mascara_cor(frame: np.ndarray, cor: str) -> np.ndarray:
    if frame is None:
        return np.zeros((1, 1), dtype=bool)

    # Se vier pilha temporal 4D [K, H, W, 3], seleciona o frame mais recente (índice 0)
    if frame.ndim == 4:
        frame = frame[0]

    # Converte para [H, W, 3] se vier [3, H, W]
    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[2] != 3:
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim != 3 or frame.shape[2] < 3:
        return np.zeros((1, 1), dtype=bool)

    r = frame[:, :, 0].astype(np.int32)
    g = frame[:, :, 1].astype(np.int32)
    b = frame[:, :, 2].astype(np.int32)

    cor = cor.lower()
    if cor == "amarelo":
        # Bloco de ouro [245, 215, 20] / amarelo vibrante
        mask = (r > 130) & (g > 130) & (b < 95) & (np.abs(r - g) < 55)
    elif cor == "azul":
        # Lapis lazuli [25, 110, 245] / azul vivo
        mask = (b > 100) & (r < 80) & (g < 140) & (b > r + 30)
    elif cor == "roxo":
        # Obsidiana do voxel_renderer [155, 38, 182] ou escura [40, 20, 50]
        mask = ((r < 75) & (g < 55) & (b < 95) & ((b > g) | ((r < 45) & (g < 45) & (b < 45)))) | \
               ((r > 90) & (b > 110) & (g < 75) & (np.abs(r - b) < 65))
    elif cor == "verde":
        # Bloco de esmeralda [42, 203, 87]
        mask = (g > 110) & (r < 85) & (b < 95) & (g > r + 30)
    elif cor == "vermelho":
        # Bloco de redstone [175, 24, 5]
        mask = (r > 120) & (g < 75) & (b < 75) & (r > g + 45)
    else:
        # Fallback genérico por luminância
        mask = (r > 100) & (g > 100) & (b > 100)

    return mask


def detectar_alvo_no_frame(frame_u8: np.ndarray, cor_alvo: str, min_pixels: int = 12) -> Dict[str, Any]:
    """
    Detecta se o pilar da cor_alvo está visível no frame RGB e sua posição horizontal relativa.
    Retorna:
      - visivel: bool
      - contagem_pixels: int
      - fracao_area: float em [0, 1]
      - centro_x: float em [-1.0, 1.0] (-1.0 = extrema esquerda, 0.0 = centro da mira, +1.0 = extrema direita)
      - centralizado: bool (|centro_x| < 0.30)
    """
    if frame_u8 is None:
        return {"visivel": False, "contagem_pixels": 0, "fracao_area": 0.0, "centro_x": 0.0, "centralizado": False}

    mask = _obter_mascara_cor(frame_u8, cor_alvo)
    contagem = int(np.sum(mask))
    total_pixels = mask.size

    if contagem < min_pixels:
        return {
            "visivel": False,
            "contagem_pixels": contagem,
            "fracao_area": float(contagem / max(1, total_pixels)),
            "centro_x": 0.0,
            "centralizado": False
        }

    # Calcula centro de massa horizontal dos pixels detectados
    coords = np.argwhere(mask)  # [N, 2] -> (y, x)
    if len(coords) == 0:
        return {
            "visivel": False,
            "contagem_pixels": 0,
            "fracao_area": 0.0,
            "centro_x": 0.0,
            "centralizado": False
        }

    x_coords = coords[:, 1]
    w_frame = mask.shape[1]
    cx_px = float(np.mean(x_coords))

    # Normaliza para [-1.0, 1.0]
    centro_x = (cx_px / (w_frame * 0.5)) - 1.0
    centralizado = abs(centro_x) < 0.30
    fracao_area = float(contagem / total_pixels)

    return {
        "visivel": True,
        "contagem_pixels": contagem,
        "fracao_area": fracao_area,
        "centro_x": centro_x,
        "centralizado": centralizado
    }


class RastreadorVisualEpisodio:
    """Acompanha métricas de percepção visual e calcula recompensas não-privilegiadas por ambiente."""

    def __init__(self, num_ambientes: int):
        self.num_ambientes = num_ambientes
        self.alvo_avistado = [False] * num_ambientes
        self.area_anterior = [0.0] * num_ambientes
        self.passos_sem_ver = [0] * num_ambientes
        self.cooldown_frenagem = [0] * num_ambientes

    def reset_ambiente(self, env_id: int):
        self.alvo_avistado[env_id] = False
        self.area_anterior[env_id] = 0.0
        self.passos_sem_ver[env_id] = 0
        self.cooldown_frenagem[env_id] = 0

    def reset_todos(self):
        for i in range(self.num_ambientes):
            self.reset_ambiente(i)

    def calcular_recompensa_passo(
        self,
        env_id: int,
        estado: dict,
        frame_u8: Optional[np.ndarray],
        cor_alvo: str,
        acao_exec: dict,
        estagio_atual: int = 0,
        shaping_geometrico: bool = True,
        lambda_potencial: float = 0.10,
        dist_atual: float = 0.0,
        dist_anterior: float = 0.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcula a recompensa total: R_total = R_visual + λ * [Φ(s') - Φ(s)] + R_terminal.
        """
        r_visual = 0.0
        r_potencial = 0.0
        r_terminal = 0.0

        # Custo basal de tempo
        r_visual -= 0.04

        # 1. Percepção Visual Não-Privilegiada
        det = detectar_alvo_no_frame(frame_u8, cor_alvo)
        visivel = det["visivel"]
        fracao = det["fracao_area"]
        cx = det["centro_x"]
        descoberta = False

        if visivel:
            # Bônus de Descoberta Visual (Busca Ativa efetiva)
            if not self.alvo_avistado[env_id]:
                r_visual += 0.50
                self.alvo_avistado[env_id] = True
                descoberta = True

            # Bônus de Centralização da Mira
            if abs(cx) < 0.25:
                r_visual += 0.23 * (1.0 - abs(cx) / 0.25)
            elif abs(cx) < 0.55:
                r_visual += 0.08 * (1.0 - (abs(cx) - 0.25) / 0.30)
            else:
                r_visual -= 0.05 * min(1.0, (abs(cx) - 0.55) / 0.45)

            # Bônus de Expansão Aparente (Looming)
            if self.area_anterior[env_id] > 0.0:
                delta_area = fracao - self.area_anterior[env_id]
                if delta_area > 0:
                    r_visual += min(0.40, delta_area * 15.0)

            self.area_anterior[env_id] = fracao
            self.passos_sem_ver[env_id] = 0

        else:
            self.passos_sem_ver[env_id] += 1
            self.area_anterior[env_id] = 0.0

            # Penalidade de Corrida Cega
            hold_keys = acao_exec.get("hold", [])
            if "W" in hold_keys:
                r_visual -= 0.25

        # Penalidade por cair na água ou lava
        if estado.get("in_water") or estado.get("in_lava"):
            r_terminal -= 3.0

        # 2. Shaping de Potencial Geométrico
        if shaping_geometrico and dist_anterior > 0.0 and dist_atual > 0.0:
            delta_d = dist_anterior - dist_atual
            delta_d_clamped = max(-1.0, min(1.0, delta_d))
            r_potencial = lambda_potencial * delta_d_clamped

        r_total = r_visual + r_potencial + r_terminal

        info = {
            "rec_visual": r_visual,
            "rec_potencial": r_potencial,
            "rec_terminal": r_terminal,
            "rec_total": r_total,
            "visivel": visivel,
            "centro_x": cx if visivel else 0.0,
            "fracao_area": fracao if visivel else 0.0,
            "descoberta": descoberta
        }

        return r_total, info
