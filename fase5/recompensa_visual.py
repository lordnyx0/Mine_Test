# coding=utf-8
"""
fase5/recompensa_visual.py — Sistema de Recompensa Não-Privilegiada Baseada em Percepção Visual.

Elimina o oráculo geométrico invisível:
  1. A recompensa de alinhamento é estritamente vinculada à presença e centralização
     do pilar-alvo no frame RGB da câmera do robô.
  2. Recompensa de "Busca Ativa" é concedida por Descoberta Visual (Delta Visão: 0 -> 1),
     e não por mera ausência de locomoção.
  3. Penalidade de "Corrida Cega" se o agente corre para frente sem que o alvo esteja visível.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Definição de máscaras RGB para os blocos de pilares do Minecraft
# Formato do frame: uint8 [H, W, 3] ou [3, H, W]
def _obter_mascara_cor(frame: np.ndarray, cor: str) -> np.ndarray:
    if frame is None:
        return np.zeros((1, 1), dtype=bool)
    
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
        # Bloco de ouro / amarelo vibrante
        mask = (r > 130) & (g > 130) & (b < 95) & (np.abs(r - g) < 55)
    elif cor == "azul":
        # Lapis lazuli / azul vivo
        mask = (b > 100) & (r < 80) & (g < 130) & (b > r + 30)
    elif cor == "roxo":
        # Obsidiana / roxo escuro
        mask = (r < 75) & (g < 55) & (b < 95) & ((b > g) | ((r < 45) & (g < 45) & (b < 45)))
    elif cor == "verde":
        mask = (g > 110) & (r < 85) & (b < 85)
    elif cor == "vermelho":
        mask = (r > 130) & (g < 75) & (b < 75)
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

    # Calcula centro de massa horizontal
    col_indices = np.where(mask)[1]
    media_col = float(np.mean(col_indices))
    largura = mask.shape[1]
    centro_x = (media_col / (largura - 1)) * 2.0 - 1.0  # Mapeia [0, W-1] para [-1.0, 1.0]

    return {
        "visivel": True,
        "contagem_pixels": contagem,
        "fracao_area": float(contagem / max(1, total_pixels)),
        "centro_x": float(np.clip(centro_x, -1.0, 1.0)),
        "centralizado": bool(abs(centro_x) < 0.30)
    }


class RastreadorVisualEpisodio:
    """Mantém o estado perceptivo do agente por episódio para cálculo de recompensas dinâmicas."""
    def __init__(self, num_ambientes: int):
        self.num_ambientes = num_ambientes
        self.alvo_avistado = [False] * num_ambientes
        self.passos_sem_foco = [0] * num_ambientes
        self.cooldown_frenagem = [0] * num_ambientes
        self.area_anterior = [0.0] * num_ambientes

    def reset_ambiente(self, env_id: int):
        self.alvo_avistado[env_id] = False
        self.passos_sem_foco[env_id] = 0
        self.cooldown_frenagem[env_id] = 0
        self.area_anterior[env_id] = 0.0

    def reset_todos(self):
        for i in range(self.num_ambientes):
            self.reset_ambiente(i)

    def calcular_recompensa_passo(
        self,
        env_id: int,
        estado: Dict[str, Any],
        frame_u8: Optional[np.ndarray],
        cor_alvo: str,
        acao_exec: Dict[str, Any],
        estagio_atual: int,
        shaping_geometrico: bool = True,
        lambda_potencial: float = 0.10,
        dist_atual: Optional[float] = None,
        dist_anterior: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcula a recompensa modular do passo para o ambiente env_id:
          R_total = R_visual + lambda * [Phi(s') - Phi(s)] + R_terminal
          com Phi(s) = -distancia ate o objetivo atual (ou seja, Phi(s') - Phi(s) = dist_anterior - dist_atual)

        O potencial geometrico e usado ESTRITAMENTE como shaping de recompensa escalar para treino GAE,
        sem nunca expor coordenadas, angulos ou direcoes do alvo as observacoes/entradas do modelo.

        Retorna:
          rec_total (float): recompensa total do passo
          info (dict): estatisticas e decomposicao (rec_visual, rec_potencial, rec_terminal, visivel, etc.)
        """
        rec_visual = 0.0
        rec_potencial = 0.0
        rec_terminal = 0.0
        info: Dict[str, Any] = {}

        # 1. Penalidade de Perigo Fatal (Agua / Lava)
        if estado.get("in_water") or estado.get("in_lava"):
            rec_terminal = -3.0
            info.update({
                "rec_visual": 0.0,
                "rec_potencial": 0.0,
                "rec_terminal": rec_terminal,
                "rec_total": rec_terminal,
                "motivo": "morte_liquido",
                "visivel": False,
                "centro_x": 0.0,
                "fracao_area": 0.0,
                "descoberta": False
            })
            return rec_terminal, info

        # 2. Custo basico de tempo (evita loop estatico)
        rec_visual -= 0.04

        # 3. Analise Visual da Camera
        det = detectar_alvo_no_frame(frame_u8, cor_alvo)
        visivel = det["visivel"]
        centro_x = det["centro_x"]
        centralizado = det["centralizado"]
        fracao_area_atual = det["fracao_area"]
        info["visivel"] = visivel
        info["centro_x"] = centro_x
        info["fracao_area"] = fracao_area_atual
        info["descoberta"] = False

        corre_frente = "W" in acao_exec.get("hold", [])
        gira = bool(acao_exec.get("mouse", [0, 0])[0] != 0)

        # 4. Recompensa de Busca e Descoberta Visual (Delta Visao)
        if visivel:
            if not self.alvo_avistado[env_id]:
                # Primeiro avistamento do pilar no estagio -> Bonus de Descoberta
                rec_visual += 0.50
                self.alvo_avistado[env_id] = True
                info["descoberta"] = True

            self.passos_sem_foco[env_id] = 0

            # Bonus por Mira Centralizada no Pilar Visivel
            alinhamento_visual = max(0.0, 1.0 - abs(centro_x))
            rec_visual += 0.15 * alinhamento_visual

            # Bonus adicional se estiver centralizado com precisao
            if centralizado:
                rec_visual += 0.08

            # 5. Progresso Visuomotor por Looming (Expansao de Area Aparente do Pilar)
            if self.area_anterior[env_id] > 0.0:
                delta_area = fracao_area_atual - self.area_anterior[env_id]
                rec_visual += float(np.clip(delta_area * 80.0, -0.20, 0.40))

            self.area_anterior[env_id] = fracao_area_atual
        else:
            self.area_anterior[env_id] = 0.0
            if self.alvo_avistado[env_id]:
                self.passos_sem_foco[env_id] += 1
                # Penalidade progressiva se perdeu de vista o alvo previamente avistado
                if self.passos_sem_foco[env_id] > 5:
                    rec_visual -= 0.04 * min(self.passos_sem_foco[env_id] - 5, 5)

            # Penalidade de Corrida Cega (sprint para frente sem pilar visivel)
            if corre_frente and not self.alvo_avistado[env_id]:
                rec_visual -= 0.25

        # 6. Recompensa de Frenagem e Amortecimento Pos-Submeta 1
        if self.cooldown_frenagem[env_id] > 0:
            self.cooldown_frenagem[env_id] -= 1
            if not corre_frente:
                rec_visual += 0.20  # Recompensa soltar W apos cruzar o Pilar 1 para reorientar a camera
            if gira and visivel:
                rec_visual += 0.25  # Recompensa girar ate encontrar o Pilar 2

        # 7. Shaping de Potencial Geometrico: lambda * [Phi(s') - Phi(s)] = lambda * [d_ant - d_atual]
        if shaping_geometrico and dist_atual is not None and dist_anterior is not None:
            delta_d = dist_anterior - dist_atual
            # Limita a variacao fisica por passo (maximo ~0.35m por tick em sprint) para evitar picos espurios
            delta_d_clamped = float(np.clip(delta_d, -1.0, 1.0))
            rec_potencial = float(lambda_potencial * delta_d_clamped)

        rec_total = rec_visual + rec_potencial + rec_terminal

        info.update({
            "rec_visual": float(rec_visual),
            "rec_potencial": float(rec_potencial),
            "rec_terminal": float(rec_terminal),
            "rec_total": float(rec_total)
        })

        return rec_total, info

