# coding=utf-8
"""
fase5/acoes_taticas.py — Mapeamento e Utilidades do Espaço de Ações Táticas Holonômico (36 Ações).

Divisão do Espaço de Ações (36 classes):
  - [00 a 08] ALINHAMENTO PURO (hold: []): 9 bins de yaw [-120, -60, -25, -5, 0, 5, 25, 60, 120]
  - [09 a 17] SPRINT RETO (hold: ['W']): 9 bins de yaw
  - [18 a 26] SPRINT COM PULO (hold: ['W', 'SPACE']): 9 bins de yaw
  - [27 a 29] STRAFE ESQUERDA (hold: ['W', 'A']): 3 micro-bins [-5, 0, +5]
  - [30 a 32] STRAFE DIREITA (hold: ['W', 'D']): 3 micro-bins [-5, 0, +5]
  - [33 a 35] RECUAR/DESENGATE (hold: ['S']): 3 micro-bins [-5, 0, +5]
"""
from __future__ import annotations
import math
import torch
from typing import Dict, Any, List, Tuple

YAW_BINS_9 = [-120, -60, -25, -5, 0, 5, 25, 60, 120]
YAW_BINS_3 = [-5, 0, 5]

def calcular_bin_yaw_9(erro_graus: float) -> int:
    """Encontra o bin mais próximo entre os 9 níveis de rotação."""
    diffs = [abs(erro_graus - b) for b in YAW_BINS_9]
    return int(diffs.index(min(diffs)))

def calcular_bin_yaw_3(erro_graus: float) -> int:
    """Encontra o bin mais próximo entre os 3 micro-níveis de rotação."""
    diffs = [abs(erro_graus - b) for b in YAW_BINS_3]
    return int(diffs.index(min(diffs)))

def decodificar_acao_36(idx: int) -> Dict[str, Any]:
    """Converte o índice 0-35 no pacote de ação física para o simulador."""
    idx = int(idx)
    # 0 a 8: Giro Parado
    if 0 <= idx <= 8:
        dx = int(YAW_BINS_9[idx])
        return {"hold": [], "mouse": [dx, 0], "duration_ms": 250, "tipo": "alinhar"}
    # 9 a 17: Sprint Frontal
    elif 9 <= idx <= 17:
        dx = int(YAW_BINS_9[idx - 9])
        return {"hold": ["W"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "sprint"}
    # 18 a 26: Sprint com Pulo
    elif 18 <= idx <= 26:
        dx = int(YAW_BINS_9[idx - 18])
        return {"hold": ["W", "SPACE"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "pulo"}
    # 27 a 29: Strafe Esquerda
    elif 27 <= idx <= 29:
        dx = int(YAW_BINS_3[idx - 27])
        return {"hold": ["W", "A"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "strafe_esq"}
    # 30 a 32: Strafe Direita
    elif 30 <= idx <= 32:
        dx = int(YAW_BINS_3[idx - 30])
        return {"hold": ["W", "D"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "strafe_dir"}
    # 33 a 35: Recuo
    elif 33 <= idx <= 35:
        dx = int(YAW_BINS_3[idx - 33])
        return {"hold": ["S"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "recuar"}
    else:
        # Fallback neutro
        return {"hold": ["W"], "mouse": [0, 0], "duration_ms": 250, "tipo": "sprint"}

def calcular_acao_otima_tatica(
    erro_yaw_graus: float,
    distancia: float,
    deve_pular: bool = False,
    is_spawn: bool = False,
    esta_colidindo: bool = False,
    offset_lateral: float = 0.0
) -> int:
    """
    Raciocínio de oráculo tático:
      1. Se está colidindo ou deu overshoot -> Recuo S (33..35)
      2. Se é spawn e erro angular > 50° -> Giro Parado [] (0..8) para não correr cego
      3. Se erro angular está entre 15° e 50° com distância curta/média -> Strafe A/D (27..32) para manter mira
      4. Se deve pular -> Sprint + Pulo (18..26)
      5. Caso contrário -> Sprint frontal W (9..17)
    """
    abs_erro = abs(erro_yaw_graus)

    # 1. Situação de desengate
    if esta_colidindo:
        bin3 = calcular_bin_yaw_3(erro_yaw_graus)
        return 33 + bin3

    # 2. Alinhamento no Spawn para alvos fora do FOV frontal
    if is_spawn and abs_erro > 45.0:
        bin9 = calcular_bin_yaw_9(erro_yaw_graus)
        return 0 + bin9

    # 3. Strafe lateral tático (fixação visual com câmera no alvo)
    if 12.0 < abs_erro <= 45.0 and distancia < 8.0:
        bin3 = calcular_bin_yaw_3(erro_yaw_graus * 0.3) # micro-giro
        if erro_yaw_graus < 0: # alvo à esquerda -> Strafe Esquerda (W+A)
            return 27 + bin3
        else: # alvo à direita -> Strafe Direita (W+D)
            return 30 + bin3

    # 4. Pulo
    if deve_pular:
        bin9 = calcular_bin_yaw_9(erro_yaw_graus)
        return 18 + bin9

    # 5. Sprint Frontal padrão
    bin9 = calcular_bin_yaw_9(erro_yaw_graus)
    return 9 + bin9
