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
    is_transicao: bool = False,
    is_alinhar: bool = False,
    esta_colidindo: bool = False,
    offset_lateral: float = 0.0
) -> int:
    """
    Raciocínio de oráculo tático balanceado e coerente (36 classes):
      1. Se está colidindo a curta distância (<1.0m) -> Recuo S (33..35)
      2. Se grande desvio angular (>45° ou spawn/transição ampla) -> Giro Parado [] (0..8)
      3. Se erro angular moderado (10° a 45°) com alvo visível -> Strafe W+A / W+D (27..32)
      4. Se deve pular relevo -> Sprint + Pulo (18..26)
      5. Caso contrário (aproximação normal ou final <2.5m com alvo na mira) -> Sprint Frontal W (9..17)
    """
    abs_erro = abs(erro_yaw_graus)

    # 1. Situação de colisão genuína contra parede a curta distância
    if esta_colidindo and distancia < 1.0:
        bin3 = calcular_bin_yaw_3(erro_yaw_graus + offset_lateral)
        return 33 + bin3

    # 2. Alinhamento Estacionário (Spawn, Transição ou Grande Desvio Angular)
    if (is_spawn or is_transicao or is_alinhar) and abs_erro > 35.0:
        bin9 = calcular_bin_yaw_9(erro_yaw_graus)
        return 0 + bin9

    if abs_erro > 50.0:
        bin9 = calcular_bin_yaw_9(erro_yaw_graus)
        return 0 + bin9

    # 3. Strafe Lateral Tático (Manutenção de Fixação Visual + Deslocamento com W)
    if 10.0 < abs_erro <= 45.0 and distancia < 9.0:
        compensacao = (erro_yaw_graus - (-25.0 if erro_yaw_graus < 0 else 25.0)) + offset_lateral
        bin3 = calcular_bin_yaw_3(compensacao)
        if erro_yaw_graus < 0:  # Alvo à esquerda -> Strafe Esquerda (W+A) (27, 28, 29)
            return 27 + bin3
        else:  # Alvo à direita -> Strafe Direita (W+D) (30, 31, 32)
            return 30 + bin3

    # 4. Sprint + Pulo para transposição
    if deve_pular:
        bin9 = calcular_bin_yaw_9(erro_yaw_graus)
        return 18 + bin9

    # 5. Sprint Frontal padrão (9..17) — cobre toda a aproximação e chegada final (< 2.5m)
    bin9 = calcular_bin_yaw_9(erro_yaw_graus)
    return 9 + bin9


MODOS = ["alinhar", "sprint", "pulo", "strafe_esq", "strafe_dir", "recuar"]
NUM_MODOS = len(MODOS)  # 6
NUM_YAW = len(YAW_BINS_9)  # 9
NUM_ACOES_FATORADAS = NUM_MODOS * NUM_YAW  # 54

# Mapeamento do bin 3 [-5, 0, 5] para os índices dos 9 bins [-120, -60, -25, -5, 0, 5, 25, 60, 120] -> Índices 3, 4, 5
_MAPA_3_PARA_9 = [3, 4, 5]


def fatorar_indice_54(idx_54: int) -> Tuple[int, int]:
    """Converte o índice canônico 54 em tupla (modo: 0..5, yaw: 0..8)."""
    idx_54 = max(0, min(NUM_ACOES_FATORADAS - 1, int(idx_54)))
    modo = idx_54 // NUM_YAW
    yaw = idx_54 % NUM_YAW
    return modo, yaw


def unificar_indice_54(modo: int, yaw: int) -> int:
    """Converte (modo: 0..5, yaw: 0..8) no índice canônico de 54 ações (0..53)."""
    modo = max(0, min(NUM_MODOS - 1, int(modo)))
    yaw = max(0, min(NUM_YAW - 1, int(yaw)))
    return modo * NUM_YAW + yaw


def converter_36_para_54(idx_36: int) -> int:
    """Converte o índice legado de 36 classes para o índice canônico de 54 classes."""
    modo, yaw = fatorar_indice_36(idx_36)
    return unificar_indice_54(modo, yaw)


def converter_54_para_36(idx_54: int) -> int:
    """Converte o índice de 54 classes de volta para o espaço aproximado de 36 classes."""
    modo, yaw = fatorar_indice_54(idx_54)
    return unificar_indices(modo, yaw)


def fatorar_indice_36(idx: int) -> Tuple[int, int]:
    """Converte o índice 36 legado em tupla (modo: 0..5, yaw: 0..8)."""
    idx = int(idx)
    if 0 <= idx <= 8:
        return 0, idx
    elif 9 <= idx <= 17:
        return 1, idx - 9
    elif 18 <= idx <= 26:
        return 2, idx - 18
    elif 27 <= idx <= 29:
        return 3, _MAPA_3_PARA_9[idx - 27]
    elif 30 <= idx <= 32:
        return 4, _MAPA_3_PARA_9[idx - 30]
    elif 33 <= idx <= 35:
        return 5, _MAPA_3_PARA_9[idx - 33]
    return 1, 4  # Fallback: sprint reto (yaw 0)


def unificar_indices(modo: int, yaw: int) -> int:
    """Converte (modo: 0..5, yaw: 0..8) de volta ao índice unificado de 36 classes (legado)."""
    modo = max(0, min(5, int(modo)))
    yaw = max(0, min(8, int(yaw)))
    if modo == 0:
        return yaw
    elif modo == 1:
        return 9 + yaw
    elif modo == 2:
        return 18 + yaw
    else:
        # Para modos com 3 micro-bins legados, mapeia yaw (0..8) para o micro-bin mais próximo (0..2)
        if yaw <= 3:
            micro = 0  # -5
        elif yaw >= 5:
            micro = 2  # +5
        else:
            micro = 1  # 0
        if modo == 3:
            return 27 + micro
        elif modo == 4:
            return 30 + micro
        else:
            return 33 + micro


def decodificar_acao_fatorada(modo: int, yaw_idx: int) -> Dict[str, Any]:
    """Converte a decisão fatorada canônica (54 combinações) diretamente no comando físico do simulador."""
    modo = max(0, min(5, int(modo)))
    yaw_idx = max(0, min(8, int(yaw_idx)))
    dx = int(YAW_BINS_9[yaw_idx])

    if modo == 0:  # Alinhar / Giro Parado
        return {"hold": [], "mouse": [dx, 0], "duration_ms": 250, "tipo": "alinhar"}
    elif modo == 1:  # Sprint
        return {"hold": ["W"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "sprint"}
    elif modo == 2:  # Pulo
        return {"hold": ["W", "SPACE"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "pulo"}
    elif modo == 3:  # Strafe Esquerda (com autoridade total de 9 yaws)
        return {"hold": ["W", "A"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "strafe_esq"}
    elif modo == 4:  # Strafe Direita (com autoridade total de 9 yaws)
        return {"hold": ["W", "D"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "strafe_dir"}
    elif modo == 5:  # Recuar (com autoridade total de 9 yaws)
        return {"hold": ["S"], "mouse": [dx, 0], "duration_ms": 250, "tipo": "recuar"}
    return {"hold": ["W"], "mouse": [0, 0], "duration_ms": 250, "tipo": "sprint"}
