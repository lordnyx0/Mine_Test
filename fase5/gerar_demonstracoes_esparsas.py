# coding=utf-8
"""
FASE 5 — Gerador de Demonstrações Esparsas de Bifurcações de Decisão (Cold-Start).

Implementa a metodologia "Sparse Policy Selection":
  - Em vez de gravar trajetórias longas e repetitivas (baixa entropia),
    foca 100% nas BIFURCAÇÕES CRÍTICAS DE DECISÃO (alta entropia):
      1. Fork 1 (Spawn): Orientação inicial em direção à Submeta 1.
      2. Fork 2 (Transição de Submeta): No exato momento em que sv[16] transiciona
         para 1.0 (Etapa 2), orientar a mira imediatamente para a Submeta 2.
      3. Fork 3 (Desvio de Barreira): No ponto de decisão de abertura, alinhar
         o vetor de aproximação com a passagem.

Gera um dataset sintético compacto (200-500 pares) para aquecimento de SFT (Cold-Start),
garantindo suporte não-nulo para o RL selecionar as políticas vencedoras.
"""
from __future__ import annotations

import math
import os
import random
import torch
from typing import List, Dict, Any, Tuple

# Constantes de Ação (18 classes: 9 bins de Yaw x 2 Jump)
YAW_BINS = [-60.0, -30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0, 60.0]
GRAUS_POR_UNIDADE = 0.15


def calcular_bin_yaw(erro_graus: float) -> int:
    """Mapeia o erro angular para o bin de yaw mais próximo (0 a 8)."""
    # mouse_x desejado = -erro_graus / 0.15
    desejado = -erro_graus / GRAUS_POR_UNIDADE
    melhor_idx = 0
    menor_dist = float("inf")
    for idx, val in enumerate(YAW_BINS):
        dist = abs(val - desejado)
        if dist < menor_dist:
            menor_dist = dist
            melhor_idx = idx
    return melhor_idx


def calcular_acao_18(bin_yaw: int, deve_pular: bool = False) -> int:
    """Combina bin de yaw (0-8) com estado de pulo (0-1) -> índice 0-17."""
    offset_pulo = 9 if deve_pular else 0
    return bin_yaw + offset_pulo


def gerar_vetor_estado(
    dx: float,
    dz: float,
    dist: float,
    erro_yaw: float,
    estagio_idx: int = 0,
    modo_varredura: bool = False,
    preso: bool = False
) -> torch.Tensor:
    """
    Constrói o vetor de estado sv de 18 dimensões:
      [0..2]: dx, dz, dist normalizados
      [3..4]: cos, sin do erro angular
      [5..15]: telemetria e flags
      [16]: Indicador de Estágio (0.0 = Etapa 1, 1.0 = Etapa 2)
      [17]: Indicador de Modo de Varredura Ativa (0.0 ou 1.0)
    """
    sv = torch.zeros(18, dtype=torch.float32)
    sv[0] = dx / 15.0
    sv[1] = dz / 15.0
    sv[2] = min(dist / 15.0, 1.0)
    rad = math.radians(erro_yaw)
    sv[3] = math.cos(rad)
    sv[4] = math.sin(rad)
    sv[5] = 1.0 if preso else 0.0
    sv[16] = float(estagio_idx)
    sv[17] = 1.0 if modo_varredura else 0.0
    return sv


def gerar_dataset_bifurcacoes(
    num_amostras: int = 500,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Gera pares sintéticos de alta fidelidade focados estritamente nas bifurcações:
      - 45% Amostras de Transição de Estágio (Etapa 1 -> Etapa 2)
      - 40% Amostras de Spawn / Orientação Inicial
      - 15% Amostras de Correção / Alinhamento
    """
    rng = random.Random(seed)
    amostras = []

    cores = ["roxo", "amarelo", "azul"]

    for _ in range(num_amostras):
        tipo = rng.choices(["transicao", "spawn", "correcao"], weights=[0.45, 0.40, 0.15])[0]

        if tipo == "transicao":
            # Bifurcação Crítica: O robô acabou de chegar na Submeta 1 e o sv[16] virou 1.0
            # O Pilar 2 está em qualquer ângulo entre -120° e +120°
            cor1, cor2 = rng.sample(cores, 2)
            prompt = f"Objetivo: vá até o bloco {cor2} [Etapa 2/2]"
            erro_yaw = rng.uniform(-120.0, 120.0)
            dist = rng.uniform(5.0, 10.0)
            rad = math.radians(erro_yaw)
            dx = -math.sin(rad) * dist
            dz = -math.cos(rad) * dist
            
            bin_yaw = calcular_bin_yaw(erro_yaw)
            acao_idx = calcular_acao_18(bin_yaw, deve_pular=False)
            
            sv = gerar_vetor_estado(dx, dz, dist, erro_yaw, estagio_idx=1, modo_varredura=False)
            
            amostras.append({
                "tipo": "fork_transicao",
                "prompt": prompt,
                "sv": sv,
                "erro_yaw": erro_yaw,
                "acao_alvo": acao_idx,
                "bin_yaw": bin_yaw
            })

        elif tipo == "spawn":
            # Bifurcação Inicial: Spawnando e identificando o Pilar 1
            cor1 = rng.choice(cores)
            prompt = f"Objetivo: vá até o bloco {cor1} [Etapa 1/2]"
            erro_yaw = rng.uniform(-45.0, 45.0)
            dist = rng.uniform(6.0, 10.0)
            rad = math.radians(erro_yaw)
            dx = -math.sin(rad) * dist
            dz = -math.cos(rad) * dist

            bin_yaw = calcular_bin_yaw(erro_yaw)
            acao_idx = calcular_acao_18(bin_yaw, deve_pular=False)

            sv = gerar_vetor_estado(dx, dz, dist, erro_yaw, estagio_idx=0, modo_varredura=False)

            amostras.append({
                "tipo": "fork_spawn",
                "prompt": prompt,
                "sv": sv,
                "erro_yaw": erro_yaw,
                "acao_alvo": acao_idx,
                "bin_yaw": bin_yaw
            })

        else:
            # Correção de Rota / Alinhamento Fino
            cor1 = rng.choice(cores)
            prompt = f"Objetivo: vá até o bloco {cor1} [Etapa 1/2]"
            erro_yaw = rng.uniform(-15.0, 15.0)
            dist = rng.uniform(2.0, 6.0)
            rad = math.radians(erro_yaw)
            dx = -math.sin(rad) * dist
            dz = -math.cos(rad) * dist

            bin_yaw = calcular_bin_yaw(erro_yaw)
            acao_idx = calcular_acao_18(bin_yaw, deve_pular=False)

            sv = gerar_vetor_estado(dx, dz, dist, erro_yaw, estagio_idx=0, modo_varredura=False)

            amostras.append({
                "tipo": "fork_correcao",
                "prompt": prompt,
                "sv": sv,
                "erro_yaw": erro_yaw,
                "acao_alvo": acao_idx,
                "bin_yaw": bin_yaw
            })

    return amostras


if __name__ == "__main__":
    dataset = gerar_dataset_bifurcacoes(num_amostras=500)
    os.makedirs("fase5/dados", exist_ok=True)
    caminho = "fase5/dados/dataset_bifurcacoes_coldstart.pt"
    torch.save(dataset, caminho)
    print(f"[OK] Geradas {len(dataset)} amostras de bifurcacao salvas em: {caminho}")
    
    tipos = {}
    for d in dataset:
        tipos[d["tipo"]] = tipos.get(d["tipo"], 0) + 1
    print(f"Distribuicao das bifurcacoes: {tipos}")
