# coding=utf-8
"""
fase5/construir_dataset_ancorado_grande.py — Construtor do Dataset Ancorado em Grande Escala.

Combina:
  1. Buffer de Locomocao Densa (70% ~ 9.600 amostras):
     - Sprint reto, micro-ajustes e pulos
     - Preserva capacidade motora e prior direcional
  2. Buffer Calibrado de Bifurcacoes (30% ~ 4.145 amostras):
     - Erros de bifurcacao reais minerados com oraculo de angulo
     - Ponderacao balanceada por magnitude de erro

Saida:
  - fase5/dados/dataset_ancorado_calibrado_grande.pt
"""
from __future__ import annotations

import os
import sys
import math
import random
import torch
from typing import List, Dict, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.gerar_dataset_ancorado import gerar_locomocao_densa


def construir_dataset_ancorado_grande(
    caminho_calibrado: str = "fase5/dados/dataset_calibrado_grande.pt",
    caminho_saida: str = "fase5/dados/dataset_ancorado_calibrado_grande.pt",
    num_densas: int = 9600,
    seed: int = 42
):
    print("=" * 80)
    print(" [FASE 5] CONSTRUINDO DATASET ANCORADO EM GRANDE ESCALA (70/30)")
    print(f"    Bifurcacoes Calibradas : {caminho_calibrado}")
    print(f"    Locomocao Densa (Base) : {num_densas} amostras")
    print(f"    Destino Final          : {caminho_saida}")
    print("=" * 80)

    if not os.path.exists(caminho_calibrado):
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_calibrado}")

    decisoes = torch.load(caminho_calibrado, weights_only=False)
    print(f"[Buffer Calibrado] {len(decisoes)} decisoes calibradas carregadas.")

    amostras_esparsas = []
    for d in decisoes:
        h_norm = float(d.get("entropia_norm", 0.5))
        erro_graus = abs(float(d.get("erro_yaw_graus", 90.0)))
        peso_erro = 1.5 if erro_graus > 60.0 else 1.0
        peso_final = max(1.0, h_norm * 2.0) * peso_erro

        item = {
            "tipo": "decisao_esparsa_causal",
            "prompt": d.get("prompt", "Objetivo: va ate o bloco roxo [Etapa 1/2]"),
            "sv": d["sv"],
            "acao_otima": int(d["acao_otima"]),
            "entropia_norm": h_norm,
            "peso": peso_final,
            "erro_yaw_graus": erro_graus,
            "tipo_fork": d.get("tipo", "erro_bifurcacao")
        }
        amostras_esparsas.append(item)

    # Gera buffer de locomocao densa para ancoragem solida
    amostras_densas = gerar_locomocao_densa(num_amostras=num_densas, seed=seed)
    print(f"[Buffer Denso] {len(amostras_densas)} amostras de locomocao continua geradas.")

    dataset_completo = amostras_densas + amostras_esparsas
    rng = random.Random(seed)
    rng.shuffle(dataset_completo)

    total = len(dataset_completo)
    pct_esparsa = (len(amostras_esparsas) / total) * 100.0
    pct_densa = (len(amostras_densas) / total) * 100.0

    print(f"[Composicao] Total: {total} amostras ({pct_densa:.1f}% Densa / {pct_esparsa:.1f}% Bifurcacoes Calibradas)")

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    torch.save(dataset_completo, caminho_saida)
    print(f"[OK] Dataset salvo em: {caminho_saida}")


if __name__ == "__main__":
    construir_dataset_ancorado_grande()
