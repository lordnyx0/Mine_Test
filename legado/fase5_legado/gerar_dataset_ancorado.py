# coding=utf-8
"""
fase5/gerar_dataset_ancorado.py — Gerador do Dataset Ancorado para Sparse Policy Selection.

Combina:
  1. Buffer de Locomoção Densa (70%):
     - Avanço reto em sprint (ação 4: YAW=0, W)
     - Pequenas correções angulares finas (ações 3 e 5: YAW=-5/+5, W)
     - Saltos e transposição de relevo (ação 13: YAW=0, W+SPACE)
     - Aproximação de submetas 1 e 2 em distâncias variadas (1.0m a 14.0m)
  2. Buffer de Bifurcações Causais de Alta Entropia (30%):
     - As 847 decisões mineradas reais filtradas causalmente em `dataset_decisoes_alta_entropia.pt`

Saída:
  - `fase5/dados/dataset_ancorado_fase5.pt`
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

from fase5.gerar_demonstracoes_esparsas import (
    calcular_bin_yaw, calcular_acao_18, gerar_vetor_estado, YAW_BINS
)

def gerar_locomocao_densa(num_amostras: int = 2000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    amostras = []
    cores = ["amarelo", "roxo", "azul"]

    for _ in range(num_amostras):
        # 1. Escolhe se é Etapa 1 ou Etapa 2
        estagio = rng.choice([0, 1])
        c1, c2 = rng.sample(cores, 2)
        cor_alvo = c1 if estagio == 0 else c2
        etapa_txt = "1/2" if estagio == 0 else "2/2"
        prompt = f"Objetivo: vá até o bloco {cor_alvo} [Etapa {etapa_txt}]"

        # 2. Perfil de locomoção de rotina (baixa incerteza):
        # 60% avanço reto perfeito (erro_yaw entre -4° e +4°)
        # 25% micro-correção (erro_yaw entre -10° e +10°)
        # 15% pulo/transposição de relevo
        modo = rng.choices(["reto", "micro_ajuste", "pulo_obstaculo"], weights=[0.60, 0.25, 0.15])[0]

        if modo == "reto":
            erro_yaw = rng.uniform(-3.5, 3.5)
            dist = rng.uniform(2.0, 14.0)
            deve_pular = False
        elif modo == "micro_ajuste":
            erro_yaw = rng.uniform(-10.0, 10.0)
            dist = rng.uniform(1.5, 12.0)
            deve_pular = False
        else: # pulo_obstaculo
            erro_yaw = rng.uniform(-5.0, 5.0)
            dist = rng.uniform(1.0, 8.0)
            deve_pular = True

        rad = math.radians(erro_yaw)
        dx = -math.sin(rad) * dist
        dz = -math.cos(rad) * dist

        bin_yaw = calcular_bin_yaw(erro_yaw)
        acao_idx = calcular_acao_18(bin_yaw, deve_pular=deve_pular)

        sv = torch.zeros(32, dtype=torch.float32)
        sv[0] = dx / 15.0
        sv[1] = dz / 15.0
        sv[2] = min(dist / 15.0, 1.0)
        rad = math.radians(erro_yaw)
        sv[3] = math.cos(rad)
        sv[4] = math.sin(rad)
        sv[16] = float(estagio)

        amostras.append({
            "tipo": "locomocao_densa",
            "prompt": prompt,
            "sv": sv,
            "acao_otima": acao_idx,
            "entropia_norm": 0.20, # Baixa entropia de rotina
            "peso": 0.50 # Peso regularizador
        })

    return amostras


def construir_dataset_ancorado(
    caminho_alta_entropia: str = "fase5/dados/dataset_decisoes_alta_entropia.pt",
    caminho_saida: str = "fase5/dados/dataset_ancorado_fase5.pt",
    num_densas: int = 2000,
    seed: int = 42
):
    print("=" * 80)
    print(" [FASE 5] CONSTRUINDO DATASET ANCORADO (MISTURA BALANCEADA)")
    print(f"    Alta Entropia (Causal) : {caminho_alta_entropia}")
    print(f"    Locomoção Densa (Base) : {num_densas} amostras")
    print(f"    Destino Final          : {caminho_saida}")
    print("=" * 80)

    # 1. Carrega decisões mineradas de alta entropia
    if not os.path.exists(caminho_alta_entropia):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_alta_entropia}")

    decisoes_entropia = torch.load(caminho_alta_entropia, weights_only=False)
    print(f"[Buffer Causal] {len(decisoes_entropia)} decisões de alta entropia carregadas.")

    # Atribui pesos de bifurcação
    amostras_esparsas = []
    for d in decisoes_entropia:
        h_norm = float(d.get("entropia_norm", 0.8))
        item = {
            "tipo": "decisao_esparsa_causal",
            "prompt": d.get("prompt", "Objetivo: vá até o bloco roxo [Etapa 2/2]"),
            "sv": d["sv"],
            "acao_otima": int(d["acao_otima"]),
            "entropia_norm": h_norm,
            "peso": max(1.0, h_norm * 2.0) # Ponderação elevada para bifurcações
        }
        amostras_esparsas.append(item)

    # 2. Gera buffer de locomoção densa
    amostras_densas = gerar_locomocao_densa(num_amostras=num_densas, seed=seed)
    print(f"[Buffer Denso] {len(amostras_densas)} amostras de locomoção contínua geradas.")

    # 3. Combina e embaralha
    dataset_completo = amostras_densas + amostras_esparsas
    rng = random.Random(seed)
    rng.shuffle(dataset_completo)

    total = len(dataset_completo)
    pct_esparsa = (len(amostras_esparsas) / total) * 100.0
    pct_densa = (len(amostras_densas) / total) * 100.0

    print(f"[Composição] Total: {total} amostras ({pct_densa:.1f}% Locomoção Densa / {pct_esparsa:.1f}% Decisões Esparsas)")

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    torch.save(dataset_completo, caminho_saida)
    print(f"[OK] Dataset ancorado salvo com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    construir_dataset_ancorado()
