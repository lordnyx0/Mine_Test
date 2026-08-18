# coding=utf-8
"""
fase5/gerar_dataset_wasd_tatico.py — Gerador do Dataset Ancorado com Raciocínio Tático WASD (36 Ações).

Gera um dataset balanceado de 16.000+ amostras contendo:
  1. Giro Parado no Spawn / Curva Fechada (2.500 amostras):
     - Para alvos fora do FOV frontal (|yaw| > 45°), ensina a girar parado (hold: [])
  2. Strafe Tático com Fixação de Olhar (3.500 amostras):
     - Para alvos descentralizados (15° < |yaw| <= 45°), ensina strafe (hold: [W, A] ou [W, D])
  3. Sprint Frontal e Pulo (6.500 amostras):
     - Locomoção contínua em linha reta e transposição de relevo (hold: [W] / [W, SPACE])
  4. Desengate e Recuo (1.000 amostras):
     - Recuperação de colisão / overshoot (hold: [S])
  5. Mapeamento das 4.145 Bifurcações Calibradas:
     - Remapeadas com a função de oráculo tático 36 classes

Saída:
  - fase5/dados/dataset_wasd_tatico_36.pt
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

from fase5.acoes_taticas import calcular_acao_otima_tatica


def gerar_amostras_sinteticas_taticas(num_amostras: int = 12000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    amostras = []
    cores = ["amarelo", "roxo", "azul"]

    for _ in range(num_amostras):
        estagio = rng.choice([0, 1])
        c1, c2 = rng.sample(cores, 2)
        cor_alvo = c1 if estagio == 0 else c2
        etapa_txt = "1/2" if estagio == 0 else "2/2"
        prompt = f"Objetivo: vá até o bloco {cor_alvo} [Etapa {etapa_txt}]"

        # Distribuição de situações táticas:
        # 40% Sprint Reto / Pulo
        # 25% Strafe Lateral (Fixação Visual)
        # 25% Alinhamento no Spawn (Giro Parado)
        # 10% Recuo / Desengate
        cenario = rng.choices(
            ["sprint_reto", "sprint_pulo", "strafe_lateral", "giro_spawn", "recuo_parede"],
            weights=[0.30, 0.10, 0.25, 0.25, 0.10]
        )[0]

        if cenario == "sprint_reto":
            erro_yaw = rng.uniform(-10.0, 10.0)
            dist = rng.uniform(2.0, 14.0)
            deve_pular = False
            is_spawn = False
            esta_colidindo = False
            tipo_nome = "sprint"
            peso = 0.60
        elif cenario == "sprint_pulo":
            erro_yaw = rng.uniform(-6.0, 6.0)
            dist = rng.uniform(1.0, 8.0)
            deve_pular = True
            is_spawn = False
            esta_colidindo = False
            tipo_nome = "pulo"
            peso = 0.80
        elif cenario == "strafe_lateral":
            # Alvo descentralizado, mantém mira
            sinal = rng.choice([-1, 1])
            erro_yaw = sinal * rng.uniform(15.0, 45.0)
            dist = rng.uniform(1.5, 7.5)
            deve_pular = False
            is_spawn = False
            esta_colidindo = False
            tipo_nome = "strafe"
            peso = 1.20
        elif cenario == "giro_spawn":
            # Spawn de costas ou em 90°
            sinal = rng.choice([-1, 1])
            erro_yaw = sinal * rng.uniform(50.0, 175.0)
            dist = rng.uniform(3.0, 15.0)
            deve_pular = False
            is_spawn = True
            esta_colidindo = False
            tipo_nome = "alinhar"
            peso = 1.50
        else: # recuo_parede
            erro_yaw = rng.uniform(-180.0, 180.0)
            dist = rng.uniform(0.5, 2.5)
            deve_pular = False
            is_spawn = False
            esta_colidindo = True
            tipo_nome = "recuar"
            peso = 1.30

        acao_otima = calcular_acao_otima_tatica(
            erro_yaw_graus=erro_yaw,
            distancia=dist,
            deve_pular=deve_pular,
            is_spawn=is_spawn,
            esta_colidindo=esta_colidindo
        )

        rad = math.radians(erro_yaw)
        dx = -math.sin(rad) * dist
        dz = -math.cos(rad) * dist

        sv = torch.zeros(32, dtype=torch.float32)
        sv[0] = dx / 15.0
        sv[1] = dz / 15.0
        sv[2] = min(dist / 15.0, 1.0)
        sv[3] = math.cos(rad)
        sv[4] = math.sin(rad)
        sv[16] = float(estagio)

        amostras.append({
            "tipo": tipo_nome,
            "prompt": prompt,
            "sv": sv,
            "acao_otima": acao_otima,
            "entropia_norm": 0.30 if tipo_nome in ["sprint", "pulo"] else 0.85,
            "peso": peso,
            "erro_yaw_graus": erro_yaw,
            "eh_decisao": (tipo_nome not in ["sprint", "pulo"])
        })

    return amostras


def construir_dataset_wasd_completo(
    caminho_calibrado_grande: str = "fase5/dados/dataset_calibrado_grande.pt",
    caminho_saida: str = "fase5/dados/dataset_wasd_tatico_36.pt",
    num_sinteticas: int = 12000,
    seed: int = 123
):
    print("=" * 80)
    print(" [FASE 5.2] CONSTRUINDO DATASET DE RACIOCÍNIO TÁTICO WASD (36 AÇÕES)")
    print(f"    Bifurcações Reais     : {caminho_calibrado_grande}")
    print(f"    Amostras Sintéticas   : {num_sinteticas}")
    print(f"    Saída                 : {caminho_saida}")
    print("=" * 80)

    # 1. Carrega amostras sintéticas táticas
    amostras_sint = gerar_amostras_sinteticas_taticas(num_sinteticas, seed=seed)
    print(f"[Sintético] {len(amostras_sint)} amostras táticas geradas.")

    # 2. Carrega e remapeia as bifurcações reais do dataset_calibrado_grande.pt
    amostras_reais_remapeadas = []
    if os.path.exists(caminho_calibrado_grande):
        reais = torch.load(caminho_calibrado_grande, weights_only=False)
        print(f"[Reais] {len(reais)} bifurcações reais carregadas para remapeamento.")
        for d in reais:
            erro_g = float(d.get("erro_yaw_graus", 85.0))
            dist = float(d.get("dist_alvo", 6.0))
            is_spawn = (d.get("tipo") == "spawn")
            
            acao_tatica = calcular_acao_otima_tatica(
                erro_yaw_graus=erro_g,
                distancia=dist,
                deve_pular=False,
                is_spawn=is_spawn,
                esta_colidindo=False
            )
            amostras_reais_remapeadas.append({
                "tipo": "bifurcacao_real",
                "prompt": d.get("prompt", "Objetivo: vá até o bloco azul [Etapa 1/2]"),
                "sv": d["sv"],
                "acao_otima": acao_tatica,
                "entropia_norm": float(d.get("entropia_norm", 0.8)),
                "peso": 1.60,
                "erro_yaw_graus": erro_g,
                "eh_decisao": True
            })

    # 3. Combina e embaralha
    dataset_final = amostras_sint + amostras_reais_remapeadas
    rng = random.Random(seed)
    rng.shuffle(dataset_final)

    total = len(dataset_final)
    print(f"[Total] {total} amostras compostas ({len(amostras_sint)} sintéticas + {len(amostras_reais_remapeadas)} reais).")

    # Contagem por tipo de ação
    contagem_modos = {"alinhar": 0, "sprint": 0, "pulo": 0, "strafe": 0, "recuar": 0}
    for item in dataset_final:
        a = item["acao_otima"]
        if 0 <= a <= 8:
            contagem_modos["alinhar"] += 1
        elif 9 <= a <= 17:
            contagem_modos["sprint"] += 1
        elif 18 <= a <= 26:
            contagem_modos["pulo"] += 1
        elif 27 <= a <= 32:
            contagem_modos["strafe"] += 1
        elif 33 <= a <= 35:
            contagem_modos["recuar"] += 1

    print("\n  Distribuição de Intenções no Dataset 36:")
    for modo, count in contagem_modos.items():
        pct = (count / total) * 100.0
        print(f"    {modo:15s}: {count:5d} ({pct:5.1f}%)")

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    torch.save(dataset_final, caminho_saida)
    print(f"\n[OK] Dataset salvo com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    construir_dataset_wasd_completo()
