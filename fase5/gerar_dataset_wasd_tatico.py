# coding=utf-8
"""
fase5/gerar_dataset_wasd_tatico.py — Gerador do Dataset Ancorado com Raciocínio Tático WASD (36 Ações).
"""
from __future__ import annotations
import os
import sys
import math
import random
import argparse
import torch
from typing import List, Dict, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.acoes_taticas import (
    calcular_acao_otima_tatica,
    fatorar_indice_36,
    MODOS,
    YAW_BINS_9
)


def gerar_amostras_sinteticas_taticas(num_amostras: int = 15000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    amostras = []
    cores = ["amarelo", "roxo", "azul", "verde", "vermelho"]

    for _ in range(num_amostras):
        estagio = rng.choice([0, 1])
        c1, c2 = rng.sample(cores, 2)
        cor_alvo = c1 if estagio == 0 else c2
        etapa_txt = "1/2" if estagio == 0 else "2/2"
        prompt = f"Objetivo: vá até o bloco {cor_alvo} [Etapa {etapa_txt}]"

        # Distribuição balanceada de cenários táticos:
        # 22% Sprint Reto / Curva Suave
        # 10% Sprint com Curva Ampla
        # 14% Sprint com Pulo (vários ângulos)
        # 22% Strafe Lateral Tático (com micro-ajustes)
        # 16% Alinhamento Estacionário / Busca / Transição
        # 16% Desengate / Recuo / Chegada Fina
        cenario = rng.choices(
            ["sprint_reto", "sprint_curva", "sprint_pulo", "strafe_lateral", "giro_alinhar", "recuo_parede", "transicao_submeta"],
            weights=[0.22, 0.10, 0.14, 0.22, 0.12, 0.10, 0.10]
        )[0]

        deve_pular = False
        is_spawn = False
        is_transicao = False
        is_alinhar = False
        esta_colidindo = False
        offset_lat = 0.0

        if cenario == "sprint_reto":
            erro_yaw = rng.uniform(-15.0, 15.0)
            dist = rng.uniform(2.0, 14.0)
            tipo_nome = "sprint"
            peso = 0.70

        elif cenario == "sprint_curva":
            # Sprint em curva com ângulos mais abertos
            sinal = rng.choice([-1, 1])
            erro_yaw = sinal * rng.uniform(20.0, 90.0)
            dist = rng.uniform(4.0, 14.0)
            tipo_nome = "sprint_curva"
            peso = 0.85

        elif cenario == "sprint_pulo":
            # Cobre todos os ângulos de transposição
            erro_yaw = rng.uniform(-75.0, 75.0)
            dist = rng.uniform(1.5, 10.0)
            deve_pular = True
            tipo_nome = "pulo"
            peso = 0.90

        elif cenario == "strafe_lateral":
            # Alvo descentralizado, micro-ajuste de câmera
            sinal = rng.choice([-1, 1])
            erro_yaw = sinal * rng.uniform(10.0, 45.0)
            dist = rng.uniform(1.5, 8.5)
            offset_lat = rng.choice([-5.0, 0.0, 5.0])
            tipo_nome = "strafe"
            peso = 1.20

        elif cenario == "giro_alinhar":
            sinal = rng.choice([-1, 1])
            erro_yaw = sinal * rng.uniform(25.0, 175.0)
            dist = rng.uniform(2.0, 15.0)
            is_alinhar = True
            is_spawn = rng.choice([True, False])
            tipo_nome = "alinhar"
            peso = 1.40

        elif cenario == "transicao_submeta":
            sinal = rng.choice([-1, 1])
            erro_yaw = sinal * rng.uniform(35.0, 160.0)
            dist = rng.uniform(4.0, 12.0)
            is_transicao = True
            tipo_nome = "transicao"
            peso = 1.50

        else:  # recuo_parede
            erro_yaw = rng.uniform(-180.0, 180.0)
            dist = rng.uniform(0.5, 2.5)
            esta_colidindo = True
            offset_lat = rng.choice([-5.0, 0.0, 5.0])
            tipo_nome = "recuar"
            peso = 1.30

        acao_otima = calcular_acao_otima_tatica(
            erro_yaw_graus=erro_yaw,
            distancia=dist,
            deve_pular=deve_pular,
            is_spawn=is_spawn,
            is_transicao=is_transicao,
            is_alinhar=is_alinhar,
            esta_colidindo=esta_colidindo,
            offset_lateral=offset_lat
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
            "entropia_norm": 0.35 if tipo_nome in ["sprint", "pulo", "sprint_curva"] else 0.85,
            "peso": peso,
            "erro_yaw_graus": erro_yaw,
            "eh_decisao": (tipo_nome not in ["sprint", "sprint_curva", "pulo"])
        })

    return amostras


def construir_dataset_wasd_completo(
    caminho_calibrado_grande: str = "fase5/dados/dataset_calibrado_grande.pt",
    caminho_saida: str = "fase5/dados/dataset_wasd_tatico_36_v2.pt",
    num_sinteticas: int = 15000,
    seed: int = 123
):
    print("=" * 80)
    print(" [FASE 5.5] CONSTRUINDO DATASET DE RACIOCÍNIO TÁTICO WASD BALANCEADO (v2)")
    print(f"    Bifurcações Reais     : {caminho_calibrado_grande}")
    print(f"    Amostras Sintéticas   : {num_sinteticas}")
    print(f"    Saída                 : {caminho_saida}")
    print("=" * 80)

    amostras_sint = gerar_amostras_sinteticas_taticas(num_sinteticas, seed=seed)
    print(f"[Sintético] {len(amostras_sint)} amostras balanceadas geradas.")

    amostras_reais_remapeadas = []
    if os.path.exists(caminho_calibrado_grande):
        reais = torch.load(caminho_calibrado_grande, weights_only=False)
        print(f"[Reais] {len(reais)} bifurcações reais carregadas para remapeamento.")
        rng_real = random.Random(seed + 1)
        for d in reais:
            erro_g = float(d.get("erro_yaw_graus", 85.0))
            dist = float(d.get("dist_alvo", 6.0))
            is_spawn = (d.get("tipo") == "spawn")
            is_transicao = (d.get("tipo") == "transicao" or d.get("tipo") == "fork_transicao")

            offset_lat = rng_real.choice([-5.0, 0.0, 5.0]) if abs(erro_g) < 45.0 else 0.0
            acao_tatica = calcular_acao_otima_tatica(
                erro_yaw_graus=erro_g,
                distancia=dist,
                deve_pular=False,
                is_spawn=is_spawn,
                is_transicao=is_transicao,
                is_alinhar=(abs(erro_g) > 40.0),
                esta_colidindo=False,
                offset_lateral=offset_lat
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

    dataset_final = amostras_sint + amostras_reais_remapeadas
    rng = random.Random(seed)
    rng.shuffle(dataset_final)

    total = len(dataset_final)
    print(f"[Total] {total} amostras compostas ({len(amostras_sint)} sintéticas + {len(amostras_reais_remapeadas)} reais).")

    cont_acoes = [0] * 36
    cont_modos = [0] * 6
    for item in dataset_final:
        a = int(item["acao_otima"])
        cont_acoes[a] += 1
        m, _ = fatorar_indice_36(a)
        cont_modos[m] += 1

    print("\n[AUDITORIA V2] Distribuição por Modo:")
    for m in range(6):
        pct = (cont_modos[m] / total) * 100.0
        print(f"  Modo {m} ({MODOS[m]:<12}): {cont_modos[m]:5d} ({pct:5.2f}%)")

    print("\n[AUDITORIA V2] Cobertura de Ações (36 classes):")
    zeradas = 0
    for a in range(36):
        pct = (cont_acoes[a] / total) * 100.0
        status = "OK" if cont_acoes[a] > 0 else "ZERADA"
        if cont_acoes[a] == 0:
            zeradas += 1
        print(f"  Ação {a:02d}: {cont_acoes[a]:5d} ({pct:5.2f}%) [{status}]")

    print(f"\nTotal de Ações Zeradas: {zeradas}/36")

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    torch.save(dataset_final, caminho_saida)
    print(f"[OK] Dataset v2 salvo com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Dataset WASD Tático v2")
    parser.add_argument("--saida", default="fase5/dados/dataset_wasd_tatico_36_v2.pt")
    parser.add_argument("--sinteticas", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    construir_dataset_wasd_completo(
        caminho_saida=args.saida,
        num_sinteticas=args.sinteticas,
        seed=args.seed
    )
