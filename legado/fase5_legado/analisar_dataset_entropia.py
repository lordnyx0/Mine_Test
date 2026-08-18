# coding=utf-8
"""
fase5/analisar_dataset_entropia.py — Analisador e Validador do Dataset de Alta Entropia.

Lê o dataset minerado em `fase5/dados/dataset_decisoes_alta_entropia.pt` e exibe:
  - Estatísticas de entropia (média, min, max, percentis).
  - Distribuição por tipo de bifurcação (Spawn, Transição de Submeta, Ajuste Fino).
  - Acurácia prévia da política nas bifurcações e desvio angular.
  - Geração de gráficos de distribuição de entropia e histograma de ações.
"""
from __future__ import annotations

import os
import sys
import math
import json
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def analisar(dataset_path: str = "fase5/dados/dataset_decisoes_alta_entropia.pt", plot_out: str = "docs/analise_dataset_entropia.png"):
    if not os.path.exists(dataset_path):
        print(f"[ERRO] Dataset não encontrado em: {dataset_path}")
        return

    amostras = torch.load(dataset_path, weights_only=False)
    total = len(amostras)
    if total == 0:
        print("[AVISO] O dataset está vazio.")
        return

    entropias = [a["entropia"] for a in amostras]
    entropias_norm = [a["entropia_norm"] for a in amostras]
    tipos = [a["tipo_fork"] for a in amostras]
    acoes_exec = [a["acao_executada"] for a in amostras]
    acoes_otimas = [a["acao_otima"] for a in amostras]
    distancias = [a["dist_alvo"] for a in amostras]

    acertos = sum(1 for e, o in zip(acoes_exec, acoes_otimas) if e == o)
    taxa_concordancia = (acertos / total) * 100.0

    contagem_tipos = {}
    for t in tipos:
        contagem_tipos[t] = contagem_tipos.get(t, 0) + 1

    print("=" * 80)
    print(" [FASE 5] RELATÓRIO DO DATASET DE ALTA ENTROPIA & BIFURCAÇÕES")
    print(f"    Arquivo               : {dataset_path}")
    print(f"    Total de Decisões     : {total}")
    print(f"    Entropia Média (Norm) : {np.mean(entropias_norm):.3f} (H = {np.mean(entropias):.3f} nats)")
    print(f"    Entropia Mín / Máx    : {np.min(entropias_norm):.3f} / {np.max(entropias_norm):.3f}")
    print(f"    Percentis 25 / 50 / 75: {np.percentile(entropias_norm, 25):.3f} / {np.percentile(entropias_norm, 50):.3f} / {np.percentile(entropias_norm, 75):.3f}")
    print(f"    Concordância Política : {acertos}/{total} ({taxa_concordancia:.1f}%)")
    print("=" * 80)
    print("\n--- Distribuição por Tipo de Decisão ---")
    for t, c in sorted(contagem_tipos.items(), key=lambda x: -x[1]):
        pct = (c / total) * 100.0
        print(f"    {t:<22}: {c:4d} amostras ({pct:5.1f}%)")

    # Amostra de exemplos
    print("\n--- Exemplos de Amostras de Alta Entropia ---")
    for idx in range(min(5, total)):
        ex = amostras[idx]
        print(f"  [{idx+1}] Tipo: {ex['tipo_fork']:<18} | H_norm: {ex['entropia_norm']:.3f} | Dist: {ex['dist_alvo']:4.1f}m | Ação Exec: {ex['acao_executada']} -> Ação Ótima: {ex['acao_otima']}")
        print(f"      Prompt: {ex['prompt']}")

    # Gráfico
    if HAS_MATPLOTLIB:
        os.makedirs(os.path.dirname(plot_out), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # 1. Histograma de Entropia
        axes[0].hist(entropias_norm, bins=20, color="#38bdf8", edgecolor="#0284c7")
        axes[0].set_title("Distribuição de Entropia Normalizada", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("H_norm")
        axes[0].set_ylabel("Frequência")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Distribuição por Tipo de Fork
        tipos_nomes = list(contagem_tipos.keys())
        tipos_vals = list(contagem_tipos.values())
        axes[1].bar(tipos_nomes, tipos_vals, color=["#10b981", "#f59e0b", "#a855f7"][:len(tipos_nomes)])
        axes[1].set_title("Decisões por Categoria", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # 3. Entropia vs Distância ao Alvo
        axes[2].scatter(distancias, entropias_norm, alpha=0.6, color="#ec4899", edgecolors="none")
        axes[2].set_title("Entropia vs Distância ao Alvo", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Distância ao Alvo (m)")
        axes[2].set_ylabel("H_norm")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"Fase 5: Análise do Dataset de Alta Entropia ({total} amostras limpas)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_out, dpi=150)
        plt.close()
        print(f"\n[OK] Gráfico estatístico salvo em: {plot_out}")


if __name__ == "__main__":
    analisar()
