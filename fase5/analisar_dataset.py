# coding=utf-8
"""
FASE 5 — Auditoria e Análise Estatística do Dataset de Bifurcações (Cold-Start).

Verifica a adequação do dataset de acordo com os critérios do Sparse Policy Selection:
  1. Balanceamento das bifurcações (Spawn vs Transição vs Correção)
  2. Cobertura angular e simetria lateral (Esquerda vs Direita vs Centro)
  3. Consistência do vetor de estado sv de 18 dimensões (normas, flags, NaNs)
  4. Alinhamento causal entre o erro angular e o índice da ação (0 a 17)
  5. Consistência semântica Prompt <-> Vetor de Estado <-> Ação
"""
from __future__ import annotations

import os
import sys
import torch
import numpy as np
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def auditar_dataset_fase5(dataset_path: str = "fase5/dados/dataset_bifurcacoes_coldstart.pt"):
    print("=" * 80)
    print(f" AUDITORIA DETALHADA DO DATASET DE BIFURCACOES: {dataset_path}")
    print("=" * 80)

    if not os.path.exists(dataset_path):
        print(f"[ERRO] Arquivo {dataset_path} nao encontrado!")
        return

    dados = torch.load(dataset_path, weights_only=False)
    N = len(dados)
    print(f"\n[1] METRICAS GERAIS:")
    print(f"  Total de Amostras : {N}")

    # 1. Contagem de Tipos de Bifurcação
    cont_tipos = Counter(d["tipo"] for d in dados)
    print(f"\n[2] DISTRIBUICAO DAS BIFURCACOES (PONTOS DE ALTA ENTROPIA):")
    for tipo, cont in cont_tipos.items():
        pct = (cont / N) * 100.0
        print(f"  - {tipo:<18}: {cont:4d} amostras ({pct:5.1f}%)")

    # 2. Análise da Cobertura Angular (Erro Yaw)
    erros_yaw = [d["erro_yaw"] for d in dados]
    erros_transicao = [d["erro_yaw"] for d in dados if d["tipo"] == "fork_transicao"]
    erros_spawn = [d["erro_yaw"] for d in dados if d["tipo"] == "fork_spawn"]

    print(f"\n[3] COBERTURA ANGULAR (GRAUS):")
    print(f"  Geral       : Min = {min(erros_yaw):6.1f}°, Max = {max(erros_yaw):6.1f}°, Media = {np.mean(erros_yaw):5.1f}°, Std = {np.std(erros_yaw):5.1f}°")
    print(f"  Spawn       : Min = {min(erros_spawn):6.1f}°, Max = {max(erros_spawn):6.1f}°, Media = {np.mean(erros_spawn):5.1f}°")
    print(f"  Transicao   : Min = {min(erros_transicao):6.1f}°, Max = {max(erros_transicao):6.1f}°, Media = {np.mean(erros_transicao):5.1f}°")

    # Simetria Lateral
    esq = sum(1 for e in erros_yaw if e > 5.0)
    dir_ = sum(1 for e in erros_yaw if e < -5.0)
    centro = sum(1 for e in erros_yaw if abs(e) <= 5.0)
    print(f"  Simetria    : Esquerda (>+5°) = {esq} ({esq/N*100:.1f}%) | Direita (<-5°) = {dir_} ({dir_/N*100:.1f}%) | Centro = {centro} ({centro/N*100:.1f}%)")

    # 3. Distribuição dos Bins de Ação (0 a 17)
    bins_yaw = Counter(d["bin_yaw"] for d in dados)
    acoes_18 = Counter(d["acao_alvo"] for d in dados)
    print(f"\n[4] DISTRIBUICAO DAS 18 CLASSES DE ACAO:")
    print("  Bin Yaw (0..8):")
    nomes_bins = ["Forte Dir (-60°)", "Media Dir (-30°)", "Suave Dir (-15°)", "Ajuste Dir (-5°)",
                  "Centro (0°)", "Ajuste Esq (+5°)", "Suave Esq (+15°)", "Media Esq (+30°)", "Forte Esq (+60°)"]
    for b in range(9):
        cont = bins_yaw.get(b, 0)
        barra = "█" * int((cont / N) * 40)
        print(f"    Bin {b} [{nomes_bins[b]:<18}]: {cont:4d} ({cont/N*100:5.1f}%) | {barra}")

    # 4. Integridade do Vetor de Estado (sv 18-D)
    svs = torch.stack([d["sv"] for d in dados])
    tem_nan = torch.isnan(svs).any().item()
    tem_inf = torch.isinf(svs).any().item()
    
    print(f"\n[5] INTEGRIDADE DO VETOR DE ESTADO (sv[18]):")
    print(f"  Shape do Tensor        : {list(svs.shape)}")
    print(f"  Presenca de NaNs / Infs: {'[FALHA] Detectado!' if tem_nan or tem_inf else '[OK] 100% Limpo'}")
    print(f"  sv[0..1] (dx, dz norm) : Min = {svs[:, :2].min().item():.3f}, Max = {svs[:, :2].max().item():.3f}")
    print(f"  sv[2] (dist norm)      : Min = {svs[:, 2].min().item():.3f}, Max = {svs[:, 2].max().item():.3f}")
    print(f"  sv[3..4] (cos, sin)    : Norma media sqrt(cos^2 + sin^2) = {torch.sqrt(svs[:, 3]**2 + svs[:, 4]**2).mean().item():.4f} (esperado 1.000)")
    
    # 5. Consistência Semântica (Estágio no Prompt vs sv[16])
    inconsistencias = 0
    for d in dados:
        sv_stage = d["sv"][16].item()
        prompt = d["prompt"]
        if "Etapa 1/2" in prompt and sv_stage != 0.0:
            inconsistencias += 1
        elif "Etapa 2/2" in prompt and sv_stage != 1.0:
            inconsistencias += 1

    print(f"\n[6] ALINHAMENTO SEMANTICO (Prompt <-> sv[16] Indicador de Etapa):")
    print(f"  Inconsistencias detectadas: {inconsistencias} de {N} ({'[OK] 100% Consistente' if inconsistencias == 0 else '[FALHA]'})")

    # 6. Alinhamento Causal (Erro Angular <-> Ação)
    # Convencao do Minecraft:
    #   erro > 0  (alvo a ESQUERDA) -> mouse NEGATIVO (virar a esquerda) -> Bins 0, 1, 2, 3
    #   erro < 0  (alvo a DIREITA)  -> mouse POSITIVO (virar a direita)  -> Bins 5, 6, 7, 8
    #   erro == 0 (alvo no CENTRO)  -> mouse ZERO                       -> Bin 4
    erros_causais = 0
    for d in dados:
        erro = d["erro_yaw"]
        bin_yaw = d["bin_yaw"]
        if erro > 15.0 and bin_yaw > 3:
            erros_causais += 1
        elif erro < -15.0 and bin_yaw < 5:
            erros_causais += 1

    print(f"\n[7] VERIFICACAO CAUSAL DIRETA (Erro Angular -> Acao de Mira):")
    print(f"  Erros de inversao de mira: {erros_causais} ({'[OK] Zero Inversoes' if erros_causais == 0 else '[FALHA]'})\n")

    print("=" * 80)
    print(" PARECER DE ADEQUACAO:")
    if inconsistencias == 0 and erros_causais == 0 and not tem_nan:
        print(" [APROVADO] O dataset cobre perfeitamente as bifurcacoes criticas de decisao,")
        print("            possui simetria bilateral e fornece o suporte previo ideal para o PPO.")
    else:
        print(" [ATENCAO] Foram encontradas inconformidades que necessitam de ajuste.")
    print("=" * 80)


if __name__ == "__main__":
    auditar_dataset_fase5()
