# coding=utf-8
"""
fase7/executar_fase7_completo.py — Pipeline Mestre de Execução e Validação da Fase 7 (CoT-GRPO VLA).

Executa:
  1. Treinamento GRPO Token-Level no Minecraft (20 iterações com muros e raciocínio de desvio).
  2. Benchmark de Desvio de Obstáculos no Minecraft (40 episódios).
  3. Conversão para GGUF True Loop Q8_0.
  4. Execução do Benchmark Oficial de Raciocínio (97 itens) medindo a emergência de capacidade cognitiva.
"""
from __future__ import annotations
import os
import sys
import subprocess
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

def main():
    print("=" * 80)
    print(" PIPELINE MESTRE DE EXECUÇÃO: FASE 7 (CoT-GRPO VLA)")
    print("=" * 80, flush=True)

    t0 = time.time()

    # 1. Treinamento GRPO
    print("\n>>> [ETAPA 1/3] Treinamento GRPO Token-Level com Obstáculos e Lógica...")
    cmd_treino = [sys.executable, os.path.join(_ROOT, "fase7", "treinar_grpo_cot_vla.py")]
    subprocess.run(cmd_treino, check=True)

    # 2. Benchmark de Desvio de Obstáculos
    print("\n>>> [ETAPA 2/3] Avaliação de Desvio de Obstáculos no Minecraft...")
    cmd_aval = [sys.executable, os.path.join(_ROOT, "fase7", "avaliar_fase7.py"), "--lotes", "5"]
    subprocess.run(cmd_aval, check=True)

    # 3. Conversão GGUF Q8_0 e Benchmark de Raciocínio
    print("\n>>> [ETAPA 3/3] Conversão GGUF Q8_0 e Benchmark Cognitivo (97 Itens)...")
    cmd_bench = [sys.executable, os.path.join(_ROOT, "fase6", "executar_benchmark_q8_fase6.py")]
    subprocess.run(cmd_bench, check=True)

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print(f" PIPELINE FASE 7 CONCLUÍDO COM SUCESSO EM {dt/60:.1f} MINUTOS!")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
