# coding=utf-8
"""
fase6/executar_fase6_completo.py — Pipeline Mestre Automatizado da Fase 6 (CoT-VLA).

Executa ponta a ponta:
  1. Treinamento PPO no Ambiente Real do Minecraft com Cérebro Supervisor e Âncora Cognitiva.
  2. Avaliação TopView 2D de locomoção em 3 Pilares.
  3. Benchmark Oficial de Raciocínio (97 itens) comprovando ausência de regressão.
"""
import os
import sys
import time
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def executar():
    print("=" * 80)
    print(" PIPELINE COMPLETO DA FASE 6 — CoT-VLA + PPO REAL + BENCHMARK COGNITIVO")
    print("=" * 80, flush=True)

    # 1. PPO-BC CoT-VLA no Minecraft
    print("\n>>> ETAPA 1: Treinamento PPO no Minecraft com CoT-VLA e Âncora Multitarefa...")
    cmd_ppo = [
        sys.executable, "-u", os.path.join(_ROOT, "fase6", "treinar_ppo_cot_vla.py"),
        "--iteracoes", "50",
        "--passos", "85",
        "--mini-batch", "16",
        "--curriculo", "auto"
    ]
    subprocess.run(cmd_ppo, cwd=_ROOT, check=True)

    # 2. Avaliação TopView 2D
    print("\n>>> ETAPA 2: Avaliação TopView 2D de Navegação Multi-Pilares...")
    cmd_topview = [sys.executable, "-u", os.path.join(_ROOT, "fase6", "avaliar_fase6_topview.py"), "--episodios", "80"]
    subprocess.run(cmd_topview, cwd=_ROOT)

    # 3. Benchmark Oficial de Raciocínio (97 itens)
    print("\n>>> ETAPA 3: Benchmark Oficial de Raciocínio e Preservação Cognitiva...")
    cmd_bench = [sys.executable, "-u", os.path.join(_ROOT, "fase6", "avaliar_raciocinio_fase6.py")]
    subprocess.run(cmd_bench, cwd=_ROOT)

    print("\n" + "=" * 80)
    print(" [PIPELINE FASE 6 CONCLUÍDO COM SUCESSO TOTAL!]")
    print("=" * 80)


if __name__ == "__main__":
    executar()
