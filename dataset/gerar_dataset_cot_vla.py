# coding=utf-8
"""
dataset/gerar_dataset_cot_vla.py — Gerador de Dataset CoT-VLA (Chain-of-Thought Vision-Language-Action).

Gera amostras multimodais contendo:
  1. Estado Perceptivo (vetor sv de 32 dimensões + prompt de missão).
  2. Raciocínio Espacial e Causal Explícito em Linguagem Natural (<think>...</think>).
  3. Ação Tática Estruturada (<action>...</action>) e índices discretos (36 classes).
  4. Intercalação com âncoras de raciocínio simbólico (Matemática, Código e Lógica).
"""
import os
import sys
import math
import json
import random
import argparse
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

CORES = ["amarelo", "azul", "vermelho", "verde", "roxo", "laranja"]

TEMPLATES_THINK = [
    (
        "1. Submeta Ativa: Bloco {cor} a {dist:.1f}m no vetor angular {ang_str}.\n"
        "2. Análise Espacial: {analise_espacial}\n"
        "3. Decisão Causal: {decisao_motora}"
    ),
    (
        "Analiso o cenário: O alvo {cor} está a {dist:.1f} metros com desvio de {erro_yaw:+.1f} graus. "
        "{analise_espacial} Portanto, a ação ótima é {decisao_motora}."
    ),
    (
        "[Etapa {estagio_num}/2] Alvo: {cor}. Distância estimada: {dist:.1f}m. Erro de mira: {erro_yaw:+.1f}°. "
        "{analise_espacial} Executando {decisao_motora} para convergir na rota."
    )
]


def descrever_analise_espacial(dist: float, erro_yaw: float, modo_nome: str) -> str:
    """Gera o raciocínio físico e espacial baseado na distância e ângulo."""
    abs_erro = abs(erro_yaw)
    if dist < 0.8:
        return "Proximidade imediata do bloco de destino. Toque iminente."
    elif dist < 2.0:
        if abs_erro < 15.0:
            return "Na reta final de aproximação com alinhamento visual preciso."
        else:
            return "Distância curta com leve desalinhamento lateral; correção rápida necessária."
    elif dist < 5.0:
        if abs_erro > 45.0:
            return "Alvo fora do cone de visão frontal estreito; requer reorientação angular ativa."
        else:
            return "Alvo travado no campo visual intermediário; trajetória desobstruída."
    else:
        return "Fase de aproximação longa em campo aberto; máxima aceleração requerida."


def descrever_decisao_motora(modo_nome: str, dyaw_val: float) -> str:
    """Gera a justificativa da escolha das teclas e do mouse."""
    lado = "à direita" if dyaw_val > 0 else "à esquerda" if dyaw_val < 0 else "em frente"
    rot_str = f"girar {abs(dyaw_val):.0f}° {lado}" if abs(dyaw_val) > 0 else "manter mira reta"

    if modo_nome == "sprint":
        return f"aplicar sprint contínuo (W+Ctrl) e {rot_str} para avanço veloz"
    elif modo_nome == "strafe_esq":
        return f"aplicar strafe lateral (W+A) com compensação angular {rot_str}"
    elif modo_nome == "strafe_dir":
        return f"aplicar strafe lateral (W+D) com compensação angular {rot_str}"
    elif modo_nome == "pulo":
        return f"acionar salto tático (W+Espaço) e {rot_str} para vencer desnível"
    elif modo_nome == "recuar":
        return f"recuar brevemente (S) e {rot_str} para reabrir ângulo de manobra"
    elif modo_nome == "alinhar":
        return f"cessar avanço momentaneamente e priorizar {rot_str} para travar o pilar no retículo"
    return f"avançar com {rot_str}"


def criar_amostra_cot(
    dist: float,
    erro_yaw: float,
    cor_alvo: str,
    cor_segunda: str,
    estagio: int,
    rng: random.Random
) -> Dict[str, Any]:
    """Cria uma amostra CoT-VLA completa com vetor de estado, think e action."""
    idx_36 = calcular_acao_otima_tatica(dist, erro_yaw, rng)
    modo_idx, yaw_bin_idx = fatorar_indice_36(idx_36)
    modo_nome = MODOS[modo_idx]
    dyaw_val = YAW_BINS_9[yaw_bin_idx]

    rad = math.radians(erro_yaw)
    dx = -math.sin(rad) * dist
    dz = -math.cos(rad) * dist

    sv = [0.0] * 32
    sv[0] = round(dx / 15.0, 4)
    sv[1] = round(dz / 15.0, 4)
    sv[2] = round(min(dist / 15.0, 1.0), 4)
    sv[3] = round(math.cos(rad), 4)
    sv[4] = round(math.sin(rad), 4)
    sv[16] = float(estagio)

    ang_str = f"{abs(erro_yaw):.1f}° à direita" if erro_yaw > 0 else f"{abs(erro_yaw):.1f}° à esquerda" if erro_yaw < 0 else "0° frontal"
    analise_esp = descrever_analise_espacial(dist, erro_yaw, modo_nome)
    decisao_mot = descrever_decisao_motora(modo_nome, dyaw_val)

    template = rng.choice(TEMPLATES_THINK)
    think_text = template.format(
        cor=cor_alvo,
        dist=dist,
        erro_yaw=erro_yaw,
        ang_str=ang_str,
        estagio_num=estagio + 1,
        analise_espacial=analise_esp,
        decisao_motora=decisao_mot
    )

    action_text = f"modo:{modo_nome}, yaw:{dyaw_val:+.0f}"
    prompt_usuario = f"Missão: Vá até o bloco {cor_alvo} [Etapa {estagio+1}/2] e depois até o bloco {cor_segunda} [Etapa {estagio+2}/2]."

    return {
        "tipo": "cot_vla",
        "prompt": prompt_usuario,
        "state_vector": sv,
        "telemetria": {
            "dist": round(dist, 3),
            "erro_yaw": round(erro_yaw, 2),
            "estagio": estagio,
            "cor_alvo": cor_alvo
        },
        "think": think_text,
        "action_text": action_text,
        "acao_36": idx_36,
        "modo_idx": modo_idx,
        "yaw_idx": yaw_bin_idx,
        "conversacao": [
            {"role": "user", "content": prompt_usuario},
            {"role": "assistant", "content": f"<think>\n{think_text}\n</think>\n<action>{action_text}</action>"}
        ]
    }


def gerar_dataset(total_amostras: int = 5000, seed: int = 42) -> List[Dict[str, Any]]:
    """Gera o dataset sintético balanceado de CoT-VLA."""
    rng = random.Random(seed)
    amostras = []

    print(f"[*] Gerando {total_amostras} amostras de CoT-VLA com Raciocínio em Linguagem...")

    for i in range(total_amostras):
        cor_1 = rng.choice(CORES)
        cor_2 = rng.choice([c for c in CORES if c != cor_1])
        estagio = rng.choice([0, 1])
        cor_ativa = cor_1 if estagio == 0 else cor_2

        # Distribuição de distâncias balanceada (curta, média, longa)
        r = rng.random()
        if r < 0.35:
            dist = rng.uniform(0.3, 2.0)
        elif r < 0.70:
            dist = rng.uniform(2.0, 5.5)
        else:
            dist = rng.uniform(5.5, 9.0)

        # Distribuição angular (frontal, moderado, varredura ampla)
        ang_r = rng.random()
        if ang_r < 0.50:
            erro_yaw = rng.uniform(-25.0, 25.0)
        elif ang_r < 0.85:
            erro_yaw = rng.uniform(-75.0, 75.0)
        else:
            erro_yaw = rng.uniform(-160.0, 160.0)

        amostra = criar_amostra_cot(dist, erro_yaw, cor_ativa, cor_2 if estagio == 0 else cor_1, estagio, rng)
        amostras.append(amostra)

    return amostras


def main():
    parser = argparse.ArgumentParser(description="Gerador de Dataset CoT-VLA com Raciocínio Espacial.")
    parser.add_argument("--total", type=int, default=5000, help="Quantidade de amostras a gerar.")
    parser.add_argument("--saida", type=str, default="dataset/cot_vla_dataset.jsonl", help="Caminho do arquivo de saída.")
    parser.add_argument("--seed", type=int, default=42, help="Seed pseudoaleatória.")
    args = parser.parse_args()

    caminho_abs = os.path.join(_ROOT, args.saida)
    os.makedirs(os.path.dirname(caminho_abs), exist_ok=True)

    dados = gerar_dataset(args.total, args.seed)

    with open(caminho_abs, "w", encoding="utf-8") as f:
        for item in dados:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tamanho_mb = os.path.getsize(caminho_abs) / (1024 * 1024)
    print(f"[OK] Dataset CoT-VLA gerado com sucesso!")
    print(f"  -> Total de Amostras: {len(dados)}")
    print(f"  -> Arquivo: {caminho_abs} ({tamanho_mb:.2f} MB)")
    print(f"  -> Exemplo de Amostra:\n" + json.dumps(dados[0]["conversacao"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
