# coding=utf-8
"""
analisar_colaboracao_loops.py — Análise da Distribuição de Colaboração nas Camadas do Loop.

Mede o delta de ativação e a contribuição relativa de cada uma das 3 passagens do miolo
(camadas 7 a 20 do LoopSplit) comparando o Modelo Base contra a Fase 4.
"""
import os
import sys
import json
import torch
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from qwen3loop import Qwen3LoopForCausalLM
from transformers import AutoTokenizer

PROMPTS_TESTE = [
    "Se 5 máquinas levam 5 minutos para fazer 5 peças, quanto tempo 100 máquinas levam para fazer 100 peças?",
    "Todas as rosas são flores. Algumas flores murcham rapidamente. Logo, algumas rosas murcham rapidamente. Isso é verdadeiro ou falso?",
    "Escreva uma função em Python para encontrar o maior número primo menor que N.",
    "Qual a capital da Austrália?",
    "Objetivo: vá até o bloco amarelo e depois vá até o bloco azul."
]


def medir_deltas_modelo(model_or_path, tokenizer, device="cuda"):
    if isinstance(model_or_path, str):
        model = Qwen3LoopForCausalLM.from_pretrained(
            model_or_path,
            dtype=torch.bfloat16,
            device_map=device
        )
    else:
        model = model_or_path
    model.eval()

    # Camadas do loop cognitivo: 7 a 20 (14 camadas físicas)
    # No LoopSplit:
    # Passo 0..6: Entrada (0..6)
    # Passo 7..20: Loop 1 (camadas 7..20)
    # Passo 21..34: Loop 2 (camadas 7..20)
    # Passo 35..48: Loop 3 (camadas 7..20)
    # Passo 49..55: Saída (camadas 21..27)

    resultados_camadas = {c: {"it1": [], "it2": [], "it3": []} for c in range(7, 21)}

    with torch.no_grad():
        for p in PROMPTS_TESTE:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            out = model.model(inputs["input_ids"], output_hidden_states=True)
            hs = out.hidden_states # Tupla com 57 tensores de ativações

            for idx, c in enumerate(range(7, 21)):
                # Índices no hidden_states
                idx_it1_in = 7 + idx
                idx_it1_out = 7 + idx + 1
                idx_it2_in = 21 + idx
                idx_it2_out = 21 + idx + 1
                idx_it3_in = 35 + idx
                idx_it3_out = 35 + idx + 1

                d1 = (hs[idx_it1_out] - hs[idx_it1_in]).norm().item() / (hs[idx_it1_in].norm().item() + 1e-6)
                d2 = (hs[idx_it2_out] - hs[idx_it2_in]).norm().item() / (hs[idx_it2_in].norm().item() + 1e-6)
                d3 = (hs[idx_it3_out] - hs[idx_it3_in]).norm().item() / (hs[idx_it3_in].norm().item() + 1e-6)

                resultados_camadas[c]["it1"].append(d1)
                resultados_camadas[c]["it2"].append(d2)
                resultados_camadas[c]["it3"].append(d3)

    resumo = {}
    for c in range(7, 21):
        m1 = float(np.mean(resultados_camadas[c]["it1"]))
        m2 = float(np.mean(resultados_camadas[c]["it2"]))
        m3 = float(np.mean(resultados_camadas[c]["it3"]))
        resumo[c] = {
            "it1": round(m1, 4),
            "it2": round(m2, 4),
            "it3": round(m3, 4),
            "it2_it1": round(m2 / max(m1, 1e-6), 4),
            "it3_it1": round(m3 / max(m1, 1e-6), 4),
            "it3_it2": round(m3 / max(m2, 1e-6), 4)
        }
    return resumo


def comparar_base_e_fase4(caminho_base=None, caminho_fase4_hf=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = caminho_base or os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
    fase4_dir = caminho_fase4_hf or os.path.join(_ROOT, "checkpoints_vla", "fase4_hf")

    tok = AutoTokenizer.from_pretrained(base_dir)

    print("=" * 80)
    print(" [ANALISE DE COLABORACAO NAS CAMADAS EM LOOP] Qwen3Loop (K=3)")
    print(f"    Base  : {base_dir}")
    print(f"    Fase 4: {fase4_dir}")
    print("=" * 80)

    print("[1/2] Analisando Modelo Base...")
    base_stats = medir_deltas_modelo(base_dir, tok, device=device)

    fase4_stats = None
    if os.path.exists(fase4_dir):
        print("[2/2] Analisando Modelo Fase 4...")
        fase4_stats = medir_deltas_modelo(fase4_dir, tok, device=device)

    print("\n" + "=" * 80)
    print(" TABELA DE COLABORACAO NO MIOLO (Camadas 7 a 20)")
    print("=" * 80)
    print(" Camada | Base it1 | Base it2 | Base it3 | Base it2/it1 | Fase4 it1 | Fase4 it2 | Fase4 it3 | Fase4 it2/it1")
    print("-" * 95)
    for c in range(7, 21):
        b = base_stats[c]
        if fase4_stats:
            f = fase4_stats[c]
            print(f"   {c:2d}   |  {b['it1']:6.4f}  |  {b['it2']:6.4f}  |  {b['it3']:6.4f}  |    {b['it2_it1']:6.4f}    |  {f['it1']:6.4f}   |  {f['it2']:6.4f}   |  {f['it3']:6.4f}   |    {f['it2_it1']:6.4f}")
        else:
            print(f"   {c:2d}   |  {b['it1']:6.4f}  |  {b['it2']:6.4f}  |  {b['it3']:6.4f}  |    {b['it2_it1']:6.4f}    |     --    |     --    |     --    |      --")
    print("=" * 80)

    resultado = {
        "base": base_stats,
        "fase4": fase4_stats
    }
    os.makedirs(os.path.join(_ROOT, "avaliacao", "resultados_colaboracao"), exist_ok=True)
    saida_json = os.path.join(_ROOT, "avaliacao", "resultados_colaboracao", "colaboracao_loops.json")
    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)
    print(f"[OK] Dados de colaboração salvos em {saida_json}")
    return resultado


if __name__ == "__main__":
    comparar_base_e_fase4()
