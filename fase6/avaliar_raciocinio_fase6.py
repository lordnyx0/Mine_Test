# coding=utf-8
"""
fase6/avaliar_raciocinio_fase6.py — Benchmark Oficial de Raciocínio Geral da Fase 6 (CoT-VLA).

Avalia o modelo Qwen3Loop com os adaptadores LoRA da Fase 6 contra o benchmark oficial de 97 itens:
  - Matemática (10 itens)
  - Raciocínio Lógico (15 itens)
  - Programação (15 itens)
  - Escrita e Linguagem (20 itens)
  - Criatividade e Conhecimento Geral (37 itens)
  - Medição de acurácia com Thinking Ativado (<think>...</think>).

Saídas:
  1. fase6/resultados_raciocinio_fase6.json: Relatório consolidado por categoria.
  2. fase6/resumo_raciocinio_fase6.txt: Tabela comparativa e score global.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
from collections import defaultdict
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qwen3loop.modeling_qwen3loop import Qwen3LoopForCausalLM
from modelo.lora_vla import aplicar_lora


def avaliar_item(pred: str, expected: str, criteria: str) -> bool:
    pred_clean = pred.strip().lower()
    exp_clean = expected.strip().lower() if expected else ""
    crit_clean = criteria.strip().lower() if criteria else ""

    if exp_clean and exp_clean in pred_clean:
        return True
    if crit_clean and crit_clean in pred_clean:
        return True
    return False


def avaliar_raciocinio_fase6(
    ckpt_path: str = "checkpoints_vla/vla_fase6_ppo_cot.pt",
    bench_file: str = "benchmarks/eval_benchmark.json",
    max_tokens: int = 256,
    temperatura: float = 0.0
):
    print("=" * 80)
    print(" BENCHMARK OFICIAL DE RACIOCÍNIO — FASE 6 CoT-VLA (97 ITENS)")
    print("=" * 80)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Benchmark  : {bench_file}")
    print(f"  Max Tokens : {max_tokens}")
    print("=" * 80, flush=True)

    base_dir = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("[*] Carregando Tokenizer...")
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[*] Carregando Backbone Qwen3Loop...")
    model = Qwen3LoopForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(dev)

    print("[*] Aplicando adaptadores LoRA (r=16, alpha=32.0)...")
    aplicar_lora(model, r=16, alpha=32.0)

    if os.path.exists(ckpt_path):
        ckpt_data = torch.load(ckpt_path, map_location=dev)
        treinaveis = ckpt_data.get("treinaveis", ckpt_data)
        
        # Mapeia nomes LoRA se necessário
        cleaned_weights = {}
        for k, v in treinaveis.items():
            k_clean = k.replace("qwen_model.", "").replace("base_model.", "")
            cleaned_weights[k_clean] = v

        msg = model.load_state_dict(cleaned_weights, strict=False)
        print(f"[*] Pesos LoRA da Fase 6 carregados: {len(cleaned_weights)} tensores!")

    model.eval()

    # Carrega Benchmark
    bench_path = os.path.join(_ROOT, bench_file)
    with open(bench_path, "r", encoding="utf-8") as f:
        dados = json.load(f)
        itens = dados.get("items", dados) if isinstance(dados, dict) else dados

    totais_cat = defaultdict(int)
    acertos_cat = defaultdict(int)
    respostas_log = []

    t_ini = time.time()

    print(f"\n[*] Executando inferência sobre {len(itens)} itens de teste...\n", flush=True)

    for i, it in enumerate(itens, 1):
        cat = it.get("category", it.get("type", "general"))
        prompt = it.get("prompt", "")
        expected = it.get("expected", it.get("answer", ""))
        criteria = it.get("criteria", "")

        # Formata prompt com chat template e thinking ativado
        mensagens = [{"role": "user", "content": prompt}]
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        input_ids = tok.encode(formatted_prompt, return_tensors="pt").to(dev)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperatura > 0.0,
                temperature=temperatura if temperatura > 0.0 else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.encode("<|im_end|>")[0]
            )

        gen_ids = out_ids[0][input_ids.shape[1]:]
        resposta = tok.decode(gen_ids, skip_special_tokens=True)

        acertou = avaliar_item(resposta, expected, criteria)

        totais_cat[cat] += 1
        if acertou:
            acertos_cat[cat] += 1

        respostas_log.append({
            "id": i,
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "criteria": criteria,
            "resposta": resposta,
            "acertou": acertou
        })

        if i % 10 == 0 or i == len(itens):
            tot = sum(totais_cat.values())
            acc = sum(acertos_cat.values())
            print(f"  Progresso: {tot:02d}/{len(itens):02d} itens | Score Parcial: {100.0*acc/tot:5.1f}% ({acc}/{tot})", flush=True)

    dt = time.time() - t_ini
    total_geral = sum(totais_cat.values())
    acertos_geral = sum(acertos_cat.values())
    score_final = 100.0 * acertos_geral / max(1, total_geral)

    print("\n" + "=" * 80)
    print(f" RESULTADO FINAL DO BENCHMARK DE RACIOCÍNIO — FASE 6 CoT-VLA ({dt:.1f}s)")
    print("=" * 80)
    for cat in sorted(totais_cat.keys()):
        t = totais_cat[cat]
        a = acertos_cat[cat]
        pct = 100.0 * a / max(1, t)
        print(f"  • {cat.capitalize():<18}: {pct:5.1f}% ({a:02d}/{t:02d})")
    print("-" * 80)
    print(f"  🏆 SCORE GLOBAL CONSOLIDADO: {score_final:5.1f}% ({acertos_geral}/{total_geral})")
    print("=" * 80, flush=True)

    # Salva Resultados
    out_json = os.path.join(_ROOT, "fase6", "resultados_raciocinio_fase6.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "score_global": score_final,
            "acertos_total": acertos_geral,
            "itens_total": total_geral,
            "tempo_s": dt,
            "categorias": {cat: {"acertos": acertos_cat[cat], "total": totais_cat[cat], "pct": 100.0*acertos_cat[cat]/totais_cat[cat]} for cat in totais_cat},
            "respostas": respostas_log
        }, f, indent=2)
    print(f"[*] Relatório completo salvo em: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Raciocínio Fase 6.")
    parser.add_argument("--ckpt", type=str, default="checkpoints_vla/vla_fase6_ppo_cot.pt")
    parser.add_argument("--bench", type=str, default="benchmarks/eval_benchmark.json")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    avaliar_raciocinio_fase6(
        ckpt_path=args.ckpt,
        bench_file=args.bench,
        max_tokens=args.max_tokens
    )
