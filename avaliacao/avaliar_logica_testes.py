# coding=utf-8
"""
avaliar_logica_testes.py — Benchmark Oficial de Raciocínio Lógico e Matemática (Auto-contido).

Implementa a metodologia com suporte a:
1. Modo Direto (PyTorch bfloat16 na GPU)
2. Modo GGUF True Loop (via llama-server.exe com 125+ tok/s e VRAM 604 MiB)
3. Scorer oficial de 97 itens contra `benchmarks/eval_benchmark.json`
"""
import os
import sys
import json
import argparse
import time
import subprocess
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR_ADAPTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIR_ADAPTER not in sys.path:
    sys.path.insert(0, DIR_ADAPTER)

from infra.run_vla_agent import load_vla_agent
from qwen3loop import Qwen3LoopForCausalLM
from transformers import AutoTokenizer

from evaluation.benchmark import load_benchmark
from evaluation.config import load_eval_config
from evaluation.generation import render_prompt, strip_thinking
from evaluation.scoring import score_response
from evaluation.types import GenerationRecord


def avaliar_checkpoint_corrigido(ckpt_path="checkpoints_vla/vla_fase4_merged.pt",
                                  categorias=("reasoning", "mathematics", "programming", "general_knowledge"),
                                  saida_json="avaliacao/resultados_raciocinio_fase4.json",
                                  saida_jsonl="avaliacao/responses_fase4.jsonl",
                                  modo="direto"):
    print("=" * 80)
    print(f" [BENCHMARK DE RACIOCINIO] PIPELINE LOCAL (MODO: {modo.upper()})")
    print(f"    Checkpoint:  {ckpt_path}")
    print(f"    Categorias:  {categorias}")
    print("=" * 80)

    if modo == "gguf":
        modelo_gguf = os.path.join(DIR_ADAPTER, "models_gguf", "fase4_loop_q8_0.gguf")
        if not os.path.exists(modelo_gguf):
            print(f"[AVISO] GGUF local '{modelo_gguf}' nao encontrado. Usando fallback de Testes2.")
            modelo_gguf = r"C:\Users\Nyx\Desktop\Testes\Testes2\models_gguf\fase4_loop_q8_0.gguf"
        
        cmd = [
            sys.executable,
            os.path.join(DIR_ADAPTER, "avaliacao", "bench_gguf.py"),
            "--modelo", modelo_gguf,
            "--nome", "fase4_loop_q8",
        ]
        if categorias:
            cmd.extend(["--categorias", *categorias])
        subprocess.run(cmd, check=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_file = os.path.join(DIR_ADAPTER, "eval_config.yaml")
    bench_file = os.path.join(DIR_ADAPTER, "benchmarks", "eval_benchmark.json")
    backbone_dir = os.path.join(DIR_ADAPTER, "checkpoints_vla", "backbone_base")
    
    if not os.path.exists(cfg_file):
        cfg_file = r"C:\Users\Nyx\Desktop\Testes\eval_config.yaml"
    if not os.path.exists(bench_file):
        bench_file = r"C:\Users\Nyx\Desktop\Testes\benchmarks\eval_benchmark.json"
    if not os.path.exists(backbone_dir):
        backbone_dir = r"C:\Users\Nyx\Desktop\Testes\checkpoints\qwen3loop_v2\final_model"

    cfg = load_eval_config(cfg_file)
    
    # 1. Carrega benchmark oficial
    itens = load_benchmark(bench_file, tuple(categorias) if categorias else None)
    print(f"[1/4] Carregados {len(itens)} itens para avaliacao.")

    # 2. Carrega modelo Qwen3LoopForCausalLM
    print(f"[2/4] Carregando pesos de '{ckpt_path}'...")
    if os.path.isdir(ckpt_path):
        causal_lm = Qwen3LoopForCausalLM.from_pretrained(
            ckpt_path,
            dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    else:
        vla, _ = load_vla_agent(ckpt_path if ckpt_path != "base" else None)
        causal_lm = Qwen3LoopForCausalLM.from_pretrained(
            backbone_dir,
            dtype=torch.bfloat16,
            device_map=device
        )
        if ckpt_path != "base":
            causal_lm.model.load_state_dict(vla.qwen_model.state_dict(), strict=False)
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir)

    causal_lm.eval()
    eos_id = tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id

    # 3. Execução das gerações com Jinja + Stop Tokens + Adaptive Token Cap
    print("\n[3/4] Gerando respostas com renderizacao Jinja e limites adaptativos...")
    registros = []
    por_categoria = {}

    for i, item in enumerate(itens, 1):
        prompt_formatado = render_prompt(tokenizer, item, enable_thinking=True)
        max_tokens = cfg.generation.max_new_tokens_for(item.category, item.max_new_tokens)
        
        inputs = tokenizer(prompt_formatado, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = causal_lm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id or eos_id,
                repetition_penalty=1.1
            )
        elapsed = time.perf_counter() - t0

        gerado_tokens = outputs[0][prompt_len:]
        n_tokens = len(gerado_tokens)
        tok_s = n_tokens / max(elapsed, 0.001)

        raw_text = tokenizer.decode(gerado_tokens, skip_special_tokens=True).strip()
        clean_text = strip_thinking(raw_text)

        rec = GenerationRecord(
            checkpoint="fase4",
            category=item.category,
            question_id=item.id,
            prompt=item.prompt,
            response=clean_text,
            generation_time=elapsed,
            tokens_generated=n_tokens,
            prompt_tokens=prompt_len,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            timestamp=""
        )
        score_res = score_response(item, rec)
        nota = float(score_res.score)

        if item.category not in por_categoria:
            por_categoria[item.category] = []
        por_categoria[item.category].append(nota)

        status = "PASS 1.0" if nota >= 0.99 else (f"PARTIAL ({nota:.2f})" if nota > 0.0 else "FAIL 0.0")
        print(f"  [{item.category[:4].upper()}] #{i:02d} ({item.id}) -> {status} ({elapsed:.2f}s, {tok_s:.1f} t/s)", flush=True)

        registros.append({
            "question_id": item.id,
            "category": item.category,
            "prompt": item.prompt,
            "score": nota,
            "clean_response": clean_text,
            "raw_response_with_think": raw_text,
            "elapsed_seconds": elapsed,
            "tokens_generated": n_tokens
        })

    # 4. Salva resultados estruturados
    os.makedirs(os.path.dirname(saida_json), exist_ok=True)
    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump(registros, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(saida_jsonl), exist_ok=True)
    with open(saida_jsonl, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. Tabela Resumo
    print("\n" + "=" * 80)
    print(" [RESUMO DO BENCHMARK DE RACIOCINIO]")
    print("=" * 80)
    total_pontos = 0
    total_itens = 0
    for cat, scores in por_categoria.items():
        media_cat = sum(scores) / len(scores) * 100.0
        total_pontos += sum(scores)
        total_itens += len(scores)
        print(f"  - {cat:<22}: {media_cat:5.1f}%  ({sum(scores):.1f}/{len(scores)} pontos)")
    
    media_geral = (total_pontos / total_itens * 100.0) if total_itens > 0 else 0.0
    print("-" * 80)
    print(f"  MEDIA GERAL              : {media_geral:5.1f}%  ({total_pontos:.1f}/{total_itens} pontos)")
    print(f"\n[OK] Resultados salvos em:")
    print(f"   - {saida_json}")
    print(f"   - {saida_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase4_merged.pt")
    ap.add_argument("--modo", choices=["direto", "gguf"], default="direto")
    ap.add_argument("--categorias", nargs="+", default=["reasoning", "mathematics", "programming", "general_knowledge"])
    args = ap.parse_args()
    avaliar_checkpoint_corrigido(ckpt_path=args.ckpt, modo=args.modo, categorias=args.categorias)
