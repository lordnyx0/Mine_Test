# coding=utf-8
"""
bench_gguf.py — Benchmark Oficial de Raciocínio em GGUF (True Loop / 28L / Q8_0).

Executa o benchmark oficial de 97 itens contra o llama-server.exe compilado com os
patches de C++/CUDA (LoopSplit e TENSOR_DUPLICATED), preservando 100% da economia de VRAM.

Uso:
    python avaliacao/bench_gguf.py --modelo models_gguf/fase4_loop_q8_0.gguf --nome fase4_loop_q8
    python avaliacao/bench_gguf.py --modelo models_gguf/fase4_loop_q8_0.gguf --nome fase4_raciocinio --categorias reasoning mathematics
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

BIN = r"C:\Users\Nyx\.unsloth\.staging\llama.cpp.staging-eu_6bjrp\build-qwen3loop\bin\Release"
SERVER = os.path.join(BIN, "llama-server.exe")
RESULTADOS = os.path.join(ROOT_DIR, "avaliacao", "results_gguf_bench")


def porta_livre() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def espera_servidor(porta: int, proc: subprocess.Popen, timeout: float = 180.0) -> None:
    alvo = f"http://127.0.0.1:{porta}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server morreu na largada (codigo {proc.returncode})")
        try:
            with urllib.request.urlopen(alvo, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError("llama-server nao respondeu /health a tempo")


def completa(porta: int, prompt: str, n_predict: int) -> dict:
    corpo = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,     # greedy, determinístico
        "top_k": 1,
        "seed": 42,
        "cache_prompt": False,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{porta}/completion", data=corpo,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    p = argparse.ArgumentParser(description="Benchmark GGUF Q8_0 True Loop")
    p.add_argument("--modelo", default="models_gguf/fase4_loop_q8_0.gguf", help="caminho do .gguf")
    p.add_argument("--nome", default="fase4_loop_q8", help="rotulo nos resultados")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--categorias", nargs="+", default=None)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--saida", default=RESULTADOS)
    a = p.parse_args()

    modelo_abs = os.path.abspath(a.modelo) if not os.path.isabs(a.modelo) else a.modelo
    saida_abs = os.path.abspath(a.saida) if not os.path.isabs(a.saida) else a.saida
    if not os.path.isfile(modelo_abs):
        sys.exit(f"\n[ERRO] GGUF nao encontrado: {modelo_abs}\n")

    from transformers import AutoTokenizer
    from evaluation.benchmark import load_benchmark
    from evaluation.config import load_eval_config
    from evaluation.generation import render_prompt, strip_thinking
    from evaluation.scoring import score_response
    from evaluation.types import GenerationRecord

    cfg_path = os.path.join(ROOT_DIR, "eval_config.yaml")
    bench_path = os.path.join(ROOT_DIR, "benchmarks", "eval_benchmark.json")
    
    cfg = load_eval_config(cfg_path)
    itens = load_benchmark(bench_path, tuple(a.categorias) if a.categorias else None)
    if a.limit:
        itens = itens[:a.limit]

    # Carrega tokenizer local
    base_tok_dir = os.path.join(ROOT_DIR, "checkpoints_vla", "backbone_base")
    if not os.path.exists(base_tok_dir):
        base_tok_dir = cfg.model.base_model_id
    tok = AutoTokenizer.from_pretrained(base_tok_dir)

    destino = os.path.join(saida_abs, a.nome)
    os.makedirs(destino, exist_ok=True)
    caminho_jsonl = os.path.join(destino, "responses.jsonl")

    porta = porta_livre()
    cmd = [SERVER, "-m", modelo_abs, "-ngl", "99",
           "-c", str(a.ctx), "--port", str(porta), "--host", "127.0.0.1",
           "-np", "1", "--no-webui"]
    print("=" * 78)
    print(f" BENCHMARK GGUF TRUE LOOP — {a.nome}")
    print("=" * 78)
    print(f"  modelo : {modelo_abs}")
    print(f"  itens  : {len(itens)}")
    print(f"  porta  : {porta}")
    print("=" * 78, flush=True)

    log = open(os.path.join(destino, "server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=BIN)

    por_cat: dict[str, list[float]] = defaultdict(list)
    todos: list[float] = []
    t_ini = time.time()
    tok_total = 0

    try:
        espera_servidor(porta, proc)
        print(f"[servidor pronto em {time.time()-t_ini:.1f}s]\n", flush=True)

        with open(caminho_jsonl, "w", encoding="utf-8") as f:
            for idx, item in enumerate(itens, 1):
                n_pred = (a.max_tokens or
                          cfg.generation.max_new_tokens_for(item.category, item.max_new_tokens))
                prompt = render_prompt(tok, item, cfg.model.enable_thinking)

                t0 = time.perf_counter()
                try:
                    r = completa(porta, prompt, n_pred)
                    dt = time.perf_counter() - t0
                    texto = strip_thinking(r.get("content", ""))
                    n_gerado = int(r.get("tokens_predicted", 0) or 0)
                    erro = None
                except Exception as exc:
                    dt = time.perf_counter() - t0
                    texto, n_gerado, erro = "", 0, f"{type(exc).__name__}: {exc}"

                rec = GenerationRecord(
                    checkpoint=a.nome, category=item.category, question_id=item.id,
                    prompt=item.prompt, response=texto, generation_time=dt,
                    tokens_generated=n_gerado,
                    prompt_tokens=int(r.get("tokens_evaluated", 0) or 0) if not erro else 0,
                    temperature=0.0, top_p=1.0, seed=42,
                    truncated=n_gerado >= n_pred,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    error=erro)
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                f.flush()

                s = score_response(item, rec)
                por_cat[item.category].append(s.score)
                todos.append(s.score)
                tok_total += n_gerado

                marca = "OK " if s.score == 1.0 else ("~  " if s.score > 0 else "X  ")
                tps = n_gerado / max(1e-6, dt)
                print(f"[{idx:>3}/{len(itens)}] {marca} {item.id:<9} {item.category:<22} "
                      f"{s.score:.2f} | {n_gerado:>3} tok {tps:>6.1f} t/s | "
                      f"acum {sum(todos)/len(todos)*100:5.1f}%", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    dt_tot = time.time() - t_ini
    print()
    print("=" * 78)
    print(f" {a.nome} — {len(todos)} itens em {dt_tot/60:.1f} min "
          f"({tok_total} tokens, {tok_total/max(1e-6, dt_tot):.1f} t/s medio)")
    print("=" * 78)
    for c in sorted(por_cat):
        v = por_cat[c]
        print(f"  {c:<24}{sum(v)/len(v)*100:>6.1f}%  (n={len(v)})")
    print("-" * 78)
    print(f"  {'GERAL':<24}{sum(todos)/len(todos)*100:>6.1f}%")
    print("=" * 78)

    json.dump({"modelo": a.modelo, "nome": a.nome,
               "geral": sum(todos) / len(todos),
               "por_categoria": {c: sum(v) / len(v) for c, v in por_cat.items()},
               "n": len(todos), "minutos": dt_tot / 60, "tokens": tok_total},
              open(os.path.join(destino, "resumo.json"), "w", encoding="utf-8"), indent=2)
    print(f"\n[OK] Resultados salvos em:\n - {caminho_jsonl}\n - {os.path.join(destino, 'resumo.json')}\n")


if __name__ == "__main__":
    main()
