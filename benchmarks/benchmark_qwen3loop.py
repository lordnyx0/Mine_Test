# coding=utf-8
""" Benchmark script comparing Qwen3 baseline vs Qwen3Loop (num_loops=1 vs num_loops=2) """

import gc
import sys
import time
import os
import psutil
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qwen3loop import Qwen3LoopConfig, Qwen3LoopForCausalLM


def get_memory_usage():
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return ram_mb, vram_mb


def measure_perplexity(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(loss)).item()
    return loss, perplexity


def run_benchmark():
    print("=" * 60)
    print("      Qwen3Loop Architectural Benchmark & Verification")
    print("=" * 60)

    # Base configuration for testing: 8 layers, 1024 hidden size
    base_config = dict(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=2048,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    batch_size = 2
    prompt_len = 128
    gen_tokens = 50
    input_ids = torch.randint(100, 10000, (batch_size, prompt_len)).to(device)

    results = []

    for num_loops in [1, 2, 3]:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        config = Qwen3LoopConfig(**base_config, num_loops=num_loops)
        model = Qwen3LoopForCausalLM(config).to(device)
        model.eval()

        ram_init, vram_init = get_memory_usage()

        # Measure loss & initial perplexity
        loss, perplexity = measure_perplexity(model, input_ids)

        # Warmup
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)

        # Measure generation speed
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)
        elapsed = time.time() - start_time

        total_gen_tokens = batch_size * gen_tokens
        tps = total_gen_tokens / elapsed

        ram_end, vram_end = get_memory_usage()

        results.append(
            {
                "num_loops": num_loops,
                "tps": tps,
                "elapsed": elapsed,
                "ram_mb": ram_end,
                "vram_mb": vram_end,
                "loss": loss,
                "perplexity": perplexity,
            }
        )

        print(f"\n--- Benchmark Results for Qwen3Loop (num_loops={num_loops}) ---")
        print(f"  Tokens/Sec: {tps:.2f} t/s")
        print(f"  Elapsed Time: {elapsed:.4f} s")
        print(f"  RAM Usage: {ram_end:.2f} MB")
        print(f"  VRAM Usage: {vram_end:.2f} MB")
        print(f"  Initial Loss: {loss:.4f}")
        print(f"  Initial Perplexity: {perplexity:.4f}")

    print("\n" + "=" * 60)
    print("Summary Table:")
    print(f"{'Loops':<6} | {'Tokens/s':<10} | {'RAM (MB)':<10} | {'VRAM (MB)':<10} | {'Loss':<8} | {'PPL':<8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['num_loops']:<6} | {r['tps']:<10.2f} | {r['ram_mb']:<10.1f} | {r['vram_mb']:<10.1f} | {r['loss']:<8.4f} | {r['perplexity']:<8.2f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
