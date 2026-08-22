# coding=utf-8
"""
executar_benchmark.py — Executa a bateria de avaliação de raciocínio lógico no modelo GGUF Q8.
"""
import os
import sys
import argparse
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def rodar_benchmark(
    modelo_nome: str = "fase6_loop_q8_pensamento",
    porta: int = 8085,
    limite: int = 0
):
    print("=" * 80)
    print(" 🧪 AVALIAÇÃO DE RACIOCÍNIO LÓGICO & BENCHMARK Q8")
    print(f"    Nome do Teste : {modelo_nome}")
    print(f"    Porta Servidor: {porta}")
    print(f"    Limite Itens  : {'Todos' if limite == 0 else limite}")
    print("=" * 80)

    bench_script = os.path.join(_ROOT, "avaliacao", "bench_gguf.py")
    saida_dir = os.path.join(_ROOT, "avaliacao", "results_gguf_bench", modelo_nome)

    cmd = [
        sys.executable,
        bench_script,
        "--name", modelo_nome,
        "--output_dir", saida_dir,
        "--port", str(porta)
    ]
    if limite > 0:
        cmd.extend(["--limit", str(limite)])

    print(f"[*] Executando: {' '.join(cmd)}")
    subprocess.run(cmd)

    # Avalia as respostas geradas
    eval_script = os.path.join(_ROOT, "avaliacao", "avaliar_logica_testes.py")
    jsonl_path = os.path.join(saida_dir, "responses.jsonl")
    if os.path.exists(jsonl_path):
        print(f"\n[*] Tabulando notas e comparando com o Modelo Professor Base (76.9%)...")
        cmd_eval = [sys.executable, eval_script, "--jsonl", jsonl_path]
        subprocess.run(cmd_eval)
    else:
        print(f"[AVISO] Arquivo {jsonl_path} não gerado. Verifique se o servidor LLM na porta {porta} está ativo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="modelo_sft_q8", help="Identificador do teste")
    parser.add_argument("--port", type=int, default=8085, help="Porta do servidor LLM")
    parser.add_argument("--limit", type=int, default=0, help="Limitar número de questões (0 = todas)")
    args = parser.parse_args()

    rodar_benchmark(modelo_nome=args.name, porta=args.port, limite=args.limit)
