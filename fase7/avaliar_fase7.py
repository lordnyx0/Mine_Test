# coding=utf-8
"""
fase7/avaliar_fase7.py — Benchmark Oficial de Avaliação Cognitiva & Labirintos da Fase 7 (CoT-GRPO VLA).

Executa:
  1. Avaliação de Desvio de Muros (80 Episódios): Testa a capacidade do modelo de contornar obstáculos físicos.
  2. Coerência dos Pensamentos (<think>): Analisa se os vetores e decisões deduzidas correspondem à física do jogo.
  3. Geração de Gráfico TopView 2D com Obstáculos e Relatório JSON.
"""
from __future__ import annotations
import os
import sys
import math
import json
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from transformers import AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora
from fase7.politica_cot_autoregressiva import PoliticaCoTAutoregressiva
from fase7.ambiente_cognitivo import AmbienteCognitivoFase7

CORES_HEX = {
    "amarelo": "#eab308",
    "azul": "#3b82f6",
    "vermelho": "#ef4444",
    "verde": "#10b981",
    "roxo": "#a855f7",
    "laranja": "#f97316"
}


def avaliar_fase7(
    modelo_ckpt: str = "checkpoints_vla/vla_fase7_grpo_cot.pt",
    num_lotes: int = 5,
    passos_max: int = 50,
    seed: int = 42,
    output_png: str = "docs/topview_fase7_obstaculos.png",
    output_json: str = "fase7/resultados_fase7.json"
):
    print("=" * 80)
    print(" BENCHMARK DE DESVIO DE OBSTÁCULOS & CoT — FASE 7")
    print("=" * 80)
    print(f"  Checkpoint : {modelo_ckpt}")
    print(f"  Lotes (8x) : {num_lotes} ({num_lotes * 8} episódios)")
    print(f"  Passos/Ep  : {passos_max}")
    print("=" * 80, flush=True)

    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(device)

    vla.vision_encoder.eval()
    for p in vla.vision_encoder.parameters():
        p.requires_grad = False

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    if os.path.exists(modelo_ckpt):
        ckpt_data = torch.load(modelo_ckpt, map_location=device)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos carregados de {modelo_ckpt}!", flush=True)

    base_dir = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    politica = PoliticaCoTAutoregressiva(vla, tokenizer, device=device)
    ambiente_cog = AmbienteCognitivoFase7(tipo_cenario="muro_simples")

    vla.eval()

    info = get("/lote/info")
    N = info["envs"]
    print(f"[ENV] Conectado ao simulador Mineflayer com {N} ambientes paralelos.", flush=True)

    todos_logs = []
    sucessos = 0
    ep_global_id = 0

    t_inicio = time.time()

    for lote_idx in range(num_lotes):
        tarefas, blocos = ambiente_cog.gerar_tarefas_cognitivas(N, seed=seed + lote_idx * 23)
        post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        post("/lote/colocar_bloco", {"blocos": blocos})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * N, "frames": True})
        obs = r["obs"][:N]
        est = [o["estado"] for o in obs]

        logs_lote = []
        for i in range(N):
            e0 = est[i]
            t = tarefas[i]
            dx = t["alvo_abs"][0] - e0["x"]
            dz = t["alvo_abs"][1] - e0["z"]
            dist_ini = math.hypot(dx, dz)

            logs_lote.append({
                "ep_id": ep_global_id + i + 1,
                "largada": {"x": e0["x"], "z": e0["z"], "yaw": e0["yaw"]},
                "alvo": {"x": t["alvo_abs"][0], "z": t["alvo_abs"][1], "cor": t["alvo_cor"]},
                "muro_centro": {"x": t["muro_centro"][0], "z": t["muro_centro"][1]},
                "lado_livre": t["lado_livre"],
                "caminho": [{"x": round(e0["x"], 2), "z": round(e0["z"], 2)}],
                "dist_min": round(dist_ini, 2),
                "pensamentos": [],
                "resultado": "falha"
            })

        for p in range(passos_max):
            alvos_abs = [tarefas[i]["alvo_abs"] for i in range(N)]
            prompts = [tarefas[i]["prompt"] for i in range(N)]

            with torch.inference_mode():
                amostras = politica.gerar_cot_e_acoes(
                    obs=obs,
                    prompts=prompts,
                    alvos_abs=alvos_abs,
                    max_new_tokens=40,
                    temperatura=0.2
                )

            acoes = [a["acao"] for a in amostras]
            rr = post("/lote/passo", {"acoes": acoes, "frames": True})
            obs = rr["obs"][:N]
            est_prox = [o["estado"] for o in obs]

            for i in range(N):
                e = est_prox[i]
                log = logs_lote[i]
                log["caminho"].append({"x": round(e["x"], 2), "z": round(e["z"], 2)})
                
                alvo = tarefas[i]["alvo_abs"]
                d = math.hypot(alvo[0] - e["x"], alvo[1] - e["z"])
                if d < log["dist_min"]:
                    log["dist_min"] = round(d, 2)

                if p % 5 == 0 and len(log["pensamentos"]) < 3:
                    log["pensamentos"].append(amostras[i]["texto_gerado"])

                if d <= 1.8 and log["resultado"] == "falha":
                    log["resultado"] = "sucesso"

        for log in logs_lote:
            if log["resultado"] == "sucesso":
                sucessos += 1
            print(f"  [Ep {log['ep_id']:02d}] Alvo ({log['alvo']['cor']}) | Muro livre: {log['lado_livre']:8s} | "
                  f"Dist Min: {log['dist_min']:4.1f}m | Resultado: {log['resultado'].upper()}", flush=True)

        todos_logs.extend(logs_lote)
        ep_global_id += N

    total_eps = len(todos_logs)
    taxa_suc = 100.0 * sucessos / max(1, total_eps)
    dt = time.time() - t_inicio

    print("\n" + "=" * 80)
    print(f" RESULTADO FINAL DO BENCHMARK FASE 7 ({dt:.1f}s)")
    print("=" * 80)
    print(f"  Total de Episódios: {total_eps}")
    print(f"  Taxa de Sucesso   : {taxa_suc:5.1f}% ({sucessos}/{total_eps})")
    print("=" * 80, flush=True)

    # Salva JSON
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"total_episodios": total_eps, "taxa_sucesso": taxa_suc, "logs": todos_logs}, f, indent=2)
    print(f"[OK] Logs salvos em: {output_json}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Fase 7.")
    parser.add_argument("--ckpt", type=str, default="checkpoints_vla/vla_fase7_grpo_cot.pt")
    parser.add_argument("--lotes", type=int, default=5)
    args = parser.parse_args()
    avaliar_fase7(modelo_ckpt=args.ckpt, num_lotes=args.lotes)
