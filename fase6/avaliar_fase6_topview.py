# coding=utf-8
"""
fase6/avaliar_fase6_topview.py — Benchmark Oficial de Avaliação TopView 2D da Fase 6 (CoT-VLA).

Executa bateria de avaliação com o modelo Fase 6 CoT-VLA PPO no simulador do Minecraft:
  - 80 Episódios de teste cego em 2 e 3 Pilares com ângulos de até 180°.
  - Rastro passo a passo com coordenadas 3D (X, Y, Z, Yaw, Ações).
  - Taxa de sucesso na Submeta 1, Submeta 2 e Sucesso Global.
  - Geração de gráficos TopView 2D e relatório HTML interativo.

Saídas:
  1. fase6/resultados_topview_cot_vla.json: Dados estruturados de todos os episódios.
  2. docs/topview_fase6_cot_vla.png: Painel 2D Top-Down com o traçado das trajetórias.
  3. fase6/relatorio_topview_fase6.html: Painel interativo com visualização no Canvas.
"""
import os
import sys
import math
import json
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from politica.cerebro import PoliticaCerebroVLA
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora
from fase5.curriculo_fase5 import CurriculoFase5, CORES_MAP
from fase5.acoes_taticas import MODOS, YAW_BINS_9

CORES_HEX = {
    "amarelo": "#eab308",
    "azul": "#3b82f6",
    "vermelho": "#ef4444",
    "verde": "#10b981",
    "roxo": "#a855f7",
    "laranja": "#f97316"
}


def avaliar(
    modelo_ckpt: str = "checkpoints_vla/vla_fase6_ppo_cot.pt",
    num_lotes: int = 10,
    passos_max: int = 85,
    usar_cerebro: bool = True,
    seed: int = 42,
    amostrar: bool = False,
    temperatura: float = 0.0,
    raio_chegada: float = 2.0,
    estagio_curriculo: str = "B",
    output_png: str = "docs/topview_fase6_cot_vla.png",
    output_json: str = "fase6/resultados_topview_cot_vla.json",
    output_html: str = "fase6/relatorio_topview_fase6.html"
):
    print("=" * 80)
    print(" BENCHMARK TOPVIEW 2D — FASE 6 CoT-VLA NO MINECRAFT")
    print("=" * 80)
    print(f"  Checkpoint : {modelo_ckpt}")
    print(f"  Lotes (8x) : {num_lotes} ({num_lotes * 8} episódios)")
    print(f"  Passos     : {passos_max} por submeta")
    print(f"  Estágio    : {estagio_curriculo}")
    print(f"  Cérebro    : {'Ativo (Laser Sprint + Transição 360°)' if usar_cerebro else 'Desativado'}")
    print("=" * 80, flush=True)

    curriculo = CurriculoFase5(modo_estagio=estagio_curriculo)
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
            msg = vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos carregados de {modelo_ckpt} ({len(ckpt_data['treinaveis'])} tensores)!", flush=True)

    pol_vla = PoliticaRaciocinioLoop(None, amostrar=amostrar, device=device, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)
    pol = PoliticaCerebroVLA(pol_vla) if usar_cerebro else pol_vla
    pol.amostrar = amostrar
    pol.temperatura = temperatura

    vla.eval()

    info = get("/lote/info")
    N = info["envs"]
    print(f"[ENV] Conectado ao simulador Mineflayer com {N} ambientes paralelos.", flush=True)

    todos_logs = []
    sucessos_totais = 0
    sucessos_parciais = 0
    falhas = 0
    ep_global_id = 0

    t_inicio = time.time()

    for lote_idx in range(num_lotes):
        tarefas, blocos_tarefas = curriculo.gerar_tarefas(N, seed=seed + lote_idx * 17)
        post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        post("/lote/colocar_bloco", {"blocos": blocos_tarefas})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * N, "frames": True})
        obs = r["obs"][:N]
        est = [o["estado"] for o in obs]
        pol.reiniciar(obs)

        for i, t in enumerate(tarefas):
            t["id_ep"] = ep_global_id + i + 1

        estagio_atual = [0] * N
        vivo = [True] * N

        logs_lote = []
        for i in range(N):
            e0 = est[i]
            tar = tarefas[i]
            p1 = tar["estagios"][0]
            p2 = tar["estagios"][1] if len(tar["estagios"]) > 1 else None

            dx1 = p1["alvo_abs"][0] - e0["x"]
            dz1 = p1["alvo_abs"][1] - e0["z"]
            dist1 = math.hypot(dx1, dz1)

            dx2 = (p2["alvo_abs"][0] - e0["x"]) if p2 else 0.0
            dz2 = (p2["alvo_abs"][1] - e0["z"]) if p2 else 0.0
            dist2 = math.hypot(dx2, dz2) if p2 else 0.0

            logs_lote.append({
                "ep_id": ep_global_id + i + 1,
                "env": i,
                "prompt": tar["prompt"],
                "largada": {"x": e0["x"], "y": e0["y"], "z": e0["z"], "yaw": e0["yaw"]},
                "pilar1": {"cor": p1["cor"], "x": p1["alvo_abs"][0], "z": p1["alvo_abs"][1], "dist_ini": round(dist1, 2)},
                "pilar2": {"cor": p2["cor"], "x": p2["alvo_abs"][0], "z": p2["alvo_abs"][1], "dist_ini": round(dist2, 2)} if p2 else None,
                "caminho": [{"passo": 0, "x": round(e0["x"], 2), "y": round(e0["y"], 2), "z": round(e0["z"], 2), "yaw": round(e0["yaw"], 1), "acao": []}],
                "resultado": "falha",
                "dist_min_pilar1": round(dist1, 2),
                "dist_min_pilar2": round(dist2, 2) if p2 else 999.0,
                "passo_pilar1": None,
                "passo_pilar2": None
            })

        max_estagios_lote = max(len(t["estagios"]) for t in tarefas)
        limite_passos_total = passos_max * max_estagios_lote

        for p in range(limite_passos_total):
            prompts_ativos = [
                tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)].get("prompt_estagio", tarefas[i]["prompt"])
                for i in range(N)
            ]
            alvos_ativos_abs = [tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)]["alvo_abs"] for i in range(N)]

            with torch.inference_mode():
                acoes = pol.agir(est, alvos_ativos_abs, obs, prompts=prompts_ativos, estagios=estagio_atual)

            for i in range(N):
                if not vivo[i]:
                    acoes[i] = {"hold": [], "mouse": [0, 0], "duration_ms": 50}

            rr = post("/lote/passo", {"acoes": acoes, "frames": True})
            obs = rr["obs"][:N]
            est = [o["estado"] for o in obs]
            pol.observar(obs)

            for i in range(N):
                if not vivo[i]:
                    continue

                e = est[i]
                log = logs_lote[i]
                tar = tarefas[i]

                log["caminho"].append({
                    "passo": p + 1,
                    "x": round(e["x"], 2),
                    "y": round(e["y"], 2),
                    "z": round(e["z"], 2),
                    "yaw": round(e["yaw"], 1),
                    "acao": acoes[i].get("hold", [])
                })

                p1_pos = tar["estagios"][0]["alvo_abs"]
                d1 = math.hypot(p1_pos[0] - e["x"], p1_pos[1] - e["z"])
                if d1 < log["dist_min_pilar1"]:
                    log["dist_min_pilar1"] = round(d1, 2)

                if len(tar["estagios"]) > 1:
                    p2_pos = tar["estagios"][1]["alvo_abs"]
                    d2 = math.hypot(p2_pos[0] - e["x"], p2_pos[1] - e["z"])
                    if d2 < log["dist_min_pilar2"]:
                        log["dist_min_pilar2"] = round(d2, 2)

                # Verifica se atingiu Submeta 1
                if estagio_atual[i] == 0 and d1 <= raio_chegada:
                    estagio_atual[i] = 1
                    log["passo_pilar1"] = p + 1
                    if len(tar["estagios"]) == 1:
                        log["resultado"] = "sucesso_total"
                        vivo[i] = False
                    else:
                        log["resultado"] = "sucesso_parcial"

                # Verifica se atingiu Submeta 2
                elif estagio_atual[i] == 1 and len(tar["estagios"]) > 1:
                    p2_pos = tar["estagios"][1]["alvo_abs"]
                    d2 = math.hypot(p2_pos[0] - e["x"], p2_pos[1] - e["z"])
                    if d2 <= raio_chegada:
                        log["resultado"] = "sucesso_total"
                        log["passo_pilar2"] = p + 1
                        vivo[i] = False

            if not any(vivo):
                break

        for log in logs_lote:
            if log["resultado"] == "sucesso_total":
                sucessos_totais += 1
            elif log["resultado"] == "sucesso_parcial":
                sucessos_parciais += 1
            else:
                falhas += 1

            s1 = "OK" if log["passo_pilar1"] is not None else "--"
            s2 = "OK" if log["passo_pilar2"] is not None else "--"
            print(f"  [Ep {log['ep_id']:02d}] Sub1 ({log['pilar1']['cor']}): {s1} (min={log['dist_min_pilar1']:.1f}m) | "
                  f"Sub2 ({log['pilar2']['cor'] if log['pilar2'] else 'N/A'}): {s2} | Resultado: {log['resultado'].upper()}", flush=True)

        todos_logs.extend(logs_lote)
        ep_global_id += N

    total_eps = len(todos_logs)
    taxa_s1 = 100.0 * (sucessos_totais + sucessos_parciais) / max(1, total_eps)
    taxa_tot = 100.0 * sucessos_totais / max(1, total_eps)
    dt = time.time() - t_inicio

    print("\n" + "=" * 80)
    print(f" RESULTADO FINAL DO BENCHMARK TOPVIEW 2D ({dt:.1f}s)")
    print("=" * 80)
    print(f"  Total de Episódios : {total_eps}")
    print(f"  Submeta 1 (Pilar 1): {taxa_s1:5.1f}% ({sucessos_totais + sucessos_parciais}/{total_eps})")
    print(f"  Sucesso Total      : {taxa_tot:5.1f}% ({sucessos_totais}/{total_eps})")
    print(f"  Falhas             : {100.0 * falhas / max(1, total_eps):5.1f}% ({falhas}/{total_eps})")
    print("=" * 80, flush=True)

    # Salva JSON
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_episodios": total_eps,
            "taxa_submeta1": taxa_s1,
            "taxa_sucesso_total": taxa_tot,
            "tempo_s": dt,
            "logs": todos_logs
        }, f, indent=2)
    print(f"[OK] JSON com todos os logs salvo em: {output_json}", flush=True)

    # Gera Imagem TopView
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    gerar_grafico_topview(todos_logs, output_png, taxa_s1, taxa_tot, raio_chegada=raio_chegada)


def gerar_grafico_topview(logs, output_png, taxa_s1, taxa_tot, raio_chegada=2.0):
    num_eps = len(logs)
    cols = 8
    rows = math.ceil(num_eps / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.8))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, log in enumerate(logs):
        ax = axes[idx]
        xs = [p["x"] for p in log["caminho"]]
        zs = [p["z"] for p in log["caminho"]]

        p1 = log["pilar1"]
        p2 = log["pilar2"]

        c1 = Circle((p1["x"], p1["z"]), raio_chegada, color=CORES_HEX.get(p1["cor"], "#eab308"), alpha=0.3)
        ax.add_patch(c1)
        ax.plot(p1["x"], p1["z"], "o", color=CORES_HEX.get(p1["cor"], "#eab308"), markersize=7)

        if p2:
            c2 = Circle((p2["x"], p2["z"]), raio_chegada, color=CORES_HEX.get(p2["cor"], "#a855f7"), alpha=0.3)
            ax.add_patch(c2)
            ax.plot(p2["x"], p2["z"], "o", color=CORES_HEX.get(p2["cor"], "#a855f7"), markersize=7)

        cor_rastro = "#10b981" if log["resultado"] == "sucesso_total" else ("#f59e0b" if log["resultado"] == "sucesso_parcial" else "#ef4444")
        ax.plot(xs, zs, "-", color=cor_rastro, linewidth=2)
        ax.plot(xs[0], zs[0], "s", color="#3b82f6", markersize=5)
        ax.plot(xs[-1], zs[-1], "x", color="#000000", markersize=6)

        ax.set_title(f"Ep {log['ep_id']} [{log['resultado'][:3].upper()}]", fontsize=8, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal", "datalim")

    for j in range(num_eps, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Fase 6 CoT-VLA: TopView 2D | Submeta 1: {taxa_s1:.1f}% | Sucesso Total: {taxa_tot:.1f}% (80 Episódios)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"[OK] Gráfico TopView 2D salvo em: {output_png}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TopView 2D Fase 6.")
    parser.add_argument("--ckpt", type=str, default="checkpoints_vla/vla_fase6_ppo_cot.pt")
    parser.add_argument("--lotes", type=int, default=10)
    parser.add_argument("--passos", type=int, default=85)
    parser.add_argument("--estagio", type=str, default="B", choices=["A", "B", "C"])
    args = parser.parse_args()

    avaliar(
        modelo_ckpt=args.ckpt,
        num_lotes=args.lotes,
        passos_max=args.passos,
        estagio_curriculo=args.estagio
    )
