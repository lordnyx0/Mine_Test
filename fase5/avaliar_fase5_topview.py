# coding=utf-8
"""
fase5/avaliar_fase5_topview.py — Benchmark Oficial de Avaliação TopView 2D da Fase 5.

Executa bateria de avaliação com o modelo Fase 5 (Cold-Start ou PPO RL) em modo determinístico ou estocástico:
  - Ponto Inicial de Partida (X0, Y0, Z0, Yaw0)
  - Posição das Submetas 1 e 2 com cores e raio de 2.5m
  - Rastro passo a passo com coordenadas 3D (X, Y, Z, Yaw, Ações)
  - Taxa de sucesso na Submeta 1, Submeta 2 e Transição Lógica
  - Geração de gráficos TopView 2D e relatório HTML interativo

Saídas:
  1. fase5/resultados_topview_coldstart.json: Dados estruturados de todos os episódios
  2. docs/topview_fase5_coldstart.png: Painel 2D Top-Down com o traçado de todos os robôs
  3. fase5/relatorio_topview_fase5.html: Painel interativo com visualização no Canvas
"""
import os
import sys
import math
import json
import time
import argparse
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import (montar_tarefas_logicas, PASSOS_MAX_F4,
                                      RAIO_CHEGADA_SUBMETA)
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

CORES_HEX = {
    "amarelo": "#eab308",
    "roxo": "#a855f7",
    "azul": "#3b82f6",
    "ouro": "#eab308",
    "obsidiana": "#a855f7",
    "lapis": "#3b82f6",
    "passagem": "#10b981",
    "alvo": "#ef4444"
}


def avaliar_fase5(
    modelo_ckpt: str = "checkpoints_vla/vla_fase5_coldstart.pt",
    num_lotes: int = 3,
    passos_max: int = 100,
    seed: int = 42,
    amostrar: bool = False,
    temperatura: float = 0.8,
    raio_chegada: float = 1.5
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" [FASE 5] AVALIACAO TOPVIEW -- DECISOES LOGICAS E TRANSIÇÃO DE SUBMETAS")
    print(f"    Checkpoint : {modelo_ckpt}")
    print(f"    Lotes      : {num_lotes} (8 robos/lote = {num_lotes * 8} episodios totais)")
    print(f"    Raio Meta  : {raio_chegada:.2f}m")
    print(f"    Modo       : {'Amostrado (Temperatura=' + str(temperatura) + ')' if amostrar else 'Deterministico (Argmax)'}")
    print("=" * 80)

    # 1. Carrega modelo VLA base
    vla, device = load_vla_agent(None)
    compactar_backbone(vla)
    if not any("lora_" in n for n, _ in vla.named_parameters()):
        vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    from politica.cerebro import PoliticaCerebroVLA

    # Detecta se o checkpoint usa cabeças fatoradas 54D, 36 ações táticas ou 18 legadas
    num_acoes = 36
    fatorada = False
    ckpt_data = None
    if os.path.exists(modelo_ckpt):
        ckpt_data = torch.load(modelo_ckpt, map_location=device)
        if "treinaveis" in ckpt_data:
            if any("cabeca_modo" in k for k in ckpt_data["treinaveis"].keys()) or ckpt_data.get("fatorada"):
                fatorada = True
            elif any("cabeca_acao_36" in k for k in ckpt_data["treinaveis"].keys()) or ckpt_data.get("num_acoes") == 36:
                num_acoes = 36
            elif ckpt_data.get("num_acoes") == 18:
                num_acoes = 18
    
    print(f"[VLA] Configuração de Ações: {'Fatorada 54D (Modo 6 x Yaw 9)' if fatorada else f'{num_acoes} classes'}", flush=True)

    pol_vla = PoliticaRaciocinioLoop(None, amostrar=amostrar, device=device, vla=vla, loops_pensamento=3, num_acoes=num_acoes, fatorada=fatorada)
    pol = PoliticaCerebroVLA(pol_vla)
    pol.amostrar = amostrar
    pol.temperatura = temperatura

    # Restaura pesos treináveis do checkpoint da Fase 5
    if ckpt_data is not None and "treinaveis" in ckpt_data:
        msg = vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
        print(f"[VLA] Pesos restaurados: {len(ckpt_data['treinaveis'])} tensores (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})", flush=True)
    elif not os.path.exists(modelo_ckpt):
        print(f"[AVISO] Checkpoint '{modelo_ckpt}' nao encontrado! Rodando com pesos base.", flush=True)

    vla.to(device)
    vla.eval()

    info = get("/lote/info")
    N = info["envs"]
    print(f"[ENV] Conectado ao simulador Mineflayer com {N} ambientes paralelos.", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    todos_logs = []
    sucessos_totais = 0
    sucessos_parciais = 0
    falhas = 0
    ep_global_id = 0

    t_inicio_total = time.time()

    from fase5.treinar_ppo_bc_hibrido import gerar_tarefas_busca_ativa

    for lote_idx in range(num_lotes):
        print(f"\n--- Executando Lote {lote_idx + 1}/{num_lotes} ---", flush=True)
        tarefas, blocos_tarefas = gerar_tarefas_busca_ativa(N, seed=seed + lote_idx * 17, nivel=2)
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
                "pilar1": {
                    "cor": p1["cor"],
                    "x": p1["alvo_abs"][0],
                    "z": p1["alvo_abs"][1],
                    "dist_inicial": round(dist1, 2)
                },
                "pilar2": {
                    "cor": p2["cor"] if p2 else "nenhum",
                    "x": p2["alvo_abs"][0] if p2 else 0.0,
                    "z": p2["alvo_abs"][1] if p2 else 0.0,
                    "dist_inicial": round(dist2, 2) if p2 else 0.0
                } if p2 else None,
                "caminho": [{"passo": 0, "x": e0["x"], "y": e0["y"], "z": e0["z"], "yaw": e0["yaw"], "pulo": False, "degrau": False}],
                "resultado": "falha",
                "passo_pilar1": None,
                "passo_pilar2": None,
                "dist_min_pilar1": dist1,
                "dist_min_pilar2": dist2,
                "pulos_executados": 0,
                "subidas_degrau": 0
            })

        for p in range(passos_max):
            prompts_ativos = [
                tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)].get("prompt_estagio", tarefas[i]["prompt"])
                for i in range(N)
            ]
            alvos_ativos_abs = [tarefas[i]["estagios"][min(estagio_atual[i], len(tarefas[i]["estagios"]) - 1)]["alvo_abs"] for i in range(N)]
            
            with torch.no_grad():
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
                y_ant = log["caminho"][-1]["y"]
                subiu = (e["y"] - y_ant) >= 0.5
                pulou = "SPACE" in acoes[i].get("hold", [])

                if pulou:
                    log["pulos_executados"] += 1
                if subiu:
                    log["subidas_degrau"] += 1

                log["caminho"].append({
                    "passo": p + 1,
                    "x": round(e["x"], 2),
                    "y": round(e["y"], 2),
                    "z": round(e["z"], 2),
                    "yaw": round(e["yaw"], 1),
                    "acao": acoes[i].get("hold", []),
                    "pulo": pulou,
                    "degrau": subiu
                })

                tar = tarefas[i]
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
                    if d2 <= raio_chegada:
                        estagio_atual[i] = 2
                        log["passo_pilar2"] = p + 1
                        log["resultado"] = "sucesso_total"
                        vivo[i] = False

            if not any(vivo):
                break

        # Contabiliza estatísticas do lote
        for i, log in enumerate(logs_lote):
            res = log["resultado"]
            if res == "sucesso_total":
                sucessos_totais += 1
                icon = "[SUCESSO TOTAL]"
            elif res == "sucesso_parcial":
                sucessos_parciais += 1
                icon = "[SUBMETA 1 ATINGIDA]"
            else:
                falhas += 1
                icon = "[FALHA]"

            print(f"  Ep {log['ep_id']:02d} (Env {i}) | {icon} | Submeta1: {log['dist_min_pilar1']}m (passo {log['passo_pilar1']}) | Submeta2: {log['dist_min_pilar2']}m (passo {log['passo_pilar2']})", flush=True)

        todos_logs.extend(logs_lote)
        ep_global_id += N

    duracao_total = time.time() - t_inicio_total
    total_eps = len(todos_logs)
    taxa_submeta1 = ((sucessos_totais + sucessos_parciais) / total_eps) * 100.0
    taxa_total = (sucessos_totais / total_eps) * 100.0

    print("\n" + "=" * 80)
    print(" [FASE 5] RESULTADOS FINAIS DO BENCHMARK TOPVIEW")
    print(f"    Episódios Totais     : {total_eps}")
    print(f"    Sucesso Submeta 1    : {sucessos_totais + sucessos_parciais}/{total_eps} ({taxa_submeta1:.1f}%)")
    print(f"    Sucesso Total (1->2) : {sucessos_totais}/{total_eps} ({taxa_total:.1f}%)")
    print(f"    Tempo de Execução    : {duracao_total:.1f}s")
    print("=" * 80)

    # Define prefixo de arquivos baseado no checkpoint
    nome_base = os.path.splitext(os.path.basename(modelo_ckpt))[0].replace("vla_fase5_", "").replace("vla_", "")
    if not nome_base:
        nome_base = "resultado"

    # 1. Salva JSON
    os.makedirs("fase5", exist_ok=True)
    json_path = f"fase5/resultados_topview_{nome_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modelo": modelo_ckpt,
            "total_episodios": total_eps,
            "taxa_submeta1": taxa_submeta1,
            "taxa_sucesso_total": taxa_total,
            "duracao_segundos": duracao_total,
            "episodios": todos_logs
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Dados JSON salvos em: {json_path}", flush=True)

    # 2. Gera Gráfico PNG TopView
    if HAS_MATPLOTLIB:
        png_path = f"docs/topview_fase5_{nome_base}.png"
        gerar_grafico_topview(todos_logs, png_path, taxa_submeta1, taxa_total, raio_chegada=raio_chegada, nome_modelo=nome_base)

    # 3. Gera Relatório HTML Interativo
    html_path = f"fase5/relatorio_topview_{nome_base}.html"
    gerar_relatorio_html(todos_logs, html_path, taxa_submeta1, taxa_total, raio_chegada=raio_chegada, nome_modelo=nome_base)


def gerar_grafico_topview(logs, output_png, taxa_s1, taxa_tot, raio_chegada=2.0, nome_modelo="PPO-BC"):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    num_eps = len(logs)
    cols = 4
    rows = math.ceil(num_eps / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.2))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, log in enumerate(logs):
        ax = axes[idx]
        xs = [p["x"] for p in log["caminho"]]
        zs = [p["z"] for p in log["caminho"]]

        p1 = log["pilar1"]
        p2 = log["pilar2"]

        # Desenha Submeta 1
        c1 = Circle((p1["x"], p1["z"]), raio_chegada, color=CORES_HEX.get(p1["cor"], "#eab308"), alpha=0.3)
        ax.add_patch(c1)
        ax.plot(p1["x"], p1["z"], "o", color=CORES_HEX.get(p1["cor"], "#eab308"), markersize=8, label="Submeta 1")

        # Desenha Submeta 2 se existir
        if p2:
            c2 = Circle((p2["x"], p2["z"]), raio_chegada, color=CORES_HEX.get(p2["cor"], "#a855f7"), alpha=0.3)
            ax.add_patch(c2)
            ax.plot(p2["x"], p2["z"], "o", color=CORES_HEX.get(p2["cor"], "#a855f7"), markersize=8, label="Submeta 2")

        # Traçado do agente
        cor_rastro = "#10b981" if log["resultado"] == "sucesso_total" else ("#f59e0b" if log["resultado"] == "sucesso_parcial" else "#ef4444")
        ax.plot(xs, zs, "-", color=cor_rastro, linewidth=2, label="Trajetória")
        ax.plot(xs[0], zs[0], "s", color="#3b82f6", markersize=6, label="Largada")
        ax.plot(xs[-1], zs[-1], "x", color="#000000", markersize=7, label="Fim")

        ax.set_title(f"Ep {log['ep_id']} [{log['resultado'].upper()}]", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", "datalim")

    # Esconde subplots vazios
    for j in range(num_eps, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Fase 5 ({nome_modelo}): TopView Trajetórias | Submeta 1: {taxa_s1:.1f}% | Sucesso Total: {taxa_tot:.1f}% (Raio={raio_chegada:.1f}m)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"[OK] Gráfico TopView 2D salvo em: {output_png}", flush=True)


def gerar_relatorio_html(logs, output_html, taxa_s1, taxa_tot, raio_chegada=2.0, nome_modelo="PPO-BC"):
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Fase 5 - Relatório TopView ({nome_modelo})</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        .header {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 20px; margin-top: 10px; }}
        .stat-card {{ background: #334155; padding: 15px 25px; border-radius: 6px; }}
        .stat-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 20px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 15px; border: 1px solid #334155; }}
        .card.sucesso_total {{ border-left: 5px solid #10b981; }}
        .card.sucesso_parcial {{ border-left: 5px solid #f59e0b; }}
        .card.falha {{ border-left: 5px solid #ef4444; }}
        canvas {{ background: #090d16; border-radius: 4px; display: block; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Fase 5: Avaliação TopView ({nome_modelo})</h1>
        <div class="stats">
            <div class="stat-card"><div>Total de Episódios</div><div class="stat-val">{len(logs)}</div></div>
            <div class="stat-card"><div>Taxa Submeta 1</div><div class="stat-val">{taxa_s1:.1f}%</div></div>
            <div class="stat-card"><div>Sucesso Completo (1 &rarr; 2)</div><div class="stat-val">{taxa_tot:.1f}%</div></div>
            <div class="stat-card"><div>Raio de Chegada</div><div class="stat-val">{raio_chegada:.1f}m</div></div>
        </div>
    </div>
    <div class="grid">
"""
    for log in logs:
        html += f"""
        <div class="card {log['resultado']}">
            <h3>Episódio {log['ep_id']} - <span style="text-transform: uppercase;">{log['resultado']}</span></h3>
            <p style="font-size: 13px; color: #94a3b8;">{log['prompt']}</p>
            <p style="font-size: 12px;">Dist Min S1: <b>{log['dist_min_pilar1']}m</b> | Dist Min S2: <b>{log['dist_min_pilar2']}m</b></p>
            <canvas id="cv_{log['ep_id']}" width="330" height="260"></canvas>
        </div>
        """

    html += """
    </div>
    <script>
        const logs = """ + json.dumps(logs) + """;
        logs.forEach(log => {
            const cv = document.getElementById('cv_' + log.ep_id);
            if (!cv) return;
            const ctx = cv.getContext('2d');
            const w = cv.width, h = cv.height;

            const xs = log.caminho.map(p => p.x);
            const zs = log.caminho.map(p => p.z);
            xs.push(log.pilar1.x); zs.push(log.pilar1.z);
            if (log.pilar2) { xs.push(log.pilar2.x); zs.push(log.pilar2.z); }

            const minX = Math.min(...xs) - 3, maxX = Math.max(...xs) + 3;
            const minZ = Math.min(...zs) - 3, maxZ = Math.max(...zs) + 3;

            function toX(x) { return 15 + ((x - minX) / (maxX - minX)) * (w - 30); }
            function toY(z) { return 15 + ((z - minZ) / (maxZ - minZ)) * (h - 30); }

            // Submeta 1
            ctx.fillStyle = '#eab30844'; ctx.strokeStyle = '#eab308';
            ctx.beginPath(); ctx.arc(toX(log.pilar1.x), toY(log.pilar1.z), 12, 0, Math.PI*2); ctx.fill(); ctx.stroke();

            // Submeta 2
            if (log.pilar2) {
                ctx.fillStyle = '#a855f744'; ctx.strokeStyle = '#a855f7';
                ctx.beginPath(); ctx.arc(toX(log.pilar2.x), toY(log.pilar2.z), 12, 0, Math.PI*2); ctx.fill(); ctx.stroke();
            }

            // Trajetória
            ctx.strokeStyle = log.resultado === 'sucesso_total' ? '#10b981' : (log.resultado === 'sucesso_parcial' ? '#f59e0b' : '#ef4444');
            ctx.lineWidth = 2;
            ctx.beginPath();
            log.caminho.forEach((p, idx) => {
                if (idx === 0) ctx.moveTo(toX(p.x), toY(p.z));
                else ctx.lineTo(toX(p.x), toY(p.z));
            });
            ctx.stroke();

            // Ponto de Partida
            ctx.fillStyle = '#38bdf8';
            ctx.beginPath(); ctx.arc(toX(log.caminho[0].x), toY(log.caminho[0].z), 4, 0, Math.PI*2); ctx.fill();
        });
    </script>
</body>
</html>"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Relatório HTML salvo em: {output_html}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase5_coldstart.pt")
    ap.add_argument("--lotes", type=int, default=3)
    ap.add_argument("--passos", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amostrar", action="store_true")
    ap.add_argument("--raio", type=float, default=1.5)
    args = ap.parse_args()

    avaliar_fase5(
        modelo_ckpt=args.ckpt,
        num_lotes=args.lotes,
        passos_max=args.passos,
        seed=args.seed,
        amostrar=args.amostrar,
        raio_chegada=args.raio
    )
