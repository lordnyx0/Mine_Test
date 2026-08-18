# coding=utf-8
"""
fase5/avaliar_oraculo_topview.py — Avaliação TopView 2D do Oráculo Heurístico (Professor do BC).

Executa 24 episódios com o modelo heurístico determinístico usando a física do Minecraft
e as 36 ações táticas WASD para demonstrar o teto teórico e o traçado de referência.

Saídas:
  1. docs/topview_oraculo_heuristico.png
  2. fase5/relatorio_topview_heuristico.html
  3. fase5/resultados_topview_heuristico.json
"""
import os
import sys
import math
import json
import time
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import RAIO_CHEGADA_SUBMETA
from fase5.acoes_taticas import calcular_acao_otima_tatica, decodificar_acao_36
from fase5.treinar_ppo_bc_hibrido import gerar_tarefas_busca_ativa

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
    "lapis": "#3b82f6"
}


def avaliar_oraculo(num_lotes: int = 3, passos_max: int = 100, seed: int = 42):
    print("=" * 80)
    print(" [ORÁCULO HEURÍSTICO] AVALIAÇÃO TOPVIEW -- TRAÇADOS DO ESPECIALISTA (PROFESSOR BC)")
    print(f"    Lotes      : {num_lotes} (8 robôs/lote = {num_lotes * 8} episódios totais)")
    print(f"    Passos/Ep  : {passos_max}")
    print(f"    Raio Meta  : {RAIO_CHEGADA_SUBMETA}m")
    print("=" * 80)

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
        print(f"\n--- Executando Lote {lote_idx + 1}/{num_lotes} ---", flush=True)
        tarefas, blocos_tarefas = gerar_tarefas_busca_ativa(N, seed=seed + lote_idx * 17, nivel=2)
        post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        post("/lote/colocar_bloco", {"blocos": blocos_tarefas})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * N, "frames": False})
        obs = r["obs"][:N]
        est = [o["estado"] for o in obs]

        for i, t in enumerate(tarefas):
            t["id_ep"] = ep_global_id + i + 1

        estagio_atual = [0] * N
        vivo = [True] * N
        cooldown_frenagem = [0] * N

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
                "caminho": [{"passo": 0, "x": e0["x"], "y": e0["y"], "z": e0["z"], "yaw": e0["yaw"]}],
                "resultado": "falha",
                "passo_pilar1": None,
                "passo_pilar2": None,
                "dist_min_pilar1": dist1,
                "dist_min_pilar2": dist2,
            })

        pos_anterior = [e0.copy() for e0 in est]
        passos_travados = [0] * N

        for p in range(passos_max):
            acoes = []
            for i in range(N):
                if not vivo[i]:
                    acoes.append({"hold": [], "mouse": [0, 0], "duration_ms": 50})
                    continue

                e = est[i]
                tar = tarefas[i]
                alvo = tar["estagios"][min(estagio_atual[i], len(tar["estagios"]) - 1)]["alvo_abs"]
                dx = alvo[0] - e["x"]
                dz = alvo[1] - e["z"]
                dist = math.hypot(dx, dz)

                ang_alvo_rad = math.atan2(-dx, -dz)
                yaw_rad = math.radians(e["yaw"])
                diff_rad = (ang_alvo_rad - yaw_rad + math.pi) % (2 * math.pi) - math.pi
                erro_yaw_graus = math.degrees(diff_rad)

                # Detecção de travamento em relevo/quina
                vel_aparente = math.hypot(e["x"] - pos_anterior[i]["x"], e["z"] - pos_anterior[i]["z"])
                if p > 0 and vel_aparente < 0.08:
                    passos_travados[i] += 1
                else:
                    passos_travados[i] = 0

                is_spawn = (p == 0)
                deve_pular = (passos_travados[i] == 1 or passos_travados[i] == 2)
                esta_colidindo = (passos_travados[i] >= 4)

                # Se estiver em frenagem pós-submeta 1, faz giro no próprio eixo
                if cooldown_frenagem[i] > 0:
                    cooldown_frenagem[i] -= 1
                    bin9 = int(np.clip(round((erro_yaw_graus / 60.0) * 4 + 4), 0, 8))
                    acao_idx = 0 + bin9
                else:
                    acao_idx = calcular_acao_otima_tatica(
                        erro_yaw_graus=erro_yaw_graus,
                        distancia=dist,
                        deve_pular=deve_pular,
                        is_spawn=is_spawn,
                        esta_colidindo=esta_colidindo
                    )

                acao_dict = decodificar_acao_36(acao_idx)
                acoes.append(acao_dict)
                pos_anterior[i] = e.copy()

            rr = post("/lote/passo", {"acoes": acoes, "frames": False})
            obs = rr["obs"][:N]
            est = [o["estado"] for o in obs]

            for i in range(N):
                if not vivo[i]:
                    continue
                e = est[i]
                log = logs_lote[i]
                log["caminho"].append({"passo": p + 1, "x": e["x"], "y": e["y"], "z": e["z"], "yaw": e["yaw"]})

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

                # Submeta 1
                if estagio_atual[i] == 0 and d1 <= RAIO_CHEGADA_SUBMETA:
                    estagio_atual[i] = 1
                    log["passo_pilar1"] = p + 1
                    cooldown_frenagem[i] = 4
                    if len(tar["estagios"]) == 1:
                        log["resultado"] = "sucesso_total"
                        vivo[i] = False
                    else:
                        log["resultado"] = "sucesso_parcial"

                # Submeta 2
                elif estagio_atual[i] == 1 and len(tar["estagios"]) > 1:
                    if d2 <= RAIO_CHEGADA_SUBMETA:
                        estagio_atual[i] = 2
                        log["passo_pilar2"] = p + 1
                        log["resultado"] = "sucesso_total"
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

            p1_str = f"{log['dist_min_pilar1']}m (passo {log['passo_pilar1']})" if log['passo_pilar1'] else f"{log['dist_min_pilar1']}m (passo None)"
            p2_str = f"{log['dist_min_pilar2']}m (passo {log['passo_pilar2']})" if log['passo_pilar2'] else f"{log['dist_min_pilar2']}m (passo None)"
            print(f"  Ep {log['ep_id']:02d} (Env {log['env']}) | [{log['resultado'].upper()}] | Submeta1: {p1_str} | Submeta2: {p2_str}", flush=True)

        todos_logs.extend(logs_lote)
        ep_global_id += N

    total_eps = len(todos_logs)
    taxa_s1 = ((sucessos_totais + sucessos_parciais) / total_eps) * 100
    taxa_tot = (sucessos_totais / total_eps) * 100
    tempo_tot = time.time() - t_inicio

    print("=" * 80)
    print(" [ORÁCULO HEURÍSTICO] RESULTADOS FINAIS DO BENCHMARK TOPVIEW")
    print(f"    Episódios Totais     : {total_eps}")
    print(f"    Sucesso Submeta 1    : {sucessos_totais + sucessos_parciais}/{total_eps} ({taxa_s1:.1f}%)")
    print(f"    Sucesso Total (1->2) : {sucessos_totais}/{total_eps} ({taxa_tot:.1f}%)")
    print(f"    Tempo de Execução    : {tempo_tot:.1f}s")
    print("=" * 80)

    # 1. Salva JSON
    json_path = "fase5/resultados_topview_heuristico.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos_logs, f, indent=2)
    print(f"[OK] Dados JSON salvos em: {json_path}")

    # 2. Gera Gráfico 2D TopView PNG
    png_path = "docs/topview_oraculo_heuristico.png"
    if HAS_MATPLOTLIB:
        gerar_grafico_topview(todos_logs, png_path, taxa_s1, taxa_tot)
        print(f"[OK] Gráfico TopView 2D salvo em: {png_path}")

    # 3. Gera Relatório HTML Interativo
    html_path = "fase5/relatorio_topview_heuristico.html"
    gerar_relatorio_html(todos_logs, html_path, taxa_s1, taxa_tot)
    print(f"[OK] Relatório HTML salvo em: {html_path}")


def gerar_grafico_topview(logs, output_png, taxa_s1, taxa_tot):
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

        c1 = Circle((p1["x"], p1["z"]), RAIO_CHEGADA_SUBMETA, color=CORES_HEX.get(p1["cor"], "#eab308"), alpha=0.3)
        ax.add_patch(c1)
        ax.plot(p1["x"], p1["z"], "o", color=CORES_HEX.get(p1["cor"], "#eab308"), markersize=8, label="Submeta 1")

        if p2:
            c2 = Circle((p2["x"], p2["z"]), RAIO_CHEGADA_SUBMETA, color=CORES_HEX.get(p2["cor"], "#a855f7"), alpha=0.3)
            ax.add_patch(c2)
            ax.plot(p2["x"], p2["z"], "o", color=CORES_HEX.get(p2["cor"], "#a855f7"), markersize=8, label="Submeta 2")

        cor_rastro = "#10b981" if log["resultado"] == "sucesso_total" else ("#f59e0b" if log["resultado"] == "sucesso_parcial" else "#ef4444")
        ax.plot(xs, zs, "-", color=cor_rastro, linewidth=2, label="Trajetória")
        ax.plot(xs[0], zs[0], "s", color="#3b82f6", markersize=6, label="Largada")
        ax.plot(xs[-1], zs[-1], "x", color="#000000", markersize=7, label="Fim")

        ax.set_title(f"Ep {log['ep_id']} [{log['resultado'].upper()}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("X (blocos)", fontsize=8)
        ax.set_ylabel("Z (blocos)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")

    for empty_idx in range(num_eps, len(axes)):
        fig.delaxes(axes[empty_idx])

    fig.suptitle(f"ORÁCULO HEURÍSTICO (PROFESSOR BC) — TopView 2D\nTaxa Submeta 1: {taxa_s1:.1f}% | Sucesso Total: {taxa_tot:.1f}% (Raio={RAIO_CHEGADA_SUBMETA}m)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()


def gerar_relatorio_html(logs, output_html, taxa_s1, taxa_tot):
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Oráculo Heurístico - Relatório TopView</title>
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
        <h1>Oráculo Heurístico: Traçados do Especialista (Professor BC)</h1>
        <div class="stats">
            <div class="stat-card"><div>Total de Episódios</div><div class="stat-val">{len(logs)}</div></div>
            <div class="stat-card"><div>Taxa Submeta 1</div><div class="stat-val">{taxa_s1:.1f}%</div></div>
            <div class="stat-card"><div>Sucesso Completo (1 &rarr; 2)</div><div class="stat-val">{taxa_tot:.1f}%</div></div>
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

            ctx.fillStyle = '#eab30844'; ctx.strokeStyle = '#eab308';
            ctx.beginPath(); ctx.arc(toX(log.pilar1.x), toY(log.pilar1.z), 12, 0, Math.PI*2); ctx.fill(); ctx.stroke();

            if (log.pilar2) {
                ctx.fillStyle = '#a855f744'; ctx.strokeStyle = '#a855f7';
                ctx.beginPath(); ctx.arc(toX(log.pilar2.x), toY(log.pilar2.z), 12, 0, Math.PI*2); ctx.fill(); ctx.stroke();
            }

            ctx.strokeStyle = log.resultado === 'sucesso_total' ? '#10b981' : (log.resultado === 'sucesso_parcial' ? '#f59e0b' : '#ef4444');
            ctx.lineWidth = 2;
            ctx.beginPath();
            log.caminho.forEach((p, idx) => {
                if (idx === 0) ctx.moveTo(toX(p.x), toY(p.z));
                else ctx.lineTo(toX(p.x), toY(p.z));
            });
            ctx.stroke();

            ctx.fillStyle = '#38bdf8';
            ctx.beginPath(); ctx.arc(toX(log.caminho[0].x), toY(log.caminho[0].z), 4, 0, Math.PI*2); ctx.fill();
        });
    </script>
</body>
</html>"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", type=int, default=3)
    ap.add_argument("--passos", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    avaliar_oraculo(num_lotes=args.lotes, passos_max=args.passos, seed=args.seed)
