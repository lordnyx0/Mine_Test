# coding=utf-8
"""
avaliar_fase4_topview.py — Benchmark Oficial de Avaliação com Topologia e Gravação TopView 2D.

Executa bateria de avaliação com o modelo Fase 4 em modo determinístico (argmax),
gravando para cada episódio:
  - Ponto Inicial de Partida (X0, Y0, Z0, Yaw0)
  - Posição dos Pilares 1 e 2 com cores e raio de 2.5m
  - Rastro passo a passo com coordenadas 3D (X, Y, Z, Yaw, Ações)
  - Perfil topológico de elevação (Y) e eventos de pulo para transposição de relevo
  - Resultado (Sucesso Completo, Parcial ou Falha)

Gera:
  1. avaliacao/resultados_fase4.json: Dados estruturados completos de todos os episódios
  2. docs/topview_avaliacao_fase4.png: Gráficos 2D Top-Down + Perfil de Elevação Topológica
  3. avaliacao/relatorio_topview.html: Painel interativo com visualização topológica no Canvas
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def avaliar(modelo_ckpt=None, num_lotes=5, passos_max=100, seed=100, amostrar=False, temperatura=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = modelo_ckpt or "checkpoints_vla/vla_fase4_logica.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints_vla/vla_fase3_merged.pt"

    print("=" * 80)
    print("[BENCHMARK OFICIAL] AVALIACAO FASE 4 -- COM TOPOLOGIA E TOPVIEW")
    print(f"    Checkpoint : {ckpt_path}")
    print(f"    Lotes      : {num_lotes} (8 robos/lote = {num_lotes * 8} episodios totais)")
    print(f"    Modo       : {'Amostrado (Temperatura=' + str(temperatura) + ')' if amostrar else 'Deterministico (Argmax)'}")
    print("=" * 80)

    # 1. Carrega modelo
    vla, device = load_vla_agent(ckpt_path)
    compactar_backbone(vla)
    if not any("lora_" in n for n, _ in vla.named_parameters()):
        vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    from politica.cerebro import PoliticaCerebroVLA

    # Instancia a política ANTES para que vla.cabeca_acao_18 exista na estrutura do modelo
    pol_vla = PoliticaRaciocinioLoop(None, amostrar=amostrar, device=device, vla=vla, loops_pensamento=3)
    pol = PoliticaCerebroVLA(pol_vla)
    pol.amostrar = amostrar
    pol.temperatura = temperatura

    # Agora restaura todos os tensores treináveis (incluindo cabeca_acao_18 e LoRA)
    if os.path.exists(ckpt_path):
        ckpt_data = torch.load(ckpt_path, map_location=device)
        if "treinaveis" in ckpt_data:
            msg = vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos treináveis restaurados por nome: {len(ckpt_data['treinaveis'])} tensores (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")

    vla.to(device)
    vla.eval()

    N = get("/lote/info")["envs"]
    historico_episodios = []

    sucessos_completos = 0
    sucessos_parciais = 0
    total_episodios = num_lotes * N

    ep_global_id = 0
    for lote_idx in range(num_lotes):
        tarefas = montar_tarefas_logicas(N, seed=seed + lote_idx * 17)
        prompts = [t["prompt"] for t in tarefas]

        r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in tarefas]})
        obs = r["obs"][:N]
        est = [o["estado"] for o in obs]
        pol.reiniciar(obs)

        estagio_atual = [0] * N
        vivo = [True] * N
        chegou_pilar1 = [None] * N
        chegou_pilar2 = [None] * N

        # Estrutura para salvar o rastro de cada episódio
        logs_lote = []
        for i in range(N):
            tar = tarefas[i]
            e0 = est[i]
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
            acoes = pol.agir(est, alvos_ativos_abs, obs, prompts=prompts_ativos, estagios=estagio_atual)
            for i in range(N):
                if not vivo[i]:
                    acoes[i] = {"hold": [], "mouse": [0, 0], "duration_ms": 50}

            u = pol.ultimo

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

                # Registra o ponto 3D no caminho
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

                # Chegada no Pilar 1
                if estagio_atual[i] == 0 and d1 <= RAIO_CHEGADA_SUBMETA:
                    chegou_pilar1[i] = p + 1
                    log["passo_pilar1"] = p + 1
                    log["resultado"] = "sucesso_parcial"
                    estagio_atual[i] = 1
                    if hasattr(pol, "ativar_varredura"):
                        pol.ativar_varredura(i, passos_varredura=3)

                # Chegada no Pilar 2
                elif estagio_atual[i] == 1 and len(tar["estagios"]) > 1:
                    if d2 <= RAIO_CHEGADA_SUBMETA:
                        chegou_pilar2[i] = p + 1
                        log["passo_pilar2"] = p + 1
                        log["resultado"] = "sucesso_completo"
                        vivo[i] = False

            if not any(vivo):
                break

        for i in range(N):
            if chegou_pilar2[i] is not None:
                sucessos_completos += 1
            elif chegou_pilar1[i] is not None:
                sucessos_parciais += 1

        historico_episodios.extend(logs_lote)
        ep_global_id += N
        print(f"Lote {lote_idx+1}/{num_lotes} concluído | Sucesso Completo Acumulado: {sucessos_completos}/{ep_global_id} ({100*sucessos_completos/ep_global_id:.1f}%)")

    # Salva dados estruturados em JSON
    os.makedirs("avaliacao", exist_ok=True)
    caminho_json = "avaliacao/resultados_fase4.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(historico_episodios, f, indent=2)
    print(f"\n[OK] Resultados estruturados salvos em {caminho_json}")

    # Gera visualizações
    if HAS_MATPLOTLIB:
        gerar_grafico_topview_topologico(historico_episodios)
    gerar_relatorio_html(historico_episodios)

    taxa_completa = (sucessos_completos / total_episodios) * 100
    taxa_parcial = (sucessos_parciais / total_episodios) * 100

    print("\n" + "=" * 80)
    print(" [RESULTADO FINAL] AVALIACAO FASE 4 (TOPOLOGIA + TOPVIEW)")
    print(f"    Total de Episodios Avaliados: {total_episodios}")
    print(f"    Taxa de Sucesso Completo (Pilar 1 -> Pilar 2): {sucessos_completos}/{total_episodios} ({taxa_completa:.1f}%)")
    print(f"    Taxa de Sucesso Parcial (Apenas Pilar 1):      {sucessos_parciais}/{total_episodios} ({taxa_parcial:.1f}%)")
    print("=" * 80)
    return historico_episodios


def gerar_grafico_topview_topologico(episodios):
    os.makedirs("docs", exist_ok=True)
    eps = episodios[:8]
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), facecolor="#0f172a")
    fig.suptitle("TopView e Perfil Topológico de Elevação — Avaliação Fase 4", color="#38bdf8", fontsize=15, fontweight="bold")

    axes = axes.flatten()
    for idx, (ax, ep) in enumerate(zip(axes, eps)):
        ax.set_facecolor("#0b0f19")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        x0 = ep["largada"]["x"]
        z0 = ep["largada"]["z"]
        caminho = ep["caminho"]
        xs = [pt["x"] for pt in caminho]
        ys = [pt["y"] for pt in caminho]
        zs = [pt["z"] for pt in caminho]

        p1 = ep["pilar1"]
        p2 = ep.get("pilar2")

        # Plota Pilar 1
        cor1 = CORES_HEX.get(p1["cor"], "#eab308")
        ax.add_patch(Circle((p1["x"], p1["z"]), 2.5, color=cor1, fill=False, linestyle="--", linewidth=1.4, alpha=0.8))
        ax.scatter([p1["x"]], [p1["z"]], color=cor1, s=130, edgecolors="white", linewidth=1.5, zorder=5)
        ax.text(p1["x"] + 0.6, p1["z"] + 0.6, f"P1:{p1['cor']}", color="white", fontsize=8, fontweight="bold")

        # Plota Pilar 2
        if p2 and p2["cor"] != "nenhum":
            cor2 = CORES_HEX.get(p2["cor"], "#3b82f6")
            ax.add_patch(Circle((p2["x"], p2["z"]), 2.5, color=cor2, fill=False, linestyle="--", linewidth=1.4, alpha=0.8))
            ax.scatter([p2["x"]], [p2["z"]], color=cor2, s=130, edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(p2["x"] + 0.6, p2["z"] + 0.6, f"P2:{p2['cor']}", color="white", fontsize=8, fontweight="bold")

        # Cor da linha com mapa de calor topológico (elevação Y)
        norm_y = plt.Normalize(vmin=min(ys) - 0.1, vmax=max(ys) + 0.1)
        scatter = ax.scatter(xs, zs, c=ys, cmap="viridis", s=25, alpha=0.9, zorder=3)

        # Destaca pontos de pulo / subida de relevo
        pulos_x = [pt["x"] for pt in caminho if pt.get("degrau")]
        pulos_z = [pt["z"] for pt in caminho if pt.get("degrau")]
        if pulos_x:
            ax.scatter(pulos_x, pulos_z, color="#f43f5e", marker="^", s=70, label="Pulo (Degrau)", zorder=5)

        # Linha contínua base
        res = ep["resultado"]
        cor_base = "#22c55e" if res == "sucesso_completo" else ("#eab308" if res == "sucesso_parcial" else "#ef4444")
        ax.plot(xs, zs, color=cor_base, linewidth=1.5, alpha=0.5, zorder=2)

        # Largada e Chegada
        ax.scatter([x0], [z0], color="#22c55e", s=90, edgecolors="white", label="Largada", zorder=4)
        ax.scatter([xs[-1]], [zs[-1]], color="#ffffff", s=90, edgecolors=cor_base, label="Final", zorder=6)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7, colors="#94a3b8")
        cbar.set_label("Elevação Y (Topologia)", color="#94a3b8", fontsize=7)

        tag = "[SUCESSO]" if res == "sucesso_completo" else ("[PILAR 1]" if res == "sucesso_parcial" else "[FALHA]")
        ax.set_title(f"Ep {ep['ep_id']} | {tag} | dY: {max(ys)-min(ys):.1f}m ({ep['subidas_degrau']} degraus)", color="#f8fafc", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    caminho_png = "docs/topview_avaliacao_fase4.png"
    plt.savefig(caminho_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Gráfico TopView com Topologia salvo em {caminho_png}")


def gerar_relatorio_html(episodios):
    json_str = json.dumps(episodios)
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>TopView — Relatório Oficial com Topologia (Fase 4)</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0b0f19; color: #f8fafc; margin: 0; padding: 16px; }}
        header {{ text-align: center; margin-bottom: 16px; }}
        h1 {{ color: #38bdf8; margin: 0 0 4px 0; font-size: 22px; }}
        p.subtitle {{ color: #94a3b8; margin: 0; font-size: 13px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; max-width: 1500px; margin: 15px auto; }}
        .card {{ background: #1e293b; border-radius: 10px; padding: 12px; border: 1px solid #334155; }}
        .card-head {{ display: flex; justify-content: space-between; font-size: 13px; font-weight: bold; margin-bottom: 8px; }}
        .badge-comp {{ background: #10b981; color: white; padding: 2px 7px; border-radius: 4px; font-size: 11px; }}
        .badge-parc {{ background: #eab308; color: black; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-falha {{ background: #ef4444; color: white; padding: 2px 7px; border-radius: 4px; font-size: 11px; }}
        canvas {{ background: #050811; border-radius: 6px; width: 100%; height: 270px; display: block; border: 1px solid #1e293b; }}
        .legend {{ display: flex; justify-content: center; gap: 15px; margin-top: 10px; font-size: 13px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .info {{ font-size: 11px; color: #94a3b8; margin-top: 6px; font-family: monospace; }}
    </style>
</head>
<body>
    <header>
        <h1>🗺️ TopView — Relatório Oficial com Topologia (Fase 4)</h1>
        <p class="subtitle">Traçado 3D do caminho, elevação topológica (Y), pilares e transposição de degraus</p>
        <div class="legend">
            <div class="legend-item"><span class="dot" style="background:#22c55e;"></span> Largada</div>
            <div class="legend-item"><span class="dot" style="background:#eab308;"></span> Pilar 1</div>
            <div class="legend-item"><span class="dot" style="background:#3b82f6;"></span> Pilar 2</div>
            <div class="legend-item"><span class="dot" style="background:#f43f5e;"></span> Pulo / Subida de Relevo</div>
            <div class="legend-item"><span class="dot" style="background:#10b981;"></span> Sucesso Completo</div>
        </div>
    </header>

    <div class="grid" id="grid"></div>

    <script>
        const eps = {json_str};
        const container = document.getElementById('grid');
        const CORES = {{ "amarelo": "#eab308", "ouro": "#eab308", "roxo": "#a855f7", "obsidiana": "#a855f7", "azul": "#3b82f6", "lapis": "#3b82f6" }};

        eps.forEach(ep => {{
            const card = document.createElement('div');
            card.className = 'card';
            
            let badgeClass = ep.resultado === 'sucesso_completo' ? 'badge-comp' : (ep.resultado === 'sucesso_parcial' ? 'badge-parc' : 'badge-falha');
            let badgeText = ep.resultado === 'sucesso_completo' ? '🏆 Sucesso Completo' : (ep.resultado === 'sucesso_parcial' ? '🟡 Pilar 1 Atingido' : '❌ Falha');

            card.innerHTML = `
                <div class="card-head">
                    <span>Episódio ${{ep.ep_id}}</span>
                    <span class="${{badgeClass}}">${{badgeText}}</span>
                </div>
                <canvas id="cv_${{ep.ep_id}}" width="360" height="280"></canvas>
                <div class="info">
                    <strong>Alvo 1:</strong> ${{ep.pilar1.cor}} (${{ep.pilar1.dist_inicial}}m) ➔ Min: ${{ep.dist_min_pilar1}}m<br>
                    <strong>Alvo 2:</strong> ${{ep.pilar2 ? ep.pilar2.cor + ' (' + ep.pilar2.dist_inicial + 'm) ➔ Min: ' + ep.dist_min_pilar2 + 'm' : 'N/A'}}<br>
                    <strong>Topologia:</strong> Elevação Y0=${{ep.largada.y.toFixed(0)}}m ➔ ${{ep.subidas_degrau}} subidas de relevo (${{ep.pulos_executados}} pulos)
                </div>
            `;
            container.appendChild(card);

            const cv = document.getElementById('cv_' + ep.ep_id);
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height;

            const xs = ep.caminho.map(p => p.x);
            const ys = ep.caminho.map(p => p.y);
            const zs = ep.caminho.map(p => p.z);
            const p1 = ep.pilar1;
            const p2 = ep.pilar2;

            let todosX = [...xs, p1.x], todosZ = [...zs, p1.z];
            if (p2 && p2.cor !== 'nenhum') {{ todosX.push(p2.x); todosZ.push(p2.z); }}

            let minX = Math.min(...todosX) - 4, maxX = Math.max(...todosX) + 4;
            let minZ = Math.min(...todosZ) - 4, maxZ = Math.max(...todosZ) + 4;
            const span = Math.max(maxX - minX, maxZ - minZ, 16);
            const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;

            function toX(x) {{ return W / 2 + ((x - cx) / span) * (W * 0.84); }}
            function toY(z) {{ return H / 2 + ((z - cz) / span) * (H * 0.84); }}

            // Grid
            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 1;
            for (let g = -50; g <= 50; g += 10) {{
                ctx.beginPath(); ctx.moveTo(toX(cx + g), 0); ctx.lineTo(toX(cx + g), H); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, toY(cz + g)); ctx.lineTo(W, toY(cz + g)); ctx.stroke();
            }}

            // Desenha Pilar 1
            const p1x = toX(p1.x), p1y = toY(p1.z);
            const c1 = CORES[p1.cor] || '#eab308';
            ctx.strokeStyle = c1; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.arc(p1x, p1y, (2.5 / span) * (W * 0.84), 0, Math.PI * 2); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = c1; ctx.beginPath(); ctx.arc(p1x, p1y, 7, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = '#ffffff'; ctx.stroke();
            ctx.fillStyle = '#ffffff'; ctx.font = 'bold 10px sans-serif'; ctx.fillText('P1:' + p1.cor, p1x + 9, p1y + 3);

            // Desenha Pilar 2
            if (p2 && p2.cor !== 'nenhum') {{
                const p2x = toX(p2.x), p2y = toY(p2.z);
                const c2 = CORES[p2.cor] || '#3b82f6';
                ctx.strokeStyle = c2; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.arc(p2x, p2y, (2.5 / span) * (W * 0.84), 0, Math.PI * 2); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = c2; ctx.beginPath(); ctx.arc(p2x, p2y, 7, 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = '#ffffff'; ctx.stroke();
                ctx.fillStyle = '#ffffff'; ctx.font = 'bold 10px sans-serif'; ctx.fillText('P2:' + p2.cor, p2x + 9, p2y + 3);
            }}

            // Desenha Trajetória
            const corLinha = ep.resultado === 'sucesso_completo' ? '#10b981' : (ep.resultado === 'sucesso_parcial' ? '#eab308' : '#ef4444');
            ctx.strokeStyle = corLinha;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            for (let k = 0; k < xs.length; k++) {{
                const px = toX(xs[k]), py = toY(zs[k]);
                if (k === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Marca Pulos / Subidas de Relevo
            ep.caminho.forEach(pt => {{
                if (pt.degrau) {{
                    const px = toX(pt.x), py = toY(pt.z);
                    ctx.fillStyle = '#f43f5e';
                    ctx.beginPath();
                    ctx.arc(px, py, 4, 0, Math.PI * 2);
                    ctx.fill();
                }}
            }});

            // Ponto Inicial (Largada)
            ctx.fillStyle = '#22c55e';
            ctx.beginPath(); ctx.arc(toX(xs[0]), toY(zs[0]), 5, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = '#ffffff'; ctx.stroke();

            // Ponto Final
            ctx.fillStyle = '#ffffff';
            ctx.beginPath(); ctx.arc(toX(xs[xs.length - 1]), toY(zs[zs.length - 1]), 5, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = corLinha; ctx.stroke();
        }});
    </script>
</body>
</html>"""
    caminho_html = "avaliacao/relatorio_topview.html"
    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Relatório interativo salvo em {caminho_html}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lotes", type=int, default=5)
    ap.add_argument("--passos", type=int, default=100)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--amostrar", action="store_true")
    ap.add_argument("--temperatura", type=float, default=0.8)
    args = ap.parse_args()

    avaliar(modelo_ckpt=args.ckpt, num_lotes=args.lotes, passos_max=args.passos,
            seed=args.seed, amostrar=args.amostrar, temperatura=args.temperatura)
