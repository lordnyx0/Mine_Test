# coding=utf-8
"""
plotar_trajetorias.py — Visualizador 2D Vista de Cima com Trajetórias e Pilares de Destino.

Desenha no Mapa 2D:
  - Ponto de Partida (Círculo Verde)
  - Trajetória dos Robôs (Linha Azul Ciano com gradiente)
  - Pilar 1 (Submeta 1) com cor real e zona de chegada (raio de 2.5m)
  - Pilar 2 (Meta Final) com cor real e zona de chegada
  - Posição Atual do Robô
  - Métricas de Proximidade em Metros aos 2 Pilares

Gera:
  1. Relatório interativo em HTML: ferramentas/mapa_trajetorias.html
  2. Gráfico PNG de alta resolução: docs/mapa_trajetorias.png
"""
import os
import sys
import json
import math
import time
import urllib.request

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

def coletar_estado_atual():
    url = "http://127.0.0.1:3002/lote/estado"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def carregar_tarefas_ativas():
    caminho = "dataset/tarefas_ativas.json"
    if os.path.exists(caminho):
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def monitorar_trajetorias(duracao_segundos=15, intervalo_amostragem=0.4):
    print("=" * 75)
    print(" [MAPA 2D] GRAVADOR DE TRAJETORIAS COM PILARES DE DESTINO")
    print(f" Gravando movimento dos 8 robos por {duracao_segundos}s...")
    print("=" * 75)

    tarefas = carregar_tarefas_ativas()
    historico = {i: {"x": [], "z": [], "y": [], "yaw": [], "pilares": []} for i in range(8)}

    # Preenche os pilares conhecidos
    for t in tarefas:
        env_id = t.get("env", 0)
        if env_id in historico:
            estagios = t.get("estagios", [])
            for est in estagios:
                historico[env_id]["pilares"].append({
                    "cor": est.get("cor", "alvo"),
                    "x": est["alvo_abs"][0],
                    "z": est["alvo_abs"][1]
                })

    t_fim = time.time() + duracao_segundos
    while time.time() < t_fim:
        d = coletar_estado_atual()
        if d and "envs" in d:
            for env_data in d["envs"]:
                i = env_data["env"]
                s = env_data["estado"]
                if i in historico:
                    historico[i]["x"].append(s["x"])
                    historico[i]["z"].append(s["z"])
                    historico[i]["y"].append(s["y"])
                    historico[i]["yaw"].append(s["yaw"])
        time.sleep(intervalo_amostragem)

    gerar_html(historico)
    if HAS_MATPLOTLIB:
        gerar_png(historico)

    imprimir_resumo(historico)

def imprimir_resumo(historico):
    print("\n" + "=" * 80)
    print(f"{'ENV':<5} | {'POS INICIAL':<17} | {'POS ATUAL':<17} | {'DIST PILAR 1':<17} | {'DIST PILAR 2':<17}")
    print("-" * 80)
    for i in range(8):
        xs = historico[i]["x"]
        zs = historico[i]["z"]
        pils = historico[i]["pilares"]
        if len(xs) > 1:
            x0, z0 = xs[0], zs[0]
            xf, zf = xs[-1], zs[-1]
            
            p1_str = "N/D"
            p2_str = "N/D"
            if len(pils) >= 1:
                px1, pz1 = pils[0]["x"], pils[0]["z"]
                d_p1_ini = math.hypot(px1 - x0, pz1 - z0)
                d_p1_min = min(math.hypot(px1 - x, pz1 - z) for x, z in zip(xs, zs))
                p1_str = f"{d_p1_ini:4.1f}m -> {d_p1_min:4.1f}m"
            if len(pils) >= 2:
                px2, pz2 = pils[1]["x"], pils[1]["z"]
                d_p2_ini = math.hypot(px2 - x0, pz2 - z0)
                d_p2_min = min(math.hypot(px2 - x, pz2 - z) for x, z in zip(xs, zs))
                p2_str = f"{d_p2_ini:4.1f}m -> {d_p2_min:4.1f}m"

            print(f"Env {i:<2} | ({x0:5.1f}, {z0:5.1f})     | ({xf:5.1f}, {zf:5.1f})     | {p1_str:<17} | {p2_str:<17}")
        else:
            print(f"Env {i:<2} | Sem dados suficientes")
    print("=" * 80)
    print("Abra o mapa 2D no seu navegador: file:///" + os.path.abspath("ferramentas/mapa_trajetorias.html").replace("\\", "/"))

def gerar_html(historico):
    os.makedirs("ferramentas", exist_ok=True)
    html_content = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Vista de Cima 2D — Trajetórias e Pilares de Destino</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }
        h1 { color: #38bdf8; text-align: center; margin-bottom: 5px; }
        p.subtitle { text-align: center; color: #94a3b8; margin-top: 0; }
        .legend { display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; font-size: 14px; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; max-width: 1500px; margin: 0 auto; }
        .card { background: #1e293b; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); }
        .card h3 { margin-top: 0; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 8px; font-size: 15px; }
        canvas { background: #0b0f19; border-radius: 8px; width: 100%; height: 280px; display: block; }
        .stats { margin-top: 10px; font-size: 12px; color: #94a3b8; display: flex; flex-direction: column; gap: 4px; }
        .badge { background: #0284c7; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🗺️ Vista Aérea 2D — Trajetórias e Pilares de Destino</h1>
    <p class="subtitle">Visualização top-down dos 8 robôs navegando entre os pilares coloridos no mundo natural</p>
    <div class="legend">
        <div class="legend-item"><span class="dot" style="background:#22c55e;"></span> Largada</div>
        <div class="legend-item"><span class="dot" style="background:#eab308;"></span> Pilar Amarelo (Ouro)</div>
        <div class="legend-item"><span class="dot" style="background:#a855f7;"></span> Pilar Roxo (Obsidiana)</div>
        <div class="legend-item"><span class="dot" style="background:#3b82f6;"></span> Pilar Azul (Lápis)</div>
        <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Rastro do Robô</div>
    </div>
    <div class="grid" id="grid"></div>

    <script>
        const data = """ + json.dumps(historico) + """;
        const coresHex = """ + json.dumps(CORES_HEX) + """;
        const container = document.getElementById('grid');

        for (let i = 0; i < 8; i++) {
            const h = data[i];
            if (!h || h.x.length === 0) continue;

            const card = document.createElement('div');
            card.className = 'card';

            const xs = h.x, zs = h.z, pils = h.pilares || [];
            const x0 = xs[0], z0 = zs[0], xf = xs[xs.length - 1], zf = zs[zs.length - 1];

            card.innerHTML = `
                <h3>Ambiente ${i} <span class="badge" style="float:right;">Env ${i}</span></h3>
                <canvas id="cv_${i}" width="400" height="320"></canvas>
                <div class="stats" id="st_${i}">
                    <div><strong>Largada:</strong> (${x0.toFixed(1)}, ${z0.toFixed(1)}) ➔ <strong>Atual:</strong> (${xf.toFixed(1)}, ${zf.toFixed(1)})</div>
                </div>
            `;
            container.appendChild(card);

            const cv = document.getElementById(`cv_${i}`);
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height;

            // Coleta todos os pontos para definir os limites de enquadramento
            let todosX = [...xs, x0, xf], todosZ = [...zs, z0, zf];
            pils.forEach(p => { todosX.push(p.x); todosZ.push(p.z); });

            let minX = Math.min(...todosX) - 4, maxX = Math.max(...todosX) + 4;
            let minZ = Math.min(...todosZ) - 4, maxZ = Math.max(...todosZ) + 4;
            const span = Math.max(maxX - minX, maxZ - minZ, 14);
            const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;

            function toScrX(x) { return W / 2 + ((x - cx) / span) * (W * 0.82); }
            function toScrY(z) { return H / 2 + ((z - cz) / span) * (H * 0.82); }

            // Grid de coordenadas
            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 1;
            for (let g = -30; g <= 30; g += 5) {
                ctx.beginPath(); ctx.moveTo(toScrX(cx + g), 0); ctx.lineTo(toScrX(cx + g), H); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, toScrY(cz + g)); ctx.lineTo(W, toScrY(cz + g)); ctx.stroke();
            }

            // Desenha Pilares de Destino com Raio de Chegada (2.5m)
            pils.forEach((p, idx) => {
                const px = toScrX(p.x), py = toScrY(p.z);
                const cor = coresHex[p.cor] || '#f43f5e';

                // Círculo de Raio de Chegada (2.5m)
                const rPix = ((2.5) / span) * (W * 0.82);
                ctx.strokeStyle = cor;
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.arc(px, py, rPix, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);

                // Marcador do Pilar
                ctx.fillStyle = cor;
                ctx.beginPath();
                ctx.arc(px, py, 8, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Rótulo
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(`P${idx+1}: ${p.cor}`, px + 10, py + 4);
            });

            // Trajetória do Robô (Linha Ciano)
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 3;
            ctx.beginPath();
            for (let k = 0; k < xs.length; k++) {
                const sx = toScrX(xs[k]), sy = toScrY(zs[k]);
                if (k === 0) ctx.moveTo(sx, sy);
                else ctx.lineTo(sx, sy);
            }
            ctx.stroke();

            // Ponto de Partida (Verde)
            ctx.fillStyle = '#22c55e';
            ctx.beginPath();
            ctx.arc(toScrX(x0), toScrY(z0), 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Posição Atual do Robô
            ctx.fillStyle = '#f8fafc';
            ctx.beginPath();
            ctx.arc(toScrX(xf), toScrY(zf), 6, 0, Math.PI * 2);
            ctx.fill();
        }
    </script>
</body>
</html>"""
    with open("ferramentas/mapa_trajetorias.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def gerar_png(historico):
    os.makedirs("docs", exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), facecolor="#0f172a")
    fig.suptitle("Vista Aérea 2D — Trajetórias e Pilares de Destino (Fase 4)", color="#38bdf8", fontsize=16, fontweight="bold")

    axes = axes.flatten()
    for i in range(8):
        ax = axes[i]
        ax.set_facecolor("#0b0f19")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        h = historico[i]
        if not h or len(h["x"]) == 0:
            ax.set_title(f"Env {i} (Sem dados)", color="#64748b")
            continue

        xs = h["x"]
        zs = h["z"]
        x0, z0 = xs[0], zs[0]
        xf, zf = xs[-1], zs[-1]
        pils = h.get("pilares", [])

        # Plota os Pilares com Raio de Chegada
        for idx, p in enumerate(pils):
            cor = CORES_HEX.get(p["cor"], "#ef4444")
            circle = Circle((p["x"], p["z"]), 2.5, color=cor, fill=False, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.add_patch(circle)
            ax.scatter([p["x"]], [p["z"]], color=cor, s=110, edgecolors="white", linewidth=1.5, zorder=5)
            ax.text(p["x"] + 0.6, p["z"] + 0.6, f"P{idx+1}:{p['cor']}", color="white", fontsize=9, fontweight="bold")

        # Plota Trajetória
        ax.plot(xs, zs, color="#38bdf8", linewidth=2.5, label="Trajetória", zorder=2)
        ax.scatter([x0], [z0], color="#22c55e", s=90, edgecolors="white", label="Largada", zorder=4)
        ax.scatter([xf], [zf], color="#ffffff", s=90, edgecolors="#38bdf8", label="Robô", zorder=6)

        ax.set_title(f"Env {i}", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig("docs/mapa_trajetorias.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

if __name__ == "__main__":
    monitorar_trajetorias(duracao_segundos=12, intervalo_amostragem=0.4)
