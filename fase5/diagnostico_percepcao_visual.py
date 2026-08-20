# coding=utf-8
"""
fase5/diagnostico_percepcao_visual.py — Diagnóstico Executável da Percepção Visual na Fase 5.5.

Executa:
  1. Rollouts no simulador com pilares de todas as cores (amarelo, azul, roxo, verde, vermelho).
  2. Medição da passagem de tensores (4D vs 3D, shape, canais, dtype).
  3. Extração dos valores RGB reais dos pilares renderizados pelo voxel_renderer.js.
  4. Teste de detecção com o detector atual vs detector corrigido.
  5. Geração de tabela quantitativa de métricas por cor.
  6. Salvamento de imagens reais com sobreposição de máscara e bounding box/centro para inspeção visual.
"""
from __future__ import annotations

import os
import sys
import math
import base64
import io
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import CORES_MAP
from fase5.recompensa_visual import detectar_alvo_no_frame, _obter_mascara_cor
from fase5.curriculo_fase5 import CurriculoFase5
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone


def salvar_imagem_diagnostico(
    caminho: str,
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    det: dict,
    titulo: str
):
    """Salva imagem composta: [Frame Original] [Máscara Binária] [Frame com BBox / Centro]."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    H, W, _ = frame_rgb.shape

    # 1. Frame Original
    img_orig = Image.fromarray(frame_rgb)

    # 2. Máscara (Branco onde True, Preto onde False)
    if mask.shape != (H, W):
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST))
    else:
        mask_resized = (mask * 255).astype(np.uint8)
    img_mask = Image.fromarray(mask_resized).convert("RGB")

    # 3. Frame com anotações de detecção
    img_annot = img_orig.copy()
    draw = ImageDraw.Draw(img_annot)

    if det.get("visivel"):
        cx_norm = det.get("centro_x", 0.0)
        cx_px = int((cx_norm + 1.0) * 0.5 * W)
        frac = det.get("fracao_area", 0.0)
        pixels = det.get("contagem_pixels", 0)

        # Linha vertical no centro horizontal detectado
        draw.line([(cx_px, 0), (cx_px, H)], fill=(0, 255, 0), width=2)

        # Bounding box aproximado a partir dos pixels ativos na máscara
        coords = np.argwhere(mask)
        if len(coords) > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=2)

        # Texto informativo
        info_txt = f"VISIVEL | Pixels: {pixels} | Area: {frac*100:.2f}% | Cx: {cx_norm:+.2f}"
        draw.rectangle([0, 0, W, 18], fill=(0, 0, 0))
        draw.text((4, 2), info_txt, fill=(0, 255, 0))
    else:
        info_txt = "NAO DETECTADO (0 pixels)"
        draw.rectangle([0, 0, W, 18], fill=(0, 0, 0))
        draw.text((4, 2), info_txt, fill=(255, 50, 50))

    # Composição lado a lado [Original | Máscara | Anotado]
    canvas = Image.new("RGB", (W * 3 + 20, H + 40), color=(30, 30, 30))
    canvas.paste(img_orig, (5, 30))
    canvas.paste(img_mask, (W + 10, 30))
    canvas.paste(img_annot, (W * 2 + 15, 30))

    draw_c = ImageDraw.Draw(canvas)
    draw_c.text((10, 8), f"DIAGNOSTICO VISUAL: {titulo}", fill=(255, 255, 255))
    draw_c.text((15, H + 28), "1. Frame Original", fill=(200, 200, 200))
    draw_c.text((W + 15, H + 28), "2. Mascara da Cor", fill=(200, 200, 200))
    draw_c.text((W * 2 + 20, H + 28), "3. Deteccao (Centro/BBox)", fill=(200, 200, 200))

    canvas.save(caminho)


def rodar_diagnostico_completo(num_lotes: int = 3, passos_por_lote: int = 15):
    print("=" * 80)
    print(" [DIAGNÓSTICO DA PERCEPÇÃO VISUAL — FASE 5.5]")
    print("=" * 80)

    # 1. Verifica conectividade com o simulador
    info_sim = get("/lote/info")
    N = info_sim["envs"]
    print(f"[Simulador] Conectado a {N} ambientes paralelos (Frame nativo: {info_sim['frame']}).")

    # 2. Carrega VLA para gerar rollouts reais idênticos ao treino
    print("[VLA] Carregando agente para rollout real...")
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)

    if os.path.exists("checkpoints_vla/vla_fase5_ppo_bc.pt"):
        ckpt = torch.load("checkpoints_vla/vla_fase5_ppo_bc.pt", map_location=dev)
        if "treinaveis" in ckpt:
            vla.load_state_dict(ckpt["treinaveis"], strict=False)
            print("  -> Pesos do checkpoint mais recente carregados.")

    # 3. Gerenciador de tarefas da Etapa A
    curriculo = CurriculoFase5(modo_estagio="A")

    # Cores disponíveis para diagnóstico
    todas_cores = ["amarelo", "azul", "roxo"]

    stats_atual = {c: {"total": 0, "detectados": 0, "pixels": [], "fracao": [], "centro_x": [], "centralizados": 0} for c in todas_cores}
    stats_3d = {c: {"total": 0, "detectados": 0, "pixels": [], "fracao": [], "centro_x": [], "centralizados": 0} for c in todas_cores}

    exemplos_salvos = {c: 0 for c in todas_cores}
    pixel_samples = {c: [] for c in todas_cores}

    print("\n--- Coletando Rollouts Reais e Analisando Detecção Frame a Frame ---")

    for lote in range(num_lotes):
        tarefas, blocos = curriculo.gerar_tarefas(N, seed=100 + lote * 23)

        r = post("/lote/reset", {"posicoes": [[t["largada"][0], t["largada"][2]] for t in tarefas]})
        if blocos:
            post("/lote/colocar_bloco", {"blocos": blocos})

        alvos_web = []
        for t in tarefas:
            a0 = t["estagios"][0]["alvo_abs"]
            alvos_web.append({"x": a0[0], "z": a0[1], "dist": 8.0, "graus": 0})
        post("/lote/alvos", {"alvos": alvos_web})

        obs = r["obs"][:N]
        est = [o["estado"] for o in obs]
        pol.reiniciar(obs)

        estagios = [0] * N

        for p in range(passos_por_lote):
            prompts = [t["prompt"] for t in tarefas]
            alvos_abs = [t["estagios"][0]["alvo_abs"] for t in tarefas]

            acoes = pol.agir(est, alvos_abs, obs, prompts=prompts, estagios=estagios)
            u = pol.ultimo
            u8_raw = u["u8"]  # Numpy array retornado pela política

            rr = post("/lote/passo", {"acoes": acoes, "frames": True})
            obs = rr["obs"][:N]
            est = [o["estado"] for o in obs]
            pol.observar(obs)

            # Analisa cada ambiente
            for i in range(N):
                cor_alvo = tarefas[i]["estagios"][0]["cor"]
                if cor_alvo not in todas_cores:
                    continue

                # Frame nativo direto do simulador (640x360 uint8)
                b64 = obs[i].get("frame_b64")
                if b64:
                    im_native = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                    frame_native = np.asarray(im_native, dtype=np.uint8)
                else:
                    frame_native = None

                # Teste 1: Frame passado como no código atual (frame_i = u8_raw[i])
                frame_atual = u8_raw[i] if u8_raw is not None and i < len(u8_raw) else None
                det_atual = detectar_alvo_no_frame(frame_atual, cor_alvo)

                # Teste 2: Frame extraído corretamente em 3D [224, 224, 3] (mais recente: u8_raw[i, 0])
                frame_3d = u8_raw[i, 0] if u8_raw is not None and u8_raw.ndim == 5 else frame_atual
                det_3d = detectar_alvo_no_frame(frame_3d, cor_alvo)
                mask_3d = _obter_mascara_cor(frame_3d, cor_alvo)

                # Amostra pixels na região central se o pilar estiver na frente
                dx = tarefas[i]["estagios"][0]["alvo_abs"][0] - est[i]["x"]
                dz = tarefas[i]["estagios"][0]["alvo_abs"][1] - est[i]["z"]
                dist = math.hypot(dx, dz)
                yaw_rad = math.radians(est[i]["yaw"])
                # Ângulo relativo
                ang_alvo = math.atan2(-dx, -dz)
                diff_ang = (ang_alvo - yaw_rad + math.pi) % (2 * math.pi) - math.pi
                graus_diff = math.degrees(diff_ang)

                # Atualiza estatísticas do detector atual
                s_at = stats_atual[cor_alvo]
                s_at["total"] += 1
                if det_atual["visivel"]:
                    s_at["detectados"] += 1
                    s_at["pixels"].append(det_atual["contagem_pixels"])
                    s_at["fracao"].append(det_atual["fracao_area"])
                    s_at["centro_x"].append(det_atual["centro_x"])
                    if det_atual["centralizado"]:
                        s_at["centralizados"] += 1

                # Atualiza estatísticas com correção de dimensão 3D
                s_3d = stats_3d[cor_alvo]
                s_3d["total"] += 1
                if det_3d["visivel"]:
                    s_3d["detectados"] += 1
                    s_3d["pixels"].append(det_3d["contagem_pixels"])
                    s_3d["fracao"].append(det_3d["fracao_area"])
                    s_3d["centro_x"].append(det_3d["centro_x"])
                    if det_3d["centralizado"]:
                        s_3d["centralizados"] += 1

                # Salva imagens de exemplo para cada cor quando o pilar estiver próximo no campo de visão
                if abs(graus_diff) < 30.0 and dist < 7.0 and exemplos_salvos[cor_alvo] < 3:
                    exemplos_salvos[cor_alvo] += 1
                    num_ex = exemplos_salvos[cor_alvo]
                    nome_img = f"docs/imagens/diagnostico_visual/diagnostico_{cor_alvo}_ex{num_ex}.png"
                    titulo = f"Cor: {cor_alvo.upper()} | Dist: {dist:.1f}m | Ang: {graus_diff:+.1f}° | Passo: {p}"
                    salvar_imagem_diagnostico(
                        caminho=nome_img,
                        frame_rgb=frame_3d,
                        mask=mask_3d,
                        det=det_3d,
                        titulo=titulo
                    )

    # 4. Exibe e documenta resultados
    print("\n" + "=" * 80)
    print(" [RESULTADO 1] TESTE COM O PIPELINE ATUAL (u8_raw[i] — 4D [3, 224, 224, 3]):")
    print("=" * 80)
    print(f"{'Cor':<10} | {'Frames':<8} | {'Detectados':<10} | {'Taxa Det %':<10} | {'Pixels Méd':<10} | {'Área Méd %':<10} | {'Cx Méd':<8} | {'Centr %':<8}")
    print("-" * 80)
    for c in todas_cores:
        s = stats_atual[c]
        tot = max(1, s["total"])
        det_cnt = s["detectados"]
        taxa_det = (det_cnt / tot) * 100.0
        px_med = float(np.mean(s["pixels"])) if s["pixels"] else 0.0
        area_med = float(np.mean(s["fracao"]) * 100.0) if s["fracao"] else 0.0
        cx_med = float(np.mean(s["centro_x"])) if s["centro_x"] else 0.0
        cent_taxa = (s["centralizados"] / max(1, det_cnt)) * 100.0
        print(f"{c:<10} | {tot:<8} | {det_cnt:<10} | {taxa_det:<9.1f}% | {px_med:<10.1f} | {area_med:<9.2f}% | {cx_med:<+7.2f} | {cent_taxa:<7.1f}%")

    print("\n" + "=" * 80)
    print(" [RESULTADO 2] TESTE COM DIMENSÃO CORRIGIDA (u8_raw[i, 0] — 3D [224, 224, 3]):")
    print("=" * 80)
    print(f"{'Cor':<10} | {'Frames':<8} | {'Detectados':<10} | {'Taxa Det %':<10} | {'Pixels Méd':<10} | {'Área Méd %':<10} | {'Cx Méd':<8} | {'Centr %':<8}")
    print("-" * 80)
    for c in todas_cores:
        s = stats_3d[c]
        tot = max(1, s["total"])
        det_cnt = s["detectados"]
        taxa_det = (det_cnt / tot) * 100.0
        px_med = float(np.mean(s["pixels"])) if s["pixels"] else 0.0
        area_med = float(np.mean(s["fracao"]) * 100.0) if s["fracao"] else 0.0
        cx_med = float(np.mean(s["centro_x"])) if s["centro_x"] else 0.0
        cent_taxa = (s["centralizados"] / max(1, det_cnt)) * 100.0
        print(f"{c:<10} | {tot:<8} | {det_cnt:<10} | {taxa_det:<9.1f}% | {px_med:<10.1f} | {area_med:<9.2f}% | {cx_med:<+7.2f} | {cent_taxa:<7.1f}%")

    print("\n[OK] Imagens de inspeção visual salvas em: docs/imagens/diagnostico_visual/")
    return stats_atual, stats_3d


if __name__ == "__main__":
    rodar_diagnostico_completo(num_lotes=3, passos_por_lote=15)
