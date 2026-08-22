# coding=utf-8
"""
fase5/investigacao_logits_yaw.py — Análise Detalhada dos Logits das Cabeças de Modo e Yaw.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.acoes_taticas import MODOS, YAW_BINS_9
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone


def investigar_logits():
    print("=" * 80)
    print(" [INVESTIGAÇÃO DE LOGITS E BIAS DA CABEÇA DE YAW/MODO]")
    print("=" * 80)

    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3, num_acoes=36, fatorada=True)

    ckpt_path = "checkpoints_vla/vla_fase5_ppo_bc.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=dev)
        if "treinaveis" in ckpt:
            vla.load_state_dict(ckpt["treinaveis"], strict=False)
            print(f"[VLA] Checkpoint '{ckpt_path}' carregado com sucesso.")

    vla.eval()

    print("\n--- 1. Estrutura das Cabeças cabeca_modo e cabeca_yaw ---")
    print(f"cabeca_yaw : {vla.cabeca_yaw}")
    print(f"cabeca_modo: {vla.cabeca_modo}")

    # 2. Teste de forward com intervenção visual
    H, W = 224, 224
    frame_esq = np.zeros((H, W, 3), dtype=np.uint8)
    frame_esq[40:190, 20:50] = [245, 215, 20]  # x ~ 35 (cx ~ -0.69)

    frame_cen = np.zeros((H, W, 3), dtype=np.uint8)
    frame_cen[40:190, 97:127] = [245, 215, 20] # x ~ 112 (cx ~ 0.00)

    frame_dir = np.zeros((H, W, 3), dtype=np.uint8)
    frame_dir[40:190, 174:204] = [245, 215, 20] # x ~ 189 (cx ~ +0.69)

    sv = torch.zeros((1, 32), dtype=torch.float32, device=dev)
    sv[0, 2] = 0.35
    sv[0, 3] = 1.0
    gv = torch.zeros((1, 4), dtype=torch.float32, device=dev)

    prompt_amarelo = "Objetivo: vá até o bloco amarelo [Etapa 1/1]"
    ids_amarelo = pol.obter_ids([prompt_amarelo])

    def get_logits(frame, ids):
        u8_stack = np.stack([frame, frame, frame], axis=0)
        u8_batch = np.expand_dims(u8_stack, axis=0)
        px = pol.normalizar(u8_batch)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg_modo, lg_yaw, val = pol.forward_pensamento(px, sv, gv, ids)
        return lg_modo[0].cpu().to(torch.float32).numpy(), lg_yaw[0].cpu().to(torch.float32).numpy()

    lm_esq, ly_esq = get_logits(frame_esq, ids_amarelo)
    lm_cen, ly_cen = get_logits(frame_cen, ids_amarelo)
    lm_dir, ly_dir = get_logits(frame_dir, ids_amarelo)

    print("\n--- 2. Logits de Yaw Pré-Softmax por Posição Visual ---")
    print(f"{'Bin':<5} | {'dx Mouse':<10} | {'Logit (Esq)':<12} | {'Logit (Centro)':<14} | {'Logit (Dir)':<12} | {'Delta(Esq - Cen)':<16} | {'Delta(Dir - Cen)'}")
    print("-" * 95)
    for i, b_val in enumerate(YAW_BINS_9):
        d_esq = ly_esq[i] - ly_cen[i]
        d_dir = ly_dir[i] - ly_cen[i]
        print(f"{i:<5} | {b_val:+5d}      | {ly_esq[i]:+10.4f}   | {ly_cen[i]:+10.4f}     | {ly_dir[i]:+10.4f}   | {d_esq:+14.4f}   | {d_dir:+14.4f}")

    print("\n--- 3. Probabilidades de Yaw Softmax (Temperatura = 1.0) ---")
    py_esq = F.softmax(torch.tensor(ly_esq), dim=-1).numpy()
    py_cen = F.softmax(torch.tensor(ly_cen), dim=-1).numpy()
    py_dir = F.softmax(torch.tensor(ly_dir), dim=-1).numpy()

    print(f"{'Bin':<5} | {'dx Mouse':<10} | {'P(Esq) %':<10} | {'P(Centro) %':<12} | {'P(Dir) %':<10}")
    print("-" * 55)
    for i, b_val in enumerate(YAW_BINS_9):
        print(f"{i:<5} | {b_val:+5d}      | {py_esq[i]*100:8.2f}% | {py_cen[i]*100:10.2f}% | {py_dir[i]*100:8.2f}%")

    print("\n--- 4. Logits de Modo Pré-Softmax por Posição Visual ---")
    print(f"{'Modo (0..5)':<14} | {'Logit (Esq)':<12} | {'Logit (Centro)':<14} | {'Logit (Dir)':<12} | {'P(Esq) %':<10} | {'P(Centro) %':<12} | {'P(Dir) %'}")
    print("-" * 95)
    pm_esq = F.softmax(torch.tensor(lm_esq), dim=-1).numpy()
    pm_cen = F.softmax(torch.tensor(lm_cen), dim=-1).numpy()
    pm_dir = F.softmax(torch.tensor(lm_dir), dim=-1).numpy()
    for i, m_name in enumerate(MODOS):
        print(f"{m_name:<14} | {lm_esq[i]:+10.4f}   | {lm_cen[i]:+10.4f}     | {lm_dir[i]:+10.4f}   | {pm_esq[i]*100:8.2f}% | {pm_cen[i]*100:10.2f}% | {pm_dir[i]*100:8.2f}%")

    # 5. Teste de intervenção de prompt
    prompt_roxo = "Objetivo: vá até o bloco roxo [Etapa 1/1]"
    ids_roxo = pol.obter_ids([prompt_roxo])
    lm_cen_roxo, ly_cen_roxo = get_logits(frame_cen, ids_roxo)

    print("\n--- 5. Intervenção de Prompt (Mesmo Frame Amarelo no Centro) ---")
    print(f"{'Prompt':<45} | {'Top Modo':<12} | {'P(Modo)':<8} | {'Top Yaw (dx)':<14} | {'P(Yaw)'}")
    print("-" * 90)
    top_m_am = int(np.argmax(pm_cen))
    top_y_am = int(np.argmax(py_cen))
    print(f"{prompt_amarelo:<45} | {MODOS[top_m_am]:<12} | {pm_cen[top_m_am]*100:5.1f}%   | {YAW_BINS_9[top_y_am]:+4d} (Bin {top_y_am})  | {py_cen[top_y_am]*100:5.1f}%")

    pm_rx = F.softmax(torch.tensor(lm_cen_roxo), dim=-1).numpy()
    py_rx = F.softmax(torch.tensor(ly_cen_roxo), dim=-1).numpy()
    top_m_rx = int(np.argmax(pm_rx))
    top_y_rx = int(np.argmax(py_rx))
    print(f"{prompt_roxo:<45} | {MODOS[top_m_rx]:<12} | {pm_rx[top_m_rx]*100:5.1f}%   | {YAW_BINS_9[top_y_rx]:+4d} (Bin {top_y_rx})  | {py_rx[top_y_rx]*100:5.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    investigar_logits()
