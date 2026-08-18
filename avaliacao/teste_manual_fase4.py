# coding=utf-8
"""
teste_manual_fase4.py — Execucao de 1 Episodio de Teste da Fase 4 com o Modelo Treinado.
"""
import os
import sys
import math
import time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import CORES_MAP, BANCO_LARGADAS, RAIO_CHEGADA_SUBMETA
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora

def main():
    print("=" * 70)
    print(" [TESTE UNITARIO] NAVEGACAO LOGICA MULTI-ALVO (FASE 4)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints_vla/vla_fase4_logica.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints_vla/vla_fase3_merged.pt"

    print(f"[1/4] Carregando modelo do checkpoint '{ckpt_path}'...")
    vla, device = load_vla_agent(ckpt_path)
    compactar_backbone(vla)
    if not any("lora_" in n for n, _ in vla.named_parameters()):
        vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)
    vla.to(device)
    vla.eval()

    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=device, vla=vla, loops_pensamento=3)
    pol.amostrar = False

    # 1. Configura 1 tarefa de 2 estágios: Amarelo -> Azul no Ambiente 0
    print("[2/4] Configurando tarefa logica no Ambiente 0: Amarelo -> Azul...")
    coord = BANCO_LARGADAS[0] # largada seca
    post("/lote/reset", {"posicoes": [[coord[0], coord[1]]] * 8})
    r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * 8, "frames": False})
    
    e0 = r["obs"][0]["estado"]
    lx, ly, lz = e0["x"], e0["y"], e0["z"]
    lyaw = math.radians(e0["yaw"])
    
    # Pilar 1: Amarelo a 10m (+15 graus)
    y1 = lyaw + math.radians(15.0)
    tx1 = round(lx - math.sin(y1) * 10.0, 1)
    tz1 = round(lz - math.cos(y1) * 10.0, 1)
    ty1 = math.floor(ly)

    # Pilar 2: Azul a 16m (-20 graus)
    y2 = lyaw + math.radians(-20.0)
    tx2 = round(lx - math.sin(y2) * 16.0, 1)
    tz2 = round(lz - math.cos(y2) * 16.0, 1)
    ty2 = math.floor(ly)

    blocos = [
        {"env": 0, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": CORES_MAP["amarelo"]},
        {"env": 0, "x": math.floor(tx2), "y": ty2, "z": math.floor(tz2), "id": CORES_MAP["azul"]}
    ]
    post("/lote/colocar_bloco", {"blocos": blocos})
    post("/lote/reset", {"posicoes": [[lx, ly, lz, lyaw]] * 8})

    prompt = "Vá até o bloco amarelo e depois vá até o bloco azul."
    estagios = [
        {"nome": "AMARELO", "pos": (tx1, tz1)},
        {"nome": "AZUL", "pos": (tx2, tz2)}
    ]
    estagio_atual = 0

    print("Largada: (%.1f, %.1f, %.1f) | Yaw: %.1f deg" % (lx, ly, lz, e0['yaw']))
    print("Pilar 1 (AMARELO): (%.1f, %.1f) a 10.0m" % (tx1, tz1))
    print("Pilar 2 (AZUL):    (%.1f, %.1f) a 16.0m" % (tx2, tz2))
    print("-" * 70)

    # 2. Executa o loop de controle de 100 passos
    r_step = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * 8, "frames": True})
    obs = r_step["obs"][:8]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    sucesso_total = False
    passo_chegou_1 = None
    passo_chegou_2 = None

    print("%-6s | %-22s | %-7s | %-12s | %-7s | %-20s" %
          ("PASSO", "POSICAO (X, Y, Z)", "YAW", "ALVO ATIVO", "DIST", "ACAO EMITIDA"))
    print("-" * 85)

    for p in range(100):
        prompts = [prompt] * 8
        acoes = pol.agir(est, [None] * 8, obs, prompts=prompts)
        u = pol.ultimo
        u["sv"][0, 16] = float(estagio_atual)

        acao_bot0 = acoes[0]
        dx = acao_bot0["mouse"][0]
        hold_str = "+".join(acao_bot0["hold"])
        acao_str = "%s | yaw %+d" % (hold_str, dx)

        r_step = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = r_step["obs"][:8]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        e = est[0]
        alvo_atual = estagios[estagio_atual]
        dist = math.hypot(alvo_atual["pos"][0] - e["x"], alvo_atual["pos"][1] - e["z"])

        if p % 5 == 0 or dist <= RAIO_CHEGADA_SUBMETA or p == 99:
            pos_str = "(%.1f, %.1f, %.1f)" % (e['x'], e['y'], e['z'])
            print("P %03d  | %-22s | %5.1f deg | %-12s | %5.1fm | %s"
                  % (p+1, pos_str, e['yaw'], alvo_atual['nome'], dist, acao_str))

        if dist <= RAIO_CHEGADA_SUBMETA:
            if estagio_atual == 0:
                print("  [OK] SUBMETA 1 ALCANCADA! (Pilar Amarelo atingido no passo %d)" % (p+1))
                passo_chegou_1 = p + 1
                estagio_atual = 1
            else:
                print("  [VITORIA] TAREFA COMPLETA CONCLUIDA! (Pilar Azul atingido no passo %d)" % (p+1))
                passo_chegou_2 = p + 1
                sucesso_total = True
                break

    print("=" * 70)
    if sucesso_total:
        print("[SUCESSO TOTAL] Amarelo em %d passos | Azul em %d passos!" % (passo_chegou_1, passo_chegou_2))
    elif passo_chegou_1:
        print("[SUCESSO PARCIAL] Amarelo em %d passos, mas Azul nao foi alcancado a tempo." % passo_chegou_1)
    else:
        print("[FALHA] O robo nao alcancou a primeira submeta dentro dos 100 passos.")

if __name__ == "__main__":
    main()
