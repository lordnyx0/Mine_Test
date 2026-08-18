# coding=utf-8
"""
FASE 3 — Alvo VISUAL (Servo-Visual Puro, sem canal de coordenadas).

A pergunta mais forte da ponte multimodal:
    O modelo consegue navegar até um objeto VISÍVEL no campo de visão, sem receber coordenadas?

Na Fase 1 e 2 o vetor de objetivo continha (frente, lado, dist, angulo), permitindo
que a rede usasse trigonometria (atan2) sem olhar os pixels.
Aqui o canal de coordenadas é REMOVIDO (ou zerado): a visão precisa ser NECESSÁRIA.
Piso cego vai a ~0%; teto é o planejador BFS (Objetivos.bloco).
"""
import os
import sys
import math
import json
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiente.arena_plana import post, get, seco, YAW_BINS, RAIO_CHEGADA

PASSOS_MAX_F3 = int(os.environ.get("PASSOS_MAX_F3", "80"))
RAIO_CHEGADA_F3 = 2.0

CORES_ALVO = [
    {"nome": "roxo", "bloco_id": 49, "prompt": "Objetivo: va ate o bloco roxo."},
    {"nome": "amarelo", "bloco_id": 41, "prompt": "Objetivo: va ate o bloco amarelo."},
    {"nome": "azul", "bloco_id": 22, "prompt": "Objetivo: va ate o bloco azul."}
]


# ── Baselines de Referência ──────────────────────────────────────────────────
class AleatorioVisual:
    """O PISO da Fase 3: vaga aleatoriamente sem ler a imagem (taxa esperada ~0%)."""
    nome = "aleatorio"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        acoes = []
        for _ in ests:
            dx = self.rng.choice(YAW_BINS)
            acoes.append({"hold": ["W"], "mouse": [int(dx), 0], "duration_ms": 250})
        return acoes


class SempreFrente:
    """Controle cego trivial: apenas anda em linha reta."""
    nome = "so_W"

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        return [{"hold": ["W"], "mouse": [0, 0], "duration_ms": 250} for _ in ests]


class PilotoBloco:
    """O TETO da Fase 3: planejador BFS com acesso a voxels rumo ao bloco alvo."""
    nome = "piloto_bloco"

    def __init__(self, raio=40):
        self.raio = raio

    def reiniciar(self, obs):
        pass

    def agir(self, ests, alvos_abs, obs):
        extras = [{"alvo": {"x": a[0], "y": e["y"], "z": a[1]}, "raio": self.raio}
                  for e, a in zip(ests, alvos_abs)]
        return post("/lote/piloto", {"objetivo": "ponto", "extras": extras})["acoes"][:len(ests)]


# ── Gerador de Tarefas Visuais Multi-Cores (Roxo, Amarelo, Azul) ─────────────
def montar_tarefas_visuais(n_ep, seed=0, verbose=True, curriculo_frac=1.0):
    """Gera tarefas onde um pilar de BLOCO COLORIDO (Roxo, Amarelo, Azul) é colocado no cone de visão."""
    rng = random.Random(seed)
    N = get("/lote/info")["envs"]
    tarefas = []

    # Currículo suave: escala a distância máxima de 12m até 24m
    dist_max = 12.0 + 12.0 * max(0.0, min(1.0, float(curriculo_frac)))

    tentativas = 0
    while len(tarefas) < n_ep and tentativas < 100:
        tentativas += 1
        post("/lote/reset", {})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * N,
                                 "frames": False, "diag": True})

        blocos_a_colocar = []
        lote_temp = []

        for env_id, o in enumerate(r["obs"][:N]):
            e = o["estado"]
            diag = o.get("diag", {})
            if seco(e, diag.get("agua_perto")) and e.get("on_ground"):
                # Sorteia uma cor alvo entre Roxo, Amarelo e Azul
                cor = rng.choice(CORES_ALVO)

                # Sorteia posição no cone de visão frontal (6.0m até dist_max, -40° a +40°)
                dist = rng.uniform(6.0, dist_max)
                desvio_ang = math.radians(rng.uniform(-40.0, 40.0))
                yaw_alvo = math.radians(e["yaw"]) + desvio_ang

                fx, fz = -math.sin(yaw_alvo), -math.cos(yaw_alvo)
                alvo_x = round(e["x"] + fx * dist, 1)
                alvo_z = round(e["z"] + fz * dist, 1)
                alvo_y = math.floor(e["y"])

                lote_temp.append({
                    "env": env_id,
                    "largada": (round(e["x"], 2), round(e["z"], 2)),
                    "alvo_abs": (alvo_x, alvo_z),
                    "alvo_y": alvo_y,
                    "y_bot": e["y"],
                    "dist_inicial": dist,
                    "alvo_nome": cor["nome"],
                    "bloco_id": cor["bloco_id"],
                    "prompt": cor["prompt"]
                })

                blocos_a_colocar.append({
                    "env": env_id,
                    "x": math.floor(alvo_x),
                    "y": alvo_y,
                    "z": math.floor(alvo_z),
                    "id": cor["bloco_id"]
                })

        # Insere os pilares coloridos firmemente no chão sólido do mundo virtual
        if blocos_a_colocar:
            resp_blocos = post("/lote/colocar_bloco", {"blocos": blocos_a_colocar})
            blocos_reais = resp_blocos.get("blocos", [])
            for idx_b, b_real in enumerate(blocos_reais):
                if idx_b < len(lote_temp):
                    lote_temp[idx_b]["alvo_y"] = b_real["y"]
                    # Filtra alvos que caíram em desníveis inacessíveis (> 3.5 blocos de altura)
                    if abs(b_real["y"] - lote_temp[idx_b]["y_bot"]) <= 3.5:
                        tarefas.append(lote_temp[idx_b])

    return tarefas[:n_ep]
