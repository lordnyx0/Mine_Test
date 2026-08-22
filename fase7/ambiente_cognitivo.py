# coding=utf-8
"""
fase7/ambiente_cognitivo.py — Gerador de Tarefas no Terreno Natural do Minecraft (100% Orgânico).

Elimina muros artificiais: o terreno procedural do jogo (árvores, folhas, colinas, desníveis, água)
fornece toda a complexidade geométrica e visual natural.
"""
from __future__ import annotations
import os
import sys
import math
import random
from typing import Dict, List, Tuple, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post, get
from fase5.curriculo_fase5 import CORES_MAP
from ambiente.tarefas_logicas import BANCO_LARGADAS


class AmbienteCognitivoFase7:
    """Gerencia o posicionamento de pilares-alvo no terreno natural do Minecraft."""
    def __init__(self, tipo_cenario: str = "natural"):
        self.tipo_cenario = tipo_cenario
        self.banco = list(BANCO_LARGADAS)
        self.nivel = 1

    def gerar_tarefas_cognitivas(
        self,
        num_ambientes: int,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Gera posições de largada no mapa natural e posiciona o pilar alvo à frente.
        Zero muros artificiais colocados.
        """
        rng = random.Random(seed)
        coords = rng.sample(self.banco, min(num_ambientes, len(self.banco)))

        post("/lote/reset", {"posicoes": [[c[0], c[1]] for c in coords]})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * num_ambientes, "frames": False})

        tarefas = []
        blocos_cenario = []

        cores_disp = list(CORES_MAP.keys())

        for env_id, o in enumerate(r["obs"][:num_ambientes]):
            e = o["estado"]
            lx, ly, lz = e["x"], e["y"], e["z"]
            lyaw = math.radians(e["yaw"])

            cor_alvo = rng.choice(cores_disp)
            id_alvo = CORES_MAP[cor_alvo]

            # Posicionamento do Pilar Alvo no terreno natural (6 a 9 metros de distância)
            desvio_alvo = math.radians(rng.uniform(-30.0, 30.0))
            yaw_alvo = lyaw + desvio_alvo
            dist_alvo = rng.uniform(6.0, 9.0)

            tx = round(lx - math.sin(yaw_alvo) * dist_alvo, 1)
            tz = round(lz + math.cos(yaw_alvo) * dist_alvo, 1)
            ty = ly

            # Coloca apenas o pilar alvo visual (3 blocos de altura)
            for dy in range(3):
                blocos_cenario.append({"x": tx, "y": ty + dy, "z": tz, "block": id_alvo})

            prompt_texto = f"Missão: Vá até o pilar {cor_alvo}."

            tarefas.append({
                "env_id": env_id,
                "largada": [lx, ly, lz, e["yaw"]],
                "alvo_abs": (tx, tz),
                "alvo_cor": cor_alvo,
                "dist_alvo": round(dist_alvo, 2),
                "prompt": prompt_texto
            })

        return tarefas, blocos_cenario
