# coding=utf-8
"""
FASE 4 — Gerador de Tarefas de Raciocínio Lógico e Submetas Sequenciais no Minecraft.

Implementa 2 famílias de tarefas lógicas com verificação causal estrita e currículo angular:
  1. Sequenciamento de Submetas ("Vá ao bloco X, DEPOIS ao bloco Y"):
     - Treina memória de curto prazo e grafo de dependência temporal.
     - Suporta 3 níveis de currículo angular:
       - Nível 1 (Cone Suave): Pilar 2 a ±25° do vetor de avanço.
       - Nível 2 (Desvio Lateral): Pilar 2 a ±70°.
       - Nível 3 (Varredura Completa): Pilar 2 em 360° (±180°).
  2. Desvio de Barreira com Sub-objetivo Visual:
     - Bloqueio em linha reta por muro de pedra, com submeta na abertura desimpedida.
"""
import os
import json
import math
import random
from typing import List, Dict, Any

from ambiente.arena_plana import post, get, seco

PASSOS_MAX_F4 = 100
RAIO_CHEGADA_SUBMETA = 1.3
BONUS_SUBMETA = 5.0
BONUS_FINAL = 15.0

CORES_MAP = {
    "roxo": 49,     # Obsidiana
    "amarelo": 41,  # Bloco de Ouro
    "azul": 22      # Bloco de Lápis-lazúli
}

# Banco de largadas pré-validadas em solo 100% seco
_SECAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "largadas_secas.json")
_ORIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "largadas_fase2.json")
CAMINHO_LARGADAS = _SECAS if os.path.exists(_SECAS) else _ORIG
if os.path.exists(CAMINHO_LARGADAS):
    BANCO_LARGADAS = json.load(open(CAMINHO_LARGADAS, encoding="utf-8"))
else:
    BANCO_LARGADAS = [[143.5, 240.5], [166.5, -236.5], [62.5, 156.5], [266.5, -59.5]]


def montar_tarefas_logicas(num_ambientes: int, seed: int = 42, proporcao_seq: float = 0.6,
                           nivel_curriculo: int = 1) -> List[Dict[str, Any]]:
    """
    Gera tarefas de raciocínio sequencial com currículo angular progressivo.
    
    nivel_curriculo:
      1: Cone suave (±25° de dispersão angular para Pilar 2)
      2: Desvio moderado (±70° de dispersão)
      3: 360° total (±180° de dispersão)
    """
    rng = random.Random(seed)
    tarefas = []
    blocos_a_colocar = []
    posicoes_largada = []

    # Sorteia largadas secas do banco validado
    coordenadas_escolhidas = rng.sample(BANCO_LARGADAS, min(num_ambientes, len(BANCO_LARGADAS)))

    # Posiciona os robôs nas coordenadas secas primeiro para obter a altura y real do solo
    post("/lote/reset", {"posicoes": [[c[0], c[1]] for c in coordenadas_escolhidas]})
    r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * num_ambientes, "frames": False})

    for env_id, o in enumerate(r["obs"][:num_ambientes]):
        e = o["estado"]
        lx, ly, lz = e["x"], e["y"], e["z"]
        lyaw = math.radians(e["yaw"])
        posicoes_largada.append([round(lx, 2), round(ly, 2), round(lz, 2), lyaw])

        if rng.random() < proporcao_seq:
            # Sequenciamento: Pilar 1 (8-12m) -> Pilar 2 (8-14m a partir de P1)
            cor1, cor2 = rng.sample(list(CORES_MAP.keys()), 2)
            id1, id2 = CORES_MAP[cor1], CORES_MAP[cor2]

            # Pilar 1: Cone frontal de partida (6.5 a 9.5m — alta densidade de alcance)
            desvio1 = math.radians(rng.uniform(-18.0, 18.0))
            y1 = lyaw + desvio1
            d1 = rng.uniform(6.5, 9.5)
            tx1 = round(lx - math.sin(y1) * d1, 1)
            tz1 = round(lz - math.cos(y1) * d1, 1)
            ty1 = math.floor(ly)

            # Pilar 2: Geometria baseada no nível de currículo (7.0 a 10.0m)
            if nivel_curriculo == 1:
                limite_ang = 25.0
            elif nivel_curriculo == 2:
                limite_ang = 60.0
            else:
                limite_ang = 180.0

            desvio2 = math.radians(rng.uniform(-limite_ang, limite_ang))
            y2 = y1 + desvio2
            d2 = rng.uniform(7.0, 10.0)
            tx2 = round(tx1 - math.sin(y2) * d2, 1)
            tz2 = round(tz1 - math.cos(y2) * d2, 1)
            ty2 = math.floor(ly)

            prompt_completo = f"Vá até o bloco {cor1} e depois vá até o bloco {cor2}."

            tarefas.append({
                "tipo": "sequencia",
                "env": env_id,
                "nivel_curriculo": nivel_curriculo,
                "largada": (round(lx, 2), round(ly, 2), round(lz, 2), lyaw),
                "prompt": prompt_completo,
                "estagios": [
                    {
                        "cor": cor1,
                        "bloco_id": id1,
                        "alvo_abs": (tx1, tz1),
                        "alvo_y": ty1,
                        "prompt_estagio": f"Objetivo: vá até o bloco {cor1} [Etapa 1/2]"
                    },
                    {
                        "cor": cor2,
                        "bloco_id": id2,
                        "alvo_abs": (tx2, tz2),
                        "alvo_y": ty2,
                        "prompt_estagio": f"Objetivo: vá até o bloco {cor2} [Etapa 2/2]"
                    }
                ]
            })
            blocos_a_colocar.append({"env": env_id, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": id1, "altura": 50})
            blocos_a_colocar.append({"env": env_id, "x": math.floor(tx2), "y": ty2, "z": math.floor(tz2), "id": id2, "altura": 50})

        else:
            # Desvio de Barreira com Submeta na abertura (12 a 15m)
            cor_alvo = rng.choice(list(CORES_MAP.keys()))
            id_alvo = CORES_MAP[cor_alvo]

            d_alvo = rng.uniform(12.0, 15.0)
            tx = round(lx - math.sin(lyaw) * d_alvo, 1)
            tz = round(lz - math.cos(lyaw) * d_alvo, 1)
            ty = math.floor(ly)

            lado_abertura = rng.choice([-1, 1])
            dm = rng.uniform(4.0, 6.0)
            mx = round(lx - math.sin(lyaw) * dm, 1)
            mz = round(lz - math.cos(lyaw) * dm, 1)
            my = math.floor(ly)

            px = -math.cos(lyaw)
            pz = math.sin(lyaw)

            ax = round(mx + lado_abertura * px * 3.0, 1)
            az = round(mz + lado_abertura * pz * 3.0, 1)

            tarefas.append({
                "tipo": "desvio_barreira",
                "env": env_id,
                "largada": (round(lx, 2), round(ly, 2), round(lz, 2), lyaw),
                "prompt": f"Vá até a abertura do muro e depois vá até o bloco {cor_alvo}.",
                "estagios": [
                    {
                        "cor": "abertura_muro",
                        "bloco_id": 0,
                        "alvo_abs": (ax, az),
                        "alvo_y": my,
                        "prompt_estagio": "Objetivo: vá até a abertura do muro [Etapa 1/2]"
                    },
                    {
                        "cor": cor_alvo,
                        "bloco_id": id_alvo,
                        "alvo_abs": (tx, tz),
                        "alvo_y": ty,
                        "prompt_estagio": f"Objetivo: vá até o bloco {cor_alvo} [Etapa 2/2]"
                    }
                ]
            })
            blocos_a_colocar.append({"env": env_id, "x": math.floor(tx), "y": ty, "z": math.floor(tz), "id": id_alvo, "altura": 50})
            for desloc in range(-4, 5):
                if (lado_abertura > 0 and desloc >= 2) or (lado_abertura < 0 and desloc <= -2):
                    continue
                bx = math.floor(mx + desloc * px * 1.0)
                bz = math.floor(mz + desloc * pz * 1.0)
                for h in range(1, 4):
                    blocos_a_colocar.append({"env": env_id, "x": bx, "y": my + h, "z": bz, "id": 4})

    # 1. Reseta todos os robôs para as largadas secas
    post("/lote/reset", {"posicoes": [[p[0], p[2]] for p in posicoes_largada]})

    # 2. Coloca os pilares no mundo após o reset
    if blocos_a_colocar:
        post("/lote/colocar_bloco", {"blocos": blocos_a_colocar})

    try:
        os.makedirs("dataset", exist_ok=True)
        with open("dataset/tarefas_ativas.json", "w", encoding="utf-8") as f:
            json.dump(tarefas, f)
    except Exception:
        pass

    return tarefas
