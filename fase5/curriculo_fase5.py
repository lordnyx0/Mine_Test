# coding=utf-8
"""
fase5/curriculo_fase5.py — Gerenciador de Currículo Progressivo da Fase 5.5.

Estrutura Curricular:
  ETAPA A (Pilar 1 Único — Fácil):
    - 1 único pilar (submeta única)
    - Distância curta: 4.0m a 6.5m
    - Dispersão angular frontal: ±35°
    - Foco: aprender descoberta visual e navegação direta
    - Critério de avanço: Submeta 1 >= 35% por 3 iterações

  ETAPA B (Pilar 1 -> Pilar 2 — Moderado):
    - 2 pilares sequenciais
    - Distâncias moderadas: 5.5m a 8.0m
    - Dispersões moderadas: ±60° (Pilar 1), ±75° (Pilar 2)
    - Foco: transição e reorientação intermediária
    - Critério de avanço: Sucesso Total >= 20% ou Submeta 1 >= 50% por 3 iterações

  ETAPA C (Tarefa Completa Atual — Plena):
    - 2 pilares sequenciais com dispersão total
    - Distâncias plenas: 6.5m a 9.5m (P1), 7.0m a 10.0m (P2)
    - Dispersão ampla: ±75° (P1), ±110° (P2)
    - Foco: generalização espacial e busca ativa completa
"""
from __future__ import annotations
import os
import sys
import math
import random
import json
from typing import List, Dict, Any, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ambiente.arena_plana import post
from ambiente.tarefas_logicas import CORES_MAP


class CurriculoFase5:
    """Gerencia a progressão e geração de tarefas em múltiplos estágios de dificuldade."""

    ESTAGIOS = ["A", "B", "C"]

    def __init__(
        self,
        modo_estagio: str = "auto",
        criterio_a: float = 0.35,
        criterio_b: float = 0.20,
        janela_estabilidade: int = 3
    ):
        """
        modo_estagio: 'A', 'B', 'C' (estágio fixo) ou 'auto' (progressão adaptativa baseada em desempenho)
        """
        self.modo = modo_estagio.upper()
        self.criterio_a = criterio_a
        self.criterio_b = criterio_b
        self.janela_estabilidade = janela_estabilidade

        if self.modo in self.ESTAGIOS:
            self.estagio_atual = self.modo
            self.auto = False
        else:
            self.estagio_atual = "A"
            self.auto = True

        self.historico_sub1: List[float] = []
        self.historico_sucesso: List[float] = []
        self.historico_rec: List[float] = []
        self.iteracoes_no_estagio = 0

        # Carrega banco de posições secas
        caminho_secas = os.path.join(_ROOT, "dataset", "largadas_secas.json")
        if os.path.exists(caminho_secas):
            with open(caminho_secas, "r", encoding="utf-8") as f:
                self.banco = json.load(f)
        else:
            from ambiente.tarefas_logicas import BANCO_LARGADAS
            self.banco = BANCO_LARGADAS

    def atualizar_desempenho(self, taxa_sub1: float, taxa_sucesso: float, recompensa: float) -> Tuple[bool, str]:
        """
        Atualiza o histórico com o resultado da iteração.
        Retorna (avancou: bool, mensagem_transicao: str).
        """
        self.historico_sub1.append(taxa_sub1)
        self.historico_sucesso.append(taxa_sucesso)
        self.historico_rec.append(recompensa)
        self.iteracoes_no_estagio += 1

        if not self.auto:
            return False, f"Estágio fixo: {self.estagio_atual}"

        if len(self.historico_sub1) < self.janela_estabilidade:
            return False, f"Acumulando histórico ({len(self.historico_sub1)}/{self.janela_estabilidade})"

        media_sub1 = float(sum(self.historico_sub1[-self.janela_estabilidade:]) / self.janela_estabilidade)
        media_sucesso = float(sum(self.historico_sucesso[-self.janela_estabilidade:]) / self.janela_estabilidade)

        if self.estagio_atual == "A":
            if media_sub1 >= self.criterio_a:
                self.estagio_atual = "B"
                self.iteracoes_no_estagio = 0
                msg = f">>> [CURRÍCULO] AVANÇO: ETAPA A -> ETAPA B (Sub1 média: {media_sub1*100:.1f}% >= {self.criterio_a*100:.1f}%)"
                return True, msg
        elif self.estagio_atual == "B":
            if media_sucesso >= self.criterio_b or media_sub1 >= 0.50:
                self.estagio_atual = "C"
                self.iteracoes_no_estagio = 0
                msg = f">>> [CURRÍCULO] AVANÇO: ETAPA B -> ETAPA C (Sucesso médio: {media_sucesso*100:.1f}%, Sub1: {media_sub1*100:.1f}%)"
                return True, msg

        return False, f"Permanecendo na ETAPA {self.estagio_atual} (Sub1: {media_sub1*100:.1f}%, Sucesso: {media_sucesso*100:.1f}%)"

    def gerar_tarefas(self, num_ambientes: int, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Gera a bateria de tarefas e blocos de acordo com a dificuldade do estágio curricular ativo."""
        rng = random.Random(seed)
        coords = rng.sample(self.banco, min(num_ambientes, len(self.banco)))

        post("/lote/reset", {"posicoes": [[c[0], c[1]] for c in coords]})
        r = post("/lote/passo", {"acoes": [{"hold": [], "mouse": [0, 0], "duration_ms": 50}] * num_ambientes, "frames": False})

        tarefas = []
        blocos = []

        for env_id, o in enumerate(r["obs"][:num_ambientes]):
            e = o["estado"]
            lx, ly, lz = e["x"], e["y"], e["z"]
            lyaw = math.radians(e["yaw"])

            cores_disp = list(CORES_MAP.keys())
            cor1, cor2 = rng.sample(cores_disp, 2)
            id1, id2 = CORES_MAP[cor1], CORES_MAP[cor2]

            if self.estagio_atual == "A":
                # ETAPA A: Pilar Único, frontal e curto
                desvio1 = math.radians(rng.uniform(-35.0, 35.0))
                y1 = lyaw + desvio1
                d1 = rng.uniform(4.0, 6.5)
                tx1 = round(lx - math.sin(y1) * d1, 1)
                tz1 = round(lz - math.cos(y1) * d1, 1)
                ty1 = math.floor(ly)

                tarefas.append({
                    "env_id": env_id,
                    "largada": (lx, ly, lz, e["yaw"]),
                    "prompt": f"Objetivo: vá até o bloco {cor1} [Etapa 1/1]",
                    "estagio_curriculo": "A",
                    "dist_media_teorica": d1,
                    "dispersao_graus": 35.0,
                    "estagios": [
                        {
                            "id": 0,
                            "cor": cor1,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor1} [Etapa 1/1]",
                            "alvo_abs": (tx1, tz1),
                            "altura_y": ty1,
                            "bloco_id": id1
                        }
                    ]
                })
                blocos.append({"env": env_id, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": id1, "altura": 50})

            elif self.estagio_atual == "B":
                # ETAPA B: Dois Pilares, distâncias e ângulos moderados
                desvio1 = math.radians(rng.uniform(-60.0, 60.0))
                y1 = lyaw + desvio1
                d1 = rng.uniform(5.5, 8.0)
                tx1 = round(lx - math.sin(y1) * d1, 1)
                tz1 = round(lz - math.cos(y1) * d1, 1)
                ty1 = math.floor(ly)

                desvio2 = math.radians(rng.uniform(-75.0, 75.0))
                y2 = y1 + desvio2
                d2 = rng.uniform(5.5, 8.0)
                tx2 = round(tx1 - math.sin(y2) * d2, 1)
                tz2 = round(tz1 - math.cos(y2) * d2, 1)
                ty2 = ty1

                tarefas.append({
                    "env_id": env_id,
                    "largada": (lx, ly, lz, e["yaw"]),
                    "prompt": f"Objetivo: vá até o bloco {cor1}, depois até o bloco {cor2}",
                    "estagio_curriculo": "B",
                    "dist_media_teorica": (d1 + d2) / 2.0,
                    "dispersao_graus": 60.0,
                    "estagios": [
                        {
                            "id": 0,
                            "cor": cor1,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor1} [Etapa 1/2]",
                            "alvo_abs": (tx1, tz1),
                            "altura_y": ty1,
                            "bloco_id": id1
                        },
                        {
                            "id": 1,
                            "cor": cor2,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor2} [Etapa 2/2]",
                            "alvo_abs": (tx2, tz2),
                            "altura_y": ty2,
                            "bloco_id": id2
                        }
                    ]
                })
                blocos.append({"env": env_id, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": id1, "altura": 50})
                blocos.append({"env": env_id, "x": math.floor(tx2), "y": ty2, "z": math.floor(tz2), "id": id2, "altura": 50})

            else:
                # ETAPA C: Tarefa Completa com Dispersão Ampla
                desvio1 = math.radians(rng.uniform(-75.0, 75.0))
                y1 = lyaw + desvio1
                d1 = rng.uniform(6.5, 9.5)
                tx1 = round(lx - math.sin(y1) * d1, 1)
                tz1 = round(lz - math.cos(y1) * d1, 1)
                ty1 = math.floor(ly)

                desvio2 = math.radians(rng.uniform(-110.0, 110.0))
                y2 = y1 + desvio2
                d2 = rng.uniform(7.0, 10.0)
                tx2 = round(tx1 - math.sin(y2) * d2, 1)
                tz2 = round(tz1 - math.cos(y2) * d2, 1)
                ty2 = ty1

                tarefas.append({
                    "env_id": env_id,
                    "largada": (lx, ly, lz, e["yaw"]),
                    "prompt": f"Objetivo: vá até o bloco {cor1}, depois até o bloco {cor2}",
                    "estagio_curriculo": "C",
                    "dist_media_teorica": (d1 + d2) / 2.0,
                    "dispersao_graus": 75.0,
                    "estagios": [
                        {
                            "id": 0,
                            "cor": cor1,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor1} [Etapa 1/2]",
                            "alvo_abs": (tx1, tz1),
                            "altura_y": ty1,
                            "bloco_id": id1
                        },
                        {
                            "id": 1,
                            "cor": cor2,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor2} [Etapa 2/2]",
                            "alvo_abs": (tx2, tz2),
                            "altura_y": ty2,
                            "bloco_id": id2
                        }
                    ]
                })
                blocos.append({"env": env_id, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": id1, "altura": 50})
                blocos.append({"env": env_id, "x": math.floor(tx2), "y": ty2, "z": math.floor(tz2), "id": id2, "altura": 50})

        return tarefas, blocos

    def obter_status(self) -> Dict[str, Any]:
        """Retorna resumo do estado curricular atual."""
        return {
            "estagio": self.estagio_atual,
            "auto": self.auto,
            "iteracoes_no_estagio": self.iteracoes_no_estagio,
            "criterio_a": self.criterio_a,
            "criterio_b": self.criterio_b
        }
