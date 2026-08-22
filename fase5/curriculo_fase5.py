# coding=utf-8
"""
fase5/curriculo_fase5.py — Gerenciador de Currículo Progressivo da Fase 5.5.

Estrutura Curricular:
  ETAPA A (Pilar 1 Único — Fácil):
    - 1 único pilar (submeta única)
    - Distância curta: 4.0m a 6.5m
    - Dispersão angular frontal: ±35°
    - Foco: aprender descoberta visual e navegação direta
    - Critério de avanço: Submeta 1 >= 35% por 3 iterações consecutivas

  ETAPA B (Pilar 1 -> Pilar 2 — Moderado):
    - 2 pilares sequenciais
    - Distâncias moderadas: 5.5m a 8.0m
    - Dispersões moderadas: ±60° (Pilar 1), ±75° (Pilar 2)
    - Foco: transição e reorientação intermediária
    - Critério de avanço: Sucesso Total >= 20% ou Submeta 1 >= 50% por 3 iterações consecutivas

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
        estagio_inicial: Optional[str] = None,
        criterio_a: float = 0.35,
        criterio_b: float = 0.20,
        consecutivas_necessarias: int = 3
    ):
        """
        modo_estagio: 'A', 'B', 'C' (estágio fixo) ou 'auto' (progressão adaptativa baseada em desempenho)
        criterio_a: float (0.35 ou 35.0) para Sub1 em A -> B
        criterio_b: float (0.20 ou 20.0) para Sucesso em B -> C (fallback Sub1 >= 50%)
        consecutivas_necessarias: número de iterações consecutivas com critério atingido para avançar
        """
        self.modo = modo_estagio.upper()
        self.criterio_a = criterio_a
        self.criterio_b = criterio_b
        self.consecutivas_necessarias = int(consecutivas_necessarias)

        if self.modo in self.ESTAGIOS:
            self.estagio_atual = self.modo
            self.auto = False
        else:
            self.estagio_atual = (estagio_inicial.upper() if estagio_inicial else "A")
            self.auto = True

        self.historico_sub1: List[float] = []
        self.historico_sucesso: List[float] = []
        self.historico_rec: List[float] = []
        self.iteracoes_no_estagio = 0
        self.streak = 0
        self.nivel = 1
        self.delta_dist = 0.0

        # Carrega banco de posições secas
        caminho_secas = os.path.join(_ROOT, "dataset", "largadas_secas.json")
        if os.path.exists(caminho_secas):
            with open(caminho_secas, "r", encoding="utf-8") as f:
                self.banco = json.load(f)
        else:
            from ambiente.tarefas_logicas import BANCO_LARGADAS
            self.banco = BANCO_LARGADAS

    def _crit_a_pct(self) -> float:
        return self.criterio_a * 100.0 if self.criterio_a <= 1.0 else self.criterio_a

    def _crit_b_pct(self) -> float:
        return self.criterio_b * 100.0 if self.criterio_b <= 1.0 else self.criterio_b

    def atualizar_desempenho(self, taxa_sub1: float, taxa_sucesso: float, recompensa: float) -> Tuple[bool, str]:
        """
        Atualiza o histórico com o resultado da iteração.
        Requer streak de N iterações CONSECUTIVAS atingindo o threshold.
        Se o desempenho cair abaixo do threshold, streak reseta para 0.
        Ao concluir 3 iterações consecutivas em C:
          -> Nível += 1, delta_dist += 1.0m, volta para Etapa A!
        Retorna (avancou: bool, mensagem_transicao: str).
        """
        self.historico_sub1.append(taxa_sub1)
        self.historico_sucesso.append(taxa_sucesso)
        self.historico_rec.append(recompensa)
        self.iteracoes_no_estagio += 1

        if not self.auto:
            return False, f"Estágio fixo: {self.estagio_atual}"

        crit_a = self._crit_a_pct()
        crit_b = self._crit_b_pct()
        crit_c = self._crit_b_pct()  # 20% de Sucesso Total para os 3 pilares

        if self.estagio_atual == "A":
            atingiu_meta = (taxa_sub1 >= crit_a)
            if atingiu_meta:
                self.streak += 1
            else:
                self.streak = 0

            if self.streak >= self.consecutivas_necessarias:
                self.estagio_atual = "B"
                self.iteracoes_no_estagio = 0
                self.streak = 0
                msg = f">>> [CURRÍCULO] AVANÇO: ETAPA A -> ETAPA B (Sub1 >= {crit_a:.1f}% por {self.consecutivas_necessarias} iterações consecutivas!)"
                return True, msg
            else:
                return False, f"ETAPA A (Nível {self.nivel}) | streak={self.streak}/{self.consecutivas_necessarias} (Sub1={taxa_sub1:.1f}%, precisa={crit_a:.1f}%)"

        elif self.estagio_atual == "B":
            atingiu_meta = (taxa_sucesso >= crit_b)
            if atingiu_meta:
                self.streak += 1
            else:
                self.streak = 0

            if self.streak >= self.consecutivas_necessarias:
                self.estagio_atual = "C"
                self.iteracoes_no_estagio = 0
                self.streak = 0
                msg = f">>> [CURRÍCULO] AVANÇO: ETAPA B -> ETAPA C (Sucesso Total >= {crit_b:.1f}% por {self.consecutivas_necessarias} iterações consecutivas!)"
                return True, msg
            else:
                return False, f"ETAPA B (Nível {self.nivel}) | streak={self.streak}/{self.consecutivas_necessarias} (Suc={taxa_sucesso:.1f}%, Sub1={taxa_sub1:.1f}%, precisa={crit_b:.1f}%)"

        elif self.estagio_atual == "C":
            atingiu_meta = (taxa_sucesso >= crit_c)
            if atingiu_meta:
                self.streak += 1
            else:
                self.streak = 0

            if self.streak >= self.consecutivas_necessarias:
                self.nivel += 1
                self.delta_dist += 1.0  # +1 METRO A MAIS EM TODOS OS PILARES!
                self.estagio_atual = "A"  # VOLTA PARA O ESTÁGIO A NO NOVO NÍVEL!
                self.iteracoes_no_estagio = 0
                self.streak = 0
                msg = f">>> [CURRÍCULO EVOLUTIVO] NÍVEL {self.nivel-1} CONCLUÍDO! Subindo para NÍVEL {self.nivel} (Distâncias +{self.delta_dist:.1f}m): ETAPA C -> ETAPA A!"
                return True, msg
            else:
                return False, f"ETAPA C (Nível {self.nivel}) | streak={self.streak}/{self.consecutivas_necessarias} (Suc={taxa_sucesso:.1f}%, precisa={crit_c:.1f}%)"

        return False, f"Estágio: {self.estagio_atual}"

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
            cor1, cor2, cor3 = rng.sample(cores_disp, 3)
            id1, id2, id3 = CORES_MAP[cor1], CORES_MAP[cor2], CORES_MAP[cor3]

            if self.estagio_atual == "A":
                # ETAPA A: Pilar Único, frontal e curto
                desvio1 = math.radians(rng.uniform(-35.0, 35.0))
                y1 = lyaw + desvio1
                d1 = rng.uniform(4.0, 6.5) + self.delta_dist
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
                desvio1 = math.radians(rng.uniform(-45.0, 45.0))
                y1 = lyaw + desvio1
                d1 = rng.uniform(4.5, 6.5) + self.delta_dist
                tx1 = round(lx - math.sin(y1) * d1, 1)
                tz1 = round(lz - math.cos(y1) * d1, 1)
                ty1 = math.floor(ly)

                desvio2 = math.radians(rng.uniform(-60.0, 60.0))
                y2 = y1 + desvio2
                d2 = rng.uniform(4.5, 6.5) + self.delta_dist
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
                # ETAPA C: 3 Pilares Sequenciais (P1 -> P2 -> P3)
                desvio1 = math.radians(rng.uniform(-45.0, 45.0))
                y1 = lyaw + desvio1
                d1 = rng.uniform(4.5, 6.5) + self.delta_dist
                tx1 = round(lx - math.sin(y1) * d1, 1)
                tz1 = round(lz - math.cos(y1) * d1, 1)
                ty1 = math.floor(ly)

                desvio2 = math.radians(rng.uniform(-60.0, 60.0))
                y2 = y1 + desvio2
                d2 = rng.uniform(4.5, 6.5) + self.delta_dist
                tx2 = round(tx1 - math.sin(y2) * d2, 1)
                tz2 = round(tz1 - math.cos(y2) * d2, 1)
                ty2 = ty1

                desvio3 = math.radians(rng.uniform(-60.0, 60.0))
                y3 = y2 + desvio3
                d3 = rng.uniform(4.5, 6.5) + self.delta_dist
                tx3 = round(tx2 - math.sin(y3) * d3, 1)
                tz3 = round(tz2 - math.cos(y3) * d3, 1)
                ty3 = ty1

                tarefas.append({
                    "env_id": env_id,
                    "largada": (lx, ly, lz, e["yaw"]),
                    "prompt": f"Objetivo: vá até o bloco {cor1}, depois {cor2}, e depois {cor3}",
                    "estagio_curriculo": "C",
                    "dist_media_teorica": (d1 + d2 + d3) / 3.0,
                    "dispersao_graus": 60.0,
                    "estagios": [
                        {
                            "id": 0,
                            "cor": cor1,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor1} [Etapa 1/3]",
                            "alvo_abs": (tx1, tz1),
                            "altura_y": ty1,
                            "bloco_id": id1
                        },
                        {
                            "id": 1,
                            "cor": cor2,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor2} [Etapa 2/3]",
                            "alvo_abs": (tx2, tz2),
                            "altura_y": ty2,
                            "bloco_id": id2
                        },
                        {
                            "id": 2,
                            "cor": cor3,
                            "prompt_estagio": f"Objetivo: vá até o bloco {cor3} [Etapa 3/3]",
                            "alvo_abs": (tx3, tz3),
                            "altura_y": ty3,
                            "bloco_id": id3
                        }
                    ]
                })
                blocos.append({"env": env_id, "x": math.floor(tx1), "y": ty1, "z": math.floor(tz1), "id": id1, "altura": 50})
                blocos.append({"env": env_id, "x": math.floor(tx2), "y": ty2, "z": math.floor(tz2), "id": id2, "altura": 50})
                blocos.append({"env": env_id, "x": math.floor(tx3), "y": ty3, "z": math.floor(tz3), "id": id3, "altura": 50})

        return tarefas, blocos

    def obter_status(self) -> Dict[str, Any]:
        """Retorna resumo do estado curricular atual."""
        crit_a = self._crit_a_pct()
        crit_b = self._crit_b_pct()
        if self.estagio_atual == "A":
            precisa_str = f"Sub1>={crit_a:.0f}%"
        elif self.estagio_atual == "B":
            precisa_str = f"Suc>={crit_b:.0f}% (Sucesso Total)"
        else:
            precisa_str = "Concluído (Plena)"

        return {
            "estagio": self.estagio_atual,
            "auto": self.auto,
            "streak": self.streak,
            "consecutivas_necessarias": self.consecutivas_necessarias,
            "iteracoes_no_estagio": self.iteracoes_no_estagio,
            "criterio_a": self.criterio_a,
            "criterio_b": self.criterio_b,
            "precisa_str": precisa_str
        }
