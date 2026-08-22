# coding=utf-8
"""
tests/test_cerebro_integracao.py — Testes Unitários da Arquitetura Hierárquica do Cérebro
"""
import os
import sys
import math
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from politica.cerebro import PoliticaCerebroVLA


class MockVLA:
    """Mock do VLA reflexo para teste isolado da lógica do Cérebro."""
    def __init__(self):
        self.device = "cpu"
        self.amostrar = False
        self.temperatura = 0.8
        self.ultimo = {"idx_modo": torch.zeros(1), "idx_yaw": torch.zeros(1)}

    def agir(self, ests, alvos_abs, obs, prompts=None, estagios=None, mascara_modo=None):
        return [{"hold": [], "mouse": [8, 0], "duration_ms": 50} for _ in ests]

    def reiniciar(self, obs):
        pass

    def observar(self, obs):
        pass

    def log_prob(self, px, sv, gv, a_idx, ids=None):
        return torch.tensor([0.0])

    def forward_pensamento(self, pixel_tensor=None, state_tensor=None, goal_tensor=None, input_ids=None, precomputed_v_emb=None):
        return torch.zeros(1, 6), torch.zeros(1, 9), torch.zeros(1, 1)


def test_cerebro_quebra_de_plato_avanco_forcado():
    mock_vla = MockVLA()
    cerebro = PoliticaCerebroVLA(mock_vla)

    # Agente em (0, 0) com yaw=0.0 (olhando para -Z). Alvo em (0.0, -5.0) -> na frente!
    ests = [{"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0, "on_ground": True}]
    alvos_abs = [[0.0, -5.0]]
    obs = [{"estado": ests[0]}]

    cerebro.reiniciar(obs)
    acoes = cerebro.agir(ests, alvos_abs, obs)

    # Como o alvo está perfeitamente alinhado na frente (erro=0), o Cérebro força avanço retilíneo laser [0, 0] com W!
    assert "W" in acoes[0]["hold"]
    assert acoes[0]["mouse"] == [0, 0]


def test_cerebro_anti_orbita_sem_avanco_desalinhado():
    class MockVLACego(MockVLA):
        def agir(self, ests, alvos_abs, obs, prompts=None, estagios=None, mascara_modo=None):
            return [{"hold": ["W"], "mouse": [45, 0], "duration_ms": 50} for _ in ests]

    mock_vla_cego = MockVLACego()
    cerebro = PoliticaCerebroVLA(mock_vla_cego)

    # Agente em (0, 0) com yaw=0.0 (olhando para -Z). Alvo em (0.0, +5.0) -> atrás!
    ests = [{"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0, "on_ground": True}]
    alvos_abs = [[0.0, 5.0]]
    obs = [{"estado": ests[0]}]

    cerebro.reiniciar(obs)
    acoes = cerebro.agir(ests, alvos_abs, obs)

    # Como o alvo está atrás, o Cérebro remove o 'W' espúrio para evitar órbita e comanda reorientação angular ativa em direção ao alvo!
    assert "W" not in acoes[0]["hold"]
    assert acoes[0]["mouse"] == [-45, 0]


def test_cerebro_parada_no_raio_terminal():
    mock_vla = MockVLA()
    cerebro = PoliticaCerebroVLA(mock_vla)

    ests = [{"x": 10.0, "y": 64.0, "z": 10.0, "yaw": 0.0, "on_ground": True}]
    alvos_abs = [[10.5, 10.5]]  # Distância = 0.707m <= 1.2m
    obs = [{"estado": ests[0]}]

    cerebro.reiniciar(obs)
    acoes = cerebro.agir(ests, alvos_abs, obs)

    assert acoes[0]["hold"] == ["W"]
    assert acoes[0]["mouse"] == [0, 0]
    assert cerebro.chegou[0] is True
