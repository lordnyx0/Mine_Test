# coding=utf-8
"""
tests/test_fase5_correcoes.py — Testes Unitários de Robustez para a Fase 5.5:
  1. Value Head Loss & MSE Shapes (Batch 1, Batch > 1, sem Warnings).
  2. Gerenciador de Currículo com Streak Consecutivo e Proteção contra Avanço Prematuro.
  3. Fatoração e Decodificação de Ações.
"""
import os
import sys
import pytest
import warnings
import torch
import torch.nn as nn
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.curriculo_fase5 import CurriculoFase5
from fase5.acoes_taticas import (
    fatorar_indice_36,
    unificar_indices,
    decodificar_acao_fatorada,
    MODOS,
    YAW_BINS_9,
    YAW_BINS_3,
    NUM_MODOS,
    NUM_YAW
)


# ==============================================================================
# 1. TESTES DO VALUE HEAD & SHAPES MSE
# ==============================================================================

def test_value_head_mse_shape_batch_1():
    """Garante que para batch size 1, val_pred e target_val têm exatamente o mesmo shape 1D [1] e MSE não emite warning."""
    raw_values = torch.tensor([[1.25]], dtype=torch.float32)  # [1, 1]
    raw_target = torch.tensor([2.50], dtype=torch.float32)   # [1]

    val_pred = torch.nan_to_num(raw_values.reshape(-1).float(), 0.0)
    target_val = raw_target.reshape(-1)

    assert val_pred.shape == target_val.shape == torch.Size([1]), f"Shapes divergem: {val_pred.shape} vs {target_val.shape}"

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        loss = nn.MSELoss()(val_pred, target_val)
        assert len(recorded_warnings) == 0, f"MSELoss emitiu warnings inesperados: {[str(w.message) for w in recorded_warnings]}"
        assert torch.isclose(loss, torch.tensor((1.25 - 2.50)**2))


def test_value_head_mse_shape_batch_multi():
    """Garante que para mini-batches de tamanho > 1 (e.g. 16), val_pred e target_val mantêm shape [16]."""
    B = 16
    raw_values = torch.randn(B, 1, dtype=torch.float32)
    raw_target = torch.randn(B, dtype=torch.float32)

    val_pred = torch.nan_to_num(raw_values.reshape(-1).float(), 0.0)
    target_val = raw_target.reshape(-1)

    assert val_pred.shape == target_val.shape == torch.Size([B])

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        loss = nn.MSELoss()(val_pred, target_val)
        assert len(recorded_warnings) == 0
        assert loss.ndim == 0


def test_value_head_mse_shape_scalar_input():
    """Garante robustez mesmo se raw_values for um escalar 0-dim tensor ou já 1-dim."""
    raw_scalar = torch.tensor(3.14)
    raw_target = torch.tensor([3.14])

    val_pred = torch.nan_to_num(raw_scalar.reshape(-1).float(), 0.0)
    target_val = raw_target.reshape(-1)

    assert val_pred.shape == target_val.shape == torch.Size([1])
    loss = nn.MSELoss()(val_pred, target_val)
    assert torch.isclose(loss, torch.tensor(0.0))


# ==============================================================================
# 2. TESTES DO CURRÍCULO COM STREAK CONSECUTIVO
# ==============================================================================

def test_curriculo_nao_avanca_com_pico_isolado():
    """Garante que 1 único pico de Sub1 >= 35% NÃO avança de A para B."""
    cur = CurriculoFase5(modo_estagio="auto", criterio_a=0.35, criterio_b=0.20, consecutivas_necessarias=3)
    assert cur.estagio_atual == "A"

    avancou, msg = cur.atualizar_desempenho(taxa_sub1=40.0, taxa_sucesso=0.0, recompensa=1.0)
    assert not avancou
    assert cur.estagio_atual == "A"
    assert cur.streak == 1


def test_curriculo_reseta_streak_quando_desempenho_cai():
    """Garante que se Sub1 cair abaixo do critério, o streak reseta para 0."""
    cur = CurriculoFase5(modo_estagio="auto", criterio_a=0.35, criterio_b=0.20, consecutivas_necessarias=3)

    cur.atualizar_desempenho(taxa_sub1=40.0, taxa_sucesso=0.0, recompensa=1.0)  # streak = 1
    cur.atualizar_desempenho(taxa_sub1=50.0, taxa_sucesso=0.0, recompensa=1.0)  # streak = 2
    assert cur.streak == 2

    # Queda na 3ª iteração
    avancou, msg = cur.atualizar_desempenho(taxa_sub1=20.0, taxa_sucesso=0.0, recompensa=0.0)
    assert not avancou
    assert cur.streak == 0
    assert cur.estagio_atual == "A"


def test_curriculo_avanco_a_para_b_com_3_consecutivas():
    """Garante avanço de A -> B somente após exatamente 3 iterações consecutivas atingindo a meta."""
    cur = CurriculoFase5(modo_estagio="auto", criterio_a=0.35, criterio_b=0.20, consecutivas_necessarias=3)

    av1, _ = cur.atualizar_desempenho(taxa_sub1=35.0, taxa_sucesso=0.0, recompensa=1.0)
    assert not av1 and cur.estagio_atual == "A" and cur.streak == 1

    av2, _ = cur.atualizar_desempenho(taxa_sub1=37.5, taxa_sucesso=0.0, recompensa=1.5)
    assert not av2 and cur.estagio_atual == "A" and cur.streak == 2

    av3, msg = cur.atualizar_desempenho(taxa_sub1=40.0, taxa_sucesso=0.0, recompensa=2.0)
    assert av3
    assert cur.estagio_atual == "B"
    assert cur.streak == 0
    assert "AVANÇO: ETAPA A -> ETAPA B" in msg


def test_curriculo_avanco_b_para_c_com_sucesso_total_obrigatorio():
    """Garante que avanço de B -> C EXIGE estritamente Sucesso Total >= 20% e NÃO avança apenas com Sub1."""
    cur = CurriculoFase5(modo_estagio="auto", criterio_a=0.35, criterio_b=0.20, consecutivas_necessarias=3)
    cur.estagio_atual = "B"

    # 1. Tenta avançar com Sub1 alto (80%), mas Sucesso = 0% -> NÃO pode avançar!
    cur.atualizar_desempenho(taxa_sub1=80.0, taxa_sucesso=0.0, recompensa=1.0)
    cur.atualizar_desempenho(taxa_sub1=90.0, taxa_sucesso=0.0, recompensa=1.0)
    avancou, msg = cur.atualizar_desempenho(taxa_sub1=85.0, taxa_sucesso=0.0, recompensa=1.0)
    assert not avancou
    assert cur.estagio_atual == "B"
    assert cur.streak == 0

    # 2. Agora com 3 iterações consecutivas de Sucesso Total >= 20% (P1 + P2 concluídos) -> AVANÇA!
    cur.atualizar_desempenho(taxa_sub1=100.0, taxa_sucesso=25.0, recompensa=1.0)
    cur.atualizar_desempenho(taxa_sub1=100.0, taxa_sucesso=20.0, recompensa=1.0)
    avancou, msg = cur.atualizar_desempenho(taxa_sub1=100.0, taxa_sucesso=22.0, recompensa=1.0)

    assert avancou
    assert cur.estagio_atual == "C"
    assert cur.streak == 0
    assert "AVANÇO: ETAPA B -> ETAPA C" in msg


def test_curriculo_status_telemetria():
    """Garante que obter_status() retorna todos os campos necessários para logging e diagnóstico."""
    cur = CurriculoFase5(modo_estagio="auto", criterio_a=0.35, criterio_b=0.20, consecutivas_necessarias=3)
    status = cur.obter_status()

    assert status["estagio"] == "A"
    assert status["auto"] is True
    assert status["streak"] == 0
    assert status["consecutivas_necessarias"] == 3
    assert "Sub1>=35%" in status["precisa_str"]


def test_curriculo_estagio_fixo_compatibilidade():
    """Garante que modos de estágio fixo ('A', 'B', 'C') não avançam automaticamente."""
    cur = CurriculoFase5(modo_estagio="B")
    assert cur.estagio_atual == "B"
    assert cur.auto is False

    avancou, _ = cur.atualizar_desempenho(taxa_sub1=100.0, taxa_sucesso=100.0, recompensa=10.0)
    assert not avancou
    assert cur.estagio_atual == "B"


# ==============================================================================
# 3. TESTES DE FATURAÇÃO E DECODIFICAÇÃO DE AÇÕES
# ==============================================================================

def test_fatoracao_bijetiva_todas_36_acoes():
    """Garante que a fatoração de todos os 36 índices é bijetiva."""
    for idx in range(36):
        m, y = fatorar_indice_36(idx)
        assert 0 <= m < NUM_MODOS
        assert 0 <= y < NUM_YAW
        recup = unificar_indices(m, y)
        assert recup == idx


def test_decodificacao_modos_corretos():
    """Valida que decodificar_acao_fatorada gera os comandos WASD e mouse corretos."""
    # Modo Sprint (1), Yaw reto (0° -> bin 4)
    acao_sprint = decodificar_acao_fatorada(1, 4)
    assert "W" in acao_sprint["hold"]
    assert "S" not in acao_sprint["hold"]
    assert acao_sprint["mouse"][0] == 0

    # Modo Strafe Esq (3), Yaw -5° (bin 3 em YAW_BINS_9)
    acao_strafe_esq = decodificar_acao_fatorada(3, 3)
    assert "W" in acao_strafe_esq["hold"]
    assert "A" in acao_strafe_esq["hold"]
    assert acao_strafe_esq["mouse"][0] == -5

    # Modo Strafe Dir (4), Yaw +5° (bin 5 em YAW_BINS_9)
    acao_strafe_dir = decodificar_acao_fatorada(4, 5)
    assert "W" in acao_strafe_dir["hold"]
    assert "D" in acao_strafe_dir["hold"]
    assert acao_strafe_dir["mouse"][0] == 5
