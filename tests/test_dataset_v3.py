# coding=utf-8
"""
tests/test_dataset_v3.py — Testes Unitários de Coerência e Qualidade para o Dataset V3 Purificado
"""
import os
import sys
import math
import pytest
import torch
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.acoes_taticas import fatorar_indice_36, MODOS

DATASET_V3_PATH = os.path.join(_ROOT, "fase5", "dados", "dataset_wasd_tatico_36_v3.pt")


@pytest.fixture(scope="module")
def dataset_v3():
    if not os.path.exists(DATASET_V3_PATH):
        pytest.skip(f"Dataset V3 não encontrado em {DATASET_V3_PATH}. Gere-o primeiro.")
    dados = torch.load(DATASET_V3_PATH, weights_only=False)

    processadas = []
    for x in dados:
        sv = x['sv']
        dx = float(sv[0]) * 15.0
        dz = float(sv[1]) * 15.0
        dist = math.hypot(dx, dz)
        if dist < 1e-4 and float(sv[2]) > 0:
            dist = float(sv[2]) * 15.0
        err = float(x.get('erro_yaw_graus', 0.0))
        if err == 0.0 and (abs(dx) > 1e-4 or abs(dz) > 1e-4):
            err = math.degrees(math.atan2(-dx, -dz))
        a = int(x['acao_otima'])
        m, y = fatorar_indice_36(a)
        m_name = MODOS[m]
        tem_w = (m_name in ['sprint', 'pulo', 'strafe_esq', 'strafe_dir'])
        processadas.append({
            'dist': dist,
            'err': err,
            'modo': m_name,
            'tipo': x.get('tipo', ''),
            'acao': a,
            'tem_w': tem_w
        })
    return processadas


def test_presenca_massiva_trajetorias_encadeadas(dataset_v3):
    """Garante que as trajetórias encadeadas contínuas constituam o núcleo principal do dataset (>75%)."""
    trajs = [p for p in dataset_v3 if p['tipo'] == 'trajetoria_encadeada']
    pct_traj = (len(trajs) / len(dataset_v3)) * 100.0
    assert len(trajs) >= 20000, f"Poucas amostras em trajetórias encadeadas: {len(trajs)}"
    assert pct_traj >= 75.0, f"Percentual de trajetórias encadeadas baixo: {pct_traj:.1f}%"


def test_cobertura_e_dominancia_avanco_aproximacao(dataset_v3):
    """Garante que na faixa 0.8m a 2.5m, pelo menos 90% das ações acionem avanço (W)."""
    criticos = [p for p in dataset_v3 if 0.8 <= p['dist'] <= 2.5]
    assert len(criticos) >= 5000, f"Poucas amostras na faixa 0.8-2.5m: {len(criticos)}"
    w_pct = (sum(1 for p in criticos if p['tem_w']) / len(criticos)) * 100.0
    assert w_pct >= 90.0, f"Taxa de avanço insuficiente em 0.8-2.5m: {w_pct:.1f}%"


def test_estados_centralizados_sem_conflito(dataset_v3):
    """Garante que para alvo centralizado (|yaw| <= 15°) em 0.8-2.5m, 100% das ações sejam de avanço (W)."""
    cent = [p for p in dataset_v3 if 0.8 <= p['dist'] <= 2.5 and abs(p['err']) <= 15.0]
    assert len(cent) >= 2000, f"Poucas amostras centralizadas: {len(cent)}"
    w_pct = (sum(1 for p in cent if p['tem_w']) / len(cent)) * 100.0
    recuar_pct = (sum(1 for p in cent if p['modo'] == 'recuar') / len(cent)) * 100.0
    alinhar_pct = (sum(1 for p in cent if p['modo'] == 'alinhar') / len(cent)) * 100.0

    assert w_pct >= 98.0, f"Avanço insuficiente quando centralizado: {w_pct:.1f}%"
    assert recuar_pct == 0.0, f"Recuo indevido quando centralizado: {recuar_pct:.1f}%"
    assert alinhar_pct <= 2.0, f"Alinhamento parado indevido quando centralizado: {alinhar_pct:.1f}%"


def test_purga_de_giro_em_proximidade_maxima(dataset_v3):
    """Garante que em distâncias < 0.8m, o avanço (W) domine e 'alinhar' seja próximo de zero."""
    sub_08 = [p for p in dataset_v3 if p['dist'] < 0.8]
    assert len(sub_08) >= 2000, f"Poucas amostras em < 0.8m: {len(sub_08)}"
    w_pct = (sum(1 for p in sub_08 if p['tem_w']) / len(sub_08)) * 100.0
    alinhar_pct = (sum(1 for p in sub_08 if p['modo'] == 'alinhar') / len(sub_08)) * 100.0
    recuar_pct = (sum(1 for p in sub_08 if p['modo'] == 'recuar') / len(sub_08)) * 100.0

    assert w_pct >= 95.0, f"Avanço insuficiente em < 0.8m: {w_pct:.1f}%"
    assert alinhar_pct <= 2.0, f"Alinhamento parado residual elevado em < 0.8m: {alinhar_pct:.1f}%"
    assert recuar_pct <= 2.0, f"Recuo indevido em < 0.8m: {recuar_pct:.1f}%"


def test_strafe_preservado_sem_colapso(dataset_v3):
    """Garante que strafe lateral (esquerda e direita) continua bem representado no dataset global."""
    cm = Counter(p['modo'] for p in dataset_v3)
    total = len(dataset_v3)
    esq_pct = (cm['strafe_esq'] / total) * 100.0
    dir_pct = (cm['strafe_dir'] / total) * 100.0
    assert esq_pct >= 10.0, f"Strafe esq baixo: {esq_pct:.1f}%"
    assert dir_pct >= 10.0, f"Strafe dir baixo: {dir_pct:.1f}%"
