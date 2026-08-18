# coding=utf-8
"""
fase5/auditar_dataset_amostras.py — Auditoria Detalhada de Amostras do Dataset WASD Tático 36.
"""
import os
import sys
import random
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fase5.acoes_taticas import decodificar_acao_36

caminho = "fase5/dados/dataset_wasd_tatico_36.pt"
dataset = torch.load(caminho, weights_only=False)

print(f"Total de amostras no dataset: {len(dataset)}")

amostras_sint = []
amostras_reais = []

for i, d in enumerate(dataset):
    if "dist_alvo" in d or "entropia" in d or "prob_max" in d or "acao_original" in d or (i >= 12000):
        amostras_reais.append((i, d))
    else:
        amostras_sint.append((i, d))

print(f"Sintéticas detectadas: {len(amostras_sint)}")
print(f"Reais/Bifurcações detectadas: {len(amostras_reais)}")

random.seed(1337)
sample_sint = random.sample(amostras_sint, min(10, len(amostras_sint)))
sample_reais = random.sample(amostras_reais, min(10, len(amostras_reais)))

def auditar_amostra(idx, d, categoria):
    sv = d["sv"]
    if isinstance(sv, torch.Tensor):
        sv = sv.numpy()
    
    acao_idx = int(d["acao_otima"])
    acao_dict = decodificar_acao_36(acao_idx)
    
    prompt = d.get("prompt", "N/A")
    tipo = d.get("tipo", "N/A")
    erro_yaw = float(d.get("erro_yaw_graus", 0.0))
    peso = float(d.get("peso", 1.0))
    estagio = int(sv[16]) if len(sv) > 16 else 0
    dist = float(sv[2] * 15.0) if len(sv) > 2 else 0.0
    
    # Avaliação lógica de consistência
    ok = True
    razao = "Consistente"
    
    # Se erro > 45 no alinhamento de spawn, não deve segurar W
    if tipo == "alinhar" and abs(erro_yaw) > 45.0:
        if "W" in acao_dict["hold"]:
            ok = False
            razao = "FALHA: Segurando W com erro angular > 45 no alinhamento"
            
    # Se recuo, deve ter S
    if tipo == "recuar" and "S" not in acao_dict["hold"]:
        ok = False
        razao = "FALHA: Tipo recuar mas nao usa S"
        
    return {
        "idx": idx,
        "cat": categoria,
        "tipo": tipo,
        "prompt": prompt,
        "estagio": estagio,
        "dist": dist,
        "erro_yaw": erro_yaw,
        "acao_idx": acao_idx,
        "hold": acao_dict["hold"],
        "mouse": acao_dict["mouse"],
        "peso": peso,
        "tipo_acao": acao_dict["tipo"],
        "status": "OK" if ok else "ALERTA",
        "detalhe": razao
    }

print("\n" + "=" * 100)
print("AUDITORIA DE 10 AMOSTRAS SINTÉTICAS ALEATÓRIAS")
print("=" * 100)
for i, d in sample_sint:
    res = auditar_amostra(i, d, "Sintetica")
    print(f"ID {res['idx']:05d} | Tipo: {res['tipo']:<14} | Est:{res['estagio']} | Dist:{res['dist']:4.1f}m | YawErr:{res['erro_yaw']:+6.1f}° | Ação:{res['acao_idx']:02d} ({res['tipo_acao']:<10} hold={str(res['hold']):<13} mouse={res['mouse']}) | Peso:{res['peso']:.2f} | Prompt: \"{res['prompt']}\"")

print("\n" + "=" * 100)
print("AUDITORIA DE 10 AMOSTRAS DE BIFURCAÇÃO REAL ALEATÓRIAS")
print("=" * 100)
for i, d in sample_reais:
    res = auditar_amostra(i, d, "Real")
    print(f"ID {res['idx']:05d} | Tipo: {res['tipo']:<14} | Est:{res['estagio']} | Dist:{res['dist']:4.1f}m | YawErr:{res['erro_yaw']:+6.1f}° | Ação:{res['acao_idx']:02d} ({res['tipo_acao']:<10} hold={str(res['hold']):<13} mouse={res['mouse']}) | Peso:{res['peso']:.2f} | Prompt: \"{res['prompt']}\"")
