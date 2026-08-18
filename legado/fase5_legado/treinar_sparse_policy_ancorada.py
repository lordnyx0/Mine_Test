# coding=utf-8
"""
fase5/treinar_sparse_policy_ancorada.py — Treinamento da Política Esparsa com Ancoragem de Buffer.

Combina o princípio do paper com a estabilidade de locomoção incorporada:
  - Preserva 100% da capacidade motora de avanço contínuo (ação 4 / straight locomotion).
  - Otimiza as decisões nas bifurcações críticas (transição de submetas e contornos).
  - Mede acurácia separada: Locomoção Densa vs. Bifurcações de Decisão.
  - Salva em 'checkpoints_vla/vla_fase5_sparse_anchored.pt'.
"""
from __future__ import annotations

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from infra.run_vla_agent import load_vla_agent
from infra.gpu_utils import compactar_backbone
from modelo.lora_vla import aplicar_lora
from politica.politica_raciocinio import PoliticaRaciocinioLoop

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def treinar_ancorado(
    dataset_path: str = "fase5/dados/dataset_ancorado_fase5.pt",
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_coldstart.pt",
    ckpt_saida: str = "checkpoints_vla/vla_fase5_sparse_anchored.pt",
    epocas: int = 15,
    batch_size: int = 32,
    lr: float = 2e-4,
    device: str = "cuda"
):
    print("=" * 80)
    print(" [FASE 5] TREINAMENTO: SPARSE POLICY COM ANCORAGEM DE BUFFER (70/30)")
    print(f"    Dataset Entrada : {dataset_path}")
    print(f"    Checkpoint Base : {ckpt_entrada}")
    print(f"    Checkpoint Fim  : {ckpt_saida}")
    print(f"    Épocas          : {epocas} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 80)

    # 1. Carrega VLA e Tokenizer
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Aplica LoRA
    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    # Instancia a política de raciocínio
    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3)

    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos base carregados de '{ckpt_entrada}' ({len(ckpt_data['treinaveis'])} tensores).")

    vla.to(dev)

    # 2. Carrega Dataset Ancorado
    dados = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset] {len(dados)} amostras carregadas.")

    todos_sv = torch.stack([d["sv"] for d in dados]).to(dev) # [N, 18]
    todas_acoes = torch.tensor([int(d["acao_otima"]) for d in dados], dtype=torch.long, device=dev) # [N]
    todos_pesos = torch.tensor([float(d.get("peso", 1.0)) for d in dados], dtype=torch.float32, device=dev) # [N]
    eh_bifurcacao = torch.tensor([1 if d.get("tipo") == "decisao_esparsa_causal" else 0 for d in dados], dtype=torch.bool, device=dev)

    # Pre-tokeniza os prompts
    prompts = [d.get("prompt", "Objetivo: vá até o bloco azul [Etapa 1/2]") for d in dados]
    enc = tokenizer(prompts, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
    tokens_dataset = enc["input_ids"].to(dev)

    # 3. Otimizador focado nos parâmetros treináveis (LoRA + StateEncoder + Acao18)
    params_treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(params_treinaveis, lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    print(f"[Treino] {len(params_treinaveis)} tensores treináveis ativos.")
    print("--- Iniciando Otimização Balanceada ---")

    t_ini = time.time()
    num_exemplos = len(dados)

    for ep in range(1, epocas + 1):
        vla.train()
        indices = torch.randperm(num_exemplos, device=dev)
        total_loss = 0.0
        
        total_acertos = 0
        total_geral = 0

        acertos_bifurc = 0
        total_bifurc = 0

        acertos_dense = 0
        total_dense = 0

        for i in range(0, num_exemplos, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_sv = todos_sv[batch_idx]
            b_toks = tokens_dataset[batch_idx]
            b_alvo = todas_acoes[batch_idx]
            b_pesos = todos_pesos[batch_idx]
            b_is_bifurc = eh_bifurcacao[batch_idx]

            otimizador.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = vla.state_encoder(b_sv) # [B, 4, H]
                t_embeds = vla.qwen_model.get_input_embeddings()(b_toks) # [B, T, H]
                inputs_embeds = torch.cat([s_embeds, t_embeds], dim=1) # [B, 4+T, H]

                outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                last_hidden = outputs.last_hidden_state[:, -1, :] # [B, H]
                logits = vla.cabeca_acao_18(last_hidden) # [B, 18]

                loss_raw = loss_fn(logits, b_alvo)
                loss = (loss_raw * b_pesos).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_treinaveis, max_norm=1.0)
            otimizador.step()

            total_loss += loss.item() * len(b_alvo)
            preds = logits.argmax(dim=-1)
            corretos = (preds == b_alvo)

            total_acertos += corretos.sum().item()
            total_geral += len(b_alvo)

            if b_is_bifurc.any():
                acertos_bifurc += corretos[b_is_bifurc].sum().item()
                total_bifurc += b_is_bifurc.sum().item()

            if (~b_is_bifurc).any():
                acertos_dense += corretos[~b_is_bifurc].sum().item()
                total_dense += (~b_is_bifurc).sum().item()

        acc_geral = (total_acertos / max(1, total_geral)) * 100.0
        acc_bifurc = (acertos_bifurc / max(1, total_bifurc)) * 100.0
        acc_dense = (acertos_dense / max(1, total_dense)) * 100.0
        loss_media = total_loss / max(1, total_geral)

        if ep % 2 == 0 or ep == 1 or ep == epocas:
            print(f"  Época {ep:2d}/{epocas} | Loss: {loss_media:.4f} | Geral: {acc_geral:5.1f}% | Densa (Sprint): {acc_dense:5.1f}% ({acertos_dense}/{total_dense}) | Bifurcações: {acc_bifurc:5.1f}% ({acertos_bifurc}/{total_bifurc})", flush=True)

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Treinamento ancorado concluído em {duracao:.1f}s.")

    # 4. Salva Checkpoint
    os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
    tensores_treinaveis = {k: v for k, v in vla.state_dict().items() if any(t in k for t in ["lora_", "state_encoder", "cabeca_acao_18"])}
    torch.save({
        "treinaveis": tensores_treinaveis,
        "epocas": epocas,
        "acuracia_geral": acc_geral,
        "acuracia_bifurc": acc_bifurc,
        "acuracia_dense": acc_dense
    }, ckpt_saida)
    print(f"[OK] Checkpoint ancorado salvo em: {ckpt_saida} ({len(tensores_treinaveis)} tensores).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fase5/dados/dataset_ancorado_fase5.pt")
    ap.add_argument("--base", default="checkpoints_vla/vla_fase5_coldstart.pt")
    ap.add_argument("--saida", default="checkpoints_vla/vla_fase5_sparse_anchored.pt")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    treinar_ancorado(
        dataset_path=args.dataset,
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        epocas=args.epocas,
        lr=args.lr
    )
