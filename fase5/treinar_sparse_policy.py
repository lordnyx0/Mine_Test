# coding=utf-8
"""
fase5/treinar_sparse_policy.py — Treinamento por Sparse Policy Selection nos Pontos de Alta Entropia.

Aplica o ajuste fino focado (Sparse Policy Selection) nas bifurcações e decisões críticas mineradas:
  - Carrega pesos de 'checkpoints_vla/vla_fase5_coldstart.pt'.
  - Carrega o dataset limpo de alta entropia de 'fase5/dados/dataset_decisoes_alta_entropia.pt'.
  - Treina por 20 épocas com ponderação por entropia (dando mais gradiente aos pontos de maior incerteza).
  - Salva o checkpoint calibrado em 'checkpoints_vla/vla_fase5_sparse_policy.pt'.
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


def treinar(
    dataset_path: str = "fase5/dados/dataset_decisoes_alta_entropia.pt",
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_coldstart.pt",
    ckpt_saida: str = "checkpoints_vla/vla_fase5_sparse_policy.pt",
    epocas: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda"
):
    print("=" * 80)
    print(" [FASE 5] TREINAMENTO: SPARSE POLICY SELECTION (PONTOS DE ALTA ENTROPIA)")
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

    # Instancia a política de raciocínio (que anexa a cabeça de 18 ações e o loop de pensamento)
    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3)

    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos base carregados de '{ckpt_entrada}' ({len(ckpt_data['treinaveis'])} tensores).")

    vla.to(dev)

    # 2. Carrega Dataset
    dados = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset] {len(dados)} decisões de alta entropia carregadas.")

    todos_sv = torch.stack([d["sv"] for d in dados]).to(dev) # [N, 18]
    todas_acoes = torch.tensor([int(d["acao_otima"]) for d in dados], dtype=torch.long, device=dev) # [N]
    todos_pesos = torch.tensor([max(0.5, float(d.get("entropia_norm", 0.5))) for d in dados], dtype=torch.float32, device=dev) # [N]

    # Pre-tokeniza os prompts
    prompts = [d.get("prompt", "Objetivo: vá até o bloco azul [Etapa 1/2]") for d in dados]
    enc = tokenizer(prompts, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
    tokens_dataset = enc["input_ids"].to(dev)

    # 3. Otimizador focado nos parâmetros treináveis (LoRA + StateEncoder + Acao18)
    params_treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(params_treinaveis, lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    print(f"[Treino] {len(params_treinaveis)} tensores treináveis ativos.")
    print("--- Iniciando Otimização Esparsa ---")

    t_ini = time.time()
    num_exemplos = len(dados)

    for ep in range(1, epocas + 1):
        vla.train()
        indices = torch.randperm(num_exemplos, device=dev)
        total_loss = 0.0
        acertos = 0
        total = 0

        for i in range(0, num_exemplos, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_sv = todos_sv[batch_idx]
            b_toks = tokens_dataset[batch_idx]
            b_alvo = todas_acoes[batch_idx]
            b_pesos = todos_pesos[batch_idx]

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
            acertos += (preds == b_alvo).sum().item()
            total += len(b_alvo)

        acc = (acertos / max(1, total)) * 100.0
        loss_media = total_loss / max(1, total)

        if ep % 2 == 0 or ep == 1 or ep == epocas:
            print(f"  Época {ep:2d}/{epocas} | Loss: {loss_media:.4f} | Acurácia nas Bifurcações: {acc:5.1f}% | Decisões Certas: {acertos}/{total}", flush=True)

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Treinamento concluído em {duracao:.1f}s.")

    # 4. Salva Checkpoint
    os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
    tensores_treinaveis = {k: v for k, v in vla.state_dict().items() if any(t in k for t in ["lora_", "state_encoder", "cabeca_acao_18"])}
    torch.save({"treinaveis": tensores_treinaveis, "epocas": epocas, "acuracia_final": acc}, ckpt_saida)
    print(f"[OK] Checkpoint calibrado salvo em: {ckpt_saida} ({len(tensores_treinaveis)} tensores).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fase5/dados/dataset_decisoes_alta_entropia.pt")
    ap.add_argument("--base", default="checkpoints_vla/vla_fase5_coldstart.pt")
    ap.add_argument("--saida", default="checkpoints_vla/vla_fase5_sparse_policy.pt")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    treinar(
        dataset_path=args.dataset,
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        epocas=args.epocas,
        lr=args.lr
    )
