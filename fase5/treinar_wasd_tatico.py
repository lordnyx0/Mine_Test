# coding=utf-8
"""
fase5/treinar_wasd_tatico.py — Treinamento da Política Tática WASD com Salvamento por Época.
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


def treinar_wasd_tatico(
    dataset_path: str = "fase5/dados/dataset_wasd_tatico_36.pt",
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_sparse_anchored_grande.pt",
    ckpt_saida:   str = "checkpoints_vla/vla_fase5_wasd_tatico.pt",
    epocas:       int = 3,
    batch_size:   int = 32,
    lr:         float = 2e-4,
    device:       str = "cuda"
):
    print("=" * 80)
    print(" [FASE 5.2] TREINAMENTO: RACIOCÍNIO TÁTICO HOLONÔMICO (36 AÇÕES WASD)")
    print(f"    Dataset Entrada : {dataset_path}")
    print(f"    Checkpoint Base : {ckpt_entrada}")
    print(f"    Checkpoint Fim  : {ckpt_saida}")
    print(f"    Épocas          : {epocas} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 80)

    # 1. Carrega modelo VLA e Tokenizer
    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    # Instancia com 36 ações
    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3, num_acoes=36)

    # Restaura tensores base
    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            state_filtrado = {k: v for k, v in ckpt_data["treinaveis"].items() if "cabeca_acao_18" not in k}
            vla.load_state_dict(state_filtrado, strict=False)
            print(f"[VLA] Pesos base carregados de '{ckpt_entrada}' ({len(state_filtrado)} tensores restaurados).")

    vla.to(dev)

    # 2. Carrega Dataset Tático de 36 classes
    dados = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset] {len(dados)} amostras táticas carregadas.")

    todos_sv    = torch.stack([d["sv"] for d in dados]).to(dev)
    todas_acoes = torch.tensor([int(d["acao_otima"]) for d in dados], dtype=torch.long, device=dev)
    todos_pesos = torch.tensor([float(d.get("peso", 1.0)) for d in dados], dtype=torch.float32, device=dev)

    acoes_np = todas_acoes.cpu().numpy()
    eh_alinhar = torch.tensor([0 <= a <= 8 for a in acoes_np], dtype=torch.bool, device=dev)
    eh_sprint  = torch.tensor([9 <= a <= 26 for a in acoes_np], dtype=torch.bool, device=dev)
    eh_strafe  = torch.tensor([27 <= a <= 32 for a in acoes_np], dtype=torch.bool, device=dev)
    eh_recuar  = torch.tensor([33 <= a <= 35 for a in acoes_np], dtype=torch.bool, device=dev)

    prompts = [d.get("prompt", "Objetivo: vá até o bloco azul [Etapa 1/2]") for d in dados]
    enc = tokenizer(prompts, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
    tokens_dataset = enc["input_ids"].to(dev)

    params_treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(params_treinaveis, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas, eta_min=lr * 0.1)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    print(f"[Treino] {len(params_treinaveis)} tensores treináveis ativos.")
    print("--- Iniciando Otimização Tática WASD ---")

    t_ini = time.time()
    num_exemplos = len(dados)

    for ep in range(1, epocas + 1):
        vla.train()
        indices = torch.randperm(num_exemplos, device=dev)
        total_loss = 0.0
        total_acertos = 0; total_geral = 0
        
        acertos_alinhar = 0; total_alinhar = 0
        acertos_sprint  = 0; total_sprint  = 0
        acertos_strafe  = 0; total_strafe  = 0
        acertos_recuar  = 0; total_recuar  = 0

        for i in range(0, num_exemplos, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_sv    = todos_sv[batch_idx]
            b_toks  = tokens_dataset[batch_idx]
            b_alvo  = todas_acoes[batch_idx]
            b_pesos = todos_pesos[batch_idx]
            
            b_is_alinhar = eh_alinhar[batch_idx]
            b_is_sprint  = eh_sprint[batch_idx]
            b_is_strafe  = eh_strafe[batch_idx]
            b_is_recuar  = eh_recuar[batch_idx]

            otimizador.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = vla.state_encoder(b_sv)
                t_embeds = vla.qwen_model.get_input_embeddings()(b_toks)
                inputs_embeds = torch.cat([s_embeds, t_embeds], dim=1)

                outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                last_hidden = outputs.last_hidden_state[:, -1, :]
                logits = vla.cabeca_acao_36(last_hidden)

                loss_raw = loss_fn(logits, b_alvo)
                loss = (loss_raw * b_pesos).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_treinaveis, max_norm=1.0)
            otimizador.step()

            total_loss += loss.item() * len(b_alvo)
            preds = logits.argmax(dim=-1)
            corretos = (preds == b_alvo)

            total_acertos += corretos.sum().item()
            total_geral   += len(b_alvo)

            if b_is_alinhar.any():
                acertos_alinhar += corretos[b_is_alinhar].sum().item()
                total_alinhar   += b_is_alinhar.sum().item()
            if b_is_sprint.any():
                acertos_sprint  += corretos[b_is_sprint].sum().item()
                total_sprint    += b_is_sprint.sum().item()
            if b_is_strafe.any():
                acertos_strafe  += corretos[b_is_strafe].sum().item()
                total_strafe    += b_is_strafe.sum().item()
            if b_is_recuar.any():
                acertos_recuar  += corretos[b_is_recuar].sum().item()
                total_recuar    += b_is_recuar.sum().item()

        scheduler.step()

        acc_geral   = (total_acertos / max(1, total_geral)) * 100.0
        acc_alinhar = (acertos_alinhar / max(1, total_alinhar)) * 100.0
        acc_sprint  = (acertos_sprint  / max(1, total_sprint))  * 100.0
        acc_strafe  = (acertos_strafe  / max(1, total_strafe))  * 100.0
        acc_recuar  = (acertos_recuar  / max(1, total_recuar))  * 100.0
        loss_media  = total_loss / max(1, total_geral)
        lr_atual    = scheduler.get_last_lr()[0]

        print(
            f"  Época {ep:2d}/{epocas} | Loss: {loss_media:.4f} | "
            f"Geral: {acc_geral:5.1f}% | "
            f"Sprint: {acc_sprint:5.1f}% | "
            f"Strafe: {acc_strafe:5.1f}% | "
            f"Giro: {acc_alinhar:5.1f}% | "
            f"Ré: {acc_recuar:5.1f}% | "
            f"LR: {lr_atual:.2e}",
            flush=True
        )

        # Salva o checkpoint a CADA ÉPOCA
        os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
        tensores_treinaveis = {
            k: v for k, v in vla.state_dict().items()
            if any(t in k for t in ["lora_", "state_encoder", "cabeca_acao_36"])
        }
        torch.save({
            "treinaveis":       tensores_treinaveis,
            "epoca":            ep,
            "acuracia_geral":   acc_geral,
            "acuracia_sprint":  acc_sprint,
            "acuracia_strafe":  acc_strafe,
            "acuracia_alinhar": acc_alinhar,
            "acuracia_recuar":  acc_recuar,
            "num_acoes":        36,
            "dataset_origem":   dataset_path,
        }, ckpt_saida)
        print(f"  [CHECKPOINT] Salvo para a Época {ep} em: {ckpt_saida}", flush=True)

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Treinamento tático concluído em {duracao:.1f}s.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fase5/dados/dataset_wasd_tatico_36.pt")
    ap.add_argument("--base",    default="checkpoints_vla/vla_fase5_sparse_anchored_grande.pt")
    ap.add_argument("--saida",   default="checkpoints_vla/vla_fase5_wasd_tatico.pt")
    ap.add_argument("--epocas",  type=int,   default=3)
    ap.add_argument("--lr",      type=float, default=2e-4)
    args = ap.parse_args()

    treinar_wasd_tatico(
        dataset_path=args.dataset,
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        epocas=args.epocas,
        lr=args.lr
    )
