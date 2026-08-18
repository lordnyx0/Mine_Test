# coding=utf-8
"""
fase5/treinar_calibracao.py — Fine-tuning de Calibracao Direcional com Early Stopping.
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


def treinar_calibracao(
    dataset_path: str = "fase5/dados/dataset_calibrado.pt",
    ckpt_entrada: str = "checkpoints_vla/vla_fase5_sparse_anchored.pt",
    ckpt_saida:   str = "checkpoints_vla/vla_fase5_calibrado.pt",
    epocas:       int = 10,
    batch_size:   int = 32,
    lr:         float = 5e-5,
    peso_alto_erro: float = 2.0,
    limiar_peso_graus: float = 60.0,
    early_stop_acc: float = 80.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" [FASE 5] FINE-TUNING DE CALIBRACAO DIRECIONAL (com Early Stopping)")
    print(f"    Dataset Entrada    : {dataset_path}")
    print(f"    Checkpoint Base    : {ckpt_entrada}")
    print(f"    Checkpoint Saida   : {ckpt_saida}")
    print(f"    Epocas Max         : {epocas} | Batch: {batch_size} | LR: {lr}")
    print(f"    Peso Erros > {limiar_peso_graus:.0f}g  : {peso_alto_erro}x")
    print(f"    Early Stop         : Bifurc >= {early_stop_acc:.0f}%  (evita overfitting)")
    print("=" * 80)

    vla, dev = load_vla_agent(None)
    compactar_backbone(vla)
    vla.to(dev)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not any("lora_" in n for n, _ in vla.named_parameters()):
        aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    pol = PoliticaRaciocinioLoop(None, amostrar=False, device=dev, vla=vla, loops_pensamento=3)

    if os.path.exists(ckpt_entrada):
        ckpt_data = torch.load(ckpt_entrada, map_location=dev)
        if "treinaveis" in ckpt_data:
            vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Pesos ancorados carregados de '{ckpt_entrada}' ({len(ckpt_data['treinaveis'])} tensores).")
    else:
        print(f"[AVISO] Checkpoint '{ckpt_entrada}' nao encontrado!")

    vla.to(dev)

    dados = torch.load(dataset_path, weights_only=False)
    print(f"[Dataset] {len(dados)} amostras de erro de bifurcacao carregadas.")

    todos_sv    = torch.stack([d["sv"] for d in dados]).to(dev)
    todas_acoes = torch.tensor([int(d["acao_otima"]) for d in dados], dtype=torch.long, device=dev)

    erros_ang = [abs(float(d.get("erro_yaw_graus", 90.0))) for d in dados]
    pesos_raw = [peso_alto_erro if e > limiar_peso_graus else 1.0 for e in erros_ang]
    todos_pesos = torch.tensor(pesos_raw, dtype=torch.float32, device=dev)

    tipos_flag = [d.get("tipo", "erro_bifurcacao") for d in dados]
    eh_spawn = torch.tensor([t == "spawn" for t in tipos_flag], dtype=torch.bool, device=dev)

    prompts = [d.get("prompt", "Objetivo: va ate o bloco azul [Etapa 1/2]") for d in dados]
    enc = tokenizer(prompts, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
    tokens_dataset = enc["input_ids"].to(dev)

    params_treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(params_treinaveis, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas, eta_min=lr * 0.1)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    print(f"[Treino] {len(params_treinaveis)} tensores | {sum(p.numel() for p in params_treinaveis):,} params")
    print(f"[Pesos] {sum(1 for w in pesos_raw if w > 1.0)} amostras com peso {peso_alto_erro}x")
    print("--- Iniciando Calibracao Direcional ---")

    num_exemplos = len(dados)
    t_ini = time.time()
    melhor_acc_bifurc = 0.0
    melhor_epoca = 0

    for ep in range(1, epocas + 1):
        vla.train()
        indices = torch.randperm(num_exemplos, device=dev)
        total_loss   = 0.0
        total_acertos = 0
        total_geral  = 0
        acertos_spawn = 0; total_spawn = 0
        acertos_bifurc = 0; total_bifurc = 0

        for i in range(0, num_exemplos, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_sv    = todos_sv[batch_idx]
            b_toks  = tokens_dataset[batch_idx]
            b_alvo  = todas_acoes[batch_idx]
            b_pesos = todos_pesos[batch_idx]
            b_spawn = eh_spawn[batch_idx]

            otimizador.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = vla.state_encoder(b_sv)
                t_embeds = vla.qwen_model.get_input_embeddings()(b_toks)
                inputs_embeds = torch.cat([s_embeds, t_embeds], dim=1)
                outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                last_hidden = outputs.last_hidden_state[:, -1, :]
                logits = vla.cabeca_acao_18(last_hidden)
                loss_raw = loss_fn(logits, b_alvo)
                loss = (loss_raw * b_pesos).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_treinaveis, max_norm=0.5)
            otimizador.step()

            total_loss    += loss.item() * len(b_alvo)
            preds          = logits.argmax(dim=-1)
            corretos       = (preds == b_alvo)
            total_acertos += corretos.sum().item()
            total_geral   += len(b_alvo)

            if b_spawn.any():
                acertos_spawn += corretos[b_spawn].sum().item()
                total_spawn   += b_spawn.sum().item()
            if (~b_spawn).any():
                acertos_bifurc += corretos[~b_spawn].sum().item()
                total_bifurc   += (~b_spawn).sum().item()

        scheduler.step()

        acc_geral  = (total_acertos / max(1, total_geral))  * 100.0
        acc_spawn  = (acertos_spawn  / max(1, total_spawn))  * 100.0
        acc_bifurc = (acertos_bifurc / max(1, total_bifurc)) * 100.0
        loss_media = total_loss / max(1, total_geral)
        lr_atual   = scheduler.get_last_lr()[0]

        print(
            f"  Epoca {ep:2d}/{epocas} | Loss: {loss_media:.4f} | "
            f"Geral: {acc_geral:5.1f}% | "
            f"Spawn: {acc_spawn:5.1f}% ({acertos_spawn}/{total_spawn}) | "
            f"Bifurc: {acc_bifurc:5.1f}% ({acertos_bifurc}/{total_bifurc}) | "
            f"LR: {lr_atual:.2e}",
            flush=True
        )

        # Salva se for o melhor checkpoint ate agora
        if acc_bifurc > melhor_acc_bifurc:
            melhor_acc_bifurc = acc_bifurc
            melhor_epoca = ep
            os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
            tensores_treinaveis = {
                k: v for k, v in vla.state_dict().items()
                if any(t in k for t in ["lora_", "state_encoder", "cabeca_acao_18"])
            }
            torch.save({
                "treinaveis":      tensores_treinaveis,
                "epoca":           ep,
                "acuracia_geral":  acc_geral,
                "acuracia_bifurc": acc_bifurc,
                "acuracia_spawn":  acc_spawn,
                "dataset_origem":  dataset_path,
                "ckpt_base":       ckpt_entrada,
            }, ckpt_saida)
            print(f"  [SAVE] Melhor checkpoint salvo (Bifurc={acc_bifurc:.1f}%)", flush=True)

        # Early stopping: para quando Bifurc >= limiar para evitar overfitting
        if acc_bifurc >= early_stop_acc:
            print(f"\n[EARLY STOP] Bifurc={acc_bifurc:.1f}% >= {early_stop_acc:.0f}% na epoca {ep}. Parando antes de overfitting.", flush=True)
            break

    duracao = time.time() - t_ini
    print("=" * 80)
    print(f"[OK] Calibracao concluida em {duracao:.1f}s.")
    print(f"[OK] Melhor checkpoint: epoca {melhor_epoca} com Bifurc={melhor_acc_bifurc:.1f}%")
    print(f"[OK] Salvo em: {ckpt_saida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default="fase5/dados/dataset_calibrado.pt")
    ap.add_argument("--base",       default="checkpoints_vla/vla_fase5_sparse_anchored.pt")
    ap.add_argument("--saida",      default="checkpoints_vla/vla_fase5_calibrado.pt")
    ap.add_argument("--epocas",     type=int,   default=10)
    ap.add_argument("--lr",         type=float, default=5e-5)
    ap.add_argument("--peso",       type=float, default=2.0)
    ap.add_argument("--limiar",     type=float, default=60.0)
    ap.add_argument("--early-stop", type=float, default=80.0, dest="early_stop")
    args = ap.parse_args()

    treinar_calibracao(
        dataset_path=args.dataset,
        ckpt_entrada=args.base,
        ckpt_saida=args.saida,
        epocas=args.epocas,
        lr=args.lr,
        peso_alto_erro=args.peso,
        limiar_peso_graus=args.limiar,
        early_stop_acc=args.early_stop,
    )
