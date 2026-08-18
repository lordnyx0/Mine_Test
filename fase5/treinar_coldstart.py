# coding=utf-8
"""
FASE 5 — Treinamento SFT Cold-Start nas Bifurcações Críticas (Sparse Policy Selection).

Aquece a cabeça de 18 ações e os adaptadores LoRA exclusivamente sobre os pontos
de alta entropia (spawn e transição de submetas).

Tempo de execução: ~60 segundos na GPU (RTX 4060).
Resultado: Implanta o suporte de prior não-nulo para o PPO selecionar as políticas.
"""
from __future__ import annotations

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transformers import AutoTokenizer
from infra.run_vla_agent import load_vla_agent
from modelo.lora_vla import aplicar_lora
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from politica.cerebro import PoliticaCerebroVLA


def treinar_coldstart(
    dataset_path: str = "fase5/dados/dataset_bifurcacoes_coldstart.pt",
    saida_ckpt: str = "checkpoints_vla/vla_fase5_coldstart.pt",
    epocas: int = 20,
    lr: float = 1.0e-4,
    batch_size: int = 16,
    device: str = "cuda"
):
    print("=" * 80)
    print(" [FASE 5] SFT COLD-START NAS BIFURCAÇÕES DE DECISÃO")
    print(f"    Dataset  : {dataset_path}")
    print(f"    Saida    : {saida_ckpt}")
    print(f"    Épocas   : {epocas} | LR: {lr} | Batch Size: {batch_size}")
    print("=" * 80)

    # 1. Carrega o modelo VLA base e tokenizer
    vla, _ = load_vla_agent(None)
    vla.to(device)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints_vla/backbone_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Aplica LoRA (rank 16, alpha 32)
    aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    # Instancia Política com a cabeça de 18 ações
    pol_loop = PoliticaRaciocinioLoop(device=device, vla=vla)
    politica = PoliticaCerebroVLA(pol_loop)
    vla.to(device)

    # 2. Carrega dataset de bifurcações
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset nao encontrado em '{dataset_path}'. Execute gerar_demonstracoes_esparsas.py primeiro.")
    
    dados: List[Dict[str, Any]] = torch.load(dataset_path, weights_only=False)
    print(f"[1/3] Carregadas {len(dados)} amostras de bifurcação.")

    # Parâmetros treináveis: LoRA + Projeção Estado + Cabeça de Ação 18
    params_treinaveis = [p for p in vla.parameters() if p.requires_grad]
    otimizador = optim.AdamW(params_treinaveis, lr=lr, weight_decay=1e-4)
    criterio = nn.CrossEntropyLoss()

    # Pre-tokeniza todos os prompts uma única vez
    print("[3/4] Pre-processando tokens de instrução...")
    todos_prompts = [item["prompt"] for item in dados]
    tokens_dataset = tokenizer(todos_prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)["input_ids"].to(device)
    todos_sv = torch.stack([item["sv"] for item in dados]).to(device)
    todas_acoes = torch.tensor([item["acao_alvo"] for item in dados], dtype=torch.long, device=device)

    print(f"[4/4] Iniciando treinamento de aquecimento ({epocas} épocas)...\n", flush=True)

    t0 = time.time()
    for epoca in range(1, epocas + 1):
        indices = torch.randperm(len(dados), device=device)
        total_loss = 0.0
        acertos = 0
        total = 0

        vla.train()
        for i in range(0, len(dados), batch_size):
            batch_idx = indices[i:i + batch_size]
            b_sv = todos_sv[batch_idx]
            b_toks = tokens_dataset[batch_idx]
            b_alvo = todas_acoes[batch_idx]

            otimizador.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = vla.state_encoder(b_sv) # [B, 4, H]
                t_embeds = vla.qwen_model.get_input_embeddings()(b_toks) # [B, T, H]
                inputs_embeds = torch.cat([s_embeds, t_embeds], dim=1) # [B, 4+T, H]

                outputs = vla.qwen_model(inputs_embeds=inputs_embeds)
                last_hidden = outputs.last_hidden_state[:, -1, :] # [B, H]
                logits = vla.cabeca_acao_18(last_hidden) # [B, 18]

                loss = criterio(logits, b_alvo)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_treinaveis, 1.0)
            otimizador.step()

            total_loss += loss.item() * len(batch_idx)
            preds = logits.argmax(dim=-1)
            acertos += (preds == b_alvo).sum().item()
            total += len(batch_idx)

        acc = (acertos / total) * 100.0
        media_loss = total_loss / total

        if epoca % 2 == 0 or epoca == 1 or epoca == epocas:
            print(f"  Época {epoca:02d}/{epocas} | Loss: {media_loss:.4f} | Acurácia nas Bifurcações: {acc:.1f}%", flush=True)

    duracao = time.time() - t0
    print(f"\n[OK] Treinamento Cold-Start concluído em {duracao:.1f}s com {acc:.1f}% de acurácia nas decisões.", flush=True)

    # Salva checkpoint com todos os tensores treináveis (LoRA + cabeça + projeção)
    os.makedirs(os.path.dirname(saida_ckpt), exist_ok=True)
    treinaveis = {n: p for n, p in vla.named_parameters() if p.requires_grad}
    torch.save({
        "treinaveis": {k: v.cpu() for k, v in treinaveis.items()},
        "epoca": epocas,
        "acuracia": acc
    }, saida_ckpt)
    print(f"[OK] Checkpoint de Cold-Start ({len(treinaveis)} tensores) salvo em: {saida_ckpt}", flush=True)


if __name__ == "__main__":
    treinar_coldstart()
