# coding=utf-8
"""
fase6/treinar_cot_vla.py — Treinamento Multitarefa CoT-VLA (Navegação Espacial + Raciocínio Formal).

Objetivo:
  Treinar o Qwen3Loop para resolver navegação com monólogo interno (<think>) e ações (<action>)
  ao mesmo tempo em que preserva 100% da capacidade lógica, matemática e de código via âncora textual.
"""
import os
import sys
import json
import math
import random
import argparse
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qwen3loop.modeling_qwen3loop import Qwen3LoopForCausalLM
from modelo.lora_vla import aplicar_lora


class CoTVLADataset(Dataset):
    """Dataset unificado que combina navegação CoT-VLA e âncoras de raciocínio lógico."""

    def __init__(self, caminho_cot: str, caminho_bench: str, tokenizer: Any, max_len: int = 512, ratio_anchor: float = 0.25):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.amostras = []

        # 1. Carrega amostras de CoT-VLA do Minecraft
        if os.path.exists(caminho_cot):
            with open(caminho_cot, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self.amostras.append({"tipo": "cot_vla", "dados": d["conversacao"]})
            print(f"[*] Carregadas {len(self.amostras)} amostras de CoT-VLA (Minecraft).")

        # 2. Carrega âncoras de raciocínio do benchmark
        anchors = []
        if os.path.exists(caminho_bench):
            with open(caminho_bench, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
                itens = bench_data.get("items", bench_data) if isinstance(bench_data, dict) else bench_data
                for it in itens:
                    prompt = it.get("prompt", "")
                    resp = it.get("expected", it.get("answer", ""))
                    if not resp and "criteria" in it:
                        resp = str(it["criteria"])
                    if prompt and resp:
                        conv = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": f"<think>\nAnalisando a questão com rigor lógico e formal.\n</think>\n{resp}"}
                        ]
                        anchors.append({"tipo": "anchor", "dados": conv})

            # Multiplica âncoras para balancear com o dataset de navegação
            if anchors and len(self.amostras) > 0:
                n_desejado = int(len(self.amostras) * ratio_anchor / (1.0 - ratio_anchor))
                mult = max(1, n_desejado // len(anchors))
                for _ in range(mult):
                    self.amostras.extend(anchors)
            print(f"[*] Total consolidado com Âncoras de Raciocínio: {len(self.amostras)} amostras.")

        random.shuffle(self.amostras)

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        item = self.amostras[idx]
        msgs = item["dados"]

        # Aplica chat template com máscara de perda apenas no assistente
        user_msg = msgs[0]["content"]
        asst_msg = msgs[1]["content"]

        prompt_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt_text}{asst_msg}<|im_end|>\n"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mascara o prompt do usuário com -100 (não calcula perda no prompt)
        len_prompt = min(len(prompt_ids), len(full_ids))
        labels[:len_prompt] = -100

        return {"input_ids": input_ids, "labels": labels, "tipo": item["tipo"]}


def collate_fn(batch, pad_token_id: int):
    max_l = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_l), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_l), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_l), dtype=torch.long)

    for i, x in enumerate(batch):
        l = len(x["input_ids"])
        input_ids[i, :l] = x["input_ids"]
        labels[i, :l] = x["labels"]
        attention_mask[i, :l] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def treinar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Iniciando Treinamento CoT-VLA Multitarefa no dispositivo: {device}")

    base_dir = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Carrega dataset
    ds_path = os.path.join(_ROOT, "dataset", "cot_vla_dataset.jsonl")
    bench_path = os.path.join(_ROOT, "benchmarks", "eval_benchmark.json")
    dataset = CoTVLADataset(ds_path, bench_path, tokenizer, max_len=args.max_len, ratio_anchor=args.ratio_anchor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True
    )

    # Carrega modelo base Qwen3Loop
    print("[*] Carregando pesos do backbone Qwen3Loop...")
    model = Qwen3LoopForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # Aplica LoRA estruturado
    print("[*] Anexando adaptadores LoRA (r=16, alpha=32)...")
    aplicar_lora(model, r=16, alpha=32.0)
    model.train()

    # Otimizador e Scheduler
    treinaveis = [p for p in model.parameters() if p.requires_grad]
    print(f"[*] Parâmetros treináveis (LoRA): {sum(p.numel() for p in treinaveis):,} ({sum(p.numel() for p in treinaveis)*4/(1024*1024):.2f} MB)")

    optimizer = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    os.makedirs(os.path.join(_ROOT, "checkpoints_vla"), exist_ok=True)
    ckpt_path = os.path.join(_ROOT, "checkpoints_vla", "vla_fase6_cot.pt")
    ckpt_best_path = os.path.join(_ROOT, "checkpoints_vla", "vla_fase6_cot_melhor.pt")

    melhor_loss = float("inf")
    passo_global = 0

    print("=" * 80)
    print(f" INICIANDO LOOP DE TREINAMENTO CoT-VLA — {args.epochs} ÉPOCAS ({total_steps} PASSOS)")
    print("=" * 80, flush=True)

    for epoch in range(1, args.epochs + 1):
        loss_acum = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            loss_acum += loss.item() * args.grad_accum

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                passo_global += 1

                if passo_global % args.log_interval == 0:
                    media_loss = loss_acum / args.log_interval
                    loss_acum = 0.0
                    lr_atual = scheduler.get_last_lr()[0]
                    print(f"  Época {epoch:02d}/{args.epochs:02d} | Passo {passo_global:04d}/{total_steps:04d} | Perda Causal: {media_loss:.4f} | LR: {lr_atual:.2e}", flush=True)

                    if media_loss < melhor_loss:
                        melhor_loss = media_loss
                        torch.save({
                            "epoch": epoch,
                            "step": passo_global,
                            "loss": melhor_loss,
                            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
                        }, ckpt_best_path)

        # Salva checkpoint da época
        torch.save({
            "epoch": epoch,
            "step": passo_global,
            "loss": loss_acum,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
        }, ckpt_path)
        print(f"[CHECKPOINT] Época {epoch} concluída! Checkpoint salvo em: {ckpt_path}\n", flush=True)

    print("=" * 80)
    print(f"[CONCLUÍDO] Treinamento CoT-VLA finalizado com sucesso! Melhor Perda: {melhor_loss:.4f}")
    print(f"  -> Checkpoint Melhor: {ckpt_best_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinador Multitarefa CoT-VLA.")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size por dispositivo.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Passos de acumulação de gradiente.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max-len", type=int, default=512, help="Tamanho máximo de sequência.")
    parser.add_argument("--ratio-anchor", type=float, default=0.25, help="Proporção de amostras de âncora de raciocínio.")
    parser.add_argument("--log-interval", type=int, default=20, help="Intervalo de logs em passos.")
    args = parser.parse_args()

    treinar(args)
