# coding=utf-8
"""
Supervised Fine-Tuning (SFT) & Action Prediction Trainer for Qwen3LoopVLA.
Trains Perceiver Resampler, Visual Projector, State Encoder, and Action Heads
on approved trajectories (+1.0 reward) in dataset/curriculum_trajectories.jsonl.
"""
import io
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from infra.gpu_utils import (limitar_recursos, limitar_vram, travar_gpu,
                             compactar_backbone, memoria_gpu)

limitar_recursos()
import time
import base64
from PIL import Image
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from qwen3loop.modeling_qwen3loop import Qwen3LoopModel
from qwen3loop.configuration_qwen3loop import Qwen3LoopConfig
from modelo.vla_model import Qwen3LoopVLA

_LOCAL_TOK = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
TOKENIZER_PATH = _LOCAL_TOK if os.path.exists(_LOCAL_TOK) else "C:/Users/Nyx/Desktop/Testes/checkpoints/qwen3loop_v2/final_model"

def _json_balanceado(texto):
    r"""
    Extrai o primeiro objeto JSON COMPLETO do texto, contando chaves.

    A versao anterior usava re.search(r"\{.*?\}") — nao-gulosa. Em
    {"action": {"hold": ["W"], ...}, "completed": false} ela capturava
    {"action": {"hold": ["W"], "mouse": [0, 0]} — fragmento desbalanceado que
    json.loads rejeitava. O except engolia o erro e o alvo virava vetor ZERO.
    Resultado: os 320 exemplos ensinavam "nao aperte nada", e a via visual
    colapsava porque o alvo nao dependia da imagem.
    """
    ini = texto.find("{")
    if ini < 0:
        return None
    nivel, dentro_str, escapa = 0, False, False
    for i in range(ini, len(texto)):
        c = texto[i]
        if escapa:
            escapa = False
            continue
        if c == "\\":
            escapa = True
            continue
        if c == '"':
            dentro_str = not dentro_str
            continue
        if dentro_str:
            continue
        if c == "{":
            nivel += 1
        elif c == "}":
            nivel -= 1
            if nivel == 0:
                try:
                    return json.loads(texto[ini:i + 1])
                except Exception:
                    return None
    return None


def extrair_acao(sample):
    """
    Acao alvo de uma amostra. Aceita os formatos que aparecem nos datasets:
      {"action": {"hold": [...], "mouse": [...]}}   (locomotion/curriculum)
      {"hold": [...], "mouse": [...]}               (live_trajectories)
    Cai para o campo action_type ("W", "A"...) se as mensagens falharem.
    """
    for m in sample.get("messages", []):
        if m.get("role") != "assistant":
            continue
        d = _json_balanceado(m.get("content", ""))
        if not d:
            continue
        acao = d.get("action", d)
        if isinstance(acao, dict) and ("hold" in acao or "mouse" in acao):
            return acao

    if isinstance(sample.get("action"), dict):
        return sample["action"]

    at = sample.get("action_type")
    if isinstance(at, str) and at.strip():
        return {"hold": [at.strip().upper()], "mouse": [0, 0]}
    return {}


class MinecraftVLADataset(Dataset):
    """Dataset loader parsing curriculum_trajectories.jsonl for PyTorch training."""
    def __init__(self, jsonl_path="dataset/curriculum_trajectories.jsonl", processor=None):
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.samples = []
        
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        # Train on positive reward trajectories (+1.0)
                        if data.get("reward", 0.0) > 0.0 and data.get("image_b64"):
                            self.samples.append(data)
                    except Exception:
                        pass
        print(f"[Dataset VLA] Carregadas {len(self.samples)} amostras de trajetória aprovadas com recompensa positiva (+1.0).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        b64_str = sample["image_b64"]
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Extract target action values
        action_dict = extrair_acao(sample)

        # Target button vector: [forward, backward, left, right, jump, attack, use, sprint]
        hold_keys = action_dict.get("hold", [])
        press_keys = action_dict.get("press", [])
        active_keys = [k.upper() for k in (hold_keys + press_keys)]
        
        target_buttons = torch.tensor([
            1.0 if "W" in active_keys else 0.0,
            1.0 if "S" in active_keys else 0.0,
            1.0 if "A" in active_keys else 0.0,
            1.0 if "D" in active_keys else 0.0,
            1.0 if "SPACE" in active_keys else 0.0,
            1.0 if "LCLICK" in active_keys else 0.0,
            1.0 if "RCLICK" in active_keys else 0.0,
            1.0 if "SHIFT" in active_keys else 0.0,
        ], dtype=torch.float32)

        # Target camera vector: [yaw, pitch]
        mouse = action_dict.get("mouse", [0, 0])
        target_camera = torch.tensor([float(mouse[0]), float(mouse[1])], dtype=torch.float32)

        # INSTRUCAO da amostra. Sem ela o treino e insoluvel: as 8 instrucoes
        # do dataset dao alvos diferentes para cenas parecidas, e o unico
        # otimo possivel e a media (botoes ~0.5, camera ~0) — exatamente o que
        # o modelo fazia.
        instrucao = ""
        for m in sample.get("messages", []):
            if m.get("role") == "user":
                instrucao = str(m.get("content", ""))
                break

        # Target hotbar index (0..8)
        target_hotbar = torch.tensor(0, dtype=torch.long)

        # Estado ALEATORIO em vez de zeros. Neste dataset a acao correta depende
        # so da instrucao, entao variar o estado ensina o modelo a ignora-lo aqui
        # — e o impede de quebrar quando, em execucao, ele chega preenchido.
        from modelo.vla_model import STATE_DIM
        state_vec = (torch.rand(STATE_DIM, dtype=torch.float32) * 2 - 1) * 0.5

        return {
            "image": img,
            "state_vec": state_vec,
            "target_buttons": target_buttons,
            "target_camera": target_camera,
            "target_hotbar": target_hotbar,
            "instrucao": instrucao,
        }

_TOKENIZER = None


def get_tokenizer():
    """Tokenizer do Qwen, com padding a ESQUERDA — a acao e lida do ULTIMO
    token, entao pad no fim colocaria a leitura em cima de um token vazio."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        for caminho in (TOKENIZER_PATH, "Qwen/Qwen3-0.6B"):
            try:
                _TOKENIZER = AutoTokenizer.from_pretrained(caminho)
                break
            except Exception:
                continue
        _TOKENIZER.padding_side = "left"
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
    return _TOKENIZER


# O agente roda com pilha temporal de N_FRAMES e vetor de estado real. Treinar
# com 1 frame e estado zerado cria um formato de entrada DIFERENTE do de
# execucao: o modelo decorava o treino (loss 0.005) e errava 8/8 na pratica.
N_FRAMES_TREINO = 3


def collate_vla(batch, processor):
    images = [b["image"] for b in batch]
    pixel_inputs = processor(images=images, return_tensors="pt")
    px = pixel_inputs["pixel_values"]
    # Repete o frame: o dataset e de imagens soltas, entao "sem movimento".
    px = px.unsqueeze(1).expand(-1, N_FRAMES_TREINO, -1, -1, -1).contiguous()
    pixel_inputs = {"pixel_values": px}

    tok = get_tokenizer()
    enc = tok([b["instrucao"] for b in batch], return_tensors="pt",
              padding=True, truncation=True, max_length=24)
    
    state_vecs = torch.stack([b["state_vec"] for b in batch])
    target_buttons = torch.stack([b["target_buttons"] for b in batch])
    target_camera = torch.stack([b["target_camera"] for b in batch])
    target_hotbar = torch.stack([b["target_hotbar"] for b in batch])
    
    return {
        "pixel_values": pixel_inputs["pixel_values"],
        "state_vec": state_vecs,
        "target_buttons": target_buttons,
        "target_camera": target_camera,
        "target_hotbar": target_hotbar,
        "input_ids": enc["input_ids"],
    }

def train_vla(epochs=5, batch_size=2, lr=1e-4, dataset_path="dataset/curriculum_trajectories.jsonl", save_path="checkpoints_vla/vla_projector.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TREINAMENTO VLA (Fase 1: Projector + Resampler + ActionHeads) ===")
    print(f"Dispositivo: {device}")

    config = Qwen3LoopConfig(
        vocab_size=151936,
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_loops=2,
        enable_double_loop_split=True
    )
    qwen_model = Qwen3LoopModel(config).to(device)

    vla_model = Qwen3LoopVLA(
        qwen_model=qwen_model,
        vision_model_name="google/siglip-base-patch16-224",
        num_visual_tokens=32,
        freeze_vision=True,
        freeze_qwen=True
    ).to(device)

    # Backbone congelado em bf16: metade da VRAM e mais rapido. Os modulos
    # treinaveis ficam em fp32.
    dt = compactar_backbone(vla_model)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # Teto DEPOIS da compactacao: o modelo nasce fp32 (8.7GB) e so entao vira
    # bf16 (4.4GB). Sem teto este script chegou a 10.4GB de 12 e arriscou
    # travar a maquina, que foi o que aconteceu uma vez.
    teto = limitar_vram(float(os.environ.get("VRAM_FRACAO", "0.72")))
    print(f"[VRAM] backbone em {dt} | teto {teto} | {memoria_gpu()}", flush=True)

    dataset = MinecraftVLADataset(jsonl_path=dataset_path, processor=vla_model.vision_processor)
    if len(dataset) == 0:
        print("[AVISO] Nenhuma amostra de trajetória aprovada encontrada. Colete trajetórias no trainer_interface.py primeiro!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_vla(b, vla_model.vision_processor)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, vla_model.parameters()),
        lr=lr,
        weight_decay=1e-2
    )

    usar_amp = torch.cuda.is_available()
    bce_loss = nn.BCEWithLogitsLoss()   # seguro sob autocast (BCELoss nao e)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        vla_model.train()
        vla_model.vision_encoder.eval() # Keep frozen
        vla_model.qwen_model.eval()     # Keep frozen

        epoch_loss = 0.0
        t0 = time.perf_counter()

        for step, batch in enumerate(dataloader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            state_vec = batch["state_vec"].to(device)
            t_buttons = batch["target_buttons"].to(device)
            t_camera = batch["target_camera"].to(device)
            t_hotbar = batch["target_hotbar"].to(device)
            ids = batch["input_ids"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # autocast e obrigatorio: o backbone esta em bf16 e as entradas em fp32
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=usar_amp):
                actions = vla_model(pixel_values=pixel_values, state_vec=state_vec,
                                    input_ids=ids)

            YB = vla_model.action_heads.YAW_BINS
            PB = vla_model.action_heads.PITCH_BINS
            t_yaw = torch.tensor([min(range(len(YB)), key=lambda i: abs(YB[i] - float(v)))
                                  for v in t_camera[:, 0]], dtype=torch.long, device=device)
            t_pit = torch.tensor([min(range(len(PB)), key=lambda i: abs(PB[i] - float(v)))
                                  for v in t_camera[:, 1]], dtype=torch.long, device=device)

            loss_b = bce_loss(actions["buttons_logits"].float(), t_buttons)
            loss_c = mse_loss(actions["camera"].float(), t_camera)
            loss_h = ce_loss(actions["hotbar_logits"].float(), t_hotbar)
            # cabecas DISCRETAS de giro: e delas que sai a acao de camera agora
            loss_y = ce_loss(actions["yaw_logits"].float(), t_yaw)
            loss_p = ce_loss(actions["pitch_logits"].float(), t_pit)

            loss = loss_b + 0.01 * loss_c + 0.1 * loss_h + loss_y + 0.3 * loss_p

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        dt = time.perf_counter() - t0
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch}/{epochs}] | Loss Total: {avg_loss:.4f} | Tempo: {dt:.2f}s", flush=True)

        # Salva a cada epoca: uma interrupcao no meio nao pode custar o treino
        # inteiro (ja aconteceu — perdi 3 epocas por salvar so no final).
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({
            "resampler": vla_model.resampler.state_dict(),
            "projector": vla_model.projector.state_dict(),
            "state_encoder": vla_model.state_encoder.state_dict(),
            "action_heads": vla_model.action_heads.state_dict(),
            "epoch": epoch, "loss": avg_loss,
        }, save_path)
        print(f"  [ckpt] epoca {epoch} salva em {save_path}", flush=True)

    torch.save({
        "resampler": vla_model.resampler.state_dict(),
        "projector": vla_model.projector.state_dict(),
        "state_encoder": vla_model.state_encoder.state_dict(),
        "action_heads": vla_model.action_heads.state_dict()
    }, save_path)

    print(f"\n[TREINAMENTO CONCLUIDO] Pesos VLA salvos com sucesso em: {save_path}", flush=True)

import re

def main():
    dataset_path = "dataset/locomotion_trajectories.jsonl" if os.path.exists("dataset/locomotion_trajectories.jsonl") else "dataset/curriculum_trajectories.jsonl"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    save_path = "checkpoints_vla/vla_locomotion.pt" if "locomotion" in dataset_path else "checkpoints_vla/vla_projector.pt"
    # 3 epocas bastam: a loss cai para ~0.001 na segunda (medido).
    travar_gpu()          # nunca dois treinos na mesma GPU
    epochs = int(os.environ.get("EPOCHS", "3"))
    train_vla(epochs=epochs, batch_size=4, lr=5e-4, dataset_path=dataset_path, save_path=save_path)

if __name__ == "__main__":
    main()

