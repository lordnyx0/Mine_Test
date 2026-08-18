# coding=utf-8
"""
fundir_lora_fase4.py — Fusão Permanente dos Pesos LoRA da Fase 4 no Backbone Qwen2-VL.

1. Instancia o VLA com camadas LoRA (112 camadas).
2. Carrega todos os 262 tensores treináveis da Fase 4 (incluindo lora_A e lora_B).
3. Mescla o delta (B @ A) * (alpha / r) diretamente nos pesos lineares base W_base.
4. Salva o checkpoint consolidado final em `checkpoints_vla/vla_fase4_merged.pt`.
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.run_vla_agent import load_vla_agent
from modelo.lora_vla import aplicar_lora, mesclar_todos_lora

def fundir_checkpoint_fase4(ckpt_entrada="checkpoints_vla/vla_fase4_logica.pt",
                            ckpt_saida="checkpoints_vla/vla_fase4_merged.pt"):
    print("=" * 80)
    print(" [FUSAO LORA FASE 4] CONSOLIDACAO DOS PESOS NO BACKBONE BASE")
    print(f"    Entrada: {ckpt_entrada}")
    print(f"    Saida:   {ckpt_saida}")
    print("=" * 80)

    # 1. Carrega modelo base
    vla, device = load_vla_agent("checkpoints_vla/vla_fase3_merged.pt")

    # 2. Injeta as camadas LoRA para receber os tensores lora_A e lora_B
    vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    # 3. Carrega os tensores da Fase 4
    sd = torch.load(ckpt_entrada, map_location=device, weights_only=False)
    vla.load_state_dict(sd, strict=False)
    print(f"[VLA] Checkpoint '{ckpt_entrada}' carregado nas camadas LoRA.")

    # 4. Mescla o delta LoRA permanentemente nos pesos base
    mesclar_todos_lora(vla.qwen_model)

    # 5. Salva modelo consolidado completo
    os.makedirs(os.path.dirname(ckpt_saida), exist_ok=True)
    torch.save(vla.state_dict(), ckpt_saida)
    
    tamanho_mb = os.path.getsize(ckpt_saida) / (1024 * 1024)
    print(f"[OK] Checkpoint consolidado salvo com sucesso em {ckpt_saida} ({tamanho_mb:.1f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    fundir_checkpoint_fase4()
