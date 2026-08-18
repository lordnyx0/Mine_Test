# coding=utf-8
"""
Script de Fusão dos Pesos LoRA no Backbone do Qwen3Loop.

Carrega o checkpoint treinado vla_fase3_lora.pt, aplica o delta LoRA
(W_novo = W_base + B @ A * alpha / r) permanentemente nas matrizes de atenção
e salva o modelo consolidado sem overhead em checkpoints_vla/vla_fase3_merged.pt.
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.run_vla_agent import load_vla_agent
from modelo.lora_vla import aplicar_lora, mesclar_todos_lora

CKPT_ENTRADA = "checkpoints_vla/vla_fase3_lora.pt"
CKPT_SAIDA = "checkpoints_vla/vla_fase3_merged.pt"


def main():
    print("=== FUSÃO DE PESOS LORA (QWEN3LOOP VLA) ===")
    vla, device = load_vla_agent(None)  # base
    vla.to("cpu")

    # Injeta a estrutura LoRA
    vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    # Carrega os tensores treinados
    print(f"[LoRA Merge] Carregando pesos de {CKPT_ENTRADA}...")
    dados = torch.load(CKPT_ENTRADA, map_location="cpu", weights_only=False)
    treinaveis = dados.get("treinaveis", dados)

    # Restaura pesos no VLA
    faltando = []
    for n, p in vla.named_parameters():
        if n in treinaveis:
            p.data.copy_(treinaveis[n].data)
        elif p.requires_grad:
            faltando.append(n)

    print(f"[LoRA Merge] {len(treinaveis)} tensores restaurados. (Faltando: {len(faltando)})")

    # Mescla as camadas LoRA nos pesos base
    mesclar_todos_lora(vla.qwen_model)

    # Salva o checkpoint fundido
    print(f"[LoRA Merge] Salvando checkpoint consolidado -> {CKPT_SAIDA}...")
    torch.save(vla.state_dict(), CKPT_SAIDA)
    print("[OK] Fusão concluída com sucesso! Modelo unificado pronto para inferência rápida e exportação GGUF.")


if __name__ == "__main__":
    main()
