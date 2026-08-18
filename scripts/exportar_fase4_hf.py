# coding=utf-8
"""
exportar_fase4_hf.py — Extrai o backbone Qwen3Loop da Fase 4 e salva em formato HF.

Aplica a fusão dos adaptadores LoRA treinados diretamente sobre o backbone base,
preservando a arquitetura True Loop (28 camadas físicas, LoopSplit) para conversão
GGUF ou inferência direta PyTorch.
"""
import os
import sys
import shutil
import json
import argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from modelo.lora_vla import aplicar_lora
from qwen3loop import Qwen3LoopForCausalLM
from transformers import AutoTokenizer


def exportar(ckpt_vla="checkpoints_vla/vla_fase4_logica.pt",
             base_hf="checkpoints_vla/backbone_base",
             saida_hf="checkpoints_vla/fase4_hf"):
    print("=" * 80)
    print(" [EXPORTACAO FASE 4 HF] Extraindo backbone Qwen3Loop preservando LoopSplit")
    print(f"    Checkpoint VLA : {ckpt_vla}")
    print(f"    Base HF        : {base_hf}")
    print(f"    Destino HF     : {saida_hf}")
    print("=" * 80)

    # 1. Carrega o modelo base
    print("[1/4] Carregando backbone base...")
    causal_lm = Qwen3LoopForCausalLM.from_pretrained(
        base_hf,
        dtype=torch.bfloat16,
        device_map="cpu"
    )

    # 2. Aplica LoRA se o checkpoint tiver adaptadores LoRA
    if os.path.exists(ckpt_vla):
        ckpt_data = torch.load(ckpt_vla, map_location="cpu")
        treinaveis = ckpt_data.get("treinaveis", {})
        
        # Identifica se há tensores de LoRA
        tem_lora = any("lora_" in k for k in treinaveis.keys())
        if tem_lora:
            print(f"[2/4] Aplicando LoRA (r=16, alpha=32.0) e restaurando {len(treinaveis)} tensores...")
            causal_lm.model = aplicar_lora(causal_lm.model, r=16, alpha=32.0)
            
            # Filtra apenas tensores do qwen_model
            lora_dict = {}
            for k, v in treinaveis.items():
                if k.startswith("qwen_model."):
                    clean_k = k.replace("qwen_model.", "")
                    lora_dict[clean_k] = v
                elif "lora_" in k:
                    lora_dict[k] = v
            
            msg = causal_lm.model.load_state_dict(lora_dict, strict=False)
            print(f"      Restaurados: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            
            # Funde e desempacota os pesos LoRA na matriz base
            print("[3/4] Fundindo adaptadores LoRA no backbone base (merge & unload)...")
            from modelo.lora_vla import descarregar_lora
            causal_lm.model = descarregar_lora(causal_lm.model)
            print("      Fusão LoRA concluída com sucesso.")
        else:
            print("[2/4] Checkpoint já consolidado (sem LoRA).")

    # Vincula o lm_head aos embeddings caso tie_word_embeddings
    if causal_lm.config.tie_word_embeddings:
        causal_lm.lm_head.weight = causal_lm.model.embed_tokens.weight

    # 3. Salva a pasta Hugging Face
    print(f"[4/4] Salvando pasta HF em {saida_hf}...")
    if os.path.exists(saida_hf):
        shutil.rmtree(saida_hf)
    os.makedirs(saida_hf, exist_ok=True)

    causal_lm.save_pretrained(saida_hf, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_hf).save_pretrained(saida_hf)

    # Copia chat_template.jinja e generation_config caso existam
    for extra in ["chat_template.jinja", "generation_config.json"]:
        src_f = os.path.join(base_hf, extra)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(saida_hf, extra))

    # Garante configuração correta
    cfg_file = os.path.join(saida_hf, "config.json")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    cfg_data["model_type"] = "qwen3loop"
    cfg_data["architectures"] = ["Qwen3LoopForCausalLM"]
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    print(f"[OK] Modelo HF Fase 4 exportado com sucesso para {saida_hf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_vla/vla_fase4_logica.pt")
    ap.add_argument("--base", default="checkpoints_vla/backbone_base")
    ap.add_argument("--saida", default="checkpoints_vla/fase4_hf")
    args = ap.parse_args()

    exportar(ckpt_vla=args.ckpt, base_hf=args.base, saida_hf=args.saida)
