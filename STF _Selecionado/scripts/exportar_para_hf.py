# coding=utf-8
"""
exportar_para_hf.py — Extrai e funde adaptadores LoRA no backbone Qwen3Loop e salva em formato HuggingFace.
"""
import os
import sys
import shutil
import json
import argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modelo.lora_vla import aplicar_lora, descarregar_lora
from qwen3loop import Qwen3LoopForCausalLM
from transformers import AutoTokenizer


def exportar_hf(
    ckpt_vla: str,
    base_hf: str,
    saida_hf: str
):
    print("=" * 80)
    print(" [EXPORTAÇÃO HUGGINGFACE] Fundindo adaptadores LoRA e gerando pasta HF")
    print(f"    Checkpoint VLA : {ckpt_vla}")
    print(f"    Base HF        : {base_hf}")
    print(f"    Destino HF     : {saida_hf}")
    print("=" * 80)

    # 1. Carrega modelo base
    print("[1/4] Carregando backbone base...")
    causal_lm = Qwen3LoopForCausalLM.from_pretrained(
        base_hf,
        dtype=torch.bfloat16,
        device_map="cpu"
    )

    # 2. Restaura e funde adaptadores LoRA
    if os.path.exists(ckpt_vla):
        print(f"[2/4] Carregando pesos do checkpoint {ckpt_vla}...")
        ckpt_data = torch.load(ckpt_vla, map_location="cpu")
        treinaveis = ckpt_data.get("treinaveis", {})
        
        tem_lora = any("lora_" in k for k in treinaveis.keys())
        if tem_lora:
            print(f"      Restaurando e aplicando LoRA ({len(treinaveis)} tensores)...")
            causal_lm.model = aplicar_lora(causal_lm.model, r=16, alpha=32.0)
            
            lora_dict = {}
            for k, v in treinaveis.items():
                if k.startswith("qwen_model."):
                    clean_k = k.replace("qwen_model.", "")
                    lora_dict[clean_k] = v
                elif "lora_" in k:
                    lora_dict[k] = v
            
            causal_lm.model.load_state_dict(lora_dict, strict=False)
            
            print("[3/4] Fundindo pesos LoRA no backbone base (merge and unload)...")
            causal_lm.model = descarregar_lora(causal_lm.model)
            print("      Fusão LoRA concluída com sucesso!")
        else:
            print("[2/4] Checkpoint já consolidado (sem LoRA pendente).")

    if causal_lm.config.tie_word_embeddings:
        causal_lm.lm_head.weight = causal_lm.model.embed_tokens.weight

    # 3. Salva os arquivos no formato HuggingFace
    print(f"[4/4] Salvando pasta Hugging Face em {saida_hf}...")
    if os.path.exists(saida_hf):
        shutil.rmtree(saida_hf)
    os.makedirs(saida_hf, exist_ok=True)

    causal_lm.save_pretrained(saida_hf, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_hf).save_pretrained(saida_hf)

    for extra in ["chat_template.jinja", "generation_config.json"]:
        src_f = os.path.join(base_hf, extra)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(saida_hf, extra))

    cfg_file = os.path.join(saida_hf, "config.json")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    cfg_data["model_type"] = "qwen3loop"
    cfg_data["architectures"] = ["Qwen3LoopForCausalLM"]
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    print(f"[OK] Modelo HF exportado com sucesso para: {saida_hf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints_vla/vla_fase6_cot_melhor.pt", help="Caminho do checkpoint .pt")
    parser.add_argument("--base", default="STF _Selecionado/modelos/backbone_base_fp16", help="Pasta do backbone base HF")
    parser.add_argument("--saida", default="STF _Selecionado/modelos/modelo_hf_exportado", help="Pasta de saída HF")
    args = parser.parse_args()

    exportar_hf(ckpt_vla=args.ckpt, base_hf=args.base, saida_hf=args.saida)
