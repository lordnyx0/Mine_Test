# coding=utf-8
"""
fase6/executar_benchmark_q8_fase6.py — Pipeline Oficial de Avaliação de Raciocínio em GGUF True Loop Q8_0 (Fase 6 CoT-VLA).

Executa:
  1. Fusão dos pesos LoRA da Fase 6 (checkpoints_vla/vla_fase6_ppo_cot.pt) no backbone Qwen3Loop.
  2. Conversão e quantização para GGUF True Loop Q8_0 (models_gguf/fase6_loop_q8_0.gguf).
  3. Inicialização do llama-server.exe compilado com True Loop (LoopSplit).
  4. Execução do Benchmark Oficial de 97 Itens com Thinking Ativado.
"""
from __future__ import annotations
import os
import sys
import shutil
import json
import subprocess
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ckpt_vla = os.path.join(_ROOT, "checkpoints_vla", "vla_fase6_ppo_cot.pt")
base_hf = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
saida_hf = os.path.join(_ROOT, "checkpoints_vla", "fase6_hf")
saida_gguf_f16 = os.path.join(_ROOT, "models_gguf", "fase6_loop_f16.gguf")
saida_gguf_q8 = os.path.join(_ROOT, "models_gguf", "fase6_loop_q8_0.gguf")
convert_script = r"C:\Users\Nyx\Desktop\Testes\Testes2\llamacpp_patch\mod\convert_hf_to_gguf.py"
llama_quantize = r"C:\Users\Nyx\.unsloth\.staging\llama.cpp.staging-eu_6bjrp\build-qwen3loop\bin\Release\llama-quantize.exe"

print("=" * 80)
print(" BENCHMARK OFICIAL DE RACIOCÍNIO GGUF Q8_0 — FASE 6 CoT-VLA")
print("=" * 80)
print(f"  Checkpoint VLA : {ckpt_vla}")
print(f"  Pasta HF Fusão : {saida_hf}")
print(f"  Modelo GGUF Q8 : {saida_gguf_q8}")
print("=" * 80, flush=True)

# 1. Exporta e funde pesos LoRA no formato Hugging Face
print("\n[1/3] Exportando e fundindo pesos LoRA no formato Hugging Face...")
from qwen3loop import Qwen3LoopForCausalLM
from transformers import AutoTokenizer
from modelo.lora_vla import aplicar_lora, descarregar_lora

causal_lm = Qwen3LoopForCausalLM.from_pretrained(
    base_hf,
    dtype=torch.bfloat16,
    device_map="cpu"
)

if os.path.exists(ckpt_vla):
    ckpt_data = torch.load(ckpt_vla, map_location="cpu")
    treinaveis = ckpt_data.get("treinaveis", {})
    if any("lora_" in k for k in treinaveis.keys()):
        print(f"      Restaurando {len(treinaveis)} tensores LoRA...")
        causal_lm.model = aplicar_lora(causal_lm.model, r=16, alpha=32.0)
        
        lora_dict = {}
        for k, v in treinaveis.items():
            if k.startswith("qwen_model."):
                clean_k = k.replace("qwen_model.", "")
                lora_dict[clean_k] = v
            elif "lora_" in k:
                lora_dict[k] = v
        
        causal_lm.model.load_state_dict(lora_dict, strict=False)
        causal_lm.model = descarregar_lora(causal_lm.model)
        print("      Fusão LoRA concluída com sucesso!")

if causal_lm.config.tie_word_embeddings:
    causal_lm.lm_head.weight = causal_lm.model.embed_tokens.weight

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

print(f"[OK] Modelo HF fundido salvo em: {saida_hf}")

# 2. Converte para GGUF True Loop Q8_0
print("\n[2/3] Convertendo para GGUF True Loop Q8_0...")
os.makedirs(os.path.dirname(saida_gguf_q8), exist_ok=True)

if not os.path.exists(convert_script):
    convert_script = r"C:\Users\Nyx\.unsloth\.staging\llama.cpp.staging-eu_6bjrp\convert_hf_to_gguf.py"

cmd_conv = [
    sys.executable,
    convert_script,
    saida_hf,
    "--outfile", saida_gguf_q8,
    "--outtype", "q8_0"
]
print(f"      Executando conversão: {' '.join(cmd_conv)}")
res = subprocess.run(cmd_conv, capture_output=True, text=True)

if res.returncode != 0:
    print("      Tentando conversão intermediária F16 -> llama-quantize Q8_0...")
    cmd_f16 = [sys.executable, convert_script, saida_hf, "--outfile", saida_gguf_f16, "--outtype", "f16"]
    subprocess.run(cmd_f16, check=True)
    cmd_q8 = [llama_quantize, saida_gguf_f16, saida_gguf_q8, "q8_0"]
    subprocess.run(cmd_q8, check=True)

if os.path.exists(saida_gguf_q8):
    tam_mb = os.path.getsize(saida_gguf_q8) / (1024 * 1024)
    print(f"[OK] Modelo GGUF Q8_0 True Loop gerado com sucesso: {tam_mb:.2f} MiB")

# 3. Executa o Benchmark GGUF Oficial
print("\n[3/3] Executando Benchmark Oficial de Raciocínio (97 Itens)...")
cmd_bench = [
    sys.executable,
    os.path.join(_ROOT, "avaliacao", "bench_gguf.py"),
    "--modelo", saida_gguf_q8,
    "--nome", "fase6_loop_q8_pensamento"
]
subprocess.run(cmd_bench, check=True)
print("\n" + "=" * 80)
print(" [BENCHMARK DE RACIOCÍNIO Q8 CONCLUÍDO COM SUCESSO!]")
print("=" * 80)
