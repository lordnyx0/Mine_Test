# coding=utf-8
"""
converter_e_quantizar.py — Pipeline completo: Checkpoint PyTorch/LoRA -> HuggingFace -> GGUF FP16 -> GGUF Q8_0.
"""
import os
import sys
import subprocess
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LLAMA_QUANTIZE_EXE = r"C:\Users\Nyx\.unsloth\.staging\llama.cpp.staging-eu_6bjrp\build-qwen3loop\bin\Release\llama-quantize.exe"


def executar_comando(cmd, descricao=""):
    print(f"\n[*] Executando: {descricao}...")
    print(f"    CMD: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERRO] Falha na etapa: {descricao} (código {res.returncode})")
        return False
    print(f"[OK] {descricao} concluído com sucesso!")
    return True


def pipeline_conversao_gguf(
    ckpt_pt: str = "checkpoints_vla/vla_fase6_cot_melhor.pt",
    base_hf: str = "STF _Selecionado/modelos/backbone_base_fp16",
    pasta_hf_saida: str = "STF _Selecionado/modelos/modelo_hf_exportado",
    gguf_f16_saida: str = "STF _Selecionado/modelos/modelo_f16.gguf",
    gguf_q8_saida: str = "STF _Selecionado/modelos/modelo_q8_0.gguf"
):
    print("=" * 80)
    print(" 🚀 PIPELINE COMPLETO DE CONVERSÃO E QUANTIZAÇÃO GGUF Q8_0")
    print("=" * 80)

    # 1. Exportar e fundir LoRA para formato HF
    from STF_Selecionado.scripts.exportar_para_hf import exportar_hf
    exportar_hf(ckpt_vla=ckpt_pt, base_hf=base_hf, saida_hf=pasta_hf_saida)

    # 2. Converter pasta HF para GGUF FP16
    convert_script = os.path.join(_ROOT, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Tenta no site-packages ou pasta gguf
        cmd_convert = f"python -m gguf.convert_hf_to_gguf \"{pasta_hf_saida}\" --outfile \"{gguf_f16_saida}\" --outtype f16"
    else:
        cmd_convert = f"python \"{convert_script}\" \"{pasta_hf_saida}\" --outfile \"{gguf_f16_saida}\" --outtype f16"

    sucesso = executar_comando(cmd_convert, "Conversão HuggingFace para GGUF FP16")
    if not sucesso:
        print("[*] Tentando método alternativo via convert_hf_to_gguf...")
        executar_comando(f"python -m gguf.convert_hf_to_gguf \"{pasta_hf_saida}\" --outfile \"{gguf_f16_saida}\"", "Conversão Fallback GGUF")

    # 3. Quantizar GGUF FP16 para Q8_0
    if os.path.exists(LLAMA_QUANTIZE_EXE) and os.path.exists(gguf_f16_saida):
        cmd_quant = f"\"{LLAMA_QUANTIZE_EXE}\" \"{gguf_f16_saida}\" \"{gguf_q8_saida}\" q8_0"
        executar_comando(cmd_quant, "Quantização GGUF para Q8_0")
    else:
        print(f"[*] Aviso: Executável llama-quantize não encontrado em {LLAMA_QUANTIZE_EXE}. O modelo FP16 está pronto.")

    print("\n" + "=" * 80)
    print(" [STATUS FINAL DA CONVERSÃO]")
    if os.path.exists(gguf_q8_saida):
        print(f"  ✅ Modelo GGUF Q8_0 Gerado com Sucesso: {gguf_q8_saida}")
    elif os.path.exists(gguf_f16_saida):
        print(f"  ✅ Modelo GGUF FP16 Gerado: {gguf_f16_saida}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints_vla/vla_fase6_cot_melhor.pt")
    parser.add_argument("--base", default="STF _Selecionado/modelos/backbone_base_fp16")
    parser.add_argument("--saida_hf", default="STF _Selecionado/modelos/modelo_hf_exportado")
    parser.add_argument("--saida_f16", default="STF _Selecionado/modelos/modelo_f16.gguf")
    parser.add_argument("--saida_q8", default="STF _Selecionado/modelos/modelo_q8_0.gguf")
    args = parser.parse_args()

    pipeline_conversao_gguf(
        ckpt_pt=args.ckpt,
        base_hf=args.base,
        pasta_hf_saida=args.saida_hf,
        gguf_f16_saida=args.saida_f16,
        gguf_q8_saida=args.saida_q8
    )
