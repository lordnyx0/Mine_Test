# coding=utf-8
"""
Utilitário para verificar a integridade e a soma dos pesos de checkpoints do VLA.

Prescrito por docs/metodo.md §13:
  "Se dois processos discordam sobre o mesmo checkpoint, compare a soma dos
   pesos antes de investigar o laço."

Uso:
  python infra/verificar_pesos.py --ckpt checkpoints_vla/vla_fase1.pt
  python infra/verificar_pesos.py --ckpt1 checkpoints_vla/vla_fase1.pt --ckpt2 checkpoints_vla/vla_fase2.pt
"""
import os
import sys
import argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from infra.run_vla_agent import load_vla_agent


def obter_sumario_pesos(caminho_ckpt=None):
    """Carrega o modelo com o checkpoint e devolve estatísticas detalhadas de tensores."""
    vla, device = load_vla_agent(caminho_ckpt)
    
    total_params = sum(p.numel() for p in vla.parameters())
    treinaveis_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    soma_total = sum(p.detach().cpu().double().sum().item() for p in vla.parameters())
    soma_treinaveis = sum(p.detach().cpu().double().sum().item() for p in vla.parameters() if p.requires_grad)
    
    submodulos = {
        "resampler": vla.resampler,
        "projector": vla.projector,
        "state_encoder": vla.state_encoder,
        "goal_encoder": getattr(vla, "goal_encoder", None),
        "action_heads": vla.action_heads,
        "vision_encoder": vla.vision_encoder,
        "qwen_model": vla.qwen_model,
    }
    
    somas_modulos = {}
    for nome, mod in submodulos.items():
        if mod is not None:
            soma = sum(p.detach().cpu().double().sum().item() for p in mod.parameters())
            qtd = sum(p.numel() for p in mod.parameters())
            somas_modulos[nome] = {"soma": soma, "numel": qtd}
            
    return {
        "checkpoint": caminho_ckpt or "BASE",
        "total_params": total_params,
        "treinaveis_params": treinaveis_params,
        "soma_total": soma_total,
        "soma_treinaveis": soma_treinaveis,
        "modulos": somas_modulos,
    }


def main():
    ap = argparse.ArgumentParser(description="Verificador de integridade de pesos do VLA")
    ap.add_argument("--ckpt", default=None, help="Caminho do checkpoint a inspecionar")
    ap.add_argument("--ckpt1", default=None, help="Primeiro checkpoint para comparação")
    ap.add_argument("--ckpt2", default=None, help="Segundo checkpoint para comparação")
    args = ap.parse_args()

    if args.ckpt1 and args.ckpt2:
        print(f"\n[Verificação Pareada] Comparando:\n  (1) {args.ckpt1}\n  (2) {args.ckpt2}\n")
        s1 = obter_sumario_pesos(args.ckpt1)
        s2 = obter_sumario_pesos(args.ckpt2)
        
        print(f"{'Módulo':<18} | {'Soma (Ckpt 1)':<20} | {'Soma (Ckpt 2)':<20} | {'Diff':<12}")
        print("-" * 76)
        for k in s1["modulos"]:
            soma1 = s1["modulos"][k]["soma"]
            soma2 = s2["modulos"][k]["soma"]
            diff = soma2 - soma1
            print(f"{k:<18} | {soma1:<20.6f} | {soma2:<20.6f} | {diff:<+12.6f}")
        print("-" * 76)
        print(f"{'TREINÁVEIS':<18} | {s1['soma_treinaveis']:<20.6f} | {s2['soma_treinaveis']:<20.6f} | {s2['soma_treinaveis'] - s1['soma_treinaveis']:<+12.6f}")
        print(f"{'TOTAL':<18} | {s1['soma_total']:<20.6f} | {s2['soma_total']:<20.6f} | {s2['soma_total'] - s1['soma_total']:<+12.6f}\n")
    else:
        s = obter_sumario_pesos(args.ckpt)
        print(f"\n[Sumário de Pesos] Checkpoint: {s['checkpoint']}")
        print(f"Total de parâmetros: {s['total_params']/1e6:.2f}M | Treináveis: {s['treinaveis_params']/1e6:.2f}M")
        print(f"Soma total: {s['soma_total']:.6f} | Soma treináveis: {s['soma_treinaveis']:.6f}\n")
        print(f"{'Módulo':<18} | {'Parâmetros':<12} | {'Soma dos Pesos':<20}")
        print("-" * 56)
        for k, v in s["modulos"].items():
            print(f"{k:<18} | {v['numel']/1e6:>8.2f}M  | {v['soma']:<20.6f}")
        print("-" * 56 + "\n")


if __name__ == "__main__":
    main()
