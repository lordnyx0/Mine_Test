# coding=utf-8
"""
Módulo de LoRA (Low-Rank Adaptation) para o Backbone Qwen3Loop.

Permite destravar a capacidade cognitiva das camadas intermediárias de loop
sem alterar os pesos originais pré-treinados, eliminando qualquer risco de
esquecimento catastrófico (catastrophic forgetting).

Estrutura:
  W_novo = W_congelado + (B @ A) * (alpha / r)
  - A: inicializado com distribuição normal N(0, 1/r)
  - B: inicializado com zeros (inicia com impacto nulo, sem choque de gradiente)
"""
import math
import torch
import torch.nn as nn


class CamadaLoRA(nn.Module):
    """Envolve uma camada linear existente com matrizes de baixo posto A e B."""
    def __init__(self, linear_original: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.linear_original = linear_original
        self.r = r
        self.alpha = alpha
        self.escala = alpha / r
        self.in_features = linear_original.in_features
        self.out_features = linear_original.out_features

        # Congela a camada linear original
        self.linear_original.weight.requires_grad = False
        if self.linear_original.bias is not None:
            self.linear_original.bias.requires_grad = False

        dev = linear_original.weight.device
        dt = linear_original.weight.dtype

        # Matrizes LoRA de baixo posto
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=dt))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.resetar_parametros()

    def resetar_parametros(self):
        # Inicialização padrão LoRA: A com He/Kaiming, B com zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resultado_base = self.linear_original(x)
        # Caminho paralelo do LoRA: (x @ A.T) @ B.T * escala
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.escala
        return resultado_base + delta.to(resultado_base.dtype)

    def mesclar(self):
        """Mescla os pesos LoRA diretamente na camada linear original (para inferência rápida)."""
        delta_w = (self.lora_B.to(self.linear_original.weight.device) @ self.lora_A.to(self.linear_original.weight.device)) * self.escala
        self.linear_original.weight.data += delta_w.to(self.linear_original.weight.dtype)


def aplicar_lora(modelo, r: int = 16, alpha: float = 32.0, modulos_alvo=("q_proj", "v_proj", "k_proj", "o_proj")):
    """Substitui as camadas lineares alvo pelas versões envolvidas com LoRA."""
    substituidos = 0
    for nome, modulo in modelo.named_modules():
        for nome_filho, filho in list(modulo.named_children()):
            if isinstance(filho, nn.Linear) and any(alvo in nome_filho for alvo in modulos_alvo):
                camada_lora = CamadaLoRA(filho, r=r, alpha=alpha)
                setattr(modulo, nome_filho, camada_lora)
                substituidos += 1

    print(f"[LoRA] {substituidos} camadas adaptadas com sucesso (rank={r}, alpha={alpha}).")
    return modelo


def obter_parametros_lora(modelo):
    """Devolve apenas os tensores treináveis gerados pelo LoRA."""
    return {n: p for n, p in modelo.named_parameters() if "lora_" in n and p.requires_grad}


def mesclar_todos_lora(modelo):
    """Mescla todas as camadas LoRA do modelo permanentemente."""
    mesclados = 0
    for modulo in modelo.modules():
        if isinstance(modulo, CamadaLoRA):
            modulo.mesclar()
            mesclados += 1
    print(f"[LoRA] {mesclados} camadas mescladas permanentemente nos pesos base.")


def descarregar_lora(modelo):
    """Mescla e desempacota CamadaLoRA de volta para nn.Linear puro (para exportação HF/GGUF)."""
    descarregados = 0
    for nome, modulo in modelo.named_modules():
        for nome_filho, filho in list(modulo.named_children()):
            if isinstance(filho, CamadaLoRA):
                filho.mesclar()
                setattr(modulo, nome_filho, filho.linear_original)
                descarregados += 1
    print(f"[LoRA] {descarregados} camadas LoRA desempacotadas de volta para nn.Linear puro.")
    return modelo
