# coding=utf-8
"""
Perda Auxiliar de Modelo de Mundo (World Model Loss) para o Qwen3Loop.

Força as camadas intermediárias de loop do Transformer a aprenderem física,
causalidade temporal e dinâmica 3D do mundo através da predição de latentes futuros:
  L_world = MSE(z_previsto_{t+1}, z_real_{t+1}) + (1 - CosineSim(z_previsto, z_real))

Isso desenvolve uma representação interna espacial que transfere diretamente
para qualquer outro jogo ou motor 3D (Unity, Unreal, Godot).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreditorDinamicaLatente(nn.Module):
    """Módulo auxiliar que projeta o estado oculto do transformer + ação no próximo latente visual."""
    def __init__(self, dim_oculta: int = 896, dim_acao: int = 9, dim_latente: int = 896):
        super().__init__()
        self.proj_acao = nn.Linear(dim_acao, 64)
        self.rede = nn.Sequential(
            nn.Linear(dim_oculta + 64, dim_oculta),
            nn.SiLU(),
            nn.Linear(dim_oculta, dim_latente)
        )

    def forward(self, h_t: torch.Tensor, a_t_onehot: torch.Tensor) -> torch.Tensor:
        emb_acao = self.proj_acao(a_t_onehot)
        entrada = torch.cat([h_t, emb_acao], dim=-1)
        return self.rede(entrada)


def perda_modelo_mundo(z_previsto: torch.Tensor, z_alvo: torch.Tensor, peso_cosseno: float = 0.5) -> torch.Tensor:
    """Calcula a perda combinada de MSE e Similaridade de Cosseno entre os vetores latentes."""
    perda_mse = F.mse_loss(z_previsto, z_alvo)
    perda_cosseno = 1.0 - F.cosine_similarity(z_previsto, z_alvo, dim=-1).mean()
    return perda_mse + peso_cosseno * perda_cosseno
