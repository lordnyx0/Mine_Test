# coding=utf-8
"""
politica_fase4_pulo.py — Política Visuomotora com Controle Neural de Yaw e Pulo (18 Ações).

Espaço de Ações Conjunto (18 Bins Discretos):
  - Bins 0 a 8:   Hold ['W'], Mouse [dx, 0]           (Andar + 9 rotações de Yaw)
  - Bins 9 a 17:  Hold ['W', 'SPACE'], Mouse [dx, 0]  (Andar + Pular + 9 rotações de Yaw)

Propriocepção no Vetor de Estado (32 dims):
  - sv[8]  = on_ground (1.0 se no chão)
  - sv[9]  = is_collided_horizontally (1.0 se colidindo com degrau/bloco na frente)
  - sv[16] = estagio_ativo (0.0 ou 1.0 na cadeia lógica)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple

YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)


class CabecaAcaoPulo(nn.Module):
    """Projeta o estado latente multimodal do Qwen3Loop para 18 classes de ação (Yaw + Pulo)."""
    def __init__(self, dim_entrada: int = 896, num_acoes: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_entrada, 256),
            nn.SiLU(),
            nn.Linear(256, num_acoes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoliticaFase4Pulo:
    """Orquestrador da Política com 18 Ações e Triplo Loop."""
    def __init__(self, vla_model, device="cuda"):
        self.vla = vla_model
        self.device = device
        self.YB = YAW_BINS
        self.ultimo = {}
        
        # Garante cabeça de 18 classes
        if not hasattr(self.vla, "cabeca_pulo") or self.vla.cabeca_pulo is None:
            self.cabeca_acao = CabecaAcaoPulo(dim_entrada=896, num_acoes=18).to(device)
            # Inicializa pesos copiando da cabeça anterior se existir
            if hasattr(self.vla, "cabeca_yaw") and self.vla.cabeca_yaw is not None:
                with torch.no_grad():
                    w_old = self.vla.cabeca_yaw.net[-1].weight.data # (9, 256)
                    b_old = self.vla.cabeca_yaw.net[-1].bias.data   # (9)
                    self.cabeca_acao.net[-1].weight.data[:9] = w_old
                    self.cabeca_acao.net[-1].weight.data[9:] = w_old
                    self.cabeca_acao.net[-1].bias.data[:9] = b_old
                    self.cabeca_acao.net[-1].bias.data[9:] = b_old - 0.5 # Leve viés inicial para correr no plano
        else:
            self.cabeca_acao = self.vla.cabeca_pulo.to(device)

    def agir(self, estados: List[Dict], frames: List[Any], prompts: List[str] = None) -> List[Dict]:
        """Gera ações discretas (Yaw + Pulo) a partir da percepção multimodal."""
        N = len(estados)
        u8 = self._processar_frames(frames)
        sv = self._extrair_propriocepcao(estados)
        
        px = self.normalizar(u8)
        svt = torch.tensor(sv, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Passa pelo Vision Encoder + Qwen3Loop (56 camadas efetivas, 3 loops)
                h_latente = self.vla.extrair_latente(px, svt, prompts=prompts)
                logits = self.cabeca_acao(h_latente)
                probs = F.softmax(logits.float(), dim=-1)
                dist = torch.distributions.Categorical(probs)
                acoes_idx = dist.sample()
                log_p = dist.log_prob(acoes_idx)

        idx_np = acoes_idx.cpu().numpy()
        acoes_formatadas = []
        for a_idx in idx_np:
            if a_idx < 9:
                # Sem Pulo: Hold W + Giro de Yaw
                dx = int(self.YB[int(a_idx)])
                acoes_formatadas.append({"hold": ["W"], "mouse": [dx, 0], "duration_ms": 250})
            else:
                # Com Pulo Neural: Hold W + SPACE + Giro de Yaw
                dx = int(self.YB[int(a_idx - 9)])
                acoes_formatadas.append({"hold": ["W", "SPACE"], "mouse": [dx, 0], "duration_ms": 250})

        self.ultimo = {
            "u8": u8,
            "sv": sv,
            "acoes_idx": idx_np,
            "log_p": log_p.cpu().numpy()
        }
        return acoes_formatadas

    def log_prob(self, px, svt, acoes_idx, prompts=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcula log_prob e entropia para a atualização do PPO."""
        h_latente = self.vla.extrair_latente(px, svt, prompts=prompts)
        logits = self.cabeca_acao(h_latente)
        probs = F.softmax(logits.float(), dim=-1)
        dist = torch.distributions.Categorical(probs)
        lp = dist.log_prob(acoes_idx)
        ent = dist.entropy().mean()
        return lp, ent

    def _extrair_propriocepcao(self, estados: List[Dict]) -> np.ndarray:
        N = len(estados)
        sv = np.zeros((N, 32), dtype=np.float32)
        for i, e in enumerate(estados):
            sv[i, 0] = e.get("x", 0.0)
            sv[i, 1] = e.get("y", 0.0)
            sv[i, 2] = e.get("z", 0.0)
            sv[i, 3] = e.get("yaw", 0.0) / 180.0
            sv[i, 4] = e.get("pitch", 0.0) / 90.0
            sv[i, 5] = e.get("vel_x", 0.0)
            sv[i, 6] = e.get("vel_y", 0.0)
            sv[i, 7] = e.get("vel_z", 0.0)
            sv[i, 8] = 1.0 if e.get("on_ground", False) else 0.0
            sv[i, 9] = 1.0 if e.get("is_collided_horizontally", False) else 0.0
        return sv

    def _processar_frames(self, frames):
        # Placeholder para normalização de frame RGB 160x120
        if isinstance(frames, np.ndarray):
            return frames
        return np.zeros((len(frames), 120, 160, 3), dtype=np.uint8)

    def normalizar(self, u8):
        t = torch.tensor(u8, dtype=torch.float32, device=self.device) / 255.0
        return t.permute(0, 3, 1, 2)  # (B, 3, H, W)
