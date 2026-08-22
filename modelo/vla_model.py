# coding=utf-8
"""
Qwen3Loop Vision-Language-Action (VLA) Model Architecture.
Real Multimodal Tensor Integration:
SigLIP (frozen) -> Perceiver Resampler (32 tokens) -> Visual Projector -> Qwen3Loop (PyTorch) -> Action Heads.
"""
import sys
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from qwen3loop.modeling_qwen3loop import Qwen3LoopModel, Qwen3LoopForCausalLM
from qwen3loop.configuration_qwen3loop import Qwen3LoopConfig

class PerceiverResampler(nn.Module):
    """Resamples N visual tokens down to num_latents (32 tokens) via learned cross-attention."""
    def __init__(self, vision_dim: int, num_latents: int = 32, num_heads: int = 8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, vision_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)
        self.norm_latents = nn.LayerNorm(vision_dim)
        self.norm_features = nn.LayerNorm(vision_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: [B, N, vision_dim]
        B = visual_features.size(0)
        latents = self.latents.repeat(B, 1, 1)
        latents_norm = self.norm_latents(latents)
        feats_norm = self.norm_features(visual_features)
        attn_out, _ = self.attn(query=latents_norm, key=feats_norm, value=feats_norm)
        return latents + attn_out  # [B, 32, vision_dim]


class VisualProjector(nn.Module):
    """Projects visual features (vision_dim) to qwen_hidden_size (obtained dynamically, e.g. 1024)."""
    def __init__(self, vision_dim: int, qwen_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, qwen_hidden_size),
            nn.GELU(),
            nn.Linear(qwen_hidden_size, qwen_hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # [B, 32, qwen_hidden_size]


STATE_DIM = 32   # 16 originais + 16 de historico temporal (ver exploration_env)


class StateEncoder(nn.Module):
    """Encodes structured game state vector into 4 state tokens [B, 4, qwen_hidden_size]."""
    def __init__(self, qwen_hidden_size: int, num_state_tokens: int = 4,
                 state_dim: int = STATE_DIM):
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.qwen_hidden_size = qwen_hidden_size
        self.state_dim = state_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, qwen_hidden_size),
            nn.GELU(),
            nn.Linear(qwen_hidden_size, num_state_tokens * qwen_hidden_size)
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        # state_vec: [B, state_dim]
        B = state_vec.size(0)
        if state_vec.size(1) < self.state_dim:      # compat com vetores antigos de 16
            state_vec = F.pad(state_vec, (0, self.state_dim - state_vec.size(1)))
        elif state_vec.size(1) > self.state_dim:
            state_vec = state_vec[:, :self.state_dim]
        out = self.mlp(state_vec)  # [B, 4 * hidden_size]
        return out.view(B, self.num_state_tokens, self.qwen_hidden_size)


class GoalEncoder(nn.Module):
    """Objetivo RELATIVO (4 dims) -> tokens que entram junto com visao e estado.

    Por que separado do StateEncoder: o objetivo e a unica entrada que muda de
    episodio para episodio sem o mundo mudar, e e ele que remove o otimo
    degenerado. Com a tarefa antiga ("afaste-se do respawn") a acao certa era
    quase sempre W, entao ignorar os pixels era quase otimo — foi assim que a
    via visual colapsou para posto efetivo 1,4 de 1024. Com um alvo sorteado a
    cada episodio, W-sempre para de pontuar e a imagem volta a ser necessaria.

    Por que RELATIVO e egocentrico (frente/lado, nao x/z do mundo): coordenada
    de mundo nao transfere entre episodios nem entre jogos. "8 blocos a frente
    e 3 a esquerda" transfere.
    """
    def __init__(self, qwen_hidden_size: int, num_goal_tokens: int = 2,
                 goal_dim: int = 4):
        super().__init__()
        self.num_goal_tokens = num_goal_tokens
        self.qwen_hidden_size = qwen_hidden_size
        self.goal_dim = goal_dim
        self.mlp = nn.Sequential(
            nn.Linear(goal_dim, qwen_hidden_size),
            nn.GELU(),
            nn.Linear(qwen_hidden_size, num_goal_tokens * qwen_hidden_size)
        )

    def forward(self, goal_vec: torch.Tensor) -> torch.Tensor:
        B = goal_vec.size(0)
        return self.mlp(goal_vec).view(B, self.num_goal_tokens, self.qwen_hidden_size)


class IndependentActionHeads(nn.Module):
    """
    Independent Action Heads for Minecraft control:
    - movement_buttons: 8 independent Sigmoid heads (forward, backward, left, right, jump, attack, use, sprint)
    - hotbar: Softmax(9)
    - camera: continuous yaw and pitch regression
    """
    # Giro de câmera DISCRETO. A regressão com MSE contra alvos ruidosos de
    # média ~0 converge para 0: o agente ficava matematicamente incapaz de
    # girar a cabeça. Classificação sobre bins não tem esse colapso.
    # Em UNIDADES DE MOUSE. O servidor aplica yaw += dx*0.003 radianos, entao
    # 60 unidades sao apenas 10.3 graus — com os bins antigos o giro maximo
    # possivel era 10 graus por passo, e virar 90 graus levava 9 passos. Isto
    # travava qualquer professor que apontasse para um alvo lateral.
    # Correspondem a -45, -20, -10, -3, 0, 3, 10, 20, 45 graus.
    YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)
    PITCH_BINS = (-58, 0, 58)        # -10, 0, +10 graus

    def __init__(self, qwen_hidden_size: int, num_rotas: int = 12):
        super().__init__()
        # 8 independent binary/sigmoid keys allowing simultaneous presses (W+LCLICK, etc.)
        self.button_keys = ["forward", "backward", "left", "right", "jump", "attack", "use", "sprint"]
        self.buttons_head = nn.Linear(qwen_hidden_size, 8)
        self.hotbar_head = nn.Linear(qwen_hidden_size, 9)
        self.camera_head = nn.Linear(qwen_hidden_size, 2)  # legado (regressão)
        self.yaw_head = nn.Linear(qwen_hidden_size, len(self.YAW_BINS))
        self.pitch_head = nn.Linear(qwen_hidden_size, len(self.PITCH_BINS))

        # PREVISÃO DE ROTA: navegabilidade em num_rotas setores ao redor.
        # Alvo denso e dependente da cena, vindo do /rotas do servidor. É o que
        # obriga a via visual a continuar informativa — treinar só com "sempre
        # aperte W" fez o projetor colapsar para um vetor constante.
        self.num_rotas = num_rotas
        self.route_head = nn.Linear(qwen_hidden_size, num_rotas)

        # Prior de "nao girar". Uma cabeca aleatoria tem argmax CONSTANTE: na
        # politica deterministica isso vira o mesmo giro todo passo, e o agente
        # anda em circulo sem nunca se afastar da origem. Zerar os pesos e
        # enviesar o bin zero faz a politica comecar reta e APRENDER a girar.
        for head, bins in ((self.yaw_head, self.YAW_BINS),
                           (self.pitch_head, self.PITCH_BINS)):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[bins.index(0)] = 2.0

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hidden_state: [B, qwen_hidden_size]
        button_logits = self.buttons_head(hidden_state)                # [B, 8]
        button_probs = torch.sigmoid(button_logits)
        hotbar_logits = self.hotbar_head(hidden_state)                 # [B, 9]
        camera_reg = self.camera_head(hidden_state)                    # [B, 2]

        yaw_logits = self.yaw_head(hidden_state)                       # [B, 9]
        pitch_logits = self.pitch_head(hidden_state)                   # [B, 3]
        rotas = torch.sigmoid(self.route_head(hidden_state))           # [B, K] em [0,1]

        return {
            # Os logits crus sao necessarios para treinar sob autocast:
            # BCELoss(sigmoid) e proibido em mixed precision, so
            # BCEWithLogitsLoss e numericamente seguro.
            "buttons_logits": button_logits,
            "buttons": button_probs,
            "hotbar_logits": hotbar_logits,
            "hotbar_probs": F.softmax(hotbar_logits, dim=-1),
            "camera": camera_reg,
            "yaw_logits": yaw_logits,
            "pitch_logits": pitch_logits,
            "rotas_logits": self.route_head(hidden_state),
            "rotas": rotas,
        }


class Qwen3LoopVLA(nn.Module):
    """
    Complete Qwen3Loop Multimodal VLA Model Architecture.
    Combines:
      - SigLIP Vision Encoder (frozen)
      - Perceiver Resampler (N -> 32 tokens, trainable)
      - Visual Projector (vision_dim -> qwen_hidden_size=1024, trainable)
      - State Encoder (16 -> 4 tokens, trainable)
      - Qwen3Loop 0.6B PyTorch Backbone (28 layers, hidden_size=1024, frozen or LoRA)
      - Independent Action Heads (trainable)
    """
    def __init__(
        self,
        qwen_model: Qwen3LoopModel,
        vision_model_name: str = "google/siglip-base-patch16-224",
        num_visual_tokens: int = 32,
        freeze_vision: bool = True,
        freeze_qwen: bool = True
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.qwen_config = qwen_model.config
        self.hidden_size = getattr(self.qwen_config, "hidden_size", 1024)

        # Vision Encoder (SigLIP / CLIP)
        from transformers import AutoModel, AutoProcessor
        self.vision_processor = AutoProcessor.from_pretrained(vision_model_name)
        full_vision_model = AutoModel.from_pretrained(vision_model_name)
        # Extract vision model component if full VLM/CLIP model loaded
        if hasattr(full_vision_model, "vision_model"):
            self.vision_encoder = full_vision_model.vision_model
        else:
            self.vision_encoder = full_vision_model
            
        self.vision_dim = getattr(self.vision_encoder.config, "hidden_size", 768)

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        if freeze_qwen:
            for p in self.qwen_model.parameters():
                p.requires_grad = False

        # Resampler & Projector
        self.resampler = PerceiverResampler(vision_dim=self.vision_dim, num_latents=num_visual_tokens)
        self.projector = VisualProjector(vision_dim=self.vision_dim, qwen_hidden_size=self.hidden_size)

        # State Encoder & Action Heads
        self.state_encoder = StateEncoder(qwen_hidden_size=self.hidden_size, num_state_tokens=4)
        self.goal_encoder = GoalEncoder(qwen_hidden_size=self.hidden_size, num_goal_tokens=2)
        self.action_heads = IndependentActionHeads(qwen_hidden_size=self.hidden_size)

        # Embedding temporal por posicao da pilha de frames. Sem isto os frames
        # entram no Qwen sem ordem e "agora" fica indistinguivel de "15s atras",
        # que e justamente a informacao que da nocao de movimento e de travado.
        self.max_frames = 4
        self.frame_time_embed = nn.Parameter(
            torch.randn(self.max_frames, self.hidden_size) * 0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        state_vec: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        goal_vec: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # pixel_values aceita [B, 3, H, W] (1 frame) ou [B, K, 3, H, W] (pilha
        # temporal de K frames, do mais recente para o mais antigo).
        if pixel_values.dim() == 5:
            B, K = pixel_values.size(0), pixel_values.size(1)
            pixel_values = pixel_values.flatten(0, 1)     # [B*K, 3, H, W]
        else:
            B, K = pixel_values.size(0), 1
        device = pixel_values.device

        # 1. Vision Encoder forward (no grad if frozen)
        if not self.vision_encoder.training:
            with torch.no_grad():
                v_out = self.vision_encoder(pixel_values=pixel_values)
                v_feats = v_out.last_hidden_state  # [B*K, N, vision_dim]
        else:
            v_out = self.vision_encoder(pixel_values=pixel_values)
            v_feats = v_out.last_hidden_state

        # 2. Resampler + Projector -> [B*K, 32, hidden_size]
        v_resample = self.resampler(v_feats)
        v_embeds = self.projector(v_resample)

        if K > 1:
            # Marca cada frame com sua posicao no tempo e concatena: [B, K*32, H]
            v_embeds = v_embeds.view(B, K, -1, self.hidden_size)
            v_embeds = v_embeds + self.frame_time_embed[:K].to(v_embeds.dtype).view(1, K, 1, -1)
            v_embeds = v_embeds.flatten(1, 2)

        embeds_list = [v_embeds]

        # 3. State Encoder -> [B, 4, hidden_size]
        if state_vec is None:
            state_vec = torch.zeros(B, STATE_DIM, device=device)
        s_embeds = self.state_encoder(state_vec)
        embeds_list.append(s_embeds)

        # 3b. Objetivo relativo, se houver. OPCIONAL de proposito: sem ele o
        # modelo se comporta exatamente como antes, entao os scripts e
        # checkpoints anteriores continuam validos sem mudanca.
        if goal_vec is not None:
            embeds_list.append(self.goal_encoder(goal_vec))

        # 4. Optional Text Embeddings
        if input_ids is not None:
            t_embeds = self.qwen_model.get_input_embeddings()(input_ids)
            embeds_list.append(t_embeds)

        # Concatenate tensor embeddings: [B, 32 + 4 (+ T), hidden_size]
        inputs_embeds = torch.cat(embeds_list, dim=1)

        # 5. Forward through Qwen3Loop (use_cache=False during training/forward)
        qwen_out = self.qwen_model(inputs_embeds=inputs_embeds, use_cache=False)
        last_hidden = qwen_out.last_hidden_state[:, -1, :]  # [B, hidden_size]

        # 6. Independent Action Heads
        actions = self.action_heads(last_hidden)
        return actions

if __name__ == "__main__":
    print("[VLA Module] Test import successful.")
