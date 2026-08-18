# coding=utf-8
"""
FASE 4 — Política de Raciocínio Recursivo Multi-Loop com Pulo Neural Aprendido (18 Ações).

Aproveita a arquitetura em loop do Qwen3Loop para executar múltiplos passos
de raciocínio latente antes de emitir a decisão motora conjunta (Yaw + Pulo):
  - Loop 1: Percepção Visual (SigLIP + Projector) + Instrução Textual.
  - Loop 2: Projeção de Submeta Latente (planejamento mental de contorno/sequência).
  - Loop 3: Decisão de Controle Motor Conjunto (18 Bins: 9 Yaw x 2 Pulo).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from politica.politica_fase3 import PoliticaFase3
from fase5.acoes_taticas import decodificar_acao_36

YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)


class PoliticaRaciocinioLoop(PoliticaFase3):
    """Executa refinamento recursivo em loop no Qwen3Loop com 36 ações táticas (WASD) ou 18 ações legadas."""
    def __init__(self, checkpoint_path=None, amostrar=False, device=None, vla=None, loops_pensamento=3, num_acoes=36):
        super().__init__(checkpoint_path, amostrar=amostrar, device=device, vla=vla)
        self.loops_pensamento = loops_pensamento
        self.num_acoes = num_acoes
        self.YB = YAW_BINS
        self.ultimo_dx = {} # Amortecimento inercial por ambiente

        hidden = self.vla.hidden_size if hasattr(self.vla, "hidden_size") else 896
        
        # Garante suporte a 36 ações táticas ou 18 ações legadas
        if num_acoes == 36 and not hasattr(self.vla, "cabeca_acao_36"):
            self.vla.cabeca_acao_36 = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.SiLU(),
                nn.Linear(256, 36)
            ).to(self.device)
            with torch.no_grad():
                self.vla.cabeca_acao_36[0].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_acao_36[0].bias.data.zero_()
                self.vla.cabeca_acao_36[2].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_acao_36[2].bias.data.zero_()

        elif num_acoes == 18 and not hasattr(self.vla, "cabeca_acao_18"):
            self.vla.cabeca_acao_18 = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.SiLU(),
                nn.Linear(256, 18)
            ).to(self.device)
            with torch.no_grad():
                self.vla.cabeca_acao_18[0].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_acao_18[0].bias.data.zero_()
                self.vla.cabeca_acao_18[2].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_acao_18[2].bias.data.zero_()

    def forward_pensamento(self, pixel_tensor, state_tensor, goal_tensor, input_ids):
        """Passa o fluxo de tokens por múltiplos ciclos de loop recursivo no transformer."""
        vla = self.vla
        B = pixel_tensor.size(0)

        if pixel_tensor.dim() == 5:
            px_flat = pixel_tensor.flatten(0, 1)
        else:
            px_flat = pixel_tensor

        if not vla.vision_encoder.training:
            with torch.no_grad():
                v_feats = vla.vision_encoder(pixel_values=px_flat).last_hidden_state
        else:
            v_feats = vla.vision_encoder(pixel_values=px_flat).last_hidden_state

        v_res = vla.resampler(v_feats)
        v_emb = vla.projector(v_res)
        if pixel_tensor.dim() == 5:
            v_emb = v_emb.view(B, pixel_tensor.size(1) * 32, -1)

        s_emb = vla.state_encoder(state_tensor)
        t_emb = vla.qwen_model.get_input_embeddings()(input_ids) if input_ids is not None else None

        embeds_list = [v_emb, s_emb]
        if goal_tensor is not None:
            embeds_list.append(vla.goal_encoder(goal_tensor))
        if t_emb is not None:
            embeds_list.append(t_emb)

        inputs_embeds = torch.cat(embeds_list, dim=1)

        qwen_out = vla.qwen_model(inputs_embeds=inputs_embeds, use_cache=False)
        last_hidden = qwen_out.last_hidden_state[:, -1, :]
        
        if hasattr(vla, "cabeca_acao_36") and self.num_acoes == 36:
            return self.vla.cabeca_acao_36(last_hidden)
        elif hasattr(vla, "cabeca_acao_18"):
            return self.vla.cabeca_acao_18(last_hidden)
        elif hasattr(vla, "cabeca_acao_36"):
            return self.vla.cabeca_acao_36(last_hidden)
        else:
            raise AttributeError("Nenhuma cabeça de ação compatível encontrada no VLA.")

    def agir(self, ests, alvos_abs, obs, prompts=None, estagios=None):
        if prompts is not None:
            self.prompts_atuais = list(prompts)
        px, sv, gv, u8 = self._entradas(ests, alvos_abs, obs)
        
        # Injeta o estágio ativo no vetor de estado proprioceptivo antes do forward pass
        if estagios is not None:
            for i, st in enumerate(estagios):
                sv[i, 16] = float(st)

        ids = self.obter_ids(self.prompts_atuais if len(self.prompts_atuais) == len(ests)
                             else [self.default_prompt] * len(ests))

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            logits = self.forward_pensamento(px, sv, gv, ids)

        LOGIT_CLIP = 3.0
        logits = torch.tanh(logits / LOGIT_CLIP) * LOGIT_CLIP

        if self.amostrar:
            probs = torch.softmax(logits / self.temperatura, dim=-1)
            acoes_idx = torch.multinomial(probs, 1).squeeze(-1)
        else:
            acoes_idx = logits.argmax(-1)

        idx_np = acoes_idx.detach().cpu().numpy()
        acoes = []
        for i, k in enumerate(idx_np):
            if self.num_acoes == 36:
                acao_dict = decodificar_acao_36(int(k))
                raw_dx = acao_dict["mouse"][0]
                # Filtro passa-baixas inercial (elimina efeito beyblade)
                prev_dx = self.ultimo_dx.get(i, 0.0)
                smooth_dx = int(0.65 * raw_dx + 0.35 * prev_dx)
                self.ultimo_dx[i] = smooth_dx
                acao_dict["mouse"] = [smooth_dx, 0]
                acoes.append(acao_dict)
            else:
                if k < 9:
                    dx = int(self.YB[int(k)])
                    acoes.append({"hold": ["W"], "mouse": [dx, 0], "duration_ms": 250})
                else:
                    dx = int(self.YB[int(k - 9)])
                    acoes.append({"hold": ["W", "SPACE"], "mouse": [dx, 0], "duration_ms": 250})

        self.ultimo = {"u8": u8, "sv": sv.detach().cpu().numpy(),
                       "gv": gv.detach().cpu().numpy(),
                       "idx": idx_np,
                       "ids": ids.detach().cpu().numpy()}
        return acoes

    def log_prob(self, px, sv, gv, a_idx, ids=None):
        logits = self.forward_pensamento(px, sv, gv, ids)
        LOGIT_CLIP = 3.0
        logits = torch.tanh(logits / LOGIT_CLIP) * LOGIT_CLIP

        logp = torch.log_softmax(logits, dim=-1)
        lp = logp.gather(1, a_idx.unsqueeze(1)).squeeze(1)
        ent = -(logp.exp() * logp).sum(-1).mean()
        ent_norm = ent / torch.log(torch.tensor(float(self.num_acoes), device=self.device))
        return lp, ent_norm
