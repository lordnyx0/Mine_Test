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
from fase5.acoes_taticas import (
    decodificar_acao_36,
    decodificar_acao_fatorada,
    fatorar_indice_36,
    unificar_indices,
    NUM_MODOS,
    NUM_YAW,
    YAW_BINS_9
)

YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)


class PoliticaRaciocinioLoop(PoliticaFase3):
    """Executa refinamento recursivo em loop no Qwen3Loop com cabeças fatoradas (Modo 6 + Yaw 9 + Value) ou 36 ações legadas."""
    def __init__(self, checkpoint_path=None, amostrar=False, device=None, vla=None, loops_pensamento=3, num_acoes=36, fatorada=True):
        super().__init__(checkpoint_path, amostrar=amostrar, device=device, vla=vla)
        self.loops_pensamento = loops_pensamento
        self.num_acoes = num_acoes
        self.fatorada = fatorada
        self.YB = YAW_BINS
        self.ultimo_dx = {}  # Amortecimento inercial por ambiente

        hidden = self.vla.hidden_size if hasattr(self.vla, "hidden_size") else 896
        
        # 1. Cabeça de Valor (Critic V(s)) para PPO / GAE
        if not hasattr(self.vla, "cabeca_valor"):
            self.vla.cabeca_valor = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.SiLU(),
                nn.Linear(256, 1)
            ).to(self.device)
            with torch.no_grad():
                self.vla.cabeca_valor[0].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_valor[0].bias.data.zero_()
                self.vla.cabeca_valor[2].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_valor[2].bias.data.zero_()

        # 2. Cabeças Fatoradas (Modo: 6 classes, Yaw: 9 classes)
        if not hasattr(self.vla, "cabeca_modo"):
            self.vla.cabeca_modo = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.SiLU(),
                nn.Linear(256, NUM_MODOS)
            ).to(self.device)
            with torch.no_grad():
                self.vla.cabeca_modo[0].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_modo[0].bias.data.zero_()
                self.vla.cabeca_modo[2].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_modo[2].bias.data.zero_()

        if not hasattr(self.vla, "cabeca_yaw"):
            self.vla.cabeca_yaw = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.SiLU(),
                nn.Linear(256, NUM_YAW)
            ).to(self.device)
            with torch.no_grad():
                self.vla.cabeca_yaw[0].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_yaw[0].bias.data.zero_()
                self.vla.cabeca_yaw[2].weight.data.normal_(0.0, 0.02)
                self.vla.cabeca_yaw[2].bias.data.zero_()

        # 3. Suporte de fallback legado (36 ações unificadas)
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

    def forward_pensamento(self, pixel_tensor, state_tensor, goal_tensor, input_ids):
        """Passa o fluxo de tokens pelo transformer e extrai representações latentes."""
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
        
        LOGIT_CLIP = 3.0
        # Cabeça de Valor V(s)
        val = vla.cabeca_valor(last_hidden).squeeze(-1)  # [B]

        if self.fatorada and hasattr(vla, "cabeca_modo") and hasattr(vla, "cabeca_yaw"):
            lg_modo = vla.cabeca_modo(last_hidden)
            lg_yaw = vla.cabeca_yaw(last_hidden)
            lg_modo = torch.tanh(lg_modo / LOGIT_CLIP) * LOGIT_CLIP
            lg_yaw = torch.tanh(lg_yaw / LOGIT_CLIP) * LOGIT_CLIP
            return lg_modo, lg_yaw, val
        elif hasattr(vla, "cabeca_acao_36"):
            lg_36 = vla.cabeca_acao_36(last_hidden)
            lg_36 = torch.tanh(lg_36 / LOGIT_CLIP) * LOGIT_CLIP
            return lg_36, val
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
            res = self.forward_pensamento(px, sv, gv, ids)

        if self.fatorada:
            lg_modo, lg_yaw, values = res
            if self.amostrar:
                p_modo = torch.softmax(lg_modo / self.temperatura, dim=-1)
                p_yaw = torch.softmax(lg_yaw / self.temperatura, dim=-1)
                dist_modo = torch.distributions.Categorical(probs=p_modo)
                dist_yaw = torch.distributions.Categorical(probs=p_yaw)
                a_modo = dist_modo.sample()
                a_yaw = dist_yaw.sample()
                logp_old = dist_modo.log_prob(a_modo) + dist_yaw.log_prob(a_yaw)
            else:
                a_modo = lg_modo.argmax(-1)
                a_yaw = lg_yaw.argmax(-1)
                logp_old = torch.zeros_like(a_modo, dtype=torch.float32)

            modo_np = a_modo.detach().cpu().numpy()
            yaw_np = a_yaw.detach().cpu().numpy()
            val_np = values.detach().cpu().numpy()
            logp_np = logp_old.detach().cpu().numpy()
            
            idx_36_np = np.array([unificar_indices(m, y) for m, y in zip(modo_np, yaw_np)], dtype=np.int64)

            acoes = []
            for i in range(len(modo_np)):
                m, y = int(modo_np[i]), int(yaw_np[i])
                acao_dict = decodificar_acao_fatorada(m, y)
                raw_dx = acao_dict["mouse"][0]
                prev_dx = self.ultimo_dx.get(i, 0.0)
                smooth_dx = int(0.65 * raw_dx + 0.35 * prev_dx)
                self.ultimo_dx[i] = smooth_dx
                acao_dict["mouse"] = [smooth_dx, 0]
                acoes.append(acao_dict)

            self.ultimo = {
                "u8": u8,
                "sv": sv.detach().cpu().numpy(),
                "gv": gv.detach().cpu().numpy(),
                "idx": idx_36_np,
                "idx_modo": modo_np,
                "idx_yaw": yaw_np,
                "logp_old": logp_np,
                "val": val_np,
                "ids": ids.detach().cpu().numpy()
            }
        else:
            logits, values = res
            if self.amostrar:
                probs = torch.softmax(logits / self.temperatura, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                acoes_idx = dist.sample()
                logp_old = dist.log_prob(acoes_idx)
            else:
                acoes_idx = logits.argmax(-1)
                logp_old = torch.zeros_like(acoes_idx, dtype=torch.float32)

            idx_np = acoes_idx.detach().cpu().numpy()
            val_np = values.detach().cpu().numpy()
            logp_np = logp_old.detach().cpu().numpy()

            acoes = []
            for i, k in enumerate(idx_np):
                acao_dict = decodificar_acao_36(int(k))
                raw_dx = acao_dict["mouse"][0]
                prev_dx = self.ultimo_dx.get(i, 0.0)
                smooth_dx = int(0.65 * raw_dx + 0.35 * prev_dx)
                self.ultimo_dx[i] = smooth_dx
                acao_dict["mouse"] = [smooth_dx, 0]
                acoes.append(acao_dict)

            self.ultimo = {
                "u8": u8,
                "sv": sv.detach().cpu().numpy(),
                "gv": gv.detach().cpu().numpy(),
                "idx": idx_np,
                "logp_old": logp_np,
                "val": val_np,
                "ids": ids.detach().cpu().numpy()
            }

        return acoes

    def log_prob(self, px, sv, gv, a_idx, ids=None):
        """Avaliação de log-probabilidade para compatibilidade com interfaces unificadas."""
        res = self.forward_pensamento(px, sv, gv, ids)
        if self.fatorada:
            lg_modo, lg_yaw, val = res
            # Converte a_idx em (modo, yaw)
            a_modo = torch.tensor([fatorar_indice_36(int(x))[0] for x in a_idx], device=self.device, dtype=torch.long)
            a_yaw  = torch.tensor([fatorar_indice_36(int(x))[1] for x in a_idx], device=self.device, dtype=torch.long)
            
            logp_m = torch.log_softmax(lg_modo, dim=-1).gather(1, a_modo.unsqueeze(1)).squeeze(1)
            logp_y = torch.log_softmax(lg_yaw, dim=-1).gather(1, a_yaw.unsqueeze(1)).squeeze(1)
            lp = logp_m + logp_y
            
            ent_m = -(torch.softmax(lg_modo, dim=-1) * torch.log_softmax(lg_modo, dim=-1)).sum(-1).mean()
            ent_y = -(torch.softmax(lg_yaw, dim=-1) * torch.log_softmax(lg_yaw, dim=-1)).sum(-1).mean()
            ent = ent_m + ent_y
            ent_norm = ent / (math.log(NUM_MODOS) + math.log(NUM_YAW))
            return lp, ent_norm, val
        else:
            logits, val = res
            logp = torch.log_softmax(logits, dim=-1)
            lp = logp.gather(1, a_idx.unsqueeze(1)).squeeze(1)
            ent = -(logp.exp() * logp).sum(-1).mean()
            ent_norm = ent / torch.log(torch.tensor(float(self.num_acoes), device=self.device))
            return lp, ent_norm, val

