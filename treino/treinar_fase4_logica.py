# coding=utf-8
"""
treinar_fase4_logica.py — RL PPO Hierárquico com Cérebro Supervisor e VLA Motor.

Melhorias de Terceira Geração (Fase 4 v3):
1. Cérebro Supervisor Hierárquico (`PoliticaCerebroVLA`):
   - Detecção automática de travamento/colisão com manobra de desengate.
   - Pulo automático em degraus e relevo.
   - Comando de varredura visual angular pós-submeta.
2. Recompensa Densa de Alinhamento Visual Angular (Visual Yaw Shaping):
   - Elimina o vício de "linha reta cega" / projétil.
   - Recompensa giros de correção e penaliza avanço desalinhado (> 40°).
3. Injeção Ativa de Estágio (`sv[:, 16]`) e Prompts Dinâmicos por Etapa.
4. Cabeça de Ação com Prior Neutro (viés zerado).
5. Suporte Completo à Retomada de Checkpoint sem perda de adaptadores LoRA.
"""
import os
import sys
import time
import math
import random
import argparse
import torch
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from ambiente.arena_plana import post, get
from ambiente.tarefas_logicas import (montar_tarefas_logicas, PASSOS_MAX_F4,
                                      RAIO_CHEGADA_SUBMETA, BONUS_SUBMETA, BONUS_FINAL)
from politica.politica_raciocinio import PoliticaRaciocinioLoop
from politica.cerebro import PoliticaCerebroVLA
from modelo.lora_vla import aplicar_lora
from infra.gpu_utils import travar_gpu, compactar_backbone
from infra.run_vla_agent import load_vla_agent
from treino.treinar_fase1 import mascara_do_passo, retornos

CKPT_SAIDA = "checkpoints_vla/vla_fase4_logica.pt"


def rollout_f4(pol, tarefas, passos=None):
    passos = passos or PASSOS_MAX_F4
    n = len(tarefas)

    # Teleporta todos os robôs exatamente para as coordenadas secas e sólidas
    r = post("/lote/reset", {"posicoes": [list(t["largada"]) for t in tarefas]})
    obs = r["obs"][:n]
    est = [o["estado"] for o in obs]
    pol.reiniciar(obs)

    estagio_atual = [0] * n  # rastreia o estágio lógico de cada robô (0 -> 1 -> concluído)
    submeta_atingida = [False] * n
    vivo = [True] * n
    concluiu_em = [None] * n
    y_ant = [e.get("y", 0.0) for e in est]

    # Distância inicial e erro angular inicial ao primeiro alvo
    dant = [math.hypot(t["estagios"][0]["alvo_abs"][0] - est[i]["x"],
                       t["estagios"][0]["alvo_abs"][1] - est[i]["z"]) for i, t in enumerate(tarefas)]
    erro_yaw_ant = []
    for i, t in enumerate(tarefas):
        alvo_0 = t["estagios"][0]["alvo_abs"]
        alvo_yaw_0 = math.atan2(-(alvo_0[0] - est[i]["x"]), -(alvo_0[1] - est[i]["z"]))
        bot_yaw_0 = math.radians(est[i]["yaw"])
        erro_yaw_ant.append(abs((bot_yaw_0 - alvo_yaw_0 + math.pi) % (2 * math.pi) - math.pi))

    U8, SV, GV, IDS, IDX, R, VIVO = [], [], [], [], [], [], []

    for p in range(passos):
        # 1. Prompts específicos do estágio ativo
        prompts_ativos = [
            t["estagios"][min(estagio_atual[i], len(t["estagios"]) - 1)].get("prompt_estagio", t["prompt"])
            for i, t in enumerate(tarefas)
        ]

        # 2. Injeção ativa do estágio no forward pass hierárquico
        alvos_ativos_abs = [t["estagios"][min(estagio_atual[i], len(t["estagios"]) - 1)]["alvo_abs"] for i, t in enumerate(tarefas)]
        acoes = pol.agir(est, alvos_ativos_abs, obs, prompts=prompts_ativos, estagios=estagio_atual)
        for i in range(n):
            if not vivo[i]:
                acoes[i] = {"hold": [], "mouse": [0, 0], "duration_ms": 50}

        u = pol.ultimo
        acoes_idx = u["idx"]

        rr = post("/lote/passo", {"acoes": acoes, "frames": True})
        obs = rr["obs"][:n]
        est = [o["estado"] for o in obs]
        pol.observar(obs)

        rec = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if not vivo[i]:
                continue
            e = est[i]
            # Elimina interferência de água e lava
            if e.get("in_water") or e.get("in_lava"):
                rec[i] -= 3.0
                vivo[i] = False
                continue

            tar = tarefas[i]
            est_idx = estagio_atual[i]
            alvo_ativo = tar["estagios"][est_idx]

            d = math.hypot(alvo_ativo["alvo_abs"][0] - e["x"],
                           alvo_ativo["alvo_abs"][1] - e["z"])

            # 3. Recompensa Diferencial de Alinhamento Angular (evita pião / spin infinito)
            alvo_yaw = math.atan2(-(alvo_ativo["alvo_abs"][0] - e["x"]),
                                  -(alvo_ativo["alvo_abs"][1] - e["z"]))
            bot_yaw = math.radians(e["yaw"])
            erro_yaw_atual = abs((bot_yaw - alvo_yaw + math.pi) % (2 * math.pi) - math.pi)

            # Recompensa proporcional à aproximação da mira correta
            delta_alinhamento = erro_yaw_ant[i] - erro_yaw_atual
            erro_yaw_ant[i] = erro_yaw_atual

            # Recompensa combinada: aproximação métrica + convergência de mira
            rec[i] = (dant[i] - d) + 1.5 * delta_alinhamento

            pulou = (acoes_idx[i] >= 9) or ("SPACE" in acoes[i].get("hold", []))
            subiu = (e.get("y", 0.0) > y_ant[i] + 0.3)

            # Custo motor / Recompensa de superação de relevo
            if pulou:
                if subiu and d < dant[i]:
                    rec[i] += 0.50
                elif not e.get("is_collided_horizontally", False):
                    rec[i] -= 0.05

            y_ant[i] = e.get("y", 0.0)
            dant[i] = d

            if d <= RAIO_CHEGADA_SUBMETA:
                if est_idx + 1 < len(tar["estagios"]):
                    # Submeta 1 alcançada! Avança e aciona varredura no Cérebro
                    rec[i] += BONUS_SUBMETA
                    estagio_atual[i] += 1
                    submeta_atingida[i] = True
                    novo_alvo = tar["estagios"][estagio_atual[i]]
                    dant[i] = math.hypot(novo_alvo["alvo_abs"][0] - est[i]["x"],
                                         novo_alvo["alvo_abs"][1] - est[i]["z"])
                    if hasattr(pol, "ativar_varredura"):
                        pol.ativar_varredura(i, passos_varredura=3)
                else:
                    # Tarefa lógica completa concluída com sucesso!
                    rec[i] += BONUS_FINAL
                    concluiu_em[i] = p + 1
                    vivo[i] = False

        U8.append(u["u8"]); SV.append(u["sv"]); GV.append(u["gv"]); IDS.append(u["ids"])
        IDX.append(u["idx"]); R.append(rec)
        VIVO.append(np.array(mascara_do_passo(vivo, concluiu_em, p), dtype=np.float32))
        if not any(vivo):
            break

    met = [{
        "concluiu": concluiu_em[i] is not None,
        "submeta_ok": submeta_atingida[i] or (concluiu_em[i] is not None),
        "estagio_final": estagio_atual[i]
    } for i in range(n)]
    return (np.stack(U8), np.stack(SV), np.stack(GV), np.stack(IDS), np.stack(IDX),
            np.stack(R), np.stack(VIVO)), met


class BufferSucesso:
    """Armazena transições de trajetórias que concluíram a cadeia lógica."""
    def __init__(self, max_transicoes=3000):
        self.max = max_transicoes
        self.u8, self.sv, self.gv, self.ids, self.idx, self.adv = [], [], [], [], [], []

    def adicionar(self, u8, sv, gv, ids, idx, adv):
        self.u8.extend(u8); self.sv.extend(sv); self.gv.extend(gv)
        self.ids.extend(ids); self.idx.extend(idx); self.adv.extend(adv)
        if len(self.u8) > self.max:
            excesso = len(self.u8) - self.max
            self.u8 = self.u8[excesso:]; self.sv = self.sv[excesso:]
            self.gv = self.gv[excesso:]; self.ids = self.ids[excesso:]
            self.idx = self.idx[excesso:]; self.adv = self.adv[excesso:]

    def amostrar(self, tamanho=32):
        if len(self.u8) < tamanho:
            return None
        idx = np.random.choice(len(self.u8), tamanho, replace=False)
        return (
            np.array([self.u8[i] for i in idx]),
            np.array([self.sv[i] for i in idx]),
            np.array([self.gv[i] for i in idx]),
            np.array([self.ids[i] for i in idx]),
            np.array([self.idx[i] for i in idx]),
            np.array([self.adv[i] for i in idx], dtype=np.float32)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteracoes", type=int, default=120)
    ap.add_argument("--it-inicio", type=int, default=1)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--clip-ppo", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--minilote", type=int, default=12)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--ckpt-entrada", default="checkpoints_vla/vla_fase3_merged.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--curriculo-nivel", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    N = get("/lote/info")["envs"]
    travar_gpu()
    
    ckpt_a_carregar = args.ckpt_entrada if os.path.exists(args.ckpt_entrada) else "checkpoints_vla/vla_fase3_merged.pt"
    vla, device = load_vla_agent(ckpt_a_carregar)
    compactar_backbone(vla)

    # Injeta LoRA se não estiver presente
    if not any("lora_" in n for n, _ in vla.named_parameters()):
        vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)

    # Instancia a política e o acoplador do Cérebro
    pol_vla = PoliticaRaciocinioLoop(None, amostrar=True, device=device, vla=vla, loops_pensamento=args.loops)
    pol = PoliticaCerebroVLA(pol_vla)

    # Se o checkpoint de entrada contém 'treinaveis' (retomando treino), restaura agora
    if os.path.exists(ckpt_a_carregar):
        ckpt_data = torch.load(ckpt_a_carregar, map_location=device)
        if "treinaveis" in ckpt_data:
            msg = vla.load_state_dict(ckpt_data["treinaveis"], strict=False)
            print(f"[VLA] Checkpoint retomado: {len(ckpt_data['treinaveis'])} tensores treináveis restaurados (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})", flush=True)

    vla.to(device)
    treinaveis = [p for p in vla.parameters() if p.requires_grad]
    print("=" * 80)
    print(f" [FASE 4 v3 — HIERARQUICA] CÉREBRO SUPERVISOR + VLA MOTOR & YAW SHAPING")
    print(f"    Tensores treináveis: {len(treinaveis)} | Loops: K={args.loops} | Ações: 18")
    print(f"    Checkpoint entrada : {ckpt_a_carregar}")
    print(f"    Nível Currículo    : {args.curriculo_nivel} (1: ±25° -> 2: ±70° -> 3: ±180°)")
    print("=" * 80)

    buffer_sucesso = BufferSucesso()
    opt = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=1e-2)
    escala = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    nivel_atual = args.curriculo_nivel
    historico_sucessos = []

    t0 = time.time()
    for it in range(args.it_inicio, args.iteracoes + 1):
        # Progressão automática de currículo
        if len(historico_sucessos) >= 8:
            taxa_media_recente = sum(historico_sucessos[-8:]) / 8.0
            if taxa_media_recente >= 0.35 and nivel_atual == 1:
                nivel_atual = 2
                print(f"\n>>> [CURRICULO] Promovido para Nível 2 (Desvio Lateral ±70°) na iteração {it}!\n", flush=True)
            elif taxa_media_recente >= 0.40 and nivel_atual == 2:
                nivel_atual = 3
                print(f"\n>>> [CURRICULO] Promovido para Nível 3 (360° Total) na iteração {it}!\n", flush=True)

        tarefas = montar_tarefas_logicas(N, seed=args.seed + it, nivel_curriculo=nivel_atual)
        vla.eval(); pol.amostrar = True
        (U8, SV, GV, IDS, IDX, R, VIVO), met = rollout_f4(pol, tarefas)

        taxa_completa = sum(x["concluiu"] for x in met) / len(met)
        taxa_submeta = sum(x["submeta_ok"] for x in met) / len(met)
        historico_sucessos.append(taxa_completa)

        G = retornos(R, VIVO, gamma=args.gamma)
        m = VIVO.reshape(-1) > 0
        g = G.reshape(-1)[m]
        g_std = float(g.std())
        PISO_G_STD = 3.0
        adv = (g - g.mean()) / (max(g_std, PISO_G_STD) + 1e-6)

        u8 = U8.reshape(-1, *U8.shape[2:])[m]
        sv = SV.reshape(-1, SV.shape[-1])[m]
        gv = GV.reshape(-1, GV.shape[-1])[m]
        ids = IDS.reshape(-1, IDS.shape[-1])[m]
        ay = IDX.reshape(-1)[m]

        # Salva trajetórias com avanço no Buffer de Sucesso
        for i, inf in enumerate(met):
            if inf["concluiu"] or inf["estagio_final"] > 0:
                passos_bot = [step for step in range(len(VIVO)) if VIVO[step][i] > 0]
                if passos_bot:
                    u8_s = [U8[step][i] for step in passos_bot]
                    sv_s = [SV[step][i] for step in passos_bot]
                    gv_s = [GV[step][i] for step in passos_bot]
                    ids_s = [IDS[step][i] for step in passos_bot]
                    idx_s = [IDX[step][i] for step in passos_bot]
                    adv_s = [float(adv[step * len(met) + i]) if (step * len(met) + i) < len(adv) else 1.0 for step in passos_bot]
                    buffer_sucesso.adicionar(u8_s, sv_s, gv_s, ids_s, idx_s, adv_s)

        vla.train(); vla.vision_encoder.eval()

        old_logp = np.zeros(len(ay), dtype=np.float32)
        with torch.no_grad():
            for b0 in range(0, len(ay), args.minilote):
                sel = slice(b0, b0 + args.minilote)
                px = pol.normalizar(u8[sel])
                svt = torch.tensor(sv[sel], dtype=torch.float32, device=device)
                gvt = torch.tensor(gv[sel], dtype=torch.float32, device=device)
                idst = torch.tensor(ids[sel], dtype=torch.long, device=device)
                yt = torch.tensor(ay[sel], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    lp_tmp, _ = pol.log_prob(px, svt, gvt, yt, ids=idst)
                old_logp[sel] = lp_tmp.float().cpu().numpy()

        ent_soma, nb = 0.0, 0
        for ep in range(args.epocas):
            ordem = np.random.permutation(len(ay))
            for b0 in range(0, len(ordem), args.minilote):
                sel = ordem[b0:b0 + args.minilote]
                px = pol.normalizar(u8[sel])
                svt = torch.tensor(sv[sel], dtype=torch.float32, device=device)
                gvt = torch.tensor(gv[sel], dtype=torch.float32, device=device)
                idst = torch.tensor(ids[sel], dtype=torch.long, device=device)
                yt = torch.tensor(ay[sel], dtype=torch.long, device=device)
                advt = torch.tensor(adv[sel], dtype=torch.float32, device=device)
                old_lpt = torch.tensor(old_logp[sel], dtype=torch.float32, device=device)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    lp, ent = pol.log_prob(px, svt, gvt, yt, ids=idst)
                    ratio = torch.exp(lp.float() - old_lpt)
                    surr1 = ratio * advt
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_ppo, 1.0 + args.clip_ppo) * advt
                    perda = -torch.min(surr1, surr2).mean() - 0.04 * ent.float()

                escala.scale(perda).backward()
                escala.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(treinaveis, 0.5)
                escala.step(opt); escala.update()
                ent_soma += float(ent.item())
                nb += 1

        print("it %3d/%d [Nivel %d] | P1 %3.0f%% | Completo %3.0f%% | ent %.2f | g_std %.2f | %4.0fs"
              % (it, args.iteracoes, nivel_atual, 100 * taxa_submeta, 100 * taxa_completa,
                 ent_soma / max(1, nb), g_std, time.time() - t0), flush=True)

        if it % 10 == 0 or it == args.iteracoes:
            os.makedirs(os.path.dirname(CKPT_SAIDA), exist_ok=True)
            torch.save({
                "treinaveis": {n_: p.detach().cpu() for n_, p in vla.named_parameters() if p.requires_grad},
                "iteracao": it,
                "curriculo_nivel": nivel_atual
            }, CKPT_SAIDA)

    print("\n[OK] Fase 4 v3 Treinamento Concluído com Sucesso -> %s" % CKPT_SAIDA)


if __name__ == "__main__":
    main()
