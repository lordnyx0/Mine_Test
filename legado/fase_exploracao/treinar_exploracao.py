# coding=utf-8
"""
Treino da tarefa de EXPLORAÇÃO com objetivo explícito.

Ciclo por rodada:
  1. COLETA — roda a política com exploração no ExplorationEnv, guardando
     (frame, state_vec, ação, reward) de cada passo.
  2. TREINA — regressão ponderada por vantagem (reward menos a média da
     rodada), só nos passos acima da média. Imitar o que deu certo em relação
     ao próprio desempenho, não em relação a um limiar fixo.
  3. SALVA — checkpoint rotativo, no máximo MAX_CHECKPOINTS no disco.
  4. MEDE — recorde médio de distância por episódio, para acompanhar a curva.

O objetivo entra no modelo como texto (`input_ids` no Qwen) + vetor de estado
de 16 dims com o andamento da tarefa.

    python treinar_exploracao.py --rodadas 20 --passos 300
"""
import os
import re
import sys
import glob
import json
import math
import time
import random
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_utils import (limitar_recursos, limitar_vram, travar_gpu,
                       compactar_backbone, memoria_gpu)

limitar_recursos()          # antes de importar torch pesado

import torch
import torch.nn as nn

from run_vla_agent import load_vla_agent
from bot_vision_capture import BotVisionCapture
from exploration_env import ExplorationEnv, GoalEncoder

CKPT_DIR = "checkpoints_vla"
CKPT_BASE = "vla_exploracao"
# A BASE pre-treinada e imutavel. O treino NUNCA escreve nela: ja perdi duas
# vezes os pesos bons porque o arquivo ativo apontava para o mesmo caminho, e
# 40 rodadas com a recompensa errada sobrescreveram a base que andava direito.
CKPT_BASE_PRETREINO = os.path.join(CKPT_DIR, "vla_locomotion.pt")
CKPT_ATIVO = os.path.join(CKPT_DIR, "vla_exploracao_ATIVO.pt")
MAX_CHECKPOINTS = 5          # teto pedido: nunca mais que isso no disco
METRICAS = "checkpoints_vla/metricas_exploracao.jsonl"

BUTTONS = ["W", "S", "A", "D", "SPACE", "LCLICK", "RCLICK", "SHIFT"]
USABLE = [0, 1, 2, 3, 4, 7]


# ── Checkpoints ───────────────────────────────────────────────────────────────
def salvar_checkpoint(vla, rodada, metrica, melhor=False):
    """
    Salva e poda. Política de retenção: 1 'melhor' + os mais recentes,
    totalizando no máximo MAX_CHECKPOINTS arquivos.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    estado = {
        "resampler":     vla.resampler.state_dict(),
        "projector":     vla.projector.state_dict(),
        "state_encoder": vla.state_encoder.state_dict(),
        "action_heads":  vla.action_heads.state_dict(),
        "rodada":        rodada,
        "metrica":       metrica,
    }

    # Checkpoint ativo (o que run_vla_agent.py carrega) — sempre sobrescrito
    torch.save(estado, CKPT_ATIVO)

    nome = os.path.join(CKPT_DIR, f"{CKPT_BASE}_r{rodada:03d}_{metrica:.1f}b.pt")
    torch.save(estado, nome)

    if melhor:
        torch.save(estado, os.path.join(CKPT_DIR, f"{CKPT_BASE}_MELHOR.pt"))

    podar_checkpoints()
    return nome


def podar_checkpoints():
    """
    Teto DURO de MAX_CHECKPOINTS arquivos no disco, contando tudo:
    o ativo (que o agente carrega) + o MELHOR + os rotativos recentes.
    """
    rotativos = sorted(
        glob.glob(os.path.join(CKPT_DIR, f"{CKPT_BASE}_r*.pt")),
        key=os.path.getmtime
    )
    melhor = os.path.join(CKPT_DIR, f"{CKPT_BASE}_MELHOR.pt")
    reservados = (sum(os.path.exists(f) for f in (CKPT_ATIVO, CKPT_BASE_PRETREINO, melhor)))
    limite = MAX_CHECKPOINTS - reservados
    while len(rotativos) > max(1, limite):
        antigo = rotativos.pop(0)
        try:
            os.remove(antigo)
            print(f"    [ckpt] removido antigo: {os.path.basename(antigo)}", flush=True)
        except OSError:
            break


# ── Política ──────────────────────────────────────────────────────────────────
def preparar_pixels(processor, pilhas, device):
    """
    pilhas: lista (batch) de listas de K frames.
    Retorna [B, K, 3, H, W] — o modelo achata para rodar o SigLIP e depois
    remonta marcando cada frame com sua posicao no tempo.
    """
    if pilhas and not isinstance(pilhas[0], (list, tuple)):
        pilhas = [[p] for p in pilhas]
    B, K = len(pilhas), len(pilhas[0])
    planas = [f for pilha in pilhas for f in pilha]
    px = processor(images=planas, return_tensors="pt")["pixel_values"]
    return px.view(B, K, *px.shape[1:]).to(device)


def decidir(vla, device, img, state_vec, goal_ids, explorar, eps):
    processor = vla.vision_processor
    px = preparar_pixels(processor, [img], device)
    sv = torch.tensor([state_vec], dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=torch.cuda.is_available()):
        out = vla(pixel_values=px, state_vec=sv, input_ids=goal_ids)
    p = out["buttons"][0].float().cpu().numpy()
    yaw_logits = out["yaw_logits"][0].float()
    pitch_logits = out["pitch_logits"][0].float()
    YB = vla.action_heads.YAW_BINS
    PB = vla.action_heads.PITCH_BINS

    if explorar and random.random() < eps:
        hold = [BUTTONS[k] for k in USABLE if random.random() < 0.3]
        dx = random.choice(YB); dy = random.choice(PB)
    elif explorar:
        hold = [BUTTONS[k] for k in USABLE if random.random() < float(p[k])]
        # amostra o bin de giro pela softmax: exploracao na CABECA tambem,
        # que antes nunca girava porque a regressao saia constante em ~0
        dx = YB[int(torch.multinomial(torch.softmax(yaw_logits, -1), 1))]
        dy = PB[int(torch.multinomial(torch.softmax(pitch_logits, -1), 1))]
    else:
        hold = [BUTTONS[k] for k in USABLE if p[k] > 0.5]
        dx = YB[int(yaw_logits.argmax())]
        dy = PB[int(pitch_logits.argmax())]

    return {"hold": hold, "mouse": [int(dx), int(dy)], "duration_ms": 250}


def bin_mais_proximo(valor, bins):
    return min(range(len(bins)), key=lambda i: abs(bins[i] - valor))


def comprimir(pilha):
    """PIL -> bytes JPEG (uma pilha de frames vira lista de bytes)."""
    import io as _io
    out = []
    for f in (pilha if isinstance(pilha, (list, tuple)) else [pilha]):
        b = _io.BytesIO()
        f.save(b, format="JPEG", quality=80)
        out.append(b.getvalue())
    return out


def descomprimir(pilha_bytes):
    import io as _io
    from PIL import Image
    return [Image.open(_io.BytesIO(b)).convert("RGB") for b in pilha_bytes]


def fechar_episodio(episodio, gamma):
    """Preenche o retorno descontado G_t de cada passo, de tras para frente."""
    G = 0.0
    for a in reversed(episodio):
        G = a["reward"] + gamma * G
        a["retorno"] = G
    return episodio


def alvo_botoes(hold):
    h = [k.upper() for k in hold]
    return torch.tensor([1.0 if b in h else 0.0 for b in BUTTONS], dtype=torch.float32)


# ── Avaliação ─────────────────────────────────────────────────────────────────
def avaliar_politica(vla, device, env, goal_ids, episodios=4, passos=70):
    """
    Mede a política SEM exploração, com protocolo fixo: N episódios, cada um
    de um respawn aleatório, contando a maior distância atingida da origem.

    A métrica por rodada da coleta é ruidosa demais para concluir qualquer
    coisa (oscilou de 15 a 154 blocos entre rodadas vizinhas): ela mistura o
    ruído da exploração com a sorte do terreno. Esta avaliação isola a
    política e usa sempre o mesmo protocolo, então é comparável no tempo.
    """
    distancias = []
    for _ in range(episodios):
        img, sv = env.reset(respawnar=True)
        melhor = 0.0
        for _ in range(passos):
            act = decidir(vla, device, img, sv, goal_ids, explorar=False, eps=0.0)
            img, sv, _r, respawnou, info = env.step(act)
            if info.get("descontinuidade") or respawnou:
                melhor = max(melhor, info["recorde"])
                break
            melhor = max(melhor, info["distancia"])
        distancias.append(melhor)
    return sum(distancias) / len(distancias), distancias


# ── Treino ────────────────────────────────────────────────────────────────────
def treinar_rodada(vla, device, amostras, goal_ids, epocas, batch, lr, usar_amp,
                   peso_rota=1.0):
    """
    Perda conjunta de dois tipos bem diferentes:

      POLITICA (esparsa) — botoes e bin de giro, ponderados pela vantagem.
      So os passos acima da media puxam; os demais tem peso zero.

      ROTA (densa) — prever a navegabilidade dos 12 setores a partir da
      imagem, em TODAS as amostras, com peso fixo. Este e o termo que impede
      o colapso visual: o alvo muda a cada cena, entao ignorar a imagem passa
      a custar caro. Medi o projetor com posto efetivo 1.4 de 1024 justamente
      porque o unico alvo anterior era constante ("sempre W").
    """
    if not amostras:
        return 0.0, 0, 0.0

    # Retorno descontado -> vantagem normalizada
    rs = [a["retorno"] for a in amostras]
    media = sum(rs) / len(rs)
    desvio = (sum((r - media) ** 2 for r in rs) / len(rs)) ** 0.5 or 1.0
    for a in amostras:
        a["peso"] = max(0.0, min(3.0, (a["retorno"] - media) / desvio))
    n_uteis = sum(1 for a in amostras if a["peso"] > 0.15)

    processor = vla.vision_processor
    params = [p for p in vla.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    escala = torch.amp.GradScaler("cuda", enabled=usar_amp)
    YB = vla.action_heads.YAW_BINS
    PB = vla.action_heads.PITCH_BINS
    K = vla.action_heads.num_rotas

    vla.train(); vla.vision_encoder.eval(); vla.qwen_model.eval()
    total, total_rota, n = 0.0, 0.0, 0

    for _ in range(epocas):
        random.shuffle(amostras)
        for i in range(0, len(amostras), batch):
            lote = amostras[i:i + batch]
            px = preparar_pixels(processor, [descomprimir(a["img"]) for a in lote], device)
            sv = torch.tensor([a["state"] for a in lote], dtype=torch.float32, device=device)
            tb = torch.stack([alvo_botoes(a["action"]["hold"]) for a in lote]).to(device)
            tyaw = torch.tensor([bin_mais_proximo(a["action"]["mouse"][0], YB) for a in lote],
                                dtype=torch.long, device=device)
            tpit = torch.tensor([bin_mais_proximo(a["action"]["mouse"][1], PB) for a in lote],
                                dtype=torch.long, device=device)
            w = torch.tensor([a["peso"] for a in lote], dtype=torch.float32, device=device)

            # alvo de rota (pode faltar se o servidor nao respondeu naquele passo)
            rot = [a.get("rotas") for a in lote]
            tem_rota = [j for j, r in enumerate(rot) if r and len(r) == K]
            ids = goal_ids.expand(len(lote), -1)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=usar_amp):
                out = vla(pixel_values=px, state_vec=sv, input_ids=ids)

            perda_pol = (bce(out["buttons_logits"].float(), tb).mean(dim=1)
                         + 0.5 * ce(out["yaw_logits"].float(), tyaw)
                         + 0.2 * ce(out["pitch_logits"].float(), tpit))
            perda = (perda_pol * w).sum() / w.sum().clamp(min=1e-6)

            perda_rota = torch.zeros((), device=device)
            if tem_rota:
                idx = torch.tensor(tem_rota, dtype=torch.long, device=device)
                alvo_r = torch.tensor([rot[j] for j in tem_rota],
                                      dtype=torch.float32, device=device)
                perda_rota = mse(out["rotas"].float()[idx], alvo_r).mean()
                perda = perda + peso_rota * perda_rota

            escala.scale(perda).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            escala.step(opt)
            escala.update()
            total += float(perda.item())
            total_rota += float(perda_rota.item())
            n += 1

    vla.eval()
    return total / max(1, n), n_uteis, total_rota / max(1, n)


# ── Principal ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rodadas", type=int, default=20)
    ap.add_argument("--passos", type=int, default=300, help="passos de coleta por rodada")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--peso-rota", type=float, default=1.0,
                    help="peso da perda de previsao de rota (denso, anti-colapso)")
    ap.add_argument("--gamma", type=float, default=0.97,
                    help="desconto do retorno (~0.97 = horizonte de ~8s)")
    ap.add_argument("--sem-progresso", type=float, default=60.0)
    ap.add_argument("--sem-amp", action="store_true")
    ap.add_argument("--do-zero", action="store_true",
                    help="ignora o ativo e parte da base pre-treinada")
    ap.add_argument("--aval-cada", type=int, default=5)
    ap.add_argument("--aval-episodios", type=int, default=4)
    ap.add_argument("--aval-passos", type=int, default=70)
    ap.add_argument("--vram", type=float, default=0.65,
                    help="fracao maxima da VRAM (o Minecraft usa o resto)")
    args = ap.parse_args()

    random.seed(0); torch.manual_seed(0)
    usar_amp = torch.cuda.is_available() and not args.sem_amp

    travar_gpu()          # um treino por vez: dois modelos no 3060 travam o PC
    # Retoma do ativo se existir; senao parte da base pre-treinada limpa.
    partida = CKPT_ATIVO if (os.path.exists(CKPT_ATIVO) and not args.do_zero)         else CKPT_BASE_PRETREINO
    print(f"[treino] partindo de: {partida}", flush=True)
    vla, device = load_vla_agent(partida)
    dt = compactar_backbone(vla)
    teto = limitar_vram(args.vram)
    print(f"[VRAM] backbone em {dt} | teto {teto} | {memoria_gpu()}", flush=True)

    cap = BotVisionCapture()
    env = ExplorationEnv(cap, segundos_sem_progresso=args.sem_progresso, verbose=True)
    ge = GoalEncoder(device=device)
    goal_ids = ge.ids("explorar")

    print(f"\n[treino] objetivo: {goal_ids.shape[1]} tokens | AMP={usar_amp} "
          f"| max {MAX_CHECKPOINTS} checkpoints", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)

    print("\n[aval] linha de base (politica atual, sem exploracao)...", flush=True)
    base_aval, base_lista = avaliar_politica(vla, device, env, goal_ids,
                                             args.aval_episodios, args.aval_passos)
    print("[aval] BASE: %.1f b  (episodios: %s)"
          % (base_aval, ", ".join("%.0f" % d for d in base_lista)), flush=True)

    melhor_metrica = base_aval
    historico_aval = [(0, base_aval)]

    for rodada in range(1, args.rodadas + 1):
        print("\n" + "=" * 70)
        print(f" RODADA {rodada}/{args.rodadas}")
        print("=" * 70, flush=True)

        # ── Coleta ────────────────────────────────────────────────────────────
        img, sv = env.reset()
        amostras, episodio = [], []
        recordes, r_total, n_reward = [], 0.0, 0
        mortes_antes = env.mortes
        t0 = time.time()

        for passo in range(args.passos):
            act = decidir(vla, device, img, sv, goal_ids, explorar=True, eps=args.eps)
            prox_img, prox_sv, reward, respawnou, info = env.step(act)

            # Passo de descontinuidade nao entra no treino: a acao nao causou
            # aquele deslocamento, entao imita-la seria aprender ruido.
            if not info.get("descontinuidade"):
                # Guarda JPEG comprimido, nao PIL cru: 3 frames x 400 passos em
                # 640x360 seriam ~830MB de RAM por rodada.
                episodio.append({"img": comprimir(img), "state": sv,
                                 "action": act, "reward": reward,
                                 "rotas": info.get("rotas")})
                r_total += reward
                n_reward += 1
            if respawnou:
                recordes.append(info["recorde"])
                amostras.extend(fechar_episodio(episodio, args.gamma))
                episodio = []

            img, sv = prox_img, prox_sv

            if (passo + 1) % 50 == 0:
                print("    passo %3d/%d | dist %6.1f | recorde %6.1f | r_medio %+.3f"
                      % (passo + 1, args.passos, info["distancia"], info["recorde"],
                         r_total / max(1, n_reward)), flush=True)

        amostras.extend(fechar_episodio(episodio, args.gamma))   # ultimo episodio
        recordes.append(env.d_recorde)
        metrica = sum(recordes) / len(recordes)
        mortes_rodada = env.mortes - mortes_antes
        dt_coleta = time.time() - t0

        print("  -> coleta: %d passos em %.0fs | recorde medio %.1f b | reward medio %+.3f "
              "| %d mortes nesta rodada"
              % (args.passos, dt_coleta, metrica, r_total / max(1, n_reward),
                 mortes_rodada), flush=True)

        # ── Treino ────────────────────────────────────────────────────────────
        t1 = time.time()
        perda, n_treino, perda_rota = treinar_rodada(
            vla, device, amostras, goal_ids, args.epocas, args.batch, args.lr,
            usar_amp, peso_rota=args.peso_rota)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()      # devolve os blocos do pico de treino
        print("  -> treino: %d/%d amostras uteis | perda %.4f (rota %.4f) | %.0fs | %s"
              % (n_treino, len(amostras), perda, perda_rota,
                 time.time() - t1, memoria_gpu()), flush=True)

        # ── Avaliação periódica (é ela que decide o MELHOR) ───────────────────
        aval = None
        if rodada % args.aval_cada == 0 or rodada == args.rodadas:
            aval, lista = avaliar_politica(vla, device, env, goal_ids,
                                           args.aval_episodios, args.aval_passos)
            historico_aval.append((rodada, aval))
            print("  -> AVALIACAO: %.1f b (%+.1f vs base %.1f) | episodios: %s"
                  % (aval, aval - base_aval, base_aval,
                     ", ".join("%.0f" % d for d in lista)), flush=True)

        # ── Checkpoint ────────────────────────────────────────────────────────
        # O MELHOR so muda com base na avaliacao limpa, nunca no ruido da coleta.
        melhor = aval is not None and aval > melhor_metrica
        if melhor:
            melhor_metrica = aval
        nome = salvar_checkpoint(vla, rodada, aval if aval is not None else metrica,
                                 melhor=melhor)
        print("  -> checkpoint: %s%s" % (os.path.basename(nome),
                                         "  [MELHOR ate agora]" if melhor else ""), flush=True)

        with open(METRICAS, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "rodada": rodada, "recorde_medio": round(metrica, 2),
                "avaliacao": round(aval, 2) if aval is not None else None,
                "reward_medio": round(r_total / max(1, n_reward), 4),
                "mortes": mortes_rodada,
                "perda": round(perda, 5), "perda_rota": round(perda_rota, 5),
                "n_treino": n_treino,
                "segundos_coleta": round(dt_coleta, 1),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")

    print("\n" + "=" * 70)
    print(" FIM - curva de avaliacao (politica sem exploracao):")
    for r_, a_ in historico_aval:
        print("   rodada %3d: %6.1f b" % (r_, a_))
    print(" base %.1f b -> final %.1f b  (%+.1f)"
          % (base_aval, historico_aval[-1][1], historico_aval[-1][1] - base_aval))
    print(" melhor avaliado: %.1f b" % melhor_metrica)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
