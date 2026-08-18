# coding=utf-8
"""
Metrica nova: TAXA DE CHEGADA em B.

A metrica antiga — distancia do respawn — satura no planejador (42.6 contra
15.6 do so-W) e tem otimo degenerado: apertar W sempre ja e quase otimo. Foi
essa degeneracao que fez a via visual ser descartada, e prever rotas tratava o
sintoma.

"Va ate B", com B sorteado e relativo, nao satura, escala com a distancia, e
e a unidade que compoe em qualquer tarefa e em qualquer jogo.

Como B e escolhido: rodando o Piloto por k passos a partir da largada e
anotando onde ele parou. Garante que B e ALCANCAVEL — um ponto sorteado
geometricamente pode estar dentro de pedra ou do outro lado de um lago, e a
taxa de chegada viraria ruido do sorteio.

Politicas comparadas:
  reto    — aponta para B e anda (pula quando trava). O baseline trivial que
            qualquer um escreveria em cinco minutos. E ELE que precisa ser
            batido, nao o so-W.
  piloto  — planejador com Objetivos.ponto. O teto: mesma tarefa, com acesso
            aos voxels que o aluno nao tem.
  modelo  — a politica condicionada a objetivo (--ckpt).

    python avaliar_objetivo.py --posicoes 60 --k-alvo 24
"""
import os
import sys
import math
import json
import time
import base64
import argparse
import statistics as st
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://127.0.0.1:3002"
GRAUS_POR_UNIDADE = 0.003 * 180 / math.pi
YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)
RAIO_CHEGADA = 2.5
# Distancia do waypoint que a politica aprendida recebe como objetivo. Escolhido
# para cair DENTRO da distribuicao de treino (K_MAX=20 passos ~ 20 blocos), nao
# por ajuste fino: passar o alvo final a 97 blocos poria a entrada 3x fora.
WAYPOINT = 15.0


def post(caminho, corpo, timeout=300):
    d = json.dumps(corpo).encode()
    r = urllib.request.Request(BASE + caminho, data=d, method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read())


def acao_reto(est, alvo, travado):
    """Aponta para o alvo e anda. Mesma convencao de giro do planejador.js:
    o sinal e invertido porque foi medido em malha fechada (mouse=+graus
    AFASTA do alvo)."""
    yaw = math.radians(est["yaw"])
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    relX, relZ = alvo[0] + 0.5 - est["x"], alvo[1] + 0.5 - est["z"]
    frente = relX * fx + relZ * fz
    lado = relX * (-fz) + relZ * fx
    graus = math.atan2(lado, frente) * 180 / math.pi
    desejado = -graus / GRAUS_POR_UNIDADE
    b = min(YAW_BINS, key=lambda v: abs(v - desejado))
    # Pular quando travado: sem isto qualquer degrau de 1 bloco trava para
    # sempre, e o baseline ficaria artificialmente fraco.
    hold = ["W", "SPACE"] if travado >= 1 else ["W"]
    return {"hold": hold, "mouse": [int(b), 0], "duration_ms": 250}


def sortear_alvos(posicoes, k, raio):
    """Roda o Piloto por k passos e devolve (largada, ponto final) por ambiente."""
    r = post("/lote/reset", {"posicoes": [[x, z] for x, z in posicoes]})
    n = len(posicoes)
    largada = [(o["estado"]["x"], o["estado"]["z"]) for o in r["obs"][:n]]
    for _ in range(k):
        acoes = post("/lote/piloto", {"objetivo": "explorar",
                                      "extra": {"raio": raio}})["acoes"]
        r = post("/lote/passo", {"acoes": acoes, "frames": False})
    # Guarda tambem o Y do alvo: se o bloqueio for elevacao (B em cima de um
    # paredao que o BFS nao sobe, porque so aceita degrau de 1), a diferenca de
    # altura entre onde empacou e o alvo diz isso direto.
    alvos = [(o["estado"]["x"], o["estado"]["z"], o["estado"]["y"]) for o in r["obs"][:n]]
    return largada, alvos


def episodio(politica, posicoes, alvos, passos, raio, ctx=None, diag=False):
    """Roda um lote. Retorna uma lista de dicts por ambiente."""
    n = len(posicoes)
    r = post("/lote/reset", {"posicoes": [[x, z] for x, z in posicoes]})
    est = [o["estado"] for o in r["obs"][:n]]
    d0 = [math.hypot(alvos[i][0] - est[i]["x"], alvos[i][1] - est[i]["z"])
          for i in range(n)]
    dmin = list(d0)
    chegou_em = [None] * n
    travado = [0] * n
    ant = [(e["x"], e["z"]) for e in est]
    # Distancia curta esconde o que quebra a locomocao de verdade: cair em
    # buraco, afogar, travar em parede. Em 70 passos e 13 blocos nada disso
    # aparece; numa viagem de centenas de blocos e o que decide.
    morreu = [False] * n
    passos_travado = [0] * n
    vivos = [0] * n            # passos contados antes de chegar
    y0 = [e.get("y", 64) for e in est]
    yq = [0.0] * n
    # Ultimo passo em que a distancia ao alvo bateu recorde. E o que separa as
    # duas causas opostas de falha: se o recorde cai perto do fim, o episodio
    # acabou o orcamento andando; se cai cedo, a distancia estacionou e o resto
    # foi oscilacao.
    t_recorde = [0] * n
    traco = [[d0[i]] for i in range(n)]
    agua_min = [None] * n      # menor distancia a agua vista no episodio
    agua_fim = [None] * n      # agua ao redor no ultimo passo (onde empacou)
    perfil_fim = [None] * n    # relevo rumo ao alvo no ultimo passo

    # A politica aprendida precisa da MESMA entrada que viu no treino: pilha de
    # 3 frames nos atrasos (0, 16, 60) passos e o state_vec de 32 dims. Montar
    # isso de outro jeito na avaliacao foi o bug que fez a base decorar o treino
    # e errar tudo na pratica — por isso reusa EstadoEpisodio, a unica definicao.
    est_ep = None
    if politica == "modelo":
        from modelo.estado_sim import EstadoEpisodio
        est_ep = [EstadoEpisodio() for _ in range(n)]
        for i, o in enumerate(r["obs"][:n]):
            est_ep[i].reiniciar(o["estado"])
            est_ep[i].registrar(o["estado"], base64.b64decode(o["frame_b64"]))

    for t in range(passos):
        if politica == "reto":
            acoes = [acao_reto(est[i], alvos[i], travado[i]) for i in range(n)]
        elif politica in ("piloto", "rumo"):
            # piloto = Objetivos.ponto (guloso) | rumo = ponto + memoria de visitas
            extras = [{"alvo": {"x": alvos[i][0], "y": est[i].get("y", 64),
                                "z": alvos[i][1]}, "raio": raio} for i in range(n)]
            obj = "ponto" if politica == "piloto" else "rumo"
            acoes = post("/lote/piloto", {"objetivo": obj,
                                          "extras": extras})["acoes"][:n]
        else:
            acoes = ctx["acao"](est_ep, est, alvos)

        # A sonda de agua custa ~676 consultas de bloco por ambiente, entao so
        # a cada 10 passos e so quando se esta diagnosticando.
        quer_diag = diag and (t % 10 == 0)
        corpo = {"acoes": acoes, "frames": politica == "modelo", "diag": quer_diag}
        if quer_diag:
            # Perfil de relevo NA DIRECAO DO ALVO: e o que distingue parede
            # (sobe), abismo/lago (desce) e "o relevo nao e o problema" (plano).
            corpo["dirs"] = [[alvos[i][0] - est[i]["x"], alvos[i][1] - est[i]["z"]]
                             for i in range(n)]
        r = post("/lote/passo", corpo)
        obs = r["obs"][:n]
        est = [o["estado"] for o in obs]
        for i in range(n):
            if obs[i].get("morreu"):
                morreu[i] = True
            if est_ep is not None:
                est_ep[i].passo(est[i], base64.b64decode(obs[i]["frame_b64"]))
            if quer_diag:
                a = obs[i].get("diag", {}).get("agua_perto")
                agua_fim[i] = a
                perfil_fim[i] = obs[i].get("diag", {}).get("perfil")
                if a is not None:
                    agua_min[i] = a if agua_min[i] is None else min(agua_min[i], a)
            andou = math.hypot(est[i]["x"] - ant[i][0], est[i]["z"] - ant[i][1])
            travado[i] = travado[i] + 1 if andou < 0.15 else 0
            # So conta ANTES de chegar: depois da chegada o episodio continua
            # ate o limite de passos e o bot fica parado no alvo, o que
            # inflaria "travado" para 60% sem nenhuma patologia real.
            if andou < 0.15 and chegou_em[i] is None:
                passos_travado[i] += 1
            if chegou_em[i] is None:
                vivos[i] += 1
            ant[i] = (est[i]["x"], est[i]["z"])
            yq[i] = min(yq[i], est[i].get("y", 64) - y0[i])
            d = math.hypot(alvos[i][0] - est[i]["x"], alvos[i][1] - est[i]["z"])
            if d < dmin[i] - 0.5:          # recorde real, nao ruido de passo
                t_recorde[i] = t + 1
            dmin[i] = min(dmin[i], d)
            if (t + 1) % 20 == 0:
                traco[i].append(d)
            if d <= RAIO_CHEGADA and chegou_em[i] is None:
                chegou_em[i] = t + 1

    return [{"d0": d0[i], "dmin": dmin[i], "chegou": chegou_em[i] is not None,
             "passos": chegou_em[i], "morreu": morreu[i],
             "travado": passos_travado[i] / max(1, vivos[i]),
             "queda": yq[i], "t_recorde": t_recorde[i], "traco": traco[i],
             "agua_min": agua_min[i], "agua_fim": agua_fim[i],
             "perfil": perfil_fim[i],
             "dy_alvo": (est[i].get("y", 64) - alvos[i][2]) if len(alvos[i]) > 2 else None,
             "fechou": max(0.0, min(1.0, 1.0 - dmin[i] / max(d0[i], 1e-6)))}
            for i in range(n)]


def carregar_modelo(ckpt, forcar_w=False):
    """Devolve acao(est_ep, est, alvos) -> lista de acoes, em UMA inferencia."""
    from infra.gpu_utils import limitar_recursos, limitar_vram, travar_gpu, compactar_backbone
    limitar_recursos()
    import torch
    from PIL import Image
    import io as _io
    from infra.run_vla_agent import load_vla_agent
    from infra.train_vla import get_tokenizer
    from modelo.estado_sim import N_FRAMES
    from infra.gate_retrospecto import objetivo_relativo

    INSTRUCAO = ("Objetivo: explorar. Ande o maximo que puder e afaste-se o maximo "
                 "possivel do ponto onde voce nasceu. Nao fique parado no mesmo lugar.")
    USABLE = [0, 1, 2, 3, 4, 7]
    BUTTONS = ["W", "S", "A", "D", "SPACE", "LCLICK", "RCLICK", "SHIFT"]

    travar_gpu()
    vla, device = load_vla_agent(ckpt)
    compactar_backbone(vla)
    torch.cuda.empty_cache()
    limitar_vram(0.62)
    vla.eval()
    tok = get_tokenizer()
    ids1 = tok([INSTRUCAO], return_tensors="pt", truncation=True,
               max_length=24)["input_ids"].to(device)
    proc = vla.vision_processor
    YB, PB = vla.action_heads.YAW_BINS, vla.action_heads.PITCH_BINS

    def acao(est_ep, est, alvos):
        n = len(est)
        pilhas = [e.pilha_frames() for e in est_ep]
        planas = [Image.open(_io.BytesIO(b)).convert("RGB") for p in pilhas for b in p]
        px = proc(images=planas, return_tensors="pt")["pixel_values"]
        px = px.view(n, N_FRAMES, *px.shape[1:]).to(device)
        sv = torch.tensor([e.vetor() for e in est_ep], dtype=torch.float32, device=device)
        # O objetivo e RECALCULADO a cada passo em relacao a pose atual: e o
        # que permite a politica corrigir sozinha em vez de divergir, e a razao
        # de 30% de discordancia por passo nao implicar falha.
        #
        # E e ENCURTADO para um waypoint. O treino so viu objetivos de 4 a 20
        # passos (~4 a 20 blocos, normalizados por 30); B na avaliacao esta a
        # ~97. Passar o alvo final cru poria a entrada 3x fora da distribuicao
        # e a politica falharia por um motivo que nao tem nada a ver com
        # navegar. Encurtar na MESMA DIRECAO e a hierarquia real: o alvo
        # distante sobrevive, e as rotas curtas se encadeiam ate ele.
        gv = []
        for i in range(n):
            dx, dz = alvos[i][0] - est[i]["x"], alvos[i][1] - est[i]["z"]
            d = math.hypot(dx, dz)
            if d > WAYPOINT:
                dx, dz = dx * WAYPOINT / d, dz * WAYPOINT / d
            gv.append(objetivo_relativo(
                [est[i]["x"], est[i]["z"], est[i]["yaw"]],
                [est[i]["x"] + dx, est[i]["z"] + dz, 0.0]))
        gv = torch.tensor(gv, dtype=torch.float32, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            out = vla(pixel_values=px, state_vec=sv,
                      input_ids=ids1.expand(n, -1), goal_vec=gv)
        p = out["buttons"].float().cpu().numpy()
        iy = out["yaw_logits"].float().argmax(-1).cpu().numpy()
        ip = out["pitch_logits"].float().argmax(-1).cpu().numpy()
        acoes = []
        for i in range(n):
            if forcar_w:
                # Isola o GIRO: as cabecas de botao emitem A em 93% e SHIFT em
                # 62% (alvos de 0%), entao o bot anda de lado agachado e nada
                # do resto pode ser avaliado. Fixando o andar, o que sobra na
                # medicao e exclusivamente o que o yaw aprendeu.
                hold = ["W"] + (["SPACE"] if p[i][4] > 0.5 else [])
            else:
                hold = [BUTTONS[k] for k in USABLE if p[i][k] > 0.5]
            acoes.append({"hold": hold,
                          "mouse": [int(YB[iy[i]]), int(PB[ip[i]])],
                          "duration_ms": 250})
        return acoes

    return {"acao": acao}


def resumo(nome, res):
    n = len(res)
    taxa = sum(r["chegou"] for r in res) / n
    passos = [r["passos"] for r in res if r["chegou"]]
    print("  %-8s chegou %4.0f%% | fechou %4.0f%% | dmin %6.1f | morreu %3.0f%% | "
          "travado %3.0f%% | queda max %5.1f | passos %s"
          % (nome, 100 * taxa, 100 * st.mean(r["fechou"] for r in res),
             st.mean(r["dmin"] for r in res),
             100 * sum(r["morreu"] for r in res) / n,
             100 * st.mean(r["travado"] for r in res),
             min(r["queda"] for r in res),
             ("%.0f" % st.mean(passos)) if passos else "-"))
    return taxa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posicoes", type=int, default=60)
    ap.add_argument("--passos", type=int, default=70)
    ap.add_argument("--k-alvo", type=int, default=24,
                    help="passos do Piloto usados para gerar B")
    ap.add_argument("--raio", type=int, default=16)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--politicas", default="reto,piloto")
    ap.add_argument("--forcar-w", action="store_true",
                    help="fixa hold=[W] e usa SO o yaw do modelo")
    ap.add_argument("--diagnosticar", action="store_true",
                    help="separa as falhas em orcamento vs travamento")
    args = ap.parse_args()

    info = json.loads(urllib.request.urlopen(BASE + "/lote/info", timeout=30).read())
    N = info["envs"]
    print("[sim] %d ambientes | B = %d passos do Piloto | raio %d"
          % (N, args.k_alvo, args.raio), flush=True)

    pos = []
    while len(pos) < args.posicoes:
        pos += [(o["estado"]["x"], o["estado"]["z"])
                for o in post("/lote/reset", {})["obs"]]
    pos = pos[:args.posicoes]

    # Alvos: uma vez so, compartilhados por todas as politicas
    largadas, alvos = [], []
    for b0 in range(0, len(pos), N):
        lg, al = sortear_alvos(pos[b0:b0 + N], args.k_alvo, args.raio)
        largadas += lg; alvos += al
    d0 = [math.hypot(alvos[i][0] - largadas[i][0], alvos[i][1] - largadas[i][1])
          for i in range(len(pos))]
    print("[alvos] %d pares | distancia media %.1f blocos (mediana %.1f, min %.1f, max %.1f)"
          % (len(pos), st.mean(d0), st.median(d0), min(d0), max(d0)), flush=True)

    uteis = sum(1 for d in d0 if d >= 5.0)
    print("[alvos] %d de %d com B a >=5 blocos da largada" % (uteis, len(pos)), flush=True)
    print()

    ctx = None
    if "modelo" in args.politicas.split(","):
        ctx = carregar_modelo(args.ckpt, args.forcar_w)
    n_lotes = (len(pos) + N - 1) // N
    for nome in args.politicas.split(","):
        t0 = time.time()
        res = []
        for k, b0 in enumerate(range(0, len(pos), N), start=1):
            res += episodio(nome, pos[b0:b0 + N], alvos[b0:b0 + N],
                            args.passos, args.raio, ctx, args.diagnosticar)
            # Progresso por lote: uma corrida longa sem isto e cega, e o
            # pipe do shell bufferiza a saida ate o fim.
            feito = sum(r["chegou"] for r in res)
            print("    [%s] lote %d/%d | %d chegaram de %d | %.0fs"
                  % (nome, k, n_lotes, feito, len(res), time.time() - t0), flush=True)
        resumo(nome, res)
        print("           (%.0fs)" % (time.time() - t0), flush=True)
        if args.diagnosticar:
            diagnosticar(nome, res, args.passos)
    print()


def diagnosticar(nome, res, passos):
    """Separa as falhas em 'acabou o orcamento' e 'estacionou'."""
    falhas = [r for r in res if not r["chegou"]]
    print("\n  --- diagnostico das falhas de '%s' (%d de %d) ---"
          % (nome, len(falhas), len(res)))
    if not falhas:
        return
    orc = [r for r in falhas if r["t_recorde"] >= 0.9 * passos]
    est = [r for r in falhas if r["t_recorde"] < 0.5 * passos]
    print("  ultimo recorde de distancia, em %% do episodio:")
    for r in sorted(falhas, key=lambda r: r["t_recorde"]):
        curva = " ".join("%.0f" % v for v in r["traco"][::2])
        ag = "agua %.1f" % r["agua_fim"] if r["agua_fim"] is not None else "sem agua"
        pf = r.get("perfil") or []
        vals = [v for v in pf if v is not None]
        if not vals:
            rel = "relevo ?"
        elif max(vals) >= 2:
            rel = "PAREDE +%d" % max(vals)
        elif min(vals) <= -4:
            rel = "ABISMO %d" % min(vals)
        else:
            rel = "plano"
        nulos = sum(1 for v in pf if v is None)
        if nulos >= 6:
            rel = "NAO CARREGADO (%d/12)" % nulos
        dy = r.get("dy_alvo")
        sdy = ("dy %+5.1f" % dy) if dy is not None else "dy ?"
        print("    d0 %6.1f -> dmin %6.1f | recorde %3d (%3.0f%%) | %-9s | %-11s | %-10s | %s"
              % (r["d0"], r["dmin"], r["t_recorde"],
                 100 * r["t_recorde"] / passos, ag, rel, sdy, curva[:24]))
    perto = [r for r in falhas if r["dmin"] <= 10]
    print("  PERTO do alvo no fim (dmin <= 10 blocos): %d de %d" % (len(perto), len(falhas)))
    quase = [r for r in falhas if r["dmin"] <= 2 * RAIO_CHEGADA]
    print("  A menos de 2x o raio de chegada (%.1f): %d de %d  <- artefato do limiar"
          % (2 * RAIO_CHEGADA, len(quase), len(falhas)))
    print("  ORCAMENTO (ainda melhorava no fim, >=90%%): %d de %d" % (len(orc), len(falhas)))
    print("  ESTACIONOU (parou de melhorar antes da metade): %d de %d" % (len(est), len(falhas)))
    com_agua = [r for r in falhas if r["agua_fim"] is not None]
    print("  COM AGUA a <=6 blocos onde empacou: %d de %d" % (len(com_agua), len(falhas)))
    ok = [r for r in res if r["chegou"] and r["agua_fim"] is not None]
    print("  (controle: %d de %d que CHEGARAM tambem tinham agua perto no fim)"
          % (len(ok), sum(1 for r in res if r["chegou"])))

    def classe(r):
        pf = r.get("perfil") or []
        vals = [v for v in pf if v is not None]
        if sum(1 for v in pf if v is None) >= 6:
            return "nao carregado"
        if not vals:
            return "?"
        if max(vals) >= 2:
            return "parede"
        if min(vals) <= -4:
            return "abismo"
        return "plano"
    from collections import Counter
    cf = Counter(classe(r) for r in falhas)
    co = Counter(classe(r) for r in res if r["chegou"] and r.get("perfil"))
    print("  relevo rumo ao alvo — FALHAS:    %s" % dict(cf))
    print("  relevo rumo ao alvo — CHEGARAM:  %s   (controle)" % dict(co))

    # ELEVACAO: quanto o alvo esta ACIMA de onde o bot empacou. O BFS so aceita
    # degrau de 1 bloco, entao B no alto de um paredao e inalcancavel por terra.
    dyf = [-r["dy_alvo"] for r in falhas if r.get("dy_alvo") is not None]
    dyo = [-r["dy_alvo"] for r in res
           if r["chegou"] and r.get("dy_alvo") is not None]
    if dyf:
        print("  alvo ACIMA do bot (blocos) — FALHAS:   media %+.1f | max %+.1f | >=2: %d de %d"
              % (st.mean(dyf), max(dyf), sum(1 for v in dyf if v >= 2), len(dyf)))
    if dyo:
        print("  alvo ACIMA do bot (blocos) — CHEGARAM: media %+.1f | max %+.1f | >=2: %d de %d"
              % (st.mean(dyo), max(dyo), sum(1 for v in dyo if v >= 2), len(dyo)))


if __name__ == "__main__":
    main()
