# coding=utf-8
"""
Gera o dataset de locomocao com FRAMES REAIS DO JOGO.

Substitui generate_locomotion_dataset.py, que desenhava as cenas com PIL
(Image.new + draw.rectangle/polygon/ellipse). Aquele dataset era arte de
placeholder: ceu chapado, chao chapado, um triangulo marrom. O modelo era
treinado nesses desenhos e executado contra o renderizador voxel real, entao
nada podia transferir — media 0/8 em obediencia a instrucao.

Cada amostra guarda o que o agente realmente ve e sabe no momento da decisao:
  - pilha de 3 frames (agora, -4s, -15s), como em execucao
  - state_vec de 32 dims
  - instrucao em texto
  - acao executada (o rotulo)
  - mapa de rotas de 12 setores (alvo denso, e o que faz a visao importar)

    python gerar_dataset_real.py --amostras 1200
"""
import os
import io
import json
import base64
import random
import argparse
import time
import math

from bot_vision_capture import BotVisionCapture, send_action
from exploration_env import ExplorationEnv

SAIDA = "dataset/locomocao_real.jsonl"

OBJETIVO_EXPLORAR = ("Objetivo: explorar. Ande o maximo que puder e afaste-se o maximo "
                     "possivel do ponto onde voce nasceu. Nao fique parado no mesmo lugar.")

# Comando -> acao. Os de camera usam os bins que a politica realmente emite.
COMANDOS = {
    "W":         {"hold": ["W"], "mouse": [0, 0]},
    "S":         {"hold": ["S"], "mouse": [0, 0]},
    "A":         {"hold": ["A"], "mouse": [0, 0]},
    "D":         {"hold": ["D"], "mouse": [0, 0]},
    "CAM_LEFT":  {"hold": [], "mouse": [-30, 0]},
    "CAM_RIGHT": {"hold": [], "mouse": [30, 0]},
    "CAM_UP":    {"hold": [], "mouse": [0, 12]},
    "CAM_DOWN":  {"hold": [], "mouse": [0, -12]},
}


# Bins de giro que a politica realmente emite (espelha IndependentActionHeads)
YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)   # = +-45 graus nas pontas
GRAUS_POR_UNIDADE = 0.003 * 180.0 / math.pi              # ~0.1719 graus/unidade


def unidades(graus):
    """
    Angulo ATE O ALVO (graus) -> unidades de mouse do /action.

    O sinal e invertido, medido em malha fechada: aplicar mouse=+graus levou o
    alvo de -38 para -76 graus (afastou), e mouse=-graus levou de -76 para 0.0
    (acertou em cheio). A magnitude tambem confere: 262 unidades = 45 graus.
    """
    return -graus / GRAUS_POR_UNIDADE

# Quao livre a frente precisa estar para valer a pena andar em vez de girar
LIMIAR_LIVRE = 0.55

# Abaixo disto ha parede colada: nao adianta tentar andar, so girar
PAREDE_COLADA = 0.18


class ProfessorEscape:
    """
    Professor de exploracao MEDIDO como o melhor disponivel.

    Regra: ande reto; se nao saiu do lugar em PASSOS_TRAVADO passos seguidos,
    vire (alternando o lado) e continue andando.

    Nao usa o mapa de rotas de proposito. Em comparacao pareada com 60 posicoes
    identicas, seguir o setor mais livre ficou -6.3 blocos ATRAS de andar cego
    (IC95% [-10.6,-2.0]): o mapa cobre 360 graus e o setor mais livre costuma
    ser ATRAS, de onde o bot veio, entao virar para la desfaz o afastamento.
    Ja o escape por propriocepcao rendeu +6.2 [+1.5,+10.8] sobre andar cego, e
    dobrou a mediana (10.5 -> 22.0).

    Depende so de "andei ou nao andei", que o agente tem na dim 5 do state_vec
    (parado_seguido) — ou seja, ele possui a entrada necessaria para imitar.
    """

    PASSOS_TRAVADO = 3
    GIRO = 262           # ~45 graus
    EPS_MOVIMENTO = 0.15

    def __init__(self):
        self.travado = 0
        self.lado = self.GIRO

    def acao(self, deslocamento):
        if deslocamento is not None and deslocamento < self.EPS_MOVIMENTO:
            self.travado += 1
        else:
            self.travado = 0

        if self.travado >= self.PASSOS_TRAVADO:
            self.lado = -self.lado          # alterna, para nao insistir no mesmo lado
            self.travado = 0
            return {"hold": ["W"], "mouse": [self.lado, 0], "duration_ms": 250}, "escapar"
        return {"hold": ["W"], "mouse": [0, 0], "duration_ms": 250}, "avancar"


def professor_de_rota(rotas, k=12):
    """
    Politica scriptada COMPETENTE que age a partir do mapa de navegabilidade.

    E ela que fecha o laco mapa -> comando. Sem isto o modelo aprende a prever
    rotas muito bem e nunca converte a previsao em acao: as duas cabecas sao
    lineares independentes no mesmo tronco, e nada obriga a politica a olhar
    para o que o previsor descobriu.

    OBSOLETO como politica: medido -6.3 blocos abaixo de andar cego. Mantido
    apenas porque o /rotas continua util como ALVO auxiliar de treino (impede
    o colapso visual) — mas nao deve ser seguido como plano. Use ProfessorEscape.
    """
    if not rotas or len(rotas) != k:
        return {"hold": ["W"], "mouse": [0, 0]}, "sem_mapa"

    if rotas[0] >= LIMIAR_LIVRE:
        return {"hold": ["W"], "mouse": [0, 0]}, "avancar"

    melhor = max(range(k), key=lambda i: rotas[i])
    graus = (melhor / k) * 360.0
    if graus > 180.0:
        graus -= 360.0                      # vira pelo lado mais curto
    alvo = min(YAW_BINS, key=lambda b: abs(b - unidades(graus)))

    # Girar PARADO desperdica o passo. Com o caminho apenas apertado (nao uma
    # parede colada), anda e vira ao mesmo tempo — que e o que uma pessoa faz,
    # e foi o que segurou o professor em 15.7 blocos na primeira versao.
    if rotas[0] >= PAREDE_COLADA:
        if alvo == 0:
            return {"hold": ["W"], "mouse": [0, 0]}, "avancar_apertado"
        return {"hold": ["W"], "mouse": [int(alvo), 0]}, "avancar_virando"

    if alvo == 0:
        # Parede na cara e o "melhor" e a frente: o mapa nao ajuda, forca giro
        alvo = -262 if rotas[k // 4] >= rotas[-k // 4] else 262
    return {"hold": [], "mouse": [int(alvo), 0]}, "girar"


def b64_jpeg(img, q=80):
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return base64.b64encode(b.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostras", type=int, default=1200)
    ap.add_argument("--respawn-cada", type=int, default=60,
                    help="troca de regiao a cada N amostras (diversidade de cena)")
    ap.add_argument("--saida", default=SAIDA)
    ap.add_argument("--modo", choices=["comandos", "rota"], default="comandos",
                    help="comandos = obedecer instrucao; rota = seguir o espaco livre")
    args = ap.parse_args()

    random.seed(0)
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    if os.path.exists(args.saida):
        os.remove(args.saida)

    cap = BotVisionCapture()
    env = ExplorationEnv(cap, segundos_sem_progresso=10 ** 9, verbose=False)
    env.reset(respawnar=True)

    nomes = list(COMANDOS)
    escritos, t0 = 0, time.time()

    with open(args.saida, "a", encoding="utf-8") as f:
        for i in range(args.amostras):
            if i % args.respawn_cada == 0 and i > 0:
                env.reset(respawnar=True)
                print(f"  [{i}] nova regiao", flush=True)

            # Observacao ANTES de agir — e sobre ela que a decisao seria tomada
            pilha = env.pilha_de_frames()
            estado = env.vetor_de_estado(env.estado())
            rotas = env.rotas()

            if args.modo == "rota":
                acao, motivo = professor_de_rota(rotas)
                acao = dict(acao)
                cmd = motivo
                # Instrucao de TAREFA, nao de direcao: e sob esta instrucao que
                # o agente vai operar em exploracao, entao e ela que precisa
                # estar associada ao comportamento de seguir o espaco livre.
                instrucao = OBJETIVO_EXPLORAR
            else:
                cmd = nomes[i % len(nomes)] if i % 3 else random.choice(nomes)
                acao = dict(COMANDOS[cmd])
                instrucao = f"Navigate direction {cmd}"
            acao["duration_ms"] = 250

            f.write(json.dumps({
                "task": "locomocao_real",
                "instrucao": instrucao,
                "action_type": cmd,
                "action": acao,
                "state_vec": [round(v, 5) for v in estado],
                "rotas": rotas,
                "frames_b64": [b64_jpeg(im) for im in pilha],
                "reward": 1.0,
            }, ensure_ascii=False) + "\n")
            escritos += 1

            env.step(acao)

            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                print("  %d/%d amostras | %.1f min | %.2fs por amostra"
                      % (i + 1, args.amostras, dt / 60, dt / (i + 1)), flush=True)

    print(f"\n[OK] {escritos} amostras REAIS em {args.saida}", flush=True)
    print(f"     {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
