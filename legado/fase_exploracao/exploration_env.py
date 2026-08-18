# coding=utf-8
"""
Ambiente de EXPLORAÇÃO com objetivo explícito.

Tarefa: nascer num ponto aleatório do mundo e se afastar o máximo possível
desse ponto.

    afastou-se da origem ANDANDO PARA FRENTE -> recompensa proporcional + bônus
    afastou-se de re                         -> penalizado (cega o agente)
    aproximou-se da origem                   -> penalidade proporcional
    parado                     -> penalidade
    girando na mesma região    -> penalidade crescente (anti-campismo)
    60s sem bater o recorde    -> respawn em outro ponto aleatório

O agente recebe o objetivo de duas formas, porque ele vai precisar distinguir
"explorar" de "coletar madeira" de "construir" mais adiante:

  1. INSTRUÇÃO EM TEXTO, tokenizada e passada como `input_ids` ao Qwen. O
     backbone é um modelo de linguagem: essa é a via natural para dizer a ele
     o que fazer, e generaliza para tarefas novas sem trocar a arquitetura.
  2. VETOR DE ESTADO de 16 dims com o andamento da tarefa (distância atual,
     quanto falta para o recorde, tempo sem progresso, vida, fome, se está no
     chão, para onde está virado...). Antes isso era `torch.zeros(1,16)`.
"""
import json
import math
import time
import urllib.request
from collections import deque

BASE_URL = "http://127.0.0.1:3001"

WALK_SPEED_BPS = 4.3     # velocidade de caminhada no Minecraft
CELULA = 8               # tamanho da célula da grade de visitação (blocos)
PARADO_EPS = 0.02        # abaixo disso o bot é considerado imóvel

# Bônus leve por andar PARA FRENTE, somado por cima do termo principal.
# Objetivo: criar tendência. O termo principal (afastar-se da origem) é neutro
# entre "andou de frente" e "andou de lado" desde que o raio cresça; este bônus
# desempata a favor de avançar de frente, que é o modo em que a câmera aponta
# para onde ele vai e a visão serve para desviar.
# Mantido pequeno de proposito: 0.15 contra 1.0 do termo principal, entao ele
# inclina sem sequestrar o objetivo de exploracao.
BONUS_FRENTE = 0.15

# Deslocamento máximo fisicamente possível num passo (~0.3s a 4.3 b/s ≈ 1.3b;
# com sprint/queda vai a ~4b). Acima disso NÃO foi locomoção: foi morte com
# respawn no spawn do mundo, ou teleporte. Tratar isso como progresso dava
# reward máximo por morrer e inflava a métrica em centenas de blocos.
SALTO_MAX = 12.0

# ── Memória temporal ──────────────────────────────────────────────────────────
# Janela de 15s. Cada passo custa ~0.55s de relógio, então 15s ≈ 27 passos.
# Empilhar 27 frames seria 864 tokens visuais no Qwen — inviável. Em vez disso,
# duas vias que cobrem escalas diferentes:
#   (a) 3 frames esparsos (agora, ~4s, ~15s) -> o agente VÊ que a árvore à
#       frente não mudou de tamanho, ou seja, que está encravado nela;
#   (b) resumo numérico da trajetória dos 15s -> deslocamento líquido, caminho
#       percorrido, e a razão entre os dois (eficiência), que é exatamente o
#       sinal de "estou andando em círculo".
JANELA_S = 15.0
ATRASOS_S = (0.0, 4.0, 15.0)   # instantes amostrados, do mais recente ao mais antigo
N_FRAMES = len(ATRASOS_S)


# ── Objetivos ─────────────────────────────────────────────────────────────────
# Um catálogo, não uma string solta: quando entrarem "coletar madeira" e
# "construir", cada uma vira uma entrada aqui com sua própria recompensa.
OBJETIVOS = {
    "explorar": (
        "Objetivo: explorar. Ande o maximo que puder e afaste-se o maximo "
        "possivel do ponto onde voce nasceu. Nao fique parado no mesmo lugar."
    ),
    "coletar_madeira": (
        "Objetivo: coletar madeira. Encontre uma arvore e quebre o tronco."
    ),
    "coletar_pedra": (
        "Objetivo: coletar pedra. Encontre pedra exposta ou cave ate achar."
    ),
}


class GoalEncoder:
    """Tokeniza a instrução da tarefa uma única vez e guarda os input_ids."""

    def __init__(self, tokenizer_path=r"C:\Users\Nyx\Desktop\Testes\models_quantized\final_28l_hf",
                 max_tokens=32, device="cpu"):
        from transformers import AutoTokenizer
        import torch
        self.torch = torch
        self.device = device
        self.max_tokens = max_tokens
        try:
            self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self._cache = {}

    def ids(self, objetivo: str):
        """Retorna input_ids [1, T] para o objetivo (com cache)."""
        if objetivo not in self._cache:
            texto = OBJETIVOS.get(objetivo, objetivo)
            enc = self.tok(texto, return_tensors="pt", truncation=True,
                           max_length=self.max_tokens)
            self._cache[objetivo] = enc["input_ids"].to(self.device)
        return self._cache[objetivo]


# ── Ambiente ──────────────────────────────────────────────────────────────────
class ExplorationEnv:
    def __init__(self, capturer, base_url=BASE_URL, raio_respawn=1500,
                 segundos_sem_progresso=60.0, verbose=True):
        self.capturer = capturer
        self.base = base_url
        self.raio = raio_respawn
        self.limite_sem_progresso = segundos_sem_progresso
        self.verbose = verbose

        self.objetivo = "explorar"
        self.origem = None
        self.d_recorde = 0.0
        self.d_anterior = 0.0
        self.t_ultimo_recorde = 0.0
        self.visitas = {}
        self.passos = 0
        self.parado_seguido = 0
        self.mortes = 0
        self._st_anterior = None
        self.historico = deque()     # (t, x, z, yaw, img)

    # ── HTTP ──────────────────────────────────────────────────────────────────
    # Com retry: o treino roda horas sem supervisao, e um soluco do servidor
    # (reconexao do bot, GC do node) nao pode derrubar a sessao inteira.
    def _get(self, path, timeout=3.0, tentativas=4):
        ultimo = None
        for i in range(tentativas):
            try:
                with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
                    return json.loads(r.read())
            except Exception as e:
                ultimo = e
                time.sleep(0.4 * (i + 1))
        raise RuntimeError(f"GET {path} falhou apos {tentativas} tentativas: {ultimo}")

    def _post(self, path, body=None, timeout=90.0):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(self.base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def rotas(self):
        """
        Navegabilidade real ao redor (12 setores, 0=parede colada, 1=livre).
        E o ALVO que o modelo aprende a prever a partir so da imagem.
        """
        try:
            return self._get("/rotas", timeout=2.0, tentativas=2)["livre"]
        except Exception:
            return None

    def estado(self):
        """Estado do bot. Se o servidor sumir de vez, devolve o ultimo conhecido
        para o loop de treino sobreviver ate ele voltar."""
        try:
            s = self._get("/state")
            self._ultimo_estado_ok = s
            return s
        except RuntimeError as e:
            if getattr(self, "_ultimo_estado_ok", None) is not None:
                print(f"  [env] AVISO: /state indisponivel ({e}); usando ultimo estado", flush=True)
                return self._ultimo_estado_ok
            raise

    # ── Episódio ──────────────────────────────────────────────────────────────
    def reset(self, respawnar=True):
        """Nasce num ponto aleatório e zera a origem do episódio."""
        if respawnar:
            try:
                r = self._post(f"/respawn?raio={self.raio}")
                if not r.get("ok") and self.verbose:
                    print(f"  [env] respawn falhou: {r.get('erro')}", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [env] respawn erro: {e}", flush=True)

        # Espera assentar no chão e os chunks chegarem
        for _ in range(24):
            time.sleep(0.25)
            try:
                s = self.estado()
            except Exception:
                continue
            if s.get("on_ground"):
                break

        s = self.estado()
        self.origem = (s["x"], s["z"])
        self.d_recorde = 0.0
        self.d_anterior = 0.0
        self.t_ultimo_recorde = time.time()
        self.visitas = {}
        self.passos = 0
        self.parado_seguido = 0
        self.historico.clear()       # memória é POR EPISÓDIO: o passado de outro
                                     # ponto do mundo não informa nada aqui
        self._st_anterior = s
        if self.verbose:
            print(f"  [env] novo episodio na origem ({s['x']:.0f}, {s['z']:.0f})", flush=True)
        return self.observar(s)

    def _distancia_da_origem(self, s):
        return math.hypot(s["x"] - self.origem[0], s["z"] - self.origem[1])

    # ── Observação ────────────────────────────────────────────────────────────
    def observar(self, s=None):
        """Retorna (pilha_de_frames, state_vec). A pilha tem N_FRAMES imagens."""
        if s is None:
            s = self.estado()
        img = self.capturer.capture()
        self._registrar(s, img)
        return self.pilha_de_frames(), self.vetor_de_estado(s)

    # ── Memória temporal ──────────────────────────────────────────────────────
    def _registrar(self, s, img):
        """Guarda o instante atual e descarta o que saiu da janela de 15s."""
        agora = time.time()
        self.historico.append((agora, s["x"], s["z"], math.radians(s["yaw"]), img))
        while self.historico and agora - self.historico[0][0] > JANELA_S * 1.4:
            self.historico.popleft()

    def pilha_de_frames(self):
        """
        Frames nos atrasos de ATRASOS_S, do mais recente ao mais antigo.
        Se o episódio ainda é novo, repete o mais antigo disponível — assim a
        forma do tensor nunca muda e o modelo não vê buracos.
        """
        if not self.historico:
            vazio = self.capturer.capture()
            return [vazio] * N_FRAMES
        agora = self.historico[-1][0]
        pilha = []
        for atraso in ATRASOS_S:
            alvo = agora - atraso
            # entrada mais próxima do instante desejado
            melhor = min(self.historico, key=lambda h: abs(h[0] - alvo))
            pilha.append(melhor[4])
        return pilha

    def _resumo_trajetoria(self):
        """
        Resumo numérico dos 15s: deslocamento líquido, caminho percorrido e a
        razão entre eles. Caminho longo com deslocamento pequeno = andando em
        círculo, que é o que a penalidade de campismo quer capturar mas o
        agente não conseguia perceber sozinho.
        """
        if len(self.historico) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        agora = self.historico[-1][0]
        pts = [h for h in self.historico if agora - h[0] <= JANELA_S]
        if len(pts) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        caminho = 0.0
        for a, b in zip(pts, pts[1:]):
            caminho += math.hypot(b[1] - a[1], b[2] - a[2])
        liquido = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
        eficiencia = liquido / caminho if caminho > 1e-6 else 0.0

        # variação de direção acumulada (o quanto girou nos 15s)
        giro = 0.0
        for a, b in zip(pts, pts[1:]):
            d = (b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi
            giro += abs(d)

        janela = agora - pts[0][0]
        return liquido, caminho, eficiencia, giro, janela

    def vetor_de_estado(self, s):
        """
        16 dims normalizadas com o que o agente precisa saber sobre a TAREFA e
        sobre o próprio corpo. Isto é o que dá a ele "noção do que está
        acontecendo" além do pixel.
        """
        d = self._distancia_da_origem(s)
        yaw = math.radians(s["yaw"])
        cel = self._celula(s)
        visitas = self.visitas.get(cel, 0)
        sem_progresso = time.time() - self.t_ultimo_recorde

        def cl(v, lo=-1.0, hi=1.0):
            return max(lo, min(hi, v))

        return [
            cl(d / 500.0),                                  # 0  distancia da origem
            cl((d - self.d_recorde) / 50.0),                # 1  quanto abaixo do recorde
            cl(self.d_recorde / 500.0),                     # 2  recorde do episodio
            cl(sem_progresso / self.limite_sem_progresso),  # 3  tempo sem bater recorde
            cl(max(0, visitas - 12) / 20.0),                # 4  campismo na celula atual
            cl(self.parado_seguido / 10.0),                 # 5  passos parado seguidos
            cl(s.get("health", 20) / 20.0),                 # 6  vida
            cl(s.get("food", 20) / 20.0),                   # 7  fome
            1.0 if s.get("on_ground") else 0.0,             # 8  no chao
            math.sin(yaw),                                  # 9  direcao (sin)
            math.cos(yaw),                                  # 10 direcao (cos)
            cl(s.get("pitch", 0) / 90.0),                   # 11 pitch
            cl(s.get("vx", 0) / 0.5),                       # 12 velocidade x
            cl(s.get("vz", 0) / 0.5),                       # 13 velocidade z
            cl(s.get("y", 64) / 128.0),                     # 14 altura
            cl(self.passos / 200.0),                        # 15 idade do episodio
        ] + self._dims_temporais()

    def _dims_temporais(self):
        """16 dims de MEMÓRIA: o que aconteceu na janela de 15s."""
        liquido, caminho, eficiencia, giro, janela = self._resumo_trajetoria()

        def cl(v, lo=-1.0, hi=1.0):
            return max(lo, min(hi, v))

        # amostras da trajetória relativa, em coordenadas centradas no bot
        pontos = []
        if len(self.historico) >= 2:
            agora, x0, z0, yaw0, _ = self.historico[-1]
            fx, fz = -math.sin(yaw0), -math.cos(yaw0)
            for atraso in (2.0, 5.0, 10.0, 15.0):
                alvo = agora - atraso
                h = min(self.historico, key=lambda k: abs(k[0] - alvo))
                dx, dz = h[1] - x0, h[2] - z0
                # projeta no referencial do bot: frente e lado
                pontos += [cl((dx * fx + dz * fz) / 30.0),
                           cl((dx * (-fz) + dz * fx) / 30.0)]
        pontos += [0.0] * (8 - len(pontos))

        return [
            cl(liquido / 40.0),        # 16 deslocamento liquido nos 15s
            cl(caminho / 60.0),        # 17 caminho percorrido nos 15s
            cl(eficiencia),            # 18 liquido/caminho: 1=reto, 0=circulo
            cl(giro / (4 * math.pi)),  # 19 quanto girou acumulado
            cl(janela / JANELA_S),     # 20 quanto da janela ja esta preenchida
            cl(len(self.historico) / 40.0),  # 21 densidade de amostras
            cl(self.mortes / 5.0),     # 22 mortes no episodio
            0.0,                       # 23 reservado
        ] + pontos                     # 24-31 trajetoria relativa (4 pontos x2)

    def _celula(self, s):
        return (int(s["x"] // CELULA), int(s["z"] // CELULA))

    # ── Passo ─────────────────────────────────────────────────────────────────
    def step(self, action_dict, espera=0.25):
        """
        Envia a ação, espera, mede. Retorna (img, state_vec, reward, respawnou, info).
        """
        from bot_vision_capture import send_action

        s0 = self._st_anterior or self.estado()
        t0 = time.time()
        send_action(action_dict)
        time.sleep(espera)
        s1 = self.estado()
        dt = max(1e-3, time.time() - t0)

        d1 = self._distancia_da_origem(s1)
        ganho = d1 - self.d_anterior
        deslocamento = math.hypot(s1["x"] - s0["x"], s1["z"] - s0["z"])

        # ── Descontinuidade: morte ou teleporte ──────────────────────────────
        if deslocamento > SALTO_MAX:
            self.mortes += 1
            recorde_perdido = self.d_recorde
            if self.verbose:
                print(f"  [env] salto de {deslocamento:.0f}b detectado "
                      f"(morte/teleporte). Episodio encerrado, recorde {recorde_perdido:.1f}b",
                      flush=True)
            # Reinicia o episodio AQUI, sem respawnar de novo, e nao recompensa
            self.reset(respawnar=False)
            info = {
                "distancia": 0.0, "recorde": recorde_perdido, "ganho": 0.0,
                "deslocamento": 0.0, "campismo": 0.0, "sem_progresso_s": 0.0,
                "descontinuidade": True,
            }
            img_d = self.capturer.capture()
            self._registrar(self._st_anterior, img_d)
            return (self.pilha_de_frames(), self.vetor_de_estado(self._st_anterior),
                    0.0, True, info)

        # ── Recompensa ───────────────────────────────────────────────────────
        # Duas componentes do MESMO deslocamento:
        #   radial = quanto se afastou da origem   (o objetivo da tarefa)
        #   frente = quanto andou para onde olha   (o modo de fazer isso)
        #
        # Usar so o radial e agnostico a direcao: afastar-se de re pontua igual
        # a afastar-se de frente. E andar de re e degenerado — a camera aponta
        # para onde ele JA esteve, entao a imagem para de prever o resultado da
        # acao, a visao desacopla e ele para de desviar de obstaculos.
        #
        # min(radial, frente) so credita o avanco que foi simultaneamente
        # afastamento E movimento para frente:
        #   frente, afastando  -> ambos altos   -> recompensa cheia
        #   re, afastando      -> frente < 0    -> penalizado
        #   frente, voltando   -> radial < 0    -> penalizado
        #   strafe             -> frente ~ 0    -> ~neutro
        yaw0 = math.radians(s0["yaw"])
        fx, fz = -math.sin(yaw0), -math.cos(yaw0)
        frente = (s1["x"] - s0["x"]) * fx + (s1["z"] - s0["z"]) * fz
        radial = ganho

        cap = max(0.05, WALK_SPEED_BPS * dt)
        avanco_util = min(radial, frente)
        reward = max(-1.0, min(1.0, avanco_util / cap))

        # Desempate a favor de avançar de frente
        if frente > 0:
            reward += BONUS_FRENTE * min(1.0, frente / cap)

        if deslocamento < PARADO_EPS:
            self.parado_seguido += 1
            reward -= 0.30                           # parado é sempre ruim
        else:
            self.parado_seguido = 0

        # Anti-campismo: conta passos gastos dentro da mesma célula. A franquia
        # precisa cobrir a travessia normal — atravessar 8 blocos a ~0.8 b/passo
        # leva ~10 passos, e punir isso seria punir andar em linha reta.
        cel = self._celula(s1)
        self.visitas[cel] = self.visitas.get(cel, 0) + 1
        campismo = min(1.0, max(0.0, (self.visitas[cel] - 12) / 20.0))
        reward -= 0.35 * campismo                    # rodar em círculo na mesma área

        # Teto sobe junto com o bônus, senão andar de frente rápido seria
        # cortado no mesmo 1.0 de andar de lado e o desempate sumiria.
        reward = max(-1.5, min(1.0 + BONUS_FRENTE, reward))

        # ── Progresso / recorde ──────────────────────────────────────────────
        bateu_recorde = d1 > self.d_recorde + 0.5
        if bateu_recorde:
            self.d_recorde = d1
            self.t_ultimo_recorde = time.time()

        self.d_anterior = d1
        self._st_anterior = s1
        self.passos += 1

        # ── Respawn por estagnação ───────────────────────────────────────────
        sem_progresso = time.time() - self.t_ultimo_recorde
        respawnou = False
        # Guarda o recorde ANTES do reset: é ele a métrica do episódio que
        # acabou. Ler self.d_recorde depois do reset devolveria 0 e afundaria
        # a curva de aprendizado com zeros falsos.
        recorde_do_episodio = self.d_recorde

        if sem_progresso >= self.limite_sem_progresso:
            if self.verbose:
                print(f"  [env] {self.limite_sem_progresso:.0f}s sem progresso "
                      f"(recorde {self.d_recorde:.1f}b) -> respawn", flush=True)
            self.reset(respawnar=True)
            respawnou = True

        info = {
            "distancia": d1,
            "recorde": recorde_do_episodio,
            "ganho": ganho,
            "deslocamento": deslocamento,
            "campismo": campismo,
            "sem_progresso_s": sem_progresso,
            "radial": radial,
            "frente": frente,
            "rotas": self.rotas(),
            "descontinuidade": False,
        }
        img = self.capturer.capture()
        self._registrar(self._st_anterior, img)
        return (self.pilha_de_frames(), self.vetor_de_estado(self._st_anterior),
                reward, respawnou, info)


if __name__ == "__main__":
    # Teste do ambiente com uma política scriptada "sempre para frente"
    from bot_vision_capture import BotVisionCapture

    cap = BotVisionCapture()
    env = ExplorationEnv(cap, segundos_sem_progresso=25.0)

    ge = GoalEncoder()
    print("objetivo tokenizado:", ge.ids("explorar").shape)

    env.reset()
    total = 0.0
    for i in range(40):
        _img, sv, r, resp, info = env.step(
            {"hold": ["W"], "mouse": [0, 0], "duration_ms": 250})
        total += r
        if i % 5 == 0:
            print("  passo %2d | r=%+.2f | dist=%6.1f | recorde=%6.1f | campismo=%.2f"
                  % (i, r, info["distancia"], info["recorde"], info["campismo"]), flush=True)
    print(f"\nreward acumulado em 40 passos: {total:+.2f}")
