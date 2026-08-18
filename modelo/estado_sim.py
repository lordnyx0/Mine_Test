# coding=utf-8
"""
Vetor de estado de 32 dims para o simulador — MESMA definicao do
exploration_env.vetor_de_estado.

Existe uma implementacao so, usada tanto para gerar dataset quanto para rodar
a politica no simulador. Duplicar isso em JS no servidor criaria duas verdades
que divergiriam silenciosamente, e treino/execucao veriam entradas diferentes —
que foi exatamente o bug que fez a base decorar o treino e errar 8/8 na pratica.
"""
import os
import math
from collections import deque

JANELA_S = 15.0
PASSO_S = 0.25          # duracao de uma acao no simulador
CELULA = 8
WALK_SPEED_BPS = 4.3
SALTO_MAX = 12.0

# Frames da pilha temporal, em PASSOS.
#
# O padrao (0,16,60) = 0s, 4s, 15s serve a episodios longos de exploracao. Para
# controle local ele e DEGENERADO: com episodios de ~15 passos, pilha_frames()
# faz clamp e os frames 2 e 3 viram o MESMO frame inicial — dois tercos da
# visao gastos num duplicado. Para a Fase 1 use ATRASOS_SIM=0,2,4 (0s, 0.5s, 1s),
# que e a escala em que dinamica de movimento e correcao de rumo aparecem.
_env = os.environ.get("ATRASOS_SIM")
ATRASOS_PASSOS = tuple(int(v) for v in _env.split(",")) if _env else (0, 16, 60)
N_FRAMES = len(ATRASOS_PASSOS)


class EstadoEpisodio:
    """Acompanha um ambiente: origem, recorde, visitas, historico de 15s."""

    def __init__(self, limite_sem_progresso_s=60.0):
        self.limite = limite_sem_progresso_s
        self.reiniciar(None)

    def reiniciar(self, estado):
        self.origem = (estado["x"], estado["z"]) if estado else (0.0, 0.0)
        self.d_recorde = 0.0
        self.d_anterior = 0.0
        self.passos_sem_recorde = 0
        self.visitas = {}
        self.passos = 0
        self.parado_seguido = 0
        self.mortes = 0
        self.hist = deque(maxlen=80)      # (passo, x, z, yaw_rad)
        self.frames = deque(maxlen=80)    # bytes JPEG
        self._st = estado

    # ── Atualizacao ───────────────────────────────────────────────────────────
    def registrar(self, estado, frame_bytes=None):
        self._st = estado
        yaw = math.radians(estado["yaw"])
        self.hist.append((self.passos, estado["x"], estado["z"], yaw))
        if frame_bytes is not None:
            self.frames.append(frame_bytes)

    def passo(self, estado_novo, frame_bytes=None):
        """Avanca um passo. Retorna (reward, descontinuidade)."""
        s0 = self._st or estado_novo
        dx = estado_novo["x"] - s0["x"]
        dz = estado_novo["z"] - s0["z"]
        deslocamento = math.hypot(dx, dz)

        if deslocamento > SALTO_MAX:      # morte/teleporte
            self.mortes += 1
            self.reiniciar(estado_novo)
            self.registrar(estado_novo, frame_bytes)
            return 0.0, True

        d1 = math.hypot(estado_novo["x"] - self.origem[0],
                        estado_novo["z"] - self.origem[1])
        radial = d1 - self.d_anterior

        yaw0 = math.radians(s0["yaw"])
        fx, fz = -math.sin(yaw0), -math.cos(yaw0)
        frente = dx * fx + dz * fz

        cap = max(0.05, WALK_SPEED_BPS * PASSO_S)
        reward = max(-1.0, min(1.0, min(radial, frente) / cap))
        if frente > 0:
            reward += 0.15 * min(1.0, frente / cap)

        if deslocamento < 0.02:
            self.parado_seguido += 1
            reward -= 0.30
        else:
            self.parado_seguido = 0

        cel = (int(estado_novo["x"] // CELULA), int(estado_novo["z"] // CELULA))
        self.visitas[cel] = self.visitas.get(cel, 0) + 1
        campismo = min(1.0, max(0.0, (self.visitas[cel] - 12) / 20.0))
        reward -= 0.35 * campismo
        reward = max(-1.5, min(1.15, reward))

        if d1 > self.d_recorde + 0.5:
            self.d_recorde = d1
            self.passos_sem_recorde = 0
        else:
            self.passos_sem_recorde += 1

        self.d_anterior = d1
        self.passos += 1
        self.registrar(estado_novo, frame_bytes)
        return reward, False

    # ── Observacao ────────────────────────────────────────────────────────────
    def pilha_frames(self):
        """Frames nos atrasos de ATRASOS_PASSOS, do mais recente ao mais antigo."""
        if not self.frames:
            return []
        n = len(self.frames)
        return [self.frames[max(0, n - 1 - a)] for a in ATRASOS_PASSOS]

    def _resumo(self):
        if len(self.hist) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        limite = self.passos - int(JANELA_S / PASSO_S)
        pts = [h for h in self.hist if h[0] >= limite]
        if len(pts) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        caminho = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(pts, pts[1:]))
        liquido = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
        eficiencia = liquido / caminho if caminho > 1e-6 else 0.0
        giro = 0.0
        for a, b in zip(pts, pts[1:]):
            d = (b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi
            giro += abs(d)
        janela = (pts[-1][0] - pts[0][0]) * PASSO_S
        return liquido, caminho, eficiencia, giro, janela

    def vetor(self, estado=None):
        """As mesmas 32 dims de exploration_env.vetor_de_estado."""
        s = estado or self._st
        if s is None:
            return [0.0] * 32
        d = math.hypot(s["x"] - self.origem[0], s["z"] - self.origem[1])
        yaw = math.radians(s["yaw"])
        cel = (int(s["x"] // CELULA), int(s["z"] // CELULA))
        visitas = self.visitas.get(cel, 0)
        sem_prog_s = self.passos_sem_recorde * PASSO_S

        def cl(v, lo=-1.0, hi=1.0):
            return max(lo, min(hi, v))

        base = [
            cl(d / 500.0),
            cl((d - self.d_recorde) / 50.0),
            cl(self.d_recorde / 500.0),
            cl(sem_prog_s / self.limite),
            cl(max(0, visitas - 12) / 20.0),
            cl(self.parado_seguido / 10.0),
            cl(s.get("health", 20) / 20.0),
            cl(s.get("food", 20) / 20.0),
            1.0 if s.get("on_ground") else 0.0,
            math.sin(yaw),
            math.cos(yaw),
            cl(s.get("pitch", 0) / 90.0),
            cl(s.get("vx", 0) / 0.5),
            cl(s.get("vz", 0) / 0.5),
            cl(s.get("y", 64) / 128.0),
            cl(self.passos / 200.0),
        ]

        liquido, caminho, eficiencia, giro, janela = self._resumo()
        pontos = []
        if len(self.hist) >= 2:
            agora, x0, z0, yaw0 = self.hist[-1]
            fx, fz = -math.sin(yaw0), -math.cos(yaw0)
            for atraso_s in (2.0, 5.0, 10.0, 15.0):
                alvo = agora - atraso_s / PASSO_S
                h = min(self.hist, key=lambda k: abs(k[0] - alvo))
                ddx, ddz = h[1] - x0, h[2] - z0
                pontos += [cl((ddx * fx + ddz * fz) / 30.0),
                           cl((ddx * (-fz) + ddz * fx) / 30.0)]
        pontos += [0.0] * (8 - len(pontos))

        temporal = [
            cl(liquido / 40.0),
            cl(caminho / 60.0),
            cl(eficiencia),
            cl(giro / (4 * math.pi)),
            cl(janela / JANELA_S),
            cl(len(self.hist) / 40.0),
            cl(self.mortes / 5.0),
            0.0,
        ] + pontos

        return base + temporal
