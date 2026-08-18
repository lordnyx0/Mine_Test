# coding=utf-8
"""
Recompensa de locomoção baseada na posição real do bot (Mineflayer HTTP).

O sinal é o PROGRESSO NA DIREÇÃO QUE O BOT ENCARA:

    progresso = deslocamento · direção_de_visão(yaw no início do passo)

Isso é o que define "andar para frente". A versão anterior usava só o módulo
da velocidade, o que produzia três formas de reward hacking:

  1. "S" (andar de ré) ganhava +1.0, igual a "W".
  2. Girar a câmera parado ganhava +0.3 INCONDICIONALMENTE — o ótimo degenerado.
     Num log real de 150 passos, 139 tinham reward positivo e 100% delas eram
     "nenhuma tecla pressionada": o agente aprendeu a ficar parado girando.
  3. Ficar preso numa parede com W dava -1.0 e o exemplo era descartado, mesmo
     sendo a ação certa.

Agora: andar de ré é negativo, girar parado é ~0, e travar com W é penalizado
de leve (incentiva virar para sair do obstáculo).
"""
import math
import json
import urllib.request

import numpy as np
from PIL import Image

MINEFLAYER_URL = "http://127.0.0.1:3001"
HTTP_TIMEOUT   = 0.15   # 150ms max por consulta, para não travar o loop

# Velocidade de caminhada no Minecraft ~4.3 blocos/s. Usada para normalizar
# o progresso para a faixa [-1, +1] independente da duração do passo.
WALK_SPEED_BPS = 4.3

# Abaixo disto o bot é considerado parado (ruído de física / sub-tick)
STUCK_EPS = 0.02


class PositionRewardEvaluator:
    """
    Recompensa = progresso na direção encarada, normalizado por passo.

    Mantém a posição do passo anterior internamente, então a janela de medição
    é exatamente um passo do agente (e não um physics tick solto como no /delta).
    """

    def __init__(self):
        self._server_available = None
        self._prev = None          # (x, z, yaw_rad, t_ms)
        self._last_frame = None
        self._check_server()

    # ── Servidor ──────────────────────────────────────────────────────────────
    def _check_server(self):
        try:
            with urllib.request.urlopen(f"{MINEFLAYER_URL}/health", timeout=1.0) as r:
                data = json.loads(r.read())
                self._server_available = data.get("ok", False)
                if self._server_available:
                    print("[Reward] Mineflayer detectado. Usando progresso real na direcao encarada.")
                else:
                    print("[Reward] Mineflayer conectado mas bot ainda nao spawnou.")
        except Exception:
            self._server_available = False
            print("[Reward] Mineflayer indisponivel. Usando fallback visual.")

    def _query_state(self):
        """Retorna (x, z, yaw_rad, t_ms) ou None."""
        try:
            with urllib.request.urlopen(f"{MINEFLAYER_URL}/state", timeout=HTTP_TIMEOUT) as r:
                s = json.loads(r.read())
            return (float(s["x"]), float(s["z"]),
                    math.radians(float(s["yaw"])), float(s["timestamp"]))
        except Exception:
            return None

    # ── Fallback visual ───────────────────────────────────────────────────────
    def _visual_reward(self, current_frame: Image.Image, action_dict: dict) -> float:
        if self._last_frame is None:
            self._last_frame = current_frame
            return 0.0
        arr_prev = np.array(self._last_frame.convert("L"), dtype=np.float32)
        arr_curr = np.array(current_frame.convert("L"), dtype=np.float32)
        self._last_frame = current_frame

        pixel_diff = float(np.mean(np.abs(arr_curr - arr_prev)))
        hold = action_dict.get("hold", [])
        if "W" in hold:
            if pixel_diff >= 8.0: return 1.0
            if pixel_diff < 2.5:  return -0.3
            return 0.5
        return 0.0

    # ── Medição de progresso ──────────────────────────────────────────────────
    def measure(self):
        """
        Retorna (progresso_frente, desvio_lateral, dt_s) em blocos desde a
        última chamada, ou None. `progresso_frente` é negativo quando o bot
        anda para trás em relação a como estava virado no início do passo.
        """
        st = self._query_state()
        if st is None:
            return None
        x, z, yaw, t = st
        if self._prev is None:
            self._prev = st
            return None

        px, pz, pyaw, pt = self._prev
        self._prev = st

        dt = max(1e-3, (t - pt) / 1000.0)
        dx, dz = x - px, z - pz

        # Direção que o bot encarava NO INÍCIO do passo (mesma convenção do
        # mineflayer: forward = (-sin yaw, -cos yaw) no plano horizontal)
        fx, fz = -math.sin(pyaw), -math.cos(pyaw)
        forward = dx * fx + dz * fz
        lateral = dx * (-fz) + dz * fx
        return forward, lateral, dt

    # ── Recompensa ────────────────────────────────────────────────────────────
    def compute_reward(self, current_frame: Image.Image, action_dict: dict):
        """
        Retorna (reward, fonte). Reward contínuo em [-1, +1]:
          +1.0  andou um passo inteiro para frente
           0.0  girou no lugar / parado
          -1.0  andou um passo inteiro para trás
          -0.3  segurou W e não saiu do lugar (obstáculo)
        """
        if not self._server_available:
            self._check_server()

        m = self.measure() if self._server_available else None

        if m is not None:
            self._server_available = True
            forward, lateral, dt = m

            # Normaliza pelo máximo que daria para andar nesse intervalo
            cap = max(0.05, WALK_SPEED_BPS * dt)
            reward = max(-1.0, min(1.0, forward / cap))

            hold = action_dict.get("hold", [])
            moved = math.hypot(forward, lateral)
            if "W" in hold and moved < STUCK_EPS:
                reward = -0.3   # travado: vale a pena virar para escapar

            return float(reward), "position"

        return self._visual_reward(current_frame, action_dict), "visual"


if __name__ == "__main__":
    import time
    from bot_vision_capture import send_action

    ev = PositionRewardEvaluator()
    img = Image.new("RGB", (224, 224))

    testes = [
        ("andar pra FRENTE (W)",     {"hold": ["W"],   "mouse": [0, 0],  "duration_ms": 400}),
        ("andar pra TRAS (S)",       {"hold": ["S"],   "mouse": [0, 0],  "duration_ms": 400}),
        ("so GIRAR camera (parado)", {"hold": [],      "mouse": [30, 0], "duration_ms": 400}),
        ("PARADO total",             {"hold": [],      "mouse": [0, 0],  "duration_ms": 400}),
        ("strafe lateral (D)",       {"hold": ["D"],   "mouse": [0, 0],  "duration_ms": 400}),
    ]
    print()
    print("%-28s %8s   %s" % ("acao", "reward", "entra no dataset?"))
    print("-" * 72)
    for nome, a in testes:
        rs = []
        for _ in range(6):
            send_action(a)
            time.sleep(0.3)
            r, _src = ev.compute_reward(img, a)
            rs.append(r)
        m = sum(rs) / len(rs)
        print("%-28s %+8.2f   %s" % (nome, m, "SIM" if m > 0.1 else "nao"))
