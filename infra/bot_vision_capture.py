# coding=utf-8
"""
Bot Vision Capture — substitui vision_encoder.py (ScreenCapturer).
Captura frames da visão em 1ª pessoa do bot Mineflayer via GET /frame.

O servidor renderiza o frame com um raycaster voxel em Node puro
(mineflayer_server/voxel_renderer.js), lendo os blocos direto de bot.world.
~10ms por frame, contra 80-200ms do antigo prismarine-viewer + Puppeteer.

Sem captura de tela do Windows. Sem browser. Sem dependência de foco de janela.
"""
import io
import time
import urllib.request
import urllib.error
from PIL import Image

FRAME_URL   = "http://127.0.0.1:3001/frame"
HEALTH_URL  = "http://127.0.0.1:3001/health"
STATS_URL   = "http://127.0.0.1:3001/stats"
TIMEOUT_SEC = 0.2   # o render leva ~10ms; 200ms já é folga larga


class BotVisionCapture:
    """
    Captura frames da visão do bot Mineflayer via HTTP.
    Interface compatível com o antigo ScreenCapturer.
    """

    def __init__(self, frame_url: str = FRAME_URL):
        self.frame_url  = frame_url
        self._last_img  = None
        self._frame_ms  = 0.0
        self._render_ms = 0.0
        self._size      = (640, 360)
        self._timeouts  = 0
        self._frames    = 0
        self._wait_ready()

    def _wait_ready(self, max_wait: float = 30.0):
        """Aguarda o servidor + viewer ficarem prontos."""
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < max_wait:
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as r:
                    import json
                    data = json.loads(r.read())
                    if data.get("viewer_ready"):
                        print(f"[BotVision] Renderer '{data.get('renderer', '?')}' pronto! "
                              f"Iniciando captura de frames.")
                        return
                    elif data.get("ok"):
                        print("[BotVision] Bot conectado. Aguardando chunks chegarem...", flush=True)
                    else:
                        print("[BotVision] Aguardando bot spawnar...", flush=True)
            except Exception:
                print("[BotVision] Aguardando servidor Mineflayer em localhost:3001...", flush=True)
            time.sleep(1.5)
        print("[BotVision] AVISO: viewer pode nao estar pronto. Tentando mesmo assim.")

    def capture(self) -> Image.Image:
        """
        Retorna PIL Image (RGB) da visão em 1ª pessoa do bot.
        Usa o último frame disponível em caso de timeout.
        """
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(self.frame_url, timeout=TIMEOUT_SEC) as r:
                raw = r.read()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                self._last_img  = img
                self._size      = img.size
                self._frame_ms  = (time.perf_counter() - t0) * 1000.0
                self._frames   += 1
                # Tempo gasto só no render, reportado pelo servidor
                try:
                    self._render_ms = float(r.headers.get("X-Render-Ms", 0.0))
                except (TypeError, ValueError):
                    self._render_ms = 0.0
                return img
        except Exception:
            self._timeouts += 1
            # Fallback: retorna último frame disponível se o servidor demorou
            if self._last_img is not None:
                return self._last_img
            # Fallback final: imagem preta se nunca houve frame
            return Image.new("RGB", self._size, color=(0, 0, 0))

    @property
    def last_capture_ms(self) -> float:
        """Latência total do GET /frame (render + HTTP + decode)."""
        return self._frame_ms

    @property
    def last_render_ms(self) -> float:
        """Tempo gasto só no raycast + JPEG, medido pelo servidor."""
        return self._render_ms

    @property
    def timeout_count(self) -> int:
        return self._timeouts


def send_action(action_dict: dict, base_url: str = "http://127.0.0.1:3001") -> bool:
    """
    Envia uma ação ao bot Mineflayer via POST /action.
    Substitui execute_action() de direct_input.py.

    action_dict = {
        "hold":        ["W", "A", "S", "D", "SPACE", "SHIFT"],
        "mouse":       [dx, dy],
        "duration_ms": 100
    }
    """
    import json
    body = json.dumps(action_dict).encode("utf-8")
    req  = urllib.request.Request(
        f"{base_url}/action",
        data    = body,
        method  = "POST",
        headers = {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=0.2) as r:
            return r.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    cap = BotVisionCapture()
    img = cap.capture()
    print(f"[OK] Frame capturado: {img.size} | {cap.last_capture_ms:.1f}ms "
          f"(render {cap.last_render_ms:.1f}ms)")
    img.save("test_bot_frame.jpg")
    print("[OK] Salvo em test_bot_frame.jpg")

    # Benchmark de captura sustentada
    N = 60
    t0 = time.perf_counter()
    lat, ren = [], []
    for _ in range(N):
        t1 = time.perf_counter()
        cap.capture()
        lat.append((time.perf_counter() - t1) * 1000.0)
        ren.append(cap.last_render_ms)
    total = time.perf_counter() - t0
    lat.sort()
    print(f"[BENCH] {N} frames em {total:.2f}s -> {N/total:.1f} fps")
    print(f"[BENCH] latencia  avg={sum(lat)/N:.2f}ms  p50={lat[N//2]:.2f}ms  "
          f"p95={lat[int(N*0.95)]:.2f}ms  max={lat[-1]:.2f}ms")
    print(f"[BENCH] render    avg={sum(ren)/N:.2f}ms  (so o raycast+JPEG, no servidor)")
    print(f"[BENCH] timeouts  {cap.timeout_count}")

    # Teste de acao
    ok = send_action({"hold": ["W"], "mouse": [0, 0], "duration_ms": 200})
    print(f"[OK] Acao W enviada ao bot: {'sucesso' if ok else 'falhou'}")
