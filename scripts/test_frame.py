"""Smoke test dos endpoints do Mineflayer VLA Server (roda com o servidor no ar)."""
import urllib.request
import json
import io
import time
from PIL import Image

BASE = "http://127.0.0.1:3001"

print("Testando endpoints do Mineflayer VLA Server...")

# /health
try:
    with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
        h = json.loads(r.read())
        print(f"[health] ok={h['ok']}  viewer_ready={h['viewer_ready']}  "
              f"renderer={h.get('renderer', '?')}")
except Exception as e:
    print(f"[health] ERRO: {e}")
    raise SystemExit(1)

# /delta
try:
    with urllib.request.urlopen(f"{BASE}/delta", timeout=3) as r:
        d = json.loads(r.read())
        print(f"[delta]  speed={d['speed']}  moved={d['moved']}")
except Exception as e:
    print(f"[delta] ERRO: {e}")

# /frame
try:
    with urllib.request.urlopen(f"{BASE}/frame", timeout=8) as r:
        ct   = r.headers.get("Content-Type", "?")
        rms  = r.headers.get("X-Render-Ms", "?")
        data = r.read()
        img  = Image.open(io.BytesIO(data))
        img.save("test_frame_bot.jpg")
        print(f"[frame]  OK! size={img.size}  bytes={len(data)}  ct={ct}  render={rms}ms")
        print("[frame]  Salvo em test_frame_bot.jpg")
except Exception as e:
    print(f"[frame]  ERRO: {e}")
    raise SystemExit(1)

# Latencia sustentada
N = 30
lat = []
t0 = time.perf_counter()
for _ in range(N):
    t1 = time.perf_counter()
    with urllib.request.urlopen(f"{BASE}/frame", timeout=8) as r:
        r.read()
    lat.append((time.perf_counter() - t1) * 1000.0)
total = time.perf_counter() - t0
lat.sort()
print(f"[bench]  {N} frames em {total:.2f}s -> {N/total:.1f} fps | "
      f"avg={sum(lat)/N:.2f}ms p50={lat[N//2]:.2f}ms p95={lat[int(N*0.95)]:.2f}ms")

# /stats
try:
    with urllib.request.urlopen(f"{BASE}/stats", timeout=3) as r:
        s = json.loads(r.read())
        print(f"[stats]  renderer={s['renderer']}  frames={s['frames']}  "
              f"avg_render={s['avg_render_ms']}ms  max_render={s['max_render_ms']}ms")
        print(f"[stats]  detail={s['detail']}")
except Exception as e:
    print(f"[stats]  ERRO: {e}")
