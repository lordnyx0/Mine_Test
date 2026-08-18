import socket
import struct
import re
import sys
import os

MCAST_GRP  = "224.0.2.60"
MCAST_PORT = 4445

# Escreve _lan_vars.bat no mesmo diretorio do script
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lan_vars.bat")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("", MCAST_PORT))
except OSError:
    sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MCAST_PORT))

mreq = struct.pack("4sL", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
sock.settimeout(8)

print("Aguardando broadcast LAN do Minecraft (ate 8s)...", flush=True)

host, port = "127.0.0.1", "25565"
try:
    data, addr = sock.recvfrom(1024)
    text = data.decode("utf-8", errors="ignore")
    m = re.search(r"\[AD\](\d+)\[/AD\]", text)
    host = addr[0]
    port = m.group(1) if m else "25565"
    print(f"Encontrado: {host}:{port}", flush=True)
except socket.timeout:
    print("AVISO: Nenhum broadcast. Usando 127.0.0.1:25565", flush=True)
except Exception as e:
    print(f"ERRO: {e}", flush=True)
finally:
    sock.close()

# Gera arquivo .bat que o batch principal faz `call` — mais confiavel que set /p
with open(OUT_FILE, "w") as f:
    f.write(f"SET MC_HOST={host}\r\n")
    f.write(f"SET MC_PORT={port}\r\n")

print(f"OK: Variaveis salvas em {OUT_FILE}", flush=True)
sys.exit(0)
