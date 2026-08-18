# coding=utf-8
"""
monitor_comandos.py — Monitor de Telemetria e Comandos em Tempo Real.

Mostra ao vivo no terminal os comandos, rotações de câmera, posições e distâncias dos 8 robôs.
Uso:
    python ferramentas/monitor_comandos.py
"""
import time
import json
import urllib.request
import os

BASE = "http://127.0.0.1:3002"

def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    print("Conectando ao simulador em %s..." % BASE)
    while True:
        try:
            with urllib.request.urlopen(BASE + "/lote/estado", timeout=2) as r:
                d = json.loads(r.read().decode("utf-8"))
            
            limpar_tela()
            print("=" * 75)
            print(" 🤖 MONITOR DE TELEMETRIA E COMANDOS DO CÉREBRO VLA (8 AMBIENTES)")
            print(" Painel Web Visual: http://127.0.0.1:3002/ver")
            print("=" * 75)
            print(f"{'ENV':<5} | {'POSIÇÃO (X, Y, Z)':<24} | {'YAW':<8} | {'PASSOS':<7} | {'STATUS':<15}")
            print("-" * 75)
            
            for e in d.get("envs", []):
                i = e["env"]
                s = e["estado"]
                p = e["passos"]
                m = e.get("morreu", False)
                w = s.get("in_water", False)
                
                pos_str = f"({s['x']:.1f}, {s['y']:.1f}, {s['z']:.1f})"
                yaw_str = f"{s['yaw']:.1f}°"
                
                status = "ATIVO"
                if m:
                    status = "MORREU"
                elif w:
                    status = "NA AGUA"
                elif p >= 100:
                    status = "CONCLUIDO"
                    
                alvo = e.get("alvo")
                alvo_str = ""
                if alvo:
                    alvo_str = f" -> Alvo: {alvo.get('dist', 0)}m ({alvo.get('graus', 0)}°)"
                
                print(f"Env {i:<2} | {pos_str:<24} | {yaw_str:<8} | {p:<7} | {status:<15}{alvo_str}")
            
            print("=" * 75)
            print("Pressione Ctrl+C para encerrar o monitor.")
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            break
        except Exception as ex:
            print("Aguardando servidor...", ex)
            time.sleep(2.0)

if __name__ == "__main__":
    main()
