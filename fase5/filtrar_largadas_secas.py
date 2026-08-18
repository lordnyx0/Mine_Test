import json
import urllib.request

largadas = json.load(open('dataset/largadas_fase2.json', 'r', encoding='utf-8'))
print(f"Total de largadas no banco original: {len(largadas)}")

largadas_validas = []

for lote_idx in range(0, min(len(largadas), 80), 8):
    lote = largadas[lote_idx:lote_idx+8]
    if len(lote) < 8:
        break
    try:
        payload = json.dumps({'posicoes': lote}).encode('utf-8')
        req = urllib.request.Request(
            'http://127.0.0.1:3002/lote/reset',
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        res = json.loads(urllib.request.urlopen(req, timeout=10).read().decode('utf-8'))
        for i, o in enumerate(res['obs']):
            e = o['estado']
            in_water = e.get('in_water', False)
            in_lava = e.get('in_lava', False)
            x, y, z = e['x'], e['y'], e['z']
            
            # Checa se o robô nasceu de fato na coordenada solicitada e se está seco
            req_x, req_z = lote[i][0], lote[i][1]
            dist_teleport = ((x - req_x)**2 + (z - req_z)**2)**0.5
            
            status = "SECO/OK"
            if in_water or in_lava:
                status = "AGUA/LAVA (INVALIDO)"
            elif dist_teleport > 5.0:
                status = "TELEPORTE FALHOU (INVALIDO)"
            elif y < 55:
                status = "CAVERNA/SUBTERRANEO (INVALIDO)"
            else:
                largadas_validas.append([round(x, 1), round(z, 1)])
                
            print(f"Largada {lote_idx + i:03d}: ({req_x:7.1f}, {req_z:7.1f}) -> Real: ({x:7.1f}, y={y:4.1f}, {z:7.1f}) | {status}")
    except Exception as err:
        print(f"Erro no lote {lote_idx}: {err}")

print(f"\nTotal de largadas 100% secas e validadas: {len(largadas_validas)}")
