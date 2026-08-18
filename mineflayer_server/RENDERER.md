# Renderizador voxel (`voxel_renderer.js`)

Substitui o pipeline **prismarine-viewer + Puppeteer.screenshot** (80–200 ms/frame)
por um **raycaster DDA em Node puro** que lê os blocos direto de `bot.world` e
escreve os pixels num canvas nativo.

Sem browser, sem WebGL, sem janela, sem dependência de foco. Só CPU.

---

## Como funciona

1. **Câmera** — origem em `bot.entity.position + eyeHeight`, direção derivada de
   `yaw`/`pitch` com a mesma convenção de `bot.blockAtCursor`
   (`forward = (-sin yaw·cos pitch, sin pitch, -cos yaw·cos pitch)`).
2. **Um raio por pixel** numa resolução interna reduzida (`FRAME_SCALE`), com
   DDA de Amanatides & Woo sobre a grade de voxels.
3. **Salto de seções vazias** — cada section 16³ carrega um flag "só ar"
   calculado uma vez. Raio que entra numa section vazia pula direto para a
   saída dela em vez de dar 16 passos. É o que torna raios de céu/horizonte
   baratos.
4. **Leitura de blocos sem alocação** — o buffer da section 1.8 é lido
   diretamente (`stateId = data[o] | data[o+1] << 8`), sem `Vec3`, sem objetos
   `Block`. O cache de section é validado por um contador de geração
   (comparação de inteiros), invalidado por section em `blockUpdate`.
5. **Sombreamento** — cor da paleta por bloco/face/metadata × sombra de face
   (topo 1.0, N/S 0.8, L/O 0.62, base 0.45) × luz do jogo (`skyLight` corrigido
   pelo ciclo dia/noite, `blockLight`) via LUT 16×16 × ruído determinístico por
   voxel (dá "grão" de textura que o encoder de visão consegue ler).
6. **Meios translúcidos** — água, vidro/gelo e sprites (grama alta, flores,
   tochas) não param o raio; acumulam tinta e ele segue.
7. **Entidades** — billboards com teste de profundidade contra o Z-buffer do
   raycast, coloridos por tipo de mob.
8. **Upscale + JPEG** — canvas pequeno → canvas de saída (nearest por padrão) →
   `toBuffer('image/jpeg')`.

---

## Custo por estágio

Medido com `node profile_renderer.js` no mundo sintético de `test_world.js`
(terreno, ~90 árvores, lago, grama alta — cenário mais pesado que uma planície).

Saída 640×360:

| `FRAME_SCALE` | raycast interno | raycast | upscale+HUD | JPEG | **total** |
|---|---|---|---|---|---|
| 3 | 213×120 | 15,5 ms | 0,95 ms | 1,6 ms | **18,1 ms** |
| **4** | **160×90** | **8,9 ms** | **0,1 ms** | **1,5 ms** | **10,5 ms** |
| 5 | 128×72 | 5,7 ms | 0,6 ms | 1,4 ms | **7,7 ms** |
| 6 | 107×60 | 3,9 ms | 0,3 ms | 1,5 ms | **5,7 ms** |
| 8 | 80×45 | 2,2 ms | 0,3 ms | 1,4 ms | **3,9 ms** |

Saída 320×180 (o SigLIP redimensiona para 224×224 de qualquer jeito, então
640×360 é resolução jogada fora):

| `FRAME_SCALE` | raycast interno | **total** |
|---|---|---|
| 2 | 160×90 | **9,2 ms** |
| 3 | 107×60 | **4,9 ms** |
| 4 | 80×45 | **3,0 ms** |

Efeito do alcance (`FRAME_DIST`, scale 4, 640×360): 32 → 9,3 ms · 48 → 9,7 ms ·
64 → 10,6 ms · 96 → 11,2 ms.

**Ponta a ponta pelo HTTP** (`node test_server_e2e.js`, scale 4, 640×360):
~15 ms/frame → ~67 fps. Contra 80–200 ms do Puppeteer.

> `FRAME_SMOOTH=1` (upscale bilinear do Cairo) custa **+5 ms**. Por isso o
> padrão é nearest.

---

## Configuração (variáveis de ambiente)

| Variável | Padrão | O que faz |
|---|---|---|
| `RENDERER` | `voxel` | `voxel` \| `puppeteer` (legado) \| `none` |
| `FRAME_W` / `FRAME_H` | `640` / `360` | Resolução do JPEG de saída |
| `FRAME_SCALE` | `4` | Divisor da resolução do raycast. Maior = mais rápido e mais grosseiro |
| `FRAME_FOV` | `70` | FOV vertical em graus |
| `FRAME_DIST` | `64` | Alcance dos raios em blocos |
| `FRAME_QUALITY` | `0.8` | Qualidade JPEG (0–1) |
| `FRAME_CROSSHAIR` | ligado | `0` desliga a mira |
| `FRAME_ENTITIES` | ligado | `0` desliga o desenho de mobs/players/itens |
| `FRAME_SMOOTH` | desligado | `1` usa upscale bilinear (+5 ms) |
| `FRAME_MIN_MS` | `0` | Reaproveita o último frame se ele tiver menos de N ms |

Para voltar ao pipeline antigo: `set RENDERER=puppeteer` (precisa de
`prismarine-viewer` e `puppeteer`, que agora são `optionalDependencies`).

---

## Endpoints afetados

- `GET /frame` — JPEG da visão em 1ª pessoa. Traz o header `X-Render-Ms`.
- `GET /health` — agora inclui `renderer`.
- `GET /stats` — **novo**: contagem de frames, média/máximo de tempo de render.

---

## Limitação: formato de chunk

O raycast lê o buffer da section no **layout do Minecraft 1.8** (4096 blocos em
uint16 LE, depois blockLight e skyLight em nibbles → 12288 bytes com skylight,
10240 sem). Em versões novas (1.13+) o formato é paletizado e completamente
diferente.

Para não renderizar lixo em silêncio, o renderer valida o tamanho da section no
primeiro frame e **aborta com mensagem explícita** se não bater. O servidor
captura o erro, desliga o renderer (`/frame` passa a devolver 503) e mantém
`/action`, `/state` e `/delta` funcionando.

---

## Testes

```bash
npm run bench      # node test_renderer.js      — benchmark + amostras em render_test/
npm run profile    # node profile_renderer.js   — custo por estágio
npm run test:e2e   # node test_server_e2e.js    — servidor HTTP real com mineflayer stubado
npm test           # os dois
```

`test_world.js` monta o mundo sintético (terreno procedural, árvores, lago,
torre de lã colorida para checar variantes de metadata, glowstone para checar
emissividade) e um bot falso — nenhum teste precisa de servidor Minecraft.

Do lado Python, com o servidor no ar:

```bash
python scripts/test_frame.py      # smoke test + latência dos endpoints
python bot_vision_capture.py      # captura + benchmark de 60 frames
```
