# Simulador offline

`mineflayer_server/servidor_offline.js` — 8 ambientes paralelos, porta 3002.

**Não precisa do Minecraft aberto.** Lê os voxels direto dos arquivos de região
(`.mca`) do save, com física real (`prismarine-physics`) e render próprio.

```bash
node mineflayer_server/servidor_offline.js
```

---

## Visualizador

<http://127.0.0.1:3002/ver>

Os 8 ambientes lado a lado, atualizando a cada 0,7 s, com posição, altura,
passos e alerta de água por baixo de cada um.

**Use-o.** Três hipóteses sobre locomoção foram refutadas por medição agregada
sem revelar a causa real; um bug de arenas geradas dentro d'água foi achado em
segundos, olhando a tela.

---

## API

### `GET /lote/info`

```json
{"envs":8,"colunas_em_memoria":900,"prontos":8,"frame":"640x360","tick_ms":50}
```

### `POST /lote/reset`

```json
{"posicoes": [[x,z], ...]}      // opcional; sem isso, respawn aleatório
```

Devolve `obs` com `estado`, `rotas` e `frame_b64` por ambiente.

### `POST /lote/passo`

```json
{
  "acoes":  [{"hold":["W","SPACE"], "mouse":[bin,0], "duration_ms":250}, ...],
  "frames": true,        // false economiza o render, que é o custo dominante
  "rotas":  true,
  "diag":   false,       // sondas caras; ver abaixo
  "dirs":   [[dx,dz], ...]   // direção do perfil de relevo, por ambiente
}
```

### `POST /lote/piloto`

Ação sugerida pelo planejador para todos os ambientes.

```json
{
  "objetivo": "explorar" | "ponto" | "rumo" | "bloco",
  "extra":  {"raio":16, "alvo":{"x":..,"y":..,"z":..}},   // igual p/ todos
  "extras": [ {...}, ... ],                               // um POR ambiente
  "compromisso": 1
}
```

`extras` existe porque avaliar "vá até B" exige um alvo diferente por ambiente.

### `POST /lote/alcancavel`

O alvo é alcançável, e **quão obstruído** está o caminho até ele?

```json
{"alvos": [{"x":.., "z":..}, ...], "raio": 40}
```

```jsonc
{"alcancavel": true, "resta": 0.7, "custo": 22, "reta": 18.4, "desvio": 1.20}
```

`desvio` = custo do caminho / distância em linha reta. **~0,86 é reta livre** —
não 1,0, porque `custo` conta células e o passo diagonal cobre 1,41 blocos.
Acima de ~1,45 há algo no meio que obriga a contornar.

**O `raio` precisa cobrir a distância do alvo.** Com raio 16 e alvo a 30 blocos
o BFS não planeja: metade dos alvos fica fora do horizonte e ele vira subida de
encosta gulosa. Isso fez o "teto" da Fase 2 medir 36% quando o real era 60%.

### `GET /lote/estado`, `GET /lote/frame?env=N`

Estado de todos os ambientes em JSON; frame atual de um ambiente em `image/jpeg`.

---

## Observação

```jsonc
{
  "env": 0, "passos": 34, "morreu": false,
  "estado": {
    "x":..,"y":..,"z":..,"yaw":..,"pitch":..,     // yaw em GRAUS
    "vx":..,"vy":..,"vz":..,"on_ground":true,
    "in_water":false,"in_lava":false,             // a física já mantinha
    "health":20,"food":20
  },
  "rotas": [0.42, 0.59, ...],   // 12 setores, 0 = parede colada, 1 = livre
  "frame_b64": "..."            // JPEG 640x360
}
```

**Setor 0 das rotas é a direção que o bot encara**, não o norte: o ângulo do
setor `i` é `yaw + (i/K)·2π`.

---

## Sondas de diagnóstico (`"diag": true`)

Custam ~676 consultas de bloco por ambiente. **Nunca no caminho quente** — só
quando se está diagnosticando, e de preferência a cada N passos.

```jsonc
"diag": {
  "agua_perto": 4.0,          // distância horizontal à água, ou null
  "bloco_pes": "air",
  "bloco_abaixo": "grass",
  "perfil": [0,0,1,1,4,4,...] // altura do chão na direção pedida, 12 blocos
}
```

**Como ler o perfil:** subindo forte = parede ou encosta íngreme (o BFS só
aceita degrau de 1, então 2+ é intransponível); despencando = abismo ou lago;
plano = o bloqueio não é relevo. `null` = coluna não carregada, que também é
intransponível para a busca.

### Duas armadilhas do perfil

**Direção degenerada.** Com `|dir| ≈ 0` a normalização daria (0,0) e as 12
amostras cairiam todas na própria célula do bot, fazendo qualquer relevo próximo
parecer parede. Isso invalidou um grupo de controle inteiro. Hoje devolve `null`
quando `|dir| < 0.5`.

**Água finge terreno plano.** Água não é bloco sólido, então o perfil atravessa
e mede o **leito** — fundo de lago raso é perfeitamente plano. Um filtro de
arenas baseado só em planura aprovou 4 de 8 arenas dentro d'água, duas a y=40.
Sempre checar `in_water` e `agua_perto` junto.

---

## Custo

O render domina. `"frames": false` acelera muito quando a política não precisa
de imagem — sortear alvos, rodar baselines geométricos, procurar arenas.

O mundo tem teto de memória por LRU (~2600 colunas ≈ 510 MB). Em viagem longa
o cache satura e passa a ler disco, o que torna os passos mais lentos no fim de
um episódio que no começo.
