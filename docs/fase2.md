# Fase 2 — terreno real, obstáculos e `SPACE`

A pergunta que a Fase 1 **não** pôde responder:

> **A política usa a IMAGEM para contornar o que as coordenadas não descrevem?**

A Fase 1 provou que o laço de RL fecha, mas a tarefa era resolvível por `atan2`
sem olhar um pixel — terreno plano, sem obstáculo, alvo em coordenadas. Aqui as
coordenadas passam a ser **insuficientes**: elas dizem "vá para o nordeste", não
dizem "tem uma parede".

---

## O que muda

| | Fase 1 | Fase 2 |
|---|---|---|
| terreno | plano filtrado (13 arenas) | **real, sem filtro** (202 largadas) |
| distância | 3-8 blocos | **14-30 blocos** |
| ações | `W` + giro | `W` + giro + **`SPACE`** |
| orçamento | 14 passos | 50 passos |
| piso | `aleatorio`, 7,8% | **`geo_pulo`** (determ., cego) / **`aleatorio_pulo`** (estoc.) |
| teto | `geometrico`, 86% | **`piloto` BFS raio 40** |

### A inversão de papel do baseline geométrico

Na Fase 1 o `geometrico` era o **teto** e a pergunta era se a rede chegava perto.
Aqui ele é o **piso**: é cego por construção, então falha exatamente onde a visão
seria necessária. A distância entre ele e o planejador é o tamanho do que há para
ganhar.

### Por que a distância teve que subir

A 3-8 blocos **não cabe obstáculo**. Medido: só 13 de 96 alvos ficaram
obstruídos, e o planejador chegou a ser *pior* que apontar-e-andar (77,1% contra
85,4%) porque a distância é curta demais para planejar compensar — o mesmo padrão
já visto a 13 blocos na locomoção longa.

---

## `desvio` — a medida que torna o resultado interpretável

```
desvio = custo do caminho da busca / distância em linha reta
```

| faixa | valor | significado |
|---|---|---|
| `reta` | < 1,15 | reta livre. O baseline cego é **ótimo**; a rede só pode empatar |
| `leve` | 1,15-1,45 | desvio moderado |
| `obstruido` | > 1,45 | há algo no meio. **Só aqui a visão tem o que contribuir** |

**~0,86 é reta livre**, não 1,0: `custo` conta células e o passo diagonal cobre
1,41 blocos por célula.

**Tudo é reportado por faixa, nunca em agregado.** Uma política que empata em
`reta` e perde em `obstruido` ainda marca bem na média — e foi a média que
enganou duas vezes na Fase 1.

> **A evidência de uso da visão é ganhar do `geo_pulo` na faixa `obstruido`.**
> É causal, e não depende de confiar na ablação.

---

## Baselines — medidos antes do modelo

Alvos condicionados a `desvio >= 1.2`, n=80, raio do planejador 40:

| política | geral | `leve` | `obstruido` |
|---|---|---|---|
| `geo_pulo` (piso, cego) | 36,2% | 39% | **33%** |
| `piloto` (teto, lê voxels) | 53,8% | 47% | **60%** |
| | | | **+27 pontos** |

### O teto estava sabotado, e quase custou a fase

Com **raio 16** contra alvos a 14-30 blocos, metade dos alvos fica fora do
horizonte de busca e o "planejador" faz subida de encosta gulosa:

```
piloto raio 16    36% em obstruido   ->  "não há o que aprender"
piloto raio 40    60% em obstruido   ->  27 pontos de espaço
```

**O raio do BFS precisa cobrir a distância do alvo.** Um teto mal configurado
produz a conclusão de que a tarefa é inútil.

---

## O banco de tarefas

Alvo obstruído tem rendimento de **~5%** no sorteio cego a 14-30 blocos: gerar 80
leva ~4 min. Num treino de centenas de iterações isso viraria metade do tempo, e
o sinal sobre contornar ficaria diluído 20:1.

```bash
DIST_MIN=14 DIST_MAX=30 python ambiente/fase2.py --banco 1200
```

Mistura controlada: **metade obstruída, metade livre**. Se a política só vir
obstáculo, aprende a contornar sempre — inclusive onde a reta era ótima.

Split **por tarefa**: 15% do banco nunca entra no treino.

---

## A ação vira conjunta

```
a = (bin de giro, pular ou não)
log p(a) = log p(giro) + log p(pulo)
```

O pulo é Bernoulli, da cabeça de botões índice 4. A log-probabilidade usa
`softplus` em vez de `log(sigmoid)` por estabilidade numérica.

**Por que `SPACE` e só ele:** a física usa `stepHeight 0.6`, então sem pulo um
degrau de 1 bloco é parede. `A`/`D`/`S` e `SHIFT` ficam para depois, quando
houver problema real de controle lateral e a latência de 4 Hz for revista.

### O risco previsto: pular indiscriminadamente

Pular é quase de graça — não custa velocidade, não causa dano de 1 bloco, e às
vezes destrava. O gradiente para "sempre pular" é fracamente positivo, e é a
mesma família de "sempre aperte W" e "seja aleatório", que já degeneraram este
projeto duas vezes.

`treinar_fase2.py` loga `pulo %` por iteração para que isso seja visível.

**Mas seria falha lateral, não fatal:** pular não resolve contornar parede — só
virar resolve. A pergunta central continua sendo testada pela quebra por desvio.

---

## Como rodar

```bash
node mineflayer_server/servidor_offline.js

DIST_MIN=14 DIST_MAX=30 python ambiente/fase2.py --largadas 240        # largadas
DIST_MIN=14 DIST_MAX=30 python ambiente/fase2.py --banco 1200          # banco (lento)

DIST_MIN=14 DIST_MAX=30 PASSOS_MAX_F2=50 python avaliacao/avaliar_fase2.py \
    --episodios 80 --desvio-min 1.2 --raio 40 \
    --politicas geo_pulo,piloto                               # BASELINES

ATRASOS_SIM=0,2,4 python treino/treinar_fase2.py --iteracoes 150

DIST_MIN=14 DIST_MAX=30 PASSOS_MAX_F2=50 python avaliacao/avaliar_fase2.py \
    --episodios 80 --desvio-min 1.2 --raio 40 \
    --politicas geo_pulo,piloto,modelo,modelo_cego
```

---

## O que decide o veredito

| resultado | leitura |
|---|---|
| modelo bate `geo_pulo` em `obstruido` | **a ponte multimodal funciona** |
| modelo empata com `geo_pulo` em todas as faixas | aprendeu `atan2` de novo |
| modelo ≈ `modelo_cego` | a visão não contribuiu, seja qual for a taxa |
| `pulo %` satura em ~100% | o pulo degenerou; não invalida o resultado do giro |

Se vier ambíguo — modelo entre piso e teto sem separação clara por faixa — o
próximo passo **não** é uma Fase 2.1. É remover o canal de coordenada por
inteiro; ver o currículo em [fase1.md](fase1.md).
