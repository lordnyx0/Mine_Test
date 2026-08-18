# Fase 1 — controle visuomotor local

A pergunta, e só ela:

> **O Qwen3Loop consegue aprender controle visuomotor local — de um objetivo
> espacial para ação — usando feedback do ambiente?**

Sem professor. Sem BFS. Sem imitação de ação.

---

## Por que decompor

Três tentativas de aprender locomoção completa falharam (ver
[experimentos.md](experimentos.md)). A hipótese desta fase é que o fracasso veio
de atacar um problema longo e complexo de uma vez. Aqui se testa só a ponte:

```
representação multimodal  →  erro espacial  →  controle de direção
```

---

## Escopo

| dimensão | Fase 1 |
|---|---|
| terreno | plano, sem obstáculo, sem água, sem buraco |
| ações | **só `W` + giro** |
| alvo | relativo, 3 a 8 blocos, 8 setores egocêntricos |
| raio de chegada | 1,5 bloco |
| orçamento | **14 passos** |

Sem `A`/`D`/`S`/`SHIFT`/`SPACE`, sem mobs, sem mineração, sem inventário.

Só o giro é aprendido — a política é um classificador de 9 bins. Isso remove
por construção o modo de falha que quebrou o treino anterior, em que as cabeças
de botão emitiam `A` em 93% e `SHIFT` em 62% contra alvos de 0%.

### As arenas

Terreno plano não existe: o mundo vem dos arquivos de região de um save real.
Em vez de gerar um, `arena_plana.py` **filtra** posições de largada — perfil de
relevo em 8 direções, tolerância de 1 bloco, num raio de 12.

Rendimento ~3%. E **checar água é obrigatório junto**: água não é bloco sólido,
então o perfil atravessa e mede o leito, e fundo de lago raso é perfeitamente
plano. A primeira versão do filtro aprovou 4 de 8 arenas dentro d'água.

### O alvo é egocêntrico

`amostrar_alvo` devolve `(frente, lado)` no referencial do bot, e
`alvo_para_mundo` converte usando o yaw da largada.

A primeira versão sorteava um ângulo **absoluto** e chamava de "frente" o que
apontava para +x do mundo. Como o respawn randomiza o yaw
(`entity.yaw = Math.random()*2π`), cada rótulo continha uma mistura uniforme de
direções relativas — a quebra por setor media ruído.

---

## Recompensa

```
r_t = distância_anterior − distância_atual        (+5 ao chegar)
```

Aproximar é positivo, afastar negativo, parado ~zero. Substitui o rótulo: não
importa qual tecla "o professor teria apertado", só se o agente encostou.

### O orçamento faz parte da recompensa

Com **40 passos**, passeio aleatório encontra o alvo (~90%) e a política aprende
a ser aleatória em vez de aprender controle. Com **14**, cai para 7,8% e o
caminho direto (~8 passos) continua viável.

**Foi o erro que custou a primeira corrida.** Ver
[experimentos.md §4.1](experimentos.md).

---

## Baselines — obrigatórios, antes do modelo

| política | chegada | o que mede |
|---|---|---|
| `so_W` | 3,1% | quanto vem de o alvo estar perto |
| `aleatorio` | 7,8% | **o piso certo** — vagar sem ler a entrada |
| `geometrico` | 85,9% | o teto: `atan2`, sem visão e sem rede |

`geometrico` é trigonometria pura. A pergunta do experimento é se a política
neural chega perto dele.

---

## Treino

REINFORCE com vantagem normalizada, backbone e visão congelados.

**Dois passos, por memória:** guardar o grafo de 14 passos × 8 ambientes através
do SigLIP e de um 0.6B estoura 12 GB. O rollout roda sem grafo, guardando pixels
em uint8, e os log-probs são recalculados em minilotes.

### Hiperparâmetros e seus porquês

| parâmetro | valor | por quê |
|---|---|---|
| `lr` | 3e-5 | com 1e-4 os logits explodem e a entropia vai a zero na 5ª iteração |
| entropia | 0,05 → 0,01 | recozido: 0,01 fixo colapsa, 0,05 fixo deixa quase uniforme |
| minilote | 4 | 2× mais rápido que 2 pelo mesmo tempo/passo; 6 dá OOM |
| `--vram` | 0,88 | pico medido de 8,81 GB no minilote 4 |
| frames | `ATRASOS_SIM=0,2,4` | o padrão (0,16,60) faz clamp em episódio curto e repete o frame inicial |

---

## Como rodar

```bash
node mineflayer_server/servidor_offline.js

python ambiente/arena_plana.py --arenas 90

ATRASOS_SIM=0,2,4 python avaliacao/avaliar_fase1.py --episodios 64 \
    --politicas so_W,aleatorio,geometrico

ATRASOS_SIM=0,2,4 python treino/treinar_fase1.py --iteracoes 300 \
    --minilote 4 --vram 0.88

ATRASOS_SIM=0,2,4 python avaliacao/avaliar_fase1.py --episodios 64 \
    --politicas geometrico,modelo,modelo_cego
```

---

## Diagnóstico — não pule

### Ablação de visão

`modelo_cego` é o mesmo checkpoint com os pixels zerados. Terreno plano com
alvo em coordenadas é resolvível por trigonometria sem olhar nada.

**Se cego e vidente empatam, a visão não contribuiu** — por mais alta que seja
a taxa, a Fase 1 não terá testado a ponte multimodal.

### Varredura de ângulo

Cena e distância fixas, ângulo do alvo de −180° a +180°. Se a distribuição de
giro não muda, a política **ignora o objetivo**.

Foi o que revelou a trapaça da primeira corrida: distribuição idêntica, dígito
por dígito, em 13 ângulos.

### Sonda do 1024 — os quatro casos

| caso | evidência | conclusão |
|---|---|---|
| **A** sucesso | chega perto do geométrico, e cego cai | a ponte funciona; escalar |
| **B** representação boa, política ruim | sonda recupera o ângulo, política não usa | problema de treino/cabeça |
| **C** representação insuficiente | sonda não recupera | problema antes da política |
| **D** incapaz no trivial | nem sonda nem política | problema estrutural |

Resultado da primeira corrida: **caso B**, com R² de 0,978/0,962 e erro angular
mediano de **3 graus** contra ~90 de acaso.

---

## Currículo — só avança se a anterior passar

```
Fase 1   ±8 blocos, plano, W + giro                      FEITA
Fase 2   terreno real + SPACE + 14-30 blocos             ver fase2.md
Fase 3   alvo VISUAL: sem coordenada, "vá até a madeira"  proposta
Fase 4   SHIFT (velocidade variável)
Fase 5   A / D / S
```

**O currículo mudou em relação ao plano original.** A Fase 2 construída fundiu
três fases que estavam separadas no papel — `SPACE`, distância maior e
obstáculos — porque as três são a mesma condição: obstáculo só existe se houver
distância para o desvio caber, e degrau de 1 bloco só é transponível com pulo.
Separá-las teria produzido três fases sem faixa dinâmica.

### A Fase 3 proposta, e por que ela é o teste mais forte

Com obstáculos, a visão precisa ser *suficiente*; o canal de coordenada continua
existindo e dá 33% de crédito de graça. **Sem coordenada, a visão precisa ser
necessária:**

| | canal de coordenada | piso |
|---|---|---|
| Fase 2 (obstáculos) | continua | 33% pelo `atan2` |
| Fase 3 (alvo visual) | **removido** | ~0% sem visão |

O alvo passa a ser um bloco — "vá até a madeira" — e a recompensa é encostar
nele. `Objetivos.bloco` já existe e dá o teto.

Uma escolha decide se é tratável: **alvo visível da largada** (servo-visual
puro: ver, apontar, ir) ou **alvo em qualquer lugar** (exige busca, exploração e
memória). Começar pelo primeiro isola a ponte `pixel → direção`.

### Por que SPACE vem logo na Fase 2

A física usa `stepHeight: 0.6` — **sem pulo o bot só sobe 0,6 bloco**, então um
degrau de 1 é parede. Isso tem duas consequências:

1. O filtro de arena com tolerância 1 admite degraus **intransponíveis** para a
   política atual. Parte dos 14% de falha do `geometrico` pode ser terreno, não
   erro de controle — o teto da Fase 1 não é limpo.
2. Com pulo, degrau de 1 vira navegável, a tolerância passa a ser válida, e o
   filtro pode ser afrouxado. Hoje o rendimento é 3% e sobram 13 arenas;
   afrouxar aumenta muito a diversidade visual, que é a maior ressalva sobre a
   ablação de visão.

O baseline ganha a regra que o `Piloto` já usa: `precisaSubir || travado >= 1`.

### Latência de controle, para as fases 5 e 6

Dois parâmetros são frequentemente confundidos:

| parâmetro | controla | valor |
|---|---|---|
| `duration_ms` | **latência de reação** | 250 ms = 4 Hz |
| `ATRASOS_PASSOS` | **janela de história** | 0s, 0,5s, 1s |

Para desvio lateral (`A`/`D`) e velocidade variável (`SHIFT`), o que aperta é a
latência, não o número de frames. A 4 Hz cada ação se compromete por um quarto
de segundo, durante o qual o bot anda ~1 bloco — granularidade grosseira demais
para manobra.

O simulador tica a 50 ms, então há espaço até 20 Hz. **Mas suba a taxa só
quando a fase exigir, e meça se ajuda** — 10 Hz custa 2,5× mais inferência por
segundo simulado, e o rollout já é ~40% do tempo de treino.
