# O planejador

`mineflayer_server/planejador.js`, 428 linhas. **Código puro, sem rede neural.**
É busca clássica, e é hoje o componente de navegação que funciona.

## Como funciona, em quatro passos

**1. Lê o mundo em blocos.** `bot.world.getColumn(cx, cz)` e vai direto no
buffer da seção. Sabe o tipo de cada bloco num raio de 16 ou 40.

**2. Decide onde dá para pisar.**

```js
pisavel(x,y,z) =  pés vazios
               && cabeça vazia
               && chão sólido
               && nada perigoso
```

**3. Busca em largura** até 9000 células, pontuando cada uma com
`objetivo.pontuar(no)` e devolvendo o caminho até a melhor.

**4. `Piloto` converte caminho em teclas.** Mira no ponto ~6 blocos à frente
(não no destino final), converte o ângulo num dos 9 bins de mouse, aperta `W`,
adiciona `SPACE` quando o próximo ponto está mais alto, e replaneja ao chegar,
ao travar, ou quando o compromisso expira.

## Objetivos plugáveis

A busca não sabe o que é "explorar" ou "achar madeira". Ela maximiza uma função
de pontuação:

```js
explorar(origem, memoria, λ)   d(origem) − λ·visitas
ponto(alvo)                    −d(alvo)                       GULOSO
rumo(alvoFinal, memoria, λ)    −d(alvo) − λ·visitas
bloco(nomes, registry)         achou ? 1e6−custo : −custo
```

**A mesma busca serve para tudo.** Trocar "explore" por "colete madeira" é
trocar uma função, não reescrever o algoritmo. É por isso que ele é boa base
para a camada de habilidades.

### Cuidado com `bloco`

Sem tronco no raio, todos os nós pontuam `−custo`, o máximo é custo zero, e **o
bot fica parado**. Chamar `procurar(madeira)` sempre falha em mundo sem árvore
por perto — o agente precisa alternar explorar e buscar. É justamente isso que
torna "colete madeira" uma tarefa não-degenerada para o cérebro.

## Desempenho medido

Alvos pareados, ~100 blocos, 400 passos:

| política | chegada | travado |
|---|---|---|
| `reto` (aponta e anda) | 52-55% | 42% |
| `piloto` (`ponto`) | 80-90% | 12-15% |

A 13 blocos, `reto` chega em 95% e o planejador em 92% — **a métrica satura**.
A vantagem dele só aparece em distância real.

## Três bugs corrigidos, cada um medido

| conserto | efeito |
|---|---|
| mirar no próximo ponto do caminho, não no destino | 14,4 → 17,5 blocos |
| impedir diagonal de cortar quina | — |
| `SPACE` em degrau | 17,5 → 41,3 |

O terceiro era o pior: a busca autorizava subir 1 bloco, mas o piloto nunca
apertava `SPACE`, então todo desnível travava para sempre.

## Limitações estruturais

**Água e abismo não existem no grafo.** Não têm chão sólido, então `pisavel` os
rejeita. Um lago torna o outro lado inalcançável e o bot fica na margem. Ele não
é cego a isso — não é representável.

**Míope.** Raio de 16 blocos, recalculado a cada passo. Se o bom caminho exige
desvio de 30 blocos, ele não vê. É a causa dos ~15% de falha.

**Precisa de voxels.** Por isso não transpõe para outro jogo, onde só há a tela.
É a razão de ele ser um *especialista*, não a resposta final.

## O que foi testado e NÃO ajuda

`Objetivos.rumo` — a composição de `ponto` com a memória de visitas do
`explorar` — foi implementada para soltar do mínimo local. **Empata com o
guloso** (80% contra 80%), com traços por lote quase idênticos: a penalidade de
visita quase nunca muda o argmax da busca.

E o diagnóstico das falhas mostrou que **não são de orçamento**: em três
rodadas, 0 de 17 falhas ainda melhoravam quando o tempo acabou. São bloqueio
duro. Água e elevação foram testadas como causa e **ambas refutadas** — ver
[experimentos.md](experimentos.md).
