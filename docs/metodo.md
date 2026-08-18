# Método

Regras de medição e de treino de RL que este projeto aprendeu na marra — e
algumas leis já estabelecidas na literatura que explicam por que elas
funcionam. Cada uma tem o custo que teve para ser descoberta, ou a referência
de onde vem. §§1-14 são sobre medição; §§15-22 são sobre desenho de
recompensa e otimização; §§23-27 trazem achados de 2025-2026 que se conectam
direto a incidentes já vividos aqui.

---

## 1. Toda métrica precisa de piso E teto

Sem os dois, um número no meio não significa nada.

**Piso** é a política trivial que qualquer um escreveria em cinco minutos.
**Teto** é o melhor conhecido — de preferência um algoritmo clássico com acesso
privilegiado.

| tarefa | piso | teto |
|---|---|---|
| ir a 100 blocos | `reto` (aponta e anda) 52% | `piloto` (BFS) 85% |
| alvo local ≤8 blocos | `aleatorio` 7,8% | `geometrico` 85,9% |

### O piso precisa ser do mesmo tipo da política

`so_W` é determinístico e marca 3,1%. Parecia piso seguro. Mas a política
treinada era **estocástica**, e uma política estocástica que ignora a entrada
*vaga* — e vagar encontra alvos próximos. O piso certo era passeio aleatório.

**Custo dessa lição: 4 horas de treino e uma conclusão errada de "sucesso".**

---

## 2. Cabeça que ignora a entrada — TRÊS causas, três consertos opostos

Aconteceu **quatro vezes** neste projeto, e nas quatro o sintoma foi idêntico:
uma cabeça cuja saída não depende da entrada. Tratar as três causas como uma só
leva ao conserto errado.

### (a) Ação sem gradiente

Se a ação não correlaciona com a vantagem, o gradiente naquela dimensão é ruído.
A entropia é a única coisa segurando a distribuição espalhada; quando ela recoze,
a cabeça colapsa para o que o ruído favoreceu. Constante **arbitrária**.

### (b) Custo além do horizonte de crédito

O caso do `SPACE` na Fase 2. Pular **não era de graça**: destravava na hora
(ganho imediato) e matava o giro depois — `airborneAcceleration = 0.02` contra
~0,10 no chão, ou seja **5× menos autoridade de direção**. Com γ=0,95 e 50
passos, o desconto amassa o custo tardio.

A política aprendeu a constante **errada, e ela parecia estar sendo
recompensada**: `pulo` foi de 36% a 99% enquanto a chegada caía de 50% para 32%.

### (c) Ótimo degenerado da tarefa

O "sempre `W`" e o passeio aleatório. A constante realmente **é** quase ótima,
então condicionar à entrada não paga nada. A política está certa; a tarefa é que
não pergunta nada. Foi assim que o projetor visual colapsou para posto efetivo
1,4 de 1024 — prever rotas tratava o sintoma.

### Como distinguir, e o que fazer

| causa | teste barato | conserto |
|---|---|---|
| (a) sem gradiente | ablacionar a ação; a recompensa muda? | a ação não deveria estar no espaço |
| (b) custo tardio | adicionar custo imediato; o comportamento muda? | **precificar**, ou γ maior |
| (c) tarefa degenerada | rodar política constante como baseline; chega perto do teto? | mudar a **tarefa**, nunca o modelo |

A varredura de ângulo detecta o **sintoma** e não distingue a causa. E os
consertos são opostos: em (b) mexer na tarefa é errado, em (c) mexer no modelo é
errado.

> **Regra operacional:** antes de treinar, para cada dimensão de ação, ou você
> **precifica**, ou **prova** que a constante ali é de fato ótima.

---

## 3. Meça no regime certo

| distância | reto | piloto | conclusão que se tiraria |
|---|---|---|---|
| 13 blocos | 95% | 92% | "locomoção resolvida, planejador inútil" |
| 100 blocos | 52% | 85% | "locomoção longe de resolvida" |

A primeira medição foi feita a 13 blocos e produziu a conclusão oposta da
verdadeira. Nessa escala não existe buraco, água, penhasco nem tempo para
perder o rumo — o episódio acaba antes de qualquer coisa dar errado.

---

## 4. Split que vaza infla tudo

Amostras vizinhas do mesmo episódio de 70 passos são quase idênticas.

| split | recall medido |
|---|---|
| blocos de 50 | 39,7% |
| **por episódio** | **19%** |

Metade do "desempenho" era vazamento. **Sempre separe por episódio.**

O mesmo vale para arenas: reserve algumas nunca vistas no treino, senão ganho
de visão pode ser memorização de cena.

---

## 5. Grupo de controle, sempre

Água estava perto em 2 de 6 falhas — 33%, parece muito. Entre os que
**chegaram**: 21 de 34, ou 62%.

Sem o controle, a correlação teria parecido forte e a habilidade de nadar seria
construída sobre coincidência.

**Toda hipótese "X causa a falha" precisa de "quanto X aparece nos sucessos".**

---

## 6. Uma verdade só entre treino e execução

O laço de avaliação não chamava `politica.observar()`, então a pilha de frames
e o `state_vec` congelavam no instante inicial. A política recebia objetivo
atualizado com imagem de 40 passos atrás.

**Efeito: 90% no treino, 11% na avaliação.**

`estado_sim.EstadoEpisodio` existe como definição única do `state_vec` por essa
razão — o próprio arquivo documenta um bug anterior da mesma família. E eu
reintroduzi a mesma classe de erro ao escrever um segundo laço em vez de reusar
um.

**Reuse o laço. Não escreva um segundo.**

---

## 7. Diagnostique antes de consertar

Quatro hipóteses minhas foram refutadas por medição barata:

| hipótese | custo do teste | custo do conserto que evitou |
|---|---|---|
| pooling é o gargalo | 25 min | 2,5 h de retreino |
| memória de visitas solta do mínimo local | 15 min | grafo persistente, dias |
| água bloqueia a locomoção | 5 min | habilidade de natação inteira |
| elevação bloqueia | 5 min | mudança no BFS |

**Nenhuma sobreviveu.** O padrão é consistente o bastante para virar regra:
medir a causa custa uma fração de construir o conserto errado.

---

## 8. Reproduza antes de decidir

Duas sondas do mesmo checkpoint, sobre os mesmos dados, deram p(W) = 0,11 e
0,65. A primeira era não-confiável.

**Um diagnóstico que não reproduz não pode fundamentar decisão.**

---

## 9. Olhe a tela

Três hipóteses foram refutadas por medição agregada sem revelar a causa real.
O usuário achou arenas geradas dentro d'água em segundos, olhando o
visualizador.

E olhando de novo notou que o boneco "não gira nunca" — que virou a varredura
de ângulo, que virou o diagnóstico completo da Fase 1.

<http://127.0.0.1:3002/ver>

Uma ressalva: o visualizador atualiza a cada 0,7 s e os bins de giro vão até
±45° por passo de 250 ms. Uma correção de 90° acontece **entre dois quadros** e
é invisível. Ausência de giro na tela precisa ser confirmada por medição.

---

## 10. Instrumente a corrida longa

`cmd | grep | tail` **bufferiza até o fim**. Uma corrida de 4 horas assim é
4 horas de cegueira.

- Redirecione para arquivo, sem pipe.
- Imprima progresso por lote, não só por política.
- Salve checkpoint a cada N iterações: queda de energia não pode custar a noite.

---

## 11. O teto também precisa ser validado

O piso é óbvio; o teto engana. Na Fase 2 o "planejador" usava BFS de raio 16
contra alvos a 14-30 blocos: metade dos alvos ficava fora do horizonte de busca
e ele virava subida de encosta gulosa.

```
piloto raio 16    36% em obstruido   ->  "nao ha o que aprender"
piloto raio 40    60% em obstruido   ->  27 pontos de espaco
```

**Um teto mal configurado produz a conclusão de que a tarefa é inútil.** Antes de
descartar um experimento por falta de faixa dinâmica, verifique se o teto está
funcionando no regime medido.

---

## 12. Todo parâmetro treinável precisa ser salvo, por nome

Listas de submódulos mantidas à mão saem de sincronia. Neste projeto isso
aconteceu **três vezes**: `frame_time_embed` fora do save, `goal_encoder` fora do
`compactar_backbone`, e a config do backbone omitindo `intermediate_size`.

```python
{n: p for n, p in vla.named_parameters() if p.requires_grad}
```

E o pior caso não é o erro — é o erro **silencioso**: um parâmetro treinado que
volta aleatório faz o treino reportar um número e o checkpoint entregar outro.

---

## 13. Semente que mascara bug

`torch.manual_seed(0)` antes de construir o modelo fazia o backbone aleatório
sair sempre igual **no treino**, e diferente em todo script sem semente. O treino
era consistente por acidente, e isso escondeu por semanas que o backbone nunca
era carregado.

**Se dois processos discordam sobre o mesmo checkpoint, compare a soma dos pesos
antes de investigar o laço.** Quatro diagnósticos meus caíram antes desse.

---

## 14. Baseline antes de modelo

Não é preferência de estilo. Nas três tentativas que falharam, o modelo foi
medido contra nada ou contra métrica saturada, e em duas delas o "resultado"
sobreviveu horas antes de cair.

Dez linhas de regra custam minutos e transformam "o modelo ajuda?" de opinião
em número.

---

## 15. Recompensa em potencial, não em preço solto

A recompensa densa da Fase 1 é `dist_anterior − dist_atual`: exatamente a
forma `Φ(s) − Φ(s')` de **potential-based shaping** (Ng, Harada & Russell,
1999), com `Φ = −distância`. Esse formato tem uma garantia forte: **não muda
a política ótima**, não importa o quão mal calibrada esteja a escala.

Preços soltos (uma constante fixa por ação, tipo `CUSTO_PULO`) não têm essa
garantia — competem contra o retorno futuro descontado por `γ^k`, e o
resultado depende de γ e do horizonte. É por isso que a mesma classe de erro
(§2b) se repetiu três vezes na Fase 2 com valores diferentes de `CUSTO_PULO`:
não existe "o" valor seguro, só valores seguros *para aquele* γ e horizonte.

**Regra:** prefira moldar a recompensa como diferença de um potencial sempre
que der. Quando um preço fixo for inevitável, calcule o horizonte de crédito
antes de escolher o valor:

```
horizonte de credito ~= 1 / (1 - gamma)     # gamma=0,95 -> 20 passos
```

Se o custo de uma ação só se manifesta depois desse horizonte — como o giro
perdido no ar, que só derruba a chegada passos depois do pulo — nenhum preço
pequeno é "seguro": o desconto já amassou o sinal antes dele chegar.

---

## 16. Entropia e taxa de aprendizado não são hiperparâmetros independentes

`lr=1e-4` com bônus de entropia 0,01 colapsou a entropia a zero na 5ª
iteração; `lr=3e-5` com recozimento 0,05 → 0,01 segurou o treino inteiro. Não
foi só o bônus que estava errado — foi a combinação: LR alto empurra os
logits para saturação mais rápido do que qualquer bônus de entropia consegue
puxar de volta.

**Regra:** trate `lr` e o coeficiente de entropia como um par, nunca como dois
botões independentes. Ao subir um, reavalie o outro. E monitore a **curva de
entropia por iteração** como métrica de primeira classe, não só a taxa de
chegada final — com N bins de ação, a entropia máxima é `ln(N)`; se ela cair
perto de zero antes de uns 30-50% do treino, o par lr/entropia está
descalibrado, mesmo que a métrica de tarefa ainda pareça subir.

---

## 17. Variância entre sementes é enorme em RL — uma corrida não é evidência

Achado estabelecido na literatura (Henderson et al., *Deep Reinforcement
Learning that Matters*, 2018): a mesma configuração, rodada com sementes
diferentes, produz curvas que divergem mais do que a diferença entre dois
algoritmos distintos. Um único traço de treino não decide nada sozinho.

Já aconteceu aqui: a primeira corrida da Fase 1 foi de 14% a 90% numa curva
"limpa" — e era um artefato de exploração (ver `experimentos.md` §4), não
aprendizado. Uma segunda semente, rodada antes de comemorar, teria levantado a
suspeita mais barato que a avaliação pareada que a derrubou depois.

**Regra:** antes de declarar que uma mudança (preço, LR, arquitetura) ajudou,
rode com pelo menos **3 sementes** (ou 3 avaliações independentes com arenas
diferentes) e reporte média e faixa, não um número só. Reserve a corrida de
semente única para sondagem rápida ("a recompensa se move?"), nunca para
decisão.

---

## 18. Parâmetro derivado de tempo precisa ser relativo à duração do episódio

`ATRASOS_PASSOS=(0,16,60)` faz sentido num episódio de 400 passos. Num
episódio de 14 (Fase 1), `pilha_frames` faz *clamp* e os atrasos de 16 e 60
colapsam no mesmo frame inicial — dois terços da entrada visual viram um
duplicado, em silêncio, sem erro nem aviso.

**Regra:** todo parâmetro em unidades absolutas de tempo/passos (atrasos,
janelas, horizontes, *warmup*) precisa ser revalidado — ou expresso como
fração do episódio — sempre que o orçamento de passos mudar entre fases. Um
valor razoável na fase anterior pode virar constante degenerada na próxima
sem nenhum sintoma além do desempenho caindo.

---

## 19. Config e estado têm UMA fonte só

Aconteceu duas vezes, com o mesmo mecanismo: a config do backbone estava
copiada à mão em `load_vla_agent` e divergia da real (`intermediate_size`
22016 contra 3072); e o `state_vec` quase ganhou uma segunda definição em JS
quando `estado_sim.EstadoEpisodio` já existia em Python.

**Regra:** toda estrutura que descreve "como o modelo é" (config) ou "o que o
modelo vê" (`state_vec`, espaço de ação) tem exatamente **um** lugar que a
define — carregada ou importada, nunca reescrita à mão num segundo arquivo.
Duas definições da mesma coisa não divergem no dia em que são escritas; elas
divergem em silêncio, meses depois, quando uma muda e a outra não acompanha.

---

## 20. Não retome de um checkpoint contaminado

A Fase 2 degenerou por preço de pulo fraco três vezes. Nas duas primeiras, o
conserto foi ajustar o preço e dar `--retomar` no checkpoint que já tinha
aprendido "pule sempre" — e a política carregava a marca d'água do erro
anterior mesmo com o preço corrigido, porque os pesos já estavam deslocados
para o ótimo degenerado e o recozimento de entropia já tinha avançado.

**Regra:** se uma métrica caiu abaixo de um pico já alcançado antes na mesma
corrida, o checkpoint mais recente está contaminado. Não dê `--retomar` nele.
Volte ao último checkpoint saudável (ou ao fim da fase anterior) e ataque a
causa antes de continuar — retomar sobre um ótimo degenerado só reaprende o
mesmo vício mais devagar.

---

## 21. Meça a feature bruta, não o proxy que ela atravessa

Perfil de relevo por altura do bloco sólido mais alto **atravessa a água**: um
lago raso mede tão plano quanto uma arena de pedra. A arena "plana" filtrada
por esse proxy incluía arenas dentro d'água — achado olhando o visualizador
(§9), não pela métrica.

**Regra:** todo filtro ou feature derivada que pode ser enganada por um caso
de borda do domínio (água, ar, blocos não-sólidos, decoração) precisa ser
checada contra o dado bruto numa amostra, não só confiada por construção.
Combine com §5 (grupo de controle) e §9 (olhe a tela): a checagem barata que
teria pego isso é a mesma.

---

## 22. Ao escalar para mais dimensões de ação: leis que ainda não foram testadas aqui

REINFORCE puro — um passe de gradiente por rollout, retorno descontado com
vantagem normalizada — é o que este projeto usa, e é adequado ao tamanho
atual do espaço de ação. Duas leis da literatura ainda não foram violadas
aqui **porque a tarefa não cresceu o suficiente para expô-las**. Registrar
antes que aconteça custa menos que descobrir depois:

- **Dado on-policy expira.** REINFORCE assume que os dados vêm da política
  atual. Hoje o rollout gera um lote e ele é consumido em minilotes dentro de
  **uma** passada (fatiar por VRAM, não reusar por época). Se algum dia isso
  virar múltiplas épocas sobre o mesmo rollout para acelerar o treino, a
  política que gerou os dados já não é mais a política sendo atualizada, e o
  gradiente fica enviesado — passa a exigir razão de importância com clipe
  (PPO, Schulman et al., 2017), deixa de ser opcional a partir desse ponto.
- **Ações que interagem fisicamente pedem uma cabeça conjunta, não duas
  independentes.** `log p(giro, pulo) = log p(giro) + log p(pulo)` (Fase 2)
  assume que as duas escolhas são independentes dado o estado. Mas
  `airborneAcceleration` mostra que pular **muda** a autoridade de giro em
  5× — as ações interagem no ambiente, então a política fatorada só consegue
  aprender uma média entre "girar bem" e "pular bem", nunca uma condicionada
  na outra. Se o pulo continuar degenerando depois de precificado
  corretamente (§15), este é o próximo suspeito, não o preço.

---

## 23. Colapso de entropia: a causa raiz é covariância entre probabilidade e vantagem

O §16 registrou o sintoma (LR alto + bônus fraco colapsa a entropia) sem
nomear o mecanismo. A literatura de RLVR de 2025-2026 fechou essa lacuna: o
colapso é dirigido pelas ações que já têm probabilidade alta **e** recebem
vantagem positiva repetidamente — cada uma dessas atualizações empurra a
massa de probabilidade ainda mais para o mesmo lugar, num ciclo que se
autoalimenta independente do LR (Cui et al., 2025; *CE-GPPO*, 2025; DAPO —
Yu et al., 2025, ByteDance Seed). É a mesma dinâmica das "ação sem preço vira
constante" do §2: o giro que já tende a ficar em `W` reto recebe vantagem
positiva por estar perto do alvo com mais frequência, e o ciclo colapsa a
distribuição mesmo sem nenhum preço errado.

**Diagnóstico prático:** quando a entropia cair, não olhe só o LR — olhe
*qual* ação está puxando. Se uma ação específica tem alta probabilidade e
alta vantagem média simultaneamente, ela é a candidata ao colapso,
independente de estar certa ou não.

**Se algum dia este projeto adotar clipe estilo PPO** (§22 já previu a
necessidade quando o rollout deixar de ser consumido em época única): use
clipe **assimétrico** (`clip-higher`, DAPO) em vez do clipe simétrico
clássico. Um teto de clipe apertado do lado de cima é o que mais acelera o
colapso, porque impede que ações raras e boas subam de probabilidade — o
mecanismo oposto do que se quer.

---

## 24. Recompensa subindo pode ser só concentração de probabilidade, não aprendizado novo

Achado de 2025 em RLVR (*Spurious Rewards: Rethinking Training Signals in
RLVR* — Shao et al., 2025): RL consegue melhorar a métrica de tarefa mesmo
com recompensa **aleatória ou sem correlação com o acerto**, porque o
gradiente de política redistribui probabilidade para as estratégias que o
modelo **já tinha** a priori — não ensina nada novo, só concentra em cima do
que já era o favorito. O ganho medido é, em boa parte, minimização de
entropia disfarçada de aprendizado.

Este projeto já viveu exatamente isso, antes da literatura nomear o
fenômeno: a primeira corrida da Fase 1 subiu de 14% a 90% concentrando
probabilidade num passeio quase aleatório que já encontrava o alvo por
exploração — a curva de recompensa subindo não distinguiu isso de
aprendizado real (ver `experimentos.md` §4.1).

**Regra:** toda vez que a recompensa de treino sobe, a primeira pergunta não
é "quanto", é **"a política está aprendendo uma habilidade nova, ou só
afiando a lâmina em cima do que a inicialização já favorecia?"**. O teste
barato é o mesmo do §1: rodar a política com a distribuição **congelada** na
inicialização (sem nenhum passo de gradiente) contra o piso `aleatorio` — se
a inicialização sozinha já bate o piso por uma margem parecida com o "ganho"
final, o ganho é suspeito de ser concentração, não capacidade nova.

---

## 25. Normalizar a vantagem pelo desvio padrão do lote pesa mais os lotes quase degenerados

O treino usa `adv = (g - g.mean()) / (g.std() + 1e-6)` — vantagem
centralizada e normalizada pelo desvio do lote, a mesma forma usada em GRPO.
Achado de 2025 (Dr. GRPO — Liu et al.) sobre exatamente essa normalização: um
lote onde os retornos já são quase todos iguais (`g.std()` pequeno) tem seu
gradiente **amplificado**, porque dividir por um desvio pequeno infla a
vantagem de cada amostra — o oposto do que se quer, já que um lote de baixa
variância normalmente significa "tarefa já resolvida" ou "tarefa impossível
daquele jeito", não "sinal forte".

Neste projeto isso é plausível quando um lote sorteia alvos muito
parecidos em dificuldade (todos triviais ou todos obstruídos ao extremo): o
`g.std()` daquele lote cai, e o passo de gradiente nele pesa mais que um lote
de dificuldade mista, sem que nada no log denuncie isso além do próprio
`g.std()`.

**Regra:** logue `g.std()` por iteração junto da entropia. Se ele variar
muito entre iterações e a curva de treino ficar instável na mesma cadência,
suspeite da normalização, não só do LR. A alternativa mais simples de testar
é centralizar sem dividir pelo desvio (só `g - g.mean()`, como em Dr. GRPO)
ou usar uma baseline *leave-one-out* por amostra (RLOO) — comparar cada
retorno com a média das *outras* amostras do lote, o que também remove a
correlação entre a amostra e sua própria baseline.

---

## 26. A mistura manual de dificuldade já é currículo — o próximo passo é priorizar por potencial de aprendizado, não por proporção fixa

O banco da Fase 2 usa mistura fixa de 50% obstruído / 50% livre (`fase2.md`)
porque sorteio cego rende ~5% de alvo obstruído. Isso já é currículo — só que
com a proporção decidida à mão. *Prioritized Level Replay* (Jiang et al.,
2021, ainda a base de trabalhos de 2024-2025 em ambientes procedurais)
formaliza a versão automática: em vez de uma proporção fixa, priorizar para
replay as tarefas onde o agente ainda tem **potencial de melhora** — medido
pelo erro de predição de valor, ou aqui, mais simples, pela distância até o
limiar de sucesso.

**Regra, para quando a Fase 3+ tiver banco de tarefas grande demais para
tunar a mão:** troque a proporção fixa por prioridade de replay — reforçar
tarefas onde o resultado ainda está perto do limiar (nem sempre acerta, nem
sempre erra) e reduzir tarefas onde a política já satura em 0% ou 100%, que
não carregam gradiente útil (ver §2a). A mistura 50/50 atual é a
aproximação de um homem só; não precisa ser jogada fora, só note que é um
caso particular disso, e o próximo ajuste manual de proporção é sinal de que
vale a pena automatizar.

---

## 27. Hackeamento de recompensa generaliza entre habilidades — relevante quando o cérebro combinar mais de uma

Achado de 2025 em modelos de agente com múltiplas ferramentas: aprender a
explorar uma brecha de recompensa numa tarefa **generaliza** para brechas
não relacionadas em outras tarefas — o comportamento de "gamejar a métrica"
não fica contido no contexto onde foi aprendido.

Isso ainda não é um risco ativo aqui — cada fase treina uma habilidade
isolada, com piso e teto medidos separadamente (§1, §14). Mas o roadmap do
projeto é explícito: `habilidades.py` existe para o `CerebroRegra` compor
várias habilidades treinadas separadamente. **Quando isso acontecer,** um
defeito de incentivo numa habilidade (o próximo "pule sempre" ou "gire
sempre para o mesmo lado") deixa de ser um problema isolado daquela
habilidade — vale auditar se o vício aparece em contextos onde a habilidade
nem deveria ter sido chamada, não só medir a taxa de sucesso dela sozinha.
