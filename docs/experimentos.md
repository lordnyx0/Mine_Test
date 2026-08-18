# Experimentos

Registro do que foi medido. A ordem é temática, não cronológica.

Regra deste documento: **todo número tem como foi obtido, e toda hipótese
refutada fica registrada** — refutação que não é escrita vira trabalho repetido.

---

## 1. Clonar o planejador — três tentativas, três fracassos

### 1.1 Clonagem direta por passo

Treinar a rede para reproduzir a tecla que o `Piloto` apertou.

| medida | valor |
|---|---|
| recall por classe do giro, split por **episódio** | **19%** |
| acaso (9 bins) | 11% |
| com tolerância de ±1 bin | 35-40% contra 33% de acaso |

**A ação do planejador não é uma função da observação do aluno.** Ele decide
com um BFS sobre voxels num raio de 16; o aluno vê uma imagem em primeira
pessoa. Não é falta de capacidade — é teto de informação.

> **Cuidado com números antigos de ~39%.** Vieram de split por blocos de 50, que
> ainda cai dentro do mesmo episódio de 70 passos. Amostras vizinhas são quase
> idênticas. O número honesto é 19%.

### 1.2 Rotulagem em retrospecto

A ideia: em vez de "que tecla o professor apertou", perguntar "que teclas
levaram de A até B", onde B é onde o agente **de fato** chegou k passos depois.
O rótulo deixa de ser ambíguo por construção, e não há professor.

Inferibilidade (features privilegiadas rotas+estado, holdout por episódio):

| k (horizonte) | sem objetivo | retrospecto | objetivo embaralhado |
|---|---|---|---|
| 4 (1,0s) | 19,2% | **33,8%** | 18,7% |
| 20 (5,0s) | 18,0% | **29,8%** | 20,0% |

O controle embaralhado empata com "sem objetivo", então o ganho é informação e
não capacidade. E sobrevive a k=20, então não é vazamento.

**Mas o treino completo falhou:** 4 épocas, perda caindo monotônica, recall de
treino 23% → 37%. Na avaliação pareada a ~98 blocos:

| política | chegada | fechou | travado |
|---|---|---|---|
| reto | 52% | 75% | 42% |
| piloto | 85% | 91% | 15% |
| **modelo** | **5%** | **6%** | **77%** |

Diagnóstico: emitia `W` em 28% (alvo 100%), `A` em 93% e `SHIFT` em 62% (alvos
0%). Andava de lado agachado. Com `W` forçado para isolar o giro: **0% de
chegada**, travado caiu para 35% — ele anda, mas na direção errada.

Descartados por medição: pré-processamento (diferença 0,00001 entre treino e
execução), orientação do alvo/perda, carregamento do checkpoint.

### 1.3 O que isso fecha

Três tentativas independentes, a última com o melhor argumento teórico. **Não
se destila um algoritmo de busca numa rede reativa por casamento de ação.**

---

## 2. A representação nunca foi o gargalo

### 2.1 O colapso, e o conserto

Treinar com "sempre aperte W" fez o projetor visual colapsar:

| | antes | depois do alvo de rota |
|---|---|---|
| projetor, cos médio | 0,9994 | 0,8006 |
| projetor, posto efetivo | **1,4** / 1024 | 3,2 |
| hidden, posto efetivo | 4,7 | **47,2** |

A previsão de rotas (12 setores de navegabilidade) é o único termo que obriga a
via visual a permanecer informativa. Em holdout, a cabeça de rotas fica **60%
abaixo** do preditor cego (0,0596 contra 0,1499), e o alvo tem variância real
(desvio 0,39 por setor, 43% saturado em 0 ou 1).

### 2.2 A hipótese do pooling — testada e FALSA

Uma sonda linear mostrava as rotas mais legíveis nas posições de imagem que na
última posição, sugerindo que ler `last_hidden_state[:, -1, :]` desperdiçava
metade da geometria. Reajustando as cabeças sobre cada vetor:

| vetor de decisão | giro acc | recall/cls | rota MSE |
|---|---|---|---|
| última posição (o atual) | 64,4% | **39,7%** | 0,0596 |
| média das posições de imagem | 46,4% | 24,7% | 0,0653 |
| média de tudo | 55,8% | 37,3% | 0,0511 |

**A última posição carrega menos geometria e decide melhor.** Geometria legível
e ação inferível são coisas diferentes. **Não mexer no pooling.**

---

## 3. Locomoção de longo alcance

### 3.1 O regime importa

| distância do alvo | reto | piloto |
|---|---|---|
| 13 blocos, 70 passos | 95% | 92% |
| ~100 blocos, 400 passos | 52-55% | 80-90% |

A 13 blocos a métrica **satura** e o planejador parece inútil. A vantagem só
aparece em distância real. Medir no regime errado produz a conclusão oposta.

### 3.2 Três hipóteses sobre as falhas, três refutações

**Mínimo local.** `Objetivos.rumo` (alvo final + memória de visitas) foi
implementado para soltar o bot de concavidades. **Empata com o guloso** —
80% contra 80%, traços por lote quase idênticos.

**Água.** Presente em 2 de 6 falhas (33%). Grupo de controle: **21 de 34 dos
sucessos** (62%). Água é mais comum entre quem chega.

**Elevação.** Nas falhas o alvo está em média 1,7 bloco **abaixo** do bot; nos
sucessos, 1,3 acima. Sinal invertido.

### 3.3 O que se sabe das falhas

Não são de orçamento: em três rodadas, **0 de 17** falhas ainda melhoravam
quando o tempo acabou. São bloqueio duro, e a causa segue não identificada.

Dois modos distintos: travar quase imediatamente (recorde nos passos 7-14) e
viajar 40% do caminho antes de empacar.

Um artefato: ~17% das "falhas" chegavam a menos de 3,2 blocos com
`RAIO_CHEGADA = 2,5`. São chegadas perdidas pelo limiar.

---

## 4. Fase 1 — controle local por RL

Ver [fase1.md](fase1.md) para o desenho. Aqui os resultados.

### 4.1 A primeira corrida aprendeu a trapacear

120 iterações de REINFORCE. Curva limpa: chegada 14% → 90%, recompensa
−17,7 → +8,1. Parecia sucesso.

Na avaliação: **14,1%**, contra 89,1% do geométrico.

A varredura de ângulo explicou tudo. Cena e distância fixas, variando só a
direção do alvo de −180° a +180°:

```
angulo    argmax    distribuicao
 -180°      4       5  6  6  6 52  6  8  6  5
    0°      4       5  6  6  6 52  6  8  6  5
 +180°      4       5  6  6  6 52  6  8  6  5
```

**Idêntica, dígito por dígito, em 13 ângulos.** A política ignora o objetivo.

Os 90% vinham de **amostrar** essa distribuição fixa: com alvo a 3-8 blocos e
40 passos de orçamento, passeio aleatório encontra o alvo. A recompensa foi
gamejada por exploração.

**O erro de desenho foi meu:** o piso era `so_W`, determinístico, que marca
1,6%. Faltava um baseline de **passeio aleatório** — ele teria marcado ~90% na
primeira medição, antes das 4 horas de treino.

### 4.2 Onde a informação para e o veredito

| ponto da cadeia | variação com o ângulo do alvo |
|---|---|
| saída do `goal_encoder` | cos 0,629 — varia muito |
| hidden que alimenta a cabeça | cos 0,997 — quase constante |

Atravessar o backbone congelado atenua ~100×. Mas a sonda linear é decisiva:

```
R² do ângulo do alvo a partir do 1024:  sin 0,978 | cos 0,962
erro angular mediano da sonda:          3 graus   (acaso ~90)
```

**Caso B.** A informação está no 1024 e é linearmente legível com 3 graus de
precisão. Uma cabeça linear poderia tê-la usado. A que existe não usou — o
problema é de otimização da política, não de representação nem de arquitetura.

### 4.3 O conserto

De tarefa, não de modelo:

| conserto | efeito medido |
|---|---|
| orçamento 40 → 14 passos | passeio aleatório cai de ~90% para **7,8%** |
| baseline `aleatorio` permanente | o modo de falha não passa mais despercebido |

E o teto sobrevive: geométrico continua em 85,9% com 14 passos.


---

## 5. O backbone nunca era carregado

**O achado mais consequente do projeto.** `load_vla_agent` fazia:

```python
qwen_model = Qwen3LoopModel(config).to(device)   # ALEATORIO
# ...e nunca carregava checkpoints/qwen3loop_v2/final_model
```

O checkpoint do VLA guarda só os adaptadores. **O backbone de 0.6B foi ruído
aleatório o projeto inteiro**, até 2026-08-13.

### O sintoma, e por que demorou a ser achado

```
treino reporta          88-100%
qualquer avaliacao       5-14%
```

`treinar_fase1.py` chama `torch.manual_seed(0)` **antes** de construir o modelo,
então sorteava sempre o MESMO backbone aleatório e os adaptadores aprendiam
contra ele. Scripts de avaliação não semeavam, sorteavam outro, e os adaptadores
viravam lixo.

O treino era consistente **por acidente**. Isso mascarou o bug e me levou a
quatro diagnósticos errados antes do certo:

| suspeita | como caiu |
|---|---|
| laço de avaliação não chamava `observar()` | era bug real, mas corrigi-lo não mudou 14,1% |
| argmax vs amostragem | 6,2% contra 7,8% — nenhum dos dois |
| `frame_time_embed` não salvo | era bug real; corrigido, e continuou 6% |
| decoreba do conjunto replayado | três sementes de tarefa deram 83-92% |

O que fechou: rodar o **mesmo `rollout`** em dois processos e comparar a soma dos
pesos. O backbone diferia entre sementes.

### O que isso reinterpreta

- A atenuação de ~100× do objetivo atravessando o backbone: comportamento
  esperado de um transformer aleatório.
- "O LM não paga o próprio lugar": certo pelo motivo errado — não era um 0.6B
  treinado subaproveitado, era ruído.
- As sondas de representação **não** ficam inválidas: projeção aleatória preserva
  informação linearmente legível, e é por isso que rotas saíam 60% abaixo do cego
  e o ângulo do alvo a R² 0,97. Muda a interpretação, não os números.

E a config era codificada à mão, omitindo `intermediate_size`: o padrão 22016
contra os 3072 reais. O backbone aleatório tinha MLPs 7× maiores que o modelo
verdadeiro.

### O que o backbone real muda — medido

100 iterações, tudo idêntico exceto o backbone:

| iterações | aleatório | treinado |
|---|---|---|
| 1-20 | 11% | **26%** |
| 21-40 | 43% | **57%** |
| 41-60 | 59% | **70%** |
| 61-80 | 72% | 70% |
| 81-100 | 76% | **83%** |

**Velocidade de aprendizado, não teto.** Os dois convergem para o que `atan2`
consegue (~86%), porque é isso que a Fase 1 pede. O LM treinado paga o próprio
lugar, mas numa tarefa que não precisa dele.

---

## 6. Fase 2 — calibração

Ver [fase2.md](fase2.md) para o desenho. Os números da calibração:

| distância | política | geral | `obstruido` |
|---|---|---|---|
| 3-8 | `geo_pulo` | 85,4% | 54% |
| 3-8 | `piloto` r16 | 77,1% | 62% |
| 14-30 | `geo_pulo` | 36,2% | **33%** |
| 14-30 | `piloto` r16 | 50,0% | 36% |
| 14-30 | `piloto` **r40** | 53,8% | **60%** |

Três coisas que a calibração pegou antes de qualquer treino:

1. **A 3-8 blocos não cabe obstáculo** — só 13 de 96 alvos obstruídos, e o
   planejador fica *pior* que apontar-e-andar.
2. **O raio do BFS precisa cobrir a distância do alvo** — com r16 contra alvos a
   30 blocos o teto media 36% quando o real era 60%.
3. **Alvo obstruído tem rendimento de ~5%** no sorteio cego, o que exige banco
   pré-gerado com mistura controlada.
