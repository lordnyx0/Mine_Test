# Handoff — Qwen3Loop + DeltaNet

Estado em 2026-08-02. Tudo pronto para rodar a Fase 1 v2.

> **Leia antes de qualquer coisa. Três coisas mudaram a natureza do projeto:**
>
> 1. **Os updates do Adam eram descartados** no arredondamento do bf16. Invalida
>    parcialmente tudo que a v1 mediu. Ver "O bug do arredondamento em bf16".
> 2. **`Qwen/Qwen3-0.6B` é o modelo PÓS-TREINADO**, não a base. O documento dizia
>    "Base" e isso induziu a escolha de um corpus 70% FineWeb — ativamente
>    destrutivo para um modelo com pós-treino. Ver "O mislabel Base/Instruct".
> 3. **A Fase 1 virou DESTILAÇÃO**, não LM loss. O corpus de texto cru custou
>    10,6 pontos de benchmark em 1M tokens. Ver "A virada: Fase 1 por destilação".
>
> As três já estão corrigidas no código. Mas o histórico de números deste
> documento foi produzido antes delas — trate qualquer valor anterior a
> 2026-08-02 como suspeito até remedido.

---

## Objetivo

1. **Fase 1** — adaptar o `Qwen/Qwen3-0.6B` (o **pós-treinado**, não a base) a
   uma arquitetura em loop (LoopSplit), **por destilação**: o aluno loopado
   aprende a reproduzir a saída do original sem loop. Produz o modelo que serve
   de **professor** para a Fase 2, com o cronograma de execução correto.
2. **Fase 2** — substituir a atenção softmax por Gated DeltaNet nas camadas do miolo,
   recuperando a capacidade via *attention transfer* (destilação a partir do professor).

Hardware: **RTX 3060 12 GB** (12.288 MiB, ~11.700 livres com o sistema limpo).

---

## Comece por aqui

```bash
cd C:\Users\Nyx\Desktop\Testes
iniciar_fase1.bat          # destilação, 15M tokens, 1103 steps, ~6,4 h
avaliar_checkpoints.bat    # depois: benchmark nos checkpoints (~1,6 h)
```

O `.bat` valida arquivos, checa VRAM livre, lista processos usando a GPU, avisa antes
de retomar de checkpoint e imprime os comandos seguintes ao terminar.

**Antes de rodar: feche o LM Studio** se tiver modelo carregado. O treino precisa de
~9 GB livres.

---

## Estado dos artefatos

| caminho | o que é |
|---|---|
| `data/mixture/mixture_train.jsonl` | corpus bruto 70/30 (FineWeb + raciocínio ChatML), 14M tokens |
| `data/mixture_mixedblocks/` | **dataset de treino**, 26.491 seqs, 14.0M tokens, integridade 100% |
| `checkpoints/qwen3loop_full/final_model` | **Fase 1 v1** (30M tokens) — ver "v1 falhou" abaixo |
| `models_quantized/fase1_q8/fase1_q8_0.gguf` | GGUF Q8 da v1, 56 camadas desenroladas |
| `logs/baseline_eval.json` | linha de base das 3 arquiteturas |
| `results/base_q8_2048/`, `results/fase1_q8_2048/` | respostas do benchmark |
| `RELATORIO_FASE1.md` | relatório completo da v1 |
| `Testes2/deltanet_fase2/` | modelo da Fase 2 já com DeltaNet no miolo |
| `scratch/bench_otim.py` | bancada de otimização — 19 configs medidas |
| `scratch/bench_parte{1..5}.json` | resultados brutos da matriz |
| `scratch/probe_dtype.py` | em que dtype o treino realmente roda |
| `scratch/test_rounding.py` | prova numérica do bug de arredondamento |
| **`data/probe_chatml/`** | **corpus de sondagem da destilação** — 30M tok, 35.275 seqs |
| `scripts/build_probe_corpus.py` | constrói o corpus de sondagem |
| `scratch/test_destilacao.py` | verifica a KL (aluno idêntico → 0,000000) |
| `scratch/bench_maxlen.py` | teto de comprimento e tamanho do bloco de logits |
| `scratch/isola_vram.py` | A/B de autocast (descartou a hipótese) |
| `scratch/comparar_runs.py` | compara duas corridas pelo `metrics.csv` |
| `avaliar_checkpoints.bat` | benchmark nos checkpoints depois do treino |
| `logs/teste1M_b2_chunked.csv` | baseline do teste de 1M (batch 2, loss em blocos) |

---

## Números de referência (não remedir)

**Eval loss na split de `data/mixture_mixedblocks`** (`scripts/baseline_eval.py`):

| arquitetura | eval loss | PPL |
|---|---|---|
| Qwen3-0.6B original — **ALVO** | **2.916** | 18.5 |
| loop N=2 simples, sem treino | 7.031 | 1131 |
| LoopSplit, sem treino | 3.610 | 37.0 |

**ATENÇÃO — existem DOIS benchmarks e eles NÃO são comparáveis entre si.**

*(a)* 20 questões, GGUF Q8, `max_tokens 2048`, via `run_q8_benchmarks.py`:

| modelo | acurácia |
|---|---|
| Qwen3-0.6B | **82.5%** |
| Fase 1 v1 | 55.0% |
| Qwen3-1.7B (relatório antigo) | 87.5% |

*(b)* **97 itens, 11 categorias, NF4, `enable_thinking: false`**, via
`run_evaluation.py` — é este que `avaliar_checkpoints.bat` usa:

| modelo | overall |
|---|---|
| `base` — Qwen3-0.6B, `num_loops=1` | **79.6%** |
| LM loss, 1M tokens (teste) | 69.0% |
| `loop-untrained` | **NUNCA MEDIDO** — falta a referência que separa dano do loop de efeito do treino |

Misturar (a) e (b) numa mesma frase é erro: quantização, número de itens e
harness são todos diferentes.

**Matriz de otimização — 19 configurações medidas** (`scratch/bench_otim.py`,
resultados em `scratch/bench_parte{1..5}.json`).

VRAM = NVML, uso da placa inteira, no PIOR caso (batch montado com as sequências
mais longas do dataset real, medido depois do primeiro `optimizer.step()`).
Velocidade = tokens reais, sem padding, em amostra representativa. Teto da placa:
**12.288 MiB**.

| config | VRAM | reserved | tok/s | vs. v1 |
|---|---|---|---|---|
| `gc1_b1` — o que a v1 rodou | 9.034 | 8.010 | 452 | 1,00× |
| `gc0_b1` — GC desligado | 12.106 ⚠ | 17.430 | 253 | 0,56× |
| `gc1_b2` | 11.119 | 10.322 | 911 | 2,02× |
| `gc1_b4` | 12.038 ⚠ | 14.066 | 535 | 1,18× |
| `fused_gc1_b2` | 11.233 | 10.322 | 900 | 1,99× |
| `exp_gc1_b2` — expandable_segments | 11.126 | 10.322 | 904 | 2,00× |
| `8bit_gc1_b2` — bnb AdamW8bit | 9.928 | 9.098 | 896 | 1,98× |
| `8bit_gc1_b4` | 12.179 ⚠ | 14.192 | 571 | 1,26× |
| `8bit_gc0_b4` | — | — | **OOM** | — |
| `fp32_gc1_b1` | 12.067 ⚠ | 13.212 | 342 | 0,76× |
| `fp32_gc1_b2` | 11.938 ⚠ | 15.662 | 164 | 0,36× |
| `fp32_8bit_gc1_b1` | 10.445 | 9.588 | 428 | 0,95× |
| `fp32_8bit_gc1_b2` | 12.145 ⚠ | 11.750 | 390 | 0,86× |
| **`sr_gc1_b2` — CONFIG ATUAL** | **9.746** | **9.098** | **894** | **1,98×** |
| `sr_gc1_b1` | 8.208 | 7.456 | 435 | 0,96× |

⚠ = passou dos 12.288 MiB e caiu em memória compartilhada. O driver WDDM aceita
em vez de dar OOM, e o treino **desaba de velocidade sem nenhum aviso** — é o
travamento perto dos 11 GB. Por isso `gc0_b1` fica mais LENTO, não mais rápido.

Tempo de uma corrida de 14M tokens: **8,6 h** no config da v1, **4,3 h** no atual.

---

## Decisões de arquitetura — todas medidas

| decisão | evidência |
|---|---|
| **LoopSplit** em vez de loop simples | dano 3.610 vs 7.031 (6× menos), mesmo custo de 56 execuções |
| **Full fine-tune** em vez de QLoRA | 1053 vs 756 tok/s **e** 596M vs 40M params treináveis |
| **DeltaNet no miolo** (camadas 7-20) | dano +7.23 vs +10.41 das pontas, convertendo 3× mais execuções |
| **Fases separadas** | a Fase 1 é o professor do attention transfer da Fase 2 |
| LR 3e-5 | full FT; os 2e-4 do config antigo eram de LoRA e destruiriam o pré-treino |

**LoopSplit**: 28 camadas → 56 execuções. `[0-6]` e `[21-27]` rodam 1×, `[7-20]` rodam 3×.
Só o miolo itera. Com LoopSplit `_get_num_loops()` devolve 1, então **não há RMSNorm entre
passagens** — por isso o desenrolamento para GGUF é exato (validado: erro 7.6e-4 em fp32,
argmax 100%).

**Nanbeige4.2**: verificado que os pesos publicados (Base e Instruct) usam o loop **simples**,
sem LoopSplit/mHC/depth-attention/n-gram — todos com default `False`. O `qwen3loop` já
estava alinhado com eles antes das minhas mudanças.

---

## O bug do arredondamento em bf16

**Este é o achado mais importante do projeto até agora.** Encontrado em
2026-08-02, presente desde o começo, invisível em todas as métricas coletadas.

O modelo é carregado em bf16, então pesos, gradientes E momentos do Adam ficam
todos em bf16 — não existe cópia mestra em fp32 (confirmado em
`scratch/probe_dtype.py`). bf16 tem 8 bits de mantissa. Na magnitude típica dos
pesos (|w| ≈ 0,023) isso dá um ULP de **9e-5**, enquanto o passo do Adam é
~lr = **3e-5** — um terço de um ULP, abaixo de meio ULP. Com arredondamento ao
mais próximo, `w - update` volta para o MESMO `w`.

Medido em `scratch/test_rounding.py` — 2000 steps, gradiente sempre no mesmo
sentido, deslocamento do peso contra o mesmo treino em fp32:

| esquema | deslocamento | recuperação |
|---|---|---|
| fp32 + AdamW fp32 (referência) | +0.058465 | — |
| **bf16 + AdamW torch — o que a v1 rodou** | **-0.000051** | **-0,1%** |
| bf16 + AdamW8bit bitsandbytes | -0.000051 | -0,1% |
| bf16 + torchao AdamW8bit SEM stochastic | -0.000051 | -0,1% |
| bf16 + torchao AdamW8bit + **stochastic round** | +0.055232 | **94,5%** |
| fp32 + AdamW8bit bitsandbytes | +0.055096 | 94,2% |

Trocar só o otimizador para 8 bits **não corrige**: o defeito está na escrita no
peso, não no estado.

**A correção adotada** é `training.optim: ao8bit_sr` — `torchao.optim.AdamW8bit`
com `bf16_stochastic_round=True`. Os pesos seguem em bf16 e o arredondamento é
sorteado com probabilidade proporcional à fração, então o update vale em
ESPERANÇA. Corrige em esperança, não em cada step: um passo de 3e-5 contra ULP
de 9e-5 vira 1/3 de chance de mover o peso um ULP inteiro. Ao longo de milhões
de pesos e 1657 steps isso converge — os 94,5% acima são essa convergência.

A alternativa determinística (`fp32_8bit`, cópia mestra em fp32) recupera o
mesmo, mas só cabe em batch 1 a 428 tok/s: sob `autocast` o PyTorch mantém
cópias bf16 dos pesos durante o forward, e com LoopSplit são **56 passagens**.
A arquitetura em loop penaliza precisão mista mais que um transformer normal.

**Trava contra reincidência:** `train.py` aborta se `optim` for `adamw_torch`
com pesos em bf16, com a explicação e o número medido. Libera com `--force`
(para reproduzir a v1 de propósito).

### O que isto muda na leitura da v1

A v1 aprendeu (3.610 → 2.5807), então o canal não estava fechado — estava
estreito. Mas a "saturação por volta de 13-16M tokens", registrada abaixo como
característica do treino, tem agora uma explicação mais simples e mais provável:
com `cosine`, o LR só cai, então a fração de updates que sobrevivem ao
arredondamento encolhe ao longo do treino. **O treino pode ter parado de
conseguir escrever, não de ter o que aprender.** Não trate os 13-16M como
orçamento suficiente comprovado — isso precisa ser remedido com o treino
corrigido.

---

## O mislabel Base/Instruct

`Qwen/Qwen3-0.6B` **é o modelo pós-treinado.** Tem chat template, tokens
`<think>`/`</think>`, e `apply_chat_template` produz papel de assistant. Na
release do Qwen3 a base é `Qwen/Qwen3-0.6B-Base`, que nunca foi usada aqui.

Este documento dizia "Qwen3-0.6B **Base**" desde o começo, e o `iniciar_fase1.bat`
repetia. Não é detalhe de nomenclatura: **a decisão de usar um corpus 70% FineWeb
faz sentido para adaptar um modelo base e é destrutiva para um pós-treinado.**
Foi essa premissa errada que moldou todo o pipeline de dados.

O padrão de dano no benchmark confirma — não é esquecimento genérico, é o
pós-treino sendo sobrescrito por texto de web (ver a tabela abaixo).

---

## A virada: Fase 1 por destilação

**O que a Fase 1 faz agora:** o aluno LoopSplit aprende a reproduzir a
distribuição de saída do `Qwen/Qwen3-0.6B` **sem loop**. Mesmos pesos,
cronogramas diferentes (56 execuções contra 28). Supervisão por KL sobre os
logits de saída.

### A evidência que forçou a mudança

Corrida de teste, 1M tokens, pipeline **já corrigido** (dataset auditado,
otimizador certo, updates chegando nos pesos). Benchmark de 97 itens:

| categoria | base | após 1M com LM loss | Δ |
|---|---|---|---|
| General Knowledge | 90,0 | 50,0 | **-40,0** |
| Robustness | 87,5 | 62,5 | **-25,0** |
| Instruction Following | 75,0 | 55,0 | **-20,0** |
| Translation | 51,7 | 36,7 | -15,0 |
| Creativity | 100,0 | 85,7 | -14,3 |
| Mathematics | 70,0 | 60,0 | -10,0 |
| Summarization | 100,0 | 95,2 | -4,8 |
| Reasoning | 70,0 | 70,0 | 0,0 |
| Writing | 100,0 | 100,0 | 0,0 |
| Programming | 81,7 | 86,7 | +5,0 |
| Context | 50,0 | 57,1 | +7,1 |
| **Overall** | **79,6** | **69,0** | **-10,6** |

Enquanto isso a eval loss MELHOROU (geral 3,610 → 3,388). O mesmo divórcio da v1,
agora com o pipeline limpo. **Não era o dataset fragmentado nem os updates
truncados — é o objetivo.** Treinar para prever texto move o modelo em direção ao
ótimo daquele texto, que não é o modelo original.

### Por que destilação resolve por construção

O alvo passa a ser a saída do próprio original. O ótimo da loss é literalmente
"comporte-se como o original" — não existe para onde derivar. Os -40 pontos em
General Knowledge deixam de ser possíveis.

### Por que KL de logits e não attention transfer

Attention transfer exige correspondência 1:1 entre camadas. O aluno tem 56
execuções contra 28 do professor, e as 3 iterações do miolo acontecem em estados
diferentes do residual stream — não existe alvo definido para elas. Forçar o
mesmo alvo nas 3 ensinaria o loop a computar a mesma coisa três vezes.

KL sobre a saída final não precisa de mapeamento nenhum. **Na Fase 2 a
correspondência existe** (professor e aluno compartilham o cronograma), e lá o
attention transfer camada a camada volta a ser o método certo.

### Verificação da implementação

`scratch/test_destilacao.py`:

| aluno | KL contra o professor |
|---|---|
| idêntico ao professor (sem loop) | **0,000000** ✓ |
| **LoopSplit, sem treino** | **0,6923 nats** |
| loop N=2 simples, sem treino | 4,5170 nats |

O zero exato prova que a loss mede o que promete. E os 0,6923 do LoopSplit batem
com o gap de eval loss medido meses antes por outro caminho (3,610 − 2,916 =
0,694) — duas medições independentes concordando na terceira casa.

**Pela primeira vez o alvo é um número conhecido e alcançável: zero.**

### A ressalva que decide o corpus

A KL só é minimizada **sobre a distribuição de entradas amostrada**. Fora desse
suporte o aluno diverge livremente. Como o modelo é usado em ChatML, o corpus tem
que ser ChatML — `data/mixture_mixedblocks` (70% FineWeb) alinharia a região que
não importa.

Medição relacionada, que corrigiu um palpite meu: o dano do LoopSplit é **igual**
nas duas regiões (0,6329 nats em FineWeb, 0,6391 em ChatML — razão 1,01×). O
argumento pelo corpus ChatML **não** é "chat está mais danificado"; é "você só
conserta onde amostra".

### O que a destilação solta

Truncar deixa de ser problema: o alvo é a distribuição do professor em cada
posição, e num corte o professor continua dando probabilidades corretas. Somem:

- o descarte de 59% dos diálogos por passarem de 1100 tokens
- a exigência de um diálogo inteiro por amostra
- o teto de 1200 como restrição de integridade
- a sensibilidade ao `<|im_end|>` duplicado (bug real: 100% das 7.351 sequências
  de raciocínio do corpus antigo têm o token repetido no fim)

Todo o trabalho do `prepare_mixed_blocks.py` resolvia um problema que esta rota
não tem.

### O corpus de sondagem

`scripts/build_probe_corpus.py` → `data/probe_chatml`, 30.000.949 tokens em
35.275 sequências, mediana 1024.

| fonte | tokens | por quê |
|---|---|---|
| `ianncity/GLM-5.2-Conversation` | 18M (60%) | Apache-2.0, ~100 tópicos, prompts sintéticos (contaminação baixa) |
| `Qyrou/reasoning-corpus-4K-5M-v1` | 9M (30%) | campo ChatML nativo do Qwen3, com `<think>` |
| FineWeb local | 3M (10%) | ancoragem para entradas que não são chat |

Filtro de contaminação por 5-gramas contra os 97 itens do benchmark: 2.858
descartados (~7,5%, provavelmente com falsos positivos de frases genéricas —
conservador de propósito).

**Descartado:** `Manusagents/...Distillation-Dataset` (18,5M linhas, 76,5 GB) —
o card diz que `instruction` contém "file path" e `response` contém "file
content", ou seja despejos de repositório e não conversas; enviesa para código
(28 de 73 fontes); e arrasta AGPL-3.0 no upstream.

**Descartado como PROFESSOR:** `r0b0tlab/qwen3.8-max-distillation-50k`. Motivo
decisivo: **texto puro, sem logits** — não dá para calcular KL. Usá-lo seria SFT
comum, que é a LM loss de volta. Além disso um professor mais forte torna o alvo
inalcançável (KL nunca chega a 0) e reabre o esquecimento. As fontes upstream
incluem GSM8K, MATH, HumanEval, MBPP, IFEval, ARC — contaminação direta do
benchmark se usado como alvo.

---

## RESULTADO — Fase 1 v2 concluída (2026-08-03)

15M tokens, 1103 steps, ~6 h. `checkpoints/qwen3loop_v2/final_model`.

| métrica | sem treino | **final_model** |
|---|---|---|
| KL contra o professor | 0,6293 | **0,0372** |
| razão PPL aluno/professor | ~1,88 | **1,038** |
| eval geral | — | 0,0645 |
| eval raciocínio | — | **0,0319** |
| dependência do loop | −0,6293 | **+0,2059** |

**93% do dano do LoopSplit reparado.** E o raciocínio ficou MELHOR que o geral
em todos os pontos de avaliação — o padrão que matou a v1 (raciocínio degradando
sob uma média que parecia boa) não apareceu em momento nenhum.

### O loop não foi desligado

Era o risco central desta rota: imitar o professor sem loop poderia ser obtido
tornando as iterações extras identidade. `scratch/diag_loop.py` mediu:

| checkpoint | KL c/ loop | KL s/ loop | dependência | it2/it1 |
|---|---|---|---|---|
| sem treino | 0,6293 | 0,0000 | −0,6293 | 0,343 |
| 100 | 0,1005 | 0,1854 | +0,0849 | 0,232 |
| 200 | 0,0868 | 0,2245 | +0,1377 | 0,164 |
| 300 | 0,0824 | 0,2362 | +0,1538 | 0,136 |
| 500 | 0,0527 | 0,2376 | +0,1849 | 0,119 |
| 800 | 0,0398 | 0,2429 | +0,2030 | 0,114 |
| **final** | **0,0372** | **0,2431** | **+0,2059** | **0,113** |

Remover as iterações extras do final leva a KL de 0,037 para 0,243 — **6,5×
pior**. Elas carregam trabalho estrutural, e cada vez mais ao longo do treino.

A razão `it2/it1` caiu de 0,343 para 0,113 e **estabilizou** a partir do step 500.
Não estava indo a zero; convergiu para um equilíbrio.

### Estrutura descoberta no miolo

Delta por camada no final_model — as camadas PROFUNDAS do miolo usam mais o loop:

```
camada    it1      it2      it3
  13    0.2704   0.0286   0.0196
  19    0.7177   0.0606   0.0733    <- it3 > it2
  20    0.5021   0.0897   0.1143    <- it3 > it2
```

Nas camadas 19 e 20 a TERCEIRA passagem contribui mais que a segunda. **O risco
da Fase 2 se concentra aí**: a camada 19 tem o maior `it1` do miolo (0,7177), e
o erro do GatedDeltaNet em cada execução pesa proporcionalmente ao delta daquela
execução.

### Saturou no step 800

De 800 a 1103 (27% da corrida, ~1,7 h): KL melhorou 6,5%, dependência 1,4%,
`it2/it1` nada. **~11M tokens teriam dado o mesmo modelo.** Confundido pelo
cosine, como sempre — mas é o dado mais concreto que existe sobre orçamento
nesta rota.

### Lição para a próxima corrida

**Salve checkpoints com espaçamento menor no INÍCIO.** A `it2/it1` percorreu de
0,343 a 0,136 nos primeiros 300 steps e depois ficou parada. Com `save_steps: 100`
uniforme, 9 dos 12 checkpoints caíram no platô e a varredura que serviria à
Fase 2 ficou com só 3 pontos úteis. Salvar a cada 25 steps nos primeiros 200
daria a varredura de verdade.

### BENCHMARK — 81,8% (base: 79,6%)

Rodado em `results_fase1v2/`, mesmo harness de 97 itens.

| categoria | base | LM loss 1M | **destilação 15M** | vs base |
|---|---|---|---|---|
| General Knowledge | 90,0 | 50,0 | 70,0 | **−20,0** |
| Reasoning | 70,0 | 70,0 | 80,0 | +10,0 |
| Mathematics | 70,0 | 60,0 | 80,0 | +10,0 |
| Programming | 81,7 | 86,7 | 91,7 | +10,0 |
| Writing | 100,0 | 100,0 | 100,0 | 0,0 |
| Translation | 51,7 | 36,7 | 48,3 | −3,4 |
| Summarization | 100,0 | 95,2 | 100,0 | 0,0 |
| Instruction Following | 75,0 | 55,0 | 65,0 | **−10,0** |
| Creativity | 100,0 | 85,7 | 100,0 | 0,0 |
| Context | 50,0 | 57,1 | 64,3 | +14,3 |
| Robustness | 87,5 | 62,5 | 100,0 | +12,5 |
| **Overall** | **79,6** | **69,0** | **81,8** | **+2,2** |

**NÃO leia o +2,2 como melhoria.** São 7-10 itens por categoria, então um item
vale 10-14 pontos: `Reasoning 70→80` é UMA resposta a mais em dez. No agregado,
+2,2 sobre 97 itens são ~2 respostas — dentro do ruído.

O que os dados sustentam: **a capacidade foi preservada.** O que é significativo
é o contraste com a outra rota: −10,6 pontos são ~10 itens, fora do ruído, e na
mesma direção em cinco categorias.

**Sinal real, não ruído:** `General Knowledge` é a categoria mais frágil nas DUAS
rotas (−40 com LM loss, −20 com destilação) e `Instruction Following` segue a
mesma direção (−20 e −10). Conhecimento factual é o que o LoopSplit mais perturba,
e a destilação reduziu o dano sem eliminá-lo.

### O que NÃO foi feito

`loop-untrained` segue sem medição no benchmark — a referência que separaria dano
do loop de efeito do treino.

**(histórico) O benchmark de 97 itens não rodou nesta v2.** A preservação de capacidade está
ARGUMENTADA (a destilação torna a degradação estruturalmente improvável) e
MEDIDA EM KL (0,037 no corpus de sondagem), mas não medida no critério que este
projeto estabeleceu com custo. `avaliar_checkpoints.bat` continua pronto; dá para
rodar depois, inclusive sobre a saída da Fase 2.

`loop-untrained` também segue sem medição no benchmark.

---

## Por que a v1 falhou (não repita)

30M tokens, eval loss **3.610 → 2.5807** (abaixo do alvo 2.916). Parecia sucesso.
Mas o benchmark caiu de **82.5% para 55.0%**.

Causa: `block_size` 512 uniforme fatiava diálogos ChatML de ~1400 tokens em ~3 pedaços.
Só **2,5%** dos blocos tinham um par `<think>…</think>` completo. O modelo aprendeu a
abrir raciocínio e nunca fechar — gera até o teto e devolve resposta vazia.

**A eval loss não viu nada disso.** Perplexidade menor ≠ capacidade maior.

Correção na v2: texto geral empacotado em 512 (contínuo, sem estrutura a quebrar) +
**um diálogo inteiro por amostra**, nunca cortado, teto 1200. Diálogos maiores são
**descartados no download** pelo campo `tok_len` (`--max_reasoning_tokens`), não baixados
e jogados fora. Integridade verificada: **0% de `<think>` sem fechar, 100% terminam em EOS**.

Custo: cobertura de ~41% dos diálogos (só os curtos).

---

## Bugs corrigidos (todos no código)

1. **Gradient checkpointing era no-op** no `qwen3loop`: o forward chamava as camadas
   direto e o `gradient_checkpointing_enable()` do HF nunca era acionado. 14.92 → 5.99 GB.
2. **`packed_dir` roteava para fallback**: `packed_dir_for()` resolvia um caminho por
   convenção, não achava, e o auto-prepare **rebaixava FineWeb puro** sem avisar. Foram
   78 steps treinados no corpus errado. Agora `dataset.packed_dir` é explícito e falha alto.
3. **`tokens_per_step` usava `block_size`**: com sequências variadas isso inflava a
   contagem 4×. O treino terminaria com 3.6M tokens reportando 14M. Agora lê a média
   real do `dataset_stats.json`.
4. **`labels` pré-computados quebram o padding dinâmico**: `tokenizer.pad()` não sabe
   padear esse campo. Só falha com `batch > 1`, então smoke tests com batch 1 mascaram.
5. **`extract_text` pegava os campos errados**: concatenava `tok_len` + conteúdo
   duplicado. O campo certo é `ChatML` (formato nativo Qwen3, com `<think>` no assistant).
6. **`group_by_length` não existe** no transformers 5.x.
7. **`torch_dtype` → `dtype`**; `device_map="auto"` quebrava a cirurgia de camadas.
8. **Updates truncados no bf16** — ver a seção própria acima. O mais grave dos
   oito: não quebrava nada, não aparecia em métrica nenhuma, e desperdiçava a
   maior parte de cada corrida.
9. **Eval loss cega ao raciocínio**: o split de validação era aleatório sobre o
   corpus misto e ficava **70% FineWeb** (medido: 349 de 500 amostras). Era essa
   diluição que deixava a degradação de formato passar. Agora o `eval_dataset` é
   um dicionário e o Trainer emite `eval_geral_loss` e `eval_raciocinio_loss`
   separadas — mesmo orçamento de 500 amostras (250+250), com a parte de
   raciocínio subindo de 151 para 250. O `eval_loss` legado continua existindo,
   recombinado por peso de TOKENS (não de amostras: as sequências de raciocínio
   são bem mais longas, e a loss do Trainer é média por token).
10. **Banner mentia sobre o contexto**: imprimia `Context: 2048 tokens`, valor de
   `dataset.block_size` — sobra do pipeline de packing antigo, que não
   corresponde a nada no corpus misto (geral 512, raciocínio até 1200, maior
   sequência real 1172). Agora lê o `dataset_stats.json`.
11. **Logits estourando a VRAM sem erro**: o vocabulário do Qwen3 tem 151.936
   entradas. Com batch 2 e sequências de 1172, `lm_head(hidden)` produz 712 MB em
   bf16, que o `loss_function` do HF promove a fp32 (mais 1.425 MB), somando o
   gradiente do mesmo tensor — mais de 3 GB concentrados na última camada.
   Medido: `reserved` **16.226 MiB numa placa de 12.288**, derramando para a RAM
   via WDDM, sem OOM, só perda silenciosa de velocidade. Corrigido com
   `ChunkedLossTrainer` (blocos de 1024, recomputação no backward):
   **16.226 → 6.550 MiB e 700 → 840 tok/s**. A classe já existia em
   `Testes2/treinar.py`, escrita para a Fase 2, com o ganho anotado na docstring
   (12,6 → 9,2 GB). A Fase 1 nunca a usou.
12. **`<|im_end|>` duplicado**: `prepare_mixed_blocks.py` acrescenta EOS a um
   texto que já termina em `<|im_end|>`. **100% das 7.351 sequências de
   raciocínio** do corpus antigo terminam com o token repetido. Irrelevante na
   rota de destilação (o professor vê a mesma entrada), mas o corpus antigo
   carrega isso.
13. **Instrumentação de VRAM enganosa**: o TUI e o CSV mostravam
   `memory_allocated`, que marcava 2.315 MB enquanto a placa estava com 12.177 de
   12.288 e o treino já derramava. Agora `get_memory_info()` devolve pico
   alocado, pico reservado e o uso da placa inteira via NVML, e o CSV tem as três
   colunas mais `tps_inst` (o `tps` antigo era média desde o início e escondia a
   queda de velocidade).

---

## Erros de método meus — evite repetir

Foram a causa de dois runs perdidos. Em todos, medi a métrica conveniente em vez da que decide:

- **`max_memory_allocated` subestima a VRAM em ~2,7 GB.** O que causa OOM é
  `max_memory_reserved` + baseline + contexto CUDA. A diferença é fragmentação do
  caching allocator, alta quando as sequências têm tamanhos variados.
- **Medir a média em vez do pior caso.** Duas sequências longas no mesmo batch tinham
  0,13% de chance — ~15 vezes num treino. Roda centenas de steps e estoura sem aviso.
  Sempre medir com as sequências **mais longas** do dataset real.
- **Medir velocidade só nas sequências longas superestima.** A GPU satura nelas; com a
  mediana de 512 e batch 1 o overhead por forward domina. Velocidade → amostra
  representativa. VRAM → pior caso. São medições **separadas**.
- **`max_tokens 512` no benchmark mascarou a degradação** (11 de 20 respostas truncadas
  nos dois modelos). Com 2048 a diferença real apareceu: 82.5% vs 55.0%.
- **`empty_cache` periódico não reduz o pico** — parecia reduzir porque eu media logo
  após a limpeza.
- **`print` sem `flush=True`** em processo longo esconde o progresso. Use `python -u`.
- **Nunca verifiquei em que dtype o treino realmente rodava.** Passei o projeto
  inteiro tratando "bf16" como um detalhe de configuração, sem uma linha
  checando `param.dtype` / `optimizer.state[p]['exp_avg'].dtype`. Três caracteres
  de `print` teriam achado o bug antes das duas corridas. Quando um número não
  fecha com a expectativa — o `adamw_8bit` economizou 1,2 GB onde a conta pedia
  3,6 — **essa discrepância É o achado**, não um arredondamento a ignorar. Foi
  ela que entregou o bug.
- **Otimização por intuição em vez de medição.** As quatro "otimizações não
  medidas" que deixei priorizadas por retorno esperado: a nº 1 estava invertida
  (desligar GC é 44% mais LENTO), a nº 3 era nula (`fused` não muda nada porque
  com `grad_accum: 16` o passo do otimizador roda 1 vez a cada 16 forward), e a
  melhor de todas não estava na lista. Ordenar por palpite deu pior que aleatório.
- **Atribuir causa sem testar a causa.** Registrei o gap de 2,7 GB entre
  `allocated` e `reserved` como fragmentação do caching allocator. Era uma
  hipótese plausível que virou fato no documento sem nunca ser testada.
  `expandable_segments:True` — que existe exatamente para isso — deixou o
  `reserved` idêntico ao byte (10.322 nos dois casos). A explicação estava errada:
  eram os logits.
- **Bancada de poucos micro-batches não reproduz o Trainer real.** A bancada dava
  9.746 MiB para a configuração que o treino real levou a 16.226. Medir 6
  micro-batches num processo limpo não captura o que centenas de steps com
  tamanhos variados fazem com o allocator. **Toda decisão de VRAM tem que ser
  confirmada numa corrida real de algumas dezenas de steps**, não só na bancada.
- **`nvidia-smi` / NVML mede a placa inteira, não o processo.** A mesma
  configuração deu `placa` 10.854 numa medição e 10.406 na outra, porque o
  desktop varia. O número reprodutível é o `reserved` do PyTorch (9.948, idêntico
  nas duas). Reportei "folga real de 1.434 MiB" com base numa leitura ruidosa; a
  folga verdadeira fica entre 1.400 e 1.900 dependendo do desktop.
- **Diagnóstico apressado por semelhança de número.** Vi `reserved` 16.226 e
  lembrei do caso `gc0_b1` (17.430), concluí que o gradient checkpointing tinha
  voltado a ser no-op, e "corrigi" passando `gradient_checkpointing_kwargs` ao
  TrainingArguments. O `reserved` continuou 16.226 **byte por byte** — sinal de
  que nada mudou. A correção do kwarg é legítima (o Trainer realmente sobrescreve
  o `use_reentrant`), mas não era a causa. Semelhança de magnitude não é
  diagnóstico.
- **Generalizar de um teste barato.** Afirmei que a região de chat estaria mais
  danificada pelo LoopSplit que a de texto cru. Medido: 0,6329 contra 0,6391 —
  iguais. A conclusão sobre o corpus continuou certa por outra razão, mas eu tinha
  apoiado ela numa afirmação inventada.

---

## Critério de parada da Fase 1

**O benchmark, não a loss.** Vale para as duas rotas, e a evidência agora é dupla:
a v1 (eval loss melhorou, benchmark caiu 27,5 pts) e o teste de 1M com o pipeline
já corrigido (eval loss geral 3,610 → 3,388, benchmark 79,6% → 69,0%).

Rode com `avaliar_checkpoints.bat` DEPOIS do treino. Rodar durante é impossível
nesta placa: o benchmark gera texto (~14 min/modelo) e precisaria de um caminho
de geração residente junto com os 9.948 MiB do treino. Não cabe.

**Na rota de destilação a loss mudou de papel.** A KL de treino já é a métrica
principal, logada a cada 2 steps a custo zero, e seu ótimo é **zero** — número
conhecido e alcançável, ao contrário do alvo 2,916 emprestado de outra medição.
O eval separado (`geral` vs `raciocinio`, 5 pontos na corrida) serve só como
detector de divergência entre regiões. O benchmark continua decidindo.

A partir da v2 há um sinal intermediário: `eval_raciocinio_loss`, separada da
`eval_geral_loss` no TUI e no CSV. Se a de raciocínio subir enquanto a geral cai,
é o padrão da v1 se repetindo — pare e rode o benchmark. Mas é sinal **precoce**,
não critério de parada: uma loss de raciocínio boa ainda pode esconder falha de
formato na geração livre, que só o benchmark pega.

Na v1 a eval loss cruzou o alvo já no step 200 e saturou por volta de **13-16M tokens** —
metade do orçamento bastava. Por isso a v2 usa 14M. **Não confie nesse número:**
a v1 rodou com os updates truncados pelo bf16, e a "saturação" pode ter sido o
treino perdendo a capacidade de escrever conforme o cosine derrubava o LR, não
convergência de verdade. Com o treino corrigido, 14M pode ser pouco.

Rode o benchmark nos checkpoints intermediários **enquanto existem** (`save_total_limit: 4`
apaga os antigos). Na v1 isso me impediu de testar se havia um ponto bom antes da degradação.

---

## Config atual (`config.yaml`)

```yaml
distill:
  enabled: true
  teacher_model_id: Qwen/Qwen3-0.6B   # o original, SEM loop
  temperature: 1.0
  chunk: 512                          # posições por bloco de logits
  alpha_lm: 0.0                       # 0 = destilação pura
dataset:
  packed_dir: data/probe_chatml       # corpus de SONDAGEM, não de alvo
training:
  target_tokens: 15000000             # -> 1103 steps, ~6,4 h
  optim: ao8bit_sr                    # obrigatório com pesos bf16
  per_device_train_batch_size: 4      # 4 x 4 = 16 seqs/step
  gradient_accumulation_steps: 4
  eval_steps: 200                     # 5 avaliações; a KL de treino é a métrica
  save_steps: 100 / save_total_limit: 20   # 12 checkpoints, nenhum apagado
```

O corpus tem 30M e a corrida usa 15M: **meia época**, nenhuma sequência vista
duas vezes. Para destilação isso é o ideal — cada token compra cobertura nova.

### Config da rota antiga (LM loss), para referência

```yaml
model:
  model_id: Qwen/Qwen3-0.6B
  enable_double_loop_split: true      # LoopSplit
dataset:
  packed_dir: data/mixture_mixedblocks   # CAMINHO EXPLÍCITO — não remover
training:
  finetune_mode: full                 # full | qlora
  target_tokens: 14000000             # -> 1657 steps
  learning_rate: 3.0e-05              # full FT (qlora seria 1e-4..2e-4)
  optim: ao8bit_sr                    # OBRIGATÓRIO com pesos bf16 — ver o bug
  per_device_train_batch_size: 2      # 2 x 8 = 16 seqs/step, o MESMO batch
  gradient_accumulation_steps: 8      #   efetivo de antes (1 x 16)
  gradient_checkpointing: true        # NÃO desligar — 44% mais lento
  save_steps: 50 / eval_steps: 50 / save_total_limit: 4
```

`batch 2 × accum 8` mantém as 16 sequências por step do config antigo de
propósito: o ganho de 2× vem de ocupar melhor a GPU, **não** de mudar a dinâmica
de otimização. `max_steps` segue 1657 e o orçamento segue 14M tokens, então as
curvas continuam comparáveis com as da v1.

`dataset.block_size: 2048` continua no arquivo mas **não descreve este corpus** —
é sobra do pipeline de packing uniforme antigo. O cálculo de orçamento ignora
quando detecta que a média real (528) diverge mais de 5%, e o banner agora lê o
`dataset_stats.json`. Mantido só porque `validate_config` exige potência de 2 e
1200 não é.

---

## Otimizações — MEDIDAS (2026-08-02)

A bancada rodou: `scratch/bench_otim.py`, teto de 5 min por caso, cada caso em
subprocesso separado (a fragmentação do allocator não atravessa processos).

| otimização | previsão antiga | medido |
|---|---|---|
| **Desligar gradient checkpointing** | "maior retorno, ~30%" | **FALSA. 44% mais lento** |
| **Batch maior** | "medir 2 e 4" | batch 2 dobra; **batch 4 é pior** |
| **`adamw_torch_fused`** | "alguns % de graça" | **nula** — 900 vs 911 tok/s |
| **`expandable_segments`** | (não listada) | **nula** — `reserved` idêntico |
| **`adamw_8bit`** | (não listada) | −1,2 GB de VRAM, velocidade igual |
| **stochastic rounding** | (não listada) | **corrige o bug + 2× de velocidade** |

**Por que desligar GC é mais lento:** com LoopSplit o miolo roda 3×, então são
**56 execuções de camada** guardando ativações, não 28. Sem checkpointing o
`reserved` pede 17,4 GB numa placa de 12. Gradient checkpointing não é opcional
nesta arquitetura — é o que a torna treinável.

**Por que batch 4 é pior que batch 2:** 16,2% de desperdício em padding (contra
6,9% no batch 2) somado ao estouro dos 12 GB. O padding vem de não haver mais
`group_by_length` no transformers 5.x: sequências de 512 e de 1172 caem no mesmo
batch e as curtas são preenchidas até a maior.

**Por que `fused` não muda nada:** com `gradient_accumulation_steps` alto, o
passo do otimizador roda uma vez a cada N forward/backward. Não é o gargalo.

**Custo do eval** (`scratch/bench_eval.py` — medido COM o estado de treino
residente, que é a condição real; medir num processo limpo daria número inútil):

| eval batch | s/eval de 500 | pico | custo em 33 evals |
|---|---|---|---|
| 1 | 125,7 | 8.549 | 69,1 min |
| **2 — adotado** | **72,1** | **8.481** | **39,7 min** |
| 4 | 114,4 | 10.246 | 62,9 min |
| 8 | 139,1 | 11.752 ⚠ | 76,5 min |

Batch 2 poupa ~29 min por corrida com pico MENOR que o batch 1. Batch 4 e 8 são
piores — mesmo padrão do treino: o desperdício de padding cresce mais rápido que
o ganho de paralelismo. A previsão de que "forward puro tem folga larga para
batch 4 ou 8" estava certa sobre a memória (8.549 com o treino residente, contra
o pico de 9.746 do treino) e **errada sobre a velocidade**.

### Teto de comprimento e bloco de logits (destilação)

`scratch/bench_maxlen.py`, batch 4, **todas** as sequências no teto (pior caso —
no corpus de sondagem a mediana bate no teto de qualquer forma):

| caso | placa | reserved | tok/s |
|---|---|---|---|
| **b4_len1024** | 10.854 | **9.948** | **692** |
| b4_len1536 | — | — | **derrama** |
| b4_len2048 | — | — | **derrama** |
| b2_len2048 | 11.030 | 10.572 | 490 |
| b2_len3072 | — | — | **derrama** |

**1024 é o teto.** Note que `b2_len2048` processa os MESMOS 4.096 tokens por
micro-batch que `b4_len1024` e é 29% mais lento gastando mais memória — a atenção
é quadrática no comprimento, então sequências longas custam dos dois lados.

Tamanho do bloco de logits (`distill.chunk`), mesma configuração:

| chunk | reserved | tok/s |
|---|---|---|
| 512 | **9.948** | 692 |
| 256 | 10.286 | 688 |
| 128 | 9.696 | 681 |

**Não é alavanca** — não é monotônico e o ganho do 128 são 252 MiB por 1,6% de
velocidade. A razão: o bloco já está sob `checkpoint()`, então suas ativações
nunca são guardadas. O pico já estava resolvido. Mantido em 512.

Folga real da corrida: **1.400 a 1.900 MiB** sobre `reserved` 9.948, dependendo
do que o desktop estiver usando. É a parte frágil do plano — qualquer app que
tome VRAM durante as ~6 h empurra para o derrame, sem erro, só lentidão.

### Ainda em aberto

1. **Avaliar checkpoints intermediários** não tem ferramenta. `run_q8_benchmarks.py`
   exige GGUF e sobe um `llama-server`; `eval_checkpoint.py` só calcula loss (e é
   do caminho QLoRA). Para amostrar a curva de benchmark contra tokens — que é o
   instrumento para decidir o orçamento — falta um script que gere as respostas
   de `benchmarks/eval_benchmark.json` direto do checkpoint PyTorch, no formato
   que `score_responses.py` já consome. **Ressalva de método:** os números de
   referência (base 82,5%, v1 55,0%) foram medidos em Q8 GGUF. Comparar um
   checkpoint bf16 contra eles misturaria quantização na conta — é preciso medir
   o base pelo MESMO caminho PyTorch primeiro.
2. **`dataloader_num_workers`** (hoje 0) — no Windows o custo de spawn é alto e
   o dataset já vem tokenizado em Arrow. Retorno provavelmente baixo.
3. **Bucketing por tamanho** — recuperaria os 6,9% de padding do batch 2. Exige
   sampler custom (o `group_by_length` do Trainer não existe mais). Cuidado: ao
   agrupar as longas no mesmo batch, o pior caso de VRAM PIORA.
4. **`torch.compile`** — arriscado com a arquitetura custom, e o gargalo são as
   56 passagens de camada.
5. **`loop-untrained` no benchmark de 97 itens** — a referência que separa dano do
   LoopSplit de efeito do treino. `avaliar_checkpoints.bat` já a inclui.
6. **Cobertura de tradução** — o GLM-5.2-Conversation é só em inglês, e Translation
   é a categoria mais fraca do base (51,7%). Sem prompts multilíngues na sondagem,
   o aluno fica sem restrição ali.
7. **A tensão do loop.** Se a destilação funcionar perfeitamente, o ótimo é as
   iterações extras virarem quase-identidade: 56 execuções entregando a qualidade
   de 28. Para a Fase 1 como INSTRUMENTO isso basta — o que ela precisa produzir é
   um professor com o cronograma certo. Mas não é ganho de capacidade.
   **Empatar com os 79,6% do base é SUCESSO, não decepção.** `distill.alpha_lm`
   existe para dosar isso; qualquer valor acima de zero reabre tanto a chance de
   superar o professor quanto a de derivar dele.

---

## Depois da Fase 1

```bash
python scripts\export_loopsplit_q8.py --model checkpoints\qwen3loop_v2\final_model --outdir models_quantized\fase1v2_q8
python scripts\run_q8_benchmarks.py --model-path models_quantized\fase1v2_q8\fase1_q8_0.gguf --model-name fase1v2 --max-tokens 2048
python scripts\score_responses.py base_q8_2048 fase1v2
```

O exportador valida a equivalência em fp32 e **aborta** se divergir, em vez de gerar um
GGUF silenciosamente errado.

**Ressalva do GGUF**: desenrolar 28 → 56 camadas duplica os pesos (610 MB → 1056 MB) e
sacrifica a vantagem central do looped transformer. O Nanbeige guarda 22 camadas físicas,
não 44. É limitação do llama.cpp, que não suporta pesos compartilhados em loop. Para
distribuir preservando a arquitetura, use o PyTorch de 28 camadas.

### Fase 2

**Estado (2026-08-03):** `deltanet_fase2/` foi RECONSTRUÍDO a partir de
`checkpoints/qwen3loop_v2/final_model`. O `FASE1` no `preparar_fase2.py` apontava
para `qwen3loop_full/final_model` — a v1, de 55% de benchmark e updates
truncados, cuja pasta hoje está VAZIA. A versão antiga foi movida para
`deltanet_fase2_OBSOLETO_v1/`.

Transplante: **42 das 56 execuções de atenção convertidas (75%)**, camadas 7-20,
herdando as projeções q/k/v/o já adaptadas ao loop pela Fase 1.

O modelo da Fase 1 **não** deve ser sobrescrito: é o professor.

**Ponto de partida escolhido: `final_model`** (KL 0,0372, dependência +0,2059,
it2/it1 0,113). Se a conversão ficar aquém, o `checkpoint-100` é a segunda
tentativa — ele tem as iterações extras com o DOBRO da magnitude (it2/it1 0,232)
ao custo de KL 2,7× pior. A hipótese de que magnitude alta ajuda o GatedDeltaNet
é especulação, não medição; a diferença entre as duas conversões responderia.

**A trava do otimizador já foi portada** para `Testes2/treinar.py`
(`escolher_otimizador()` + `optimizer_cls_and_kwargs`). Era necessária: o lr de lá
é 8e-5, que dá **0,89 ULP** — já na fronteira antes de o cosine começar a
derrubá-lo. E o alvo de MSE do attention transfer tem magnitude muito menor que
uma loss de linguagem, então o passo efetivo é ainda menor. **A Fase 2 estava
mais exposta ao bug de arredondamento que a Fase 1, não menos.**

O `ChunkedLossTrainer` de lá foi a origem da correção de logits da Fase 1 — o
caminho inverso do usual, vale olhar o que mais já está resolvido lá.

Falta implementar o **attention transfer**: treinar cada `GatedDeltaNet` para imitar a
saída da `Qwen3Attention` correspondente (MSE, camada a camada). É o caminho viável —
o transplante direto parte de loss ~9.8 contra ~2.6 do professor.

O `Testes2/treinar.py` já tem a Gated Delta Rule chunkwise implementada e validada
(erro 5e-16 contra a recorrência sequencial, causalidade verificada, cache recorrente
para geração incremental funcionando).

---

## Se for escalar

- **Qwen3-1.7B** rende mais que otimizar o 0.6B: 87.5% contra 82.5% da base. Todo o
  código é indiferente ao tamanho — é trocar `model_id`.
- **Kaggle T4/P100**: P100 não vale (sem tensor cores). T4 tem 16 GB mas **não suporta
  bf16** (precisa compute 8.0+), forçando fp16 e `GradScaler`.
- **5090 alugada** (~$0,50-1/h) domina: 32 GB, bf16 nativo, sem limite de sessão.
  ~$5 por uma corrida de 100M tokens.
