# Ajustes pendentes — base para o plano de conserto

Documento de trabalho, não de referência permanente. Consolida a auditoria de
`docs/metodo.md` §1-27 contra o código e os checkpoints reais, feita em
2026-08-13 por 5 agentes (um por faixa de regras) mais uma reconstrução
adicional a partir do conteúdo bruto dos `.pt` em `checkpoints_vla/`. Serve de
insumo para decidir o próximo passo da Fase 2 — não é o plano em si, é o
levantamento que o plano deveria usar.

---

## 1. Qual checkpoint usar — resposta direta

```
python -c "import torch; ck=torch.load('checkpoints_vla/NOME.pt', map_location='cpu', weights_only=False); print(ck['iteracao'], len(ck['hist']))"
```

foi rodado contra os 8 arquivos de `checkpoints_vla/`. Resultado:

| checkpoint | mtime | iteração salva | taxa (início→fim) | pulo (início→fim) | veredito |
|---|---|---|---|---|---|
| `BASE_locomocao_limpa.pt` | 10/08 21:01 | — | — | — | base pré-Fase-1, não mexer |
| `vla_locomotion.pt` | 11/08 04:26 | — | — | — | base, não mexer (nome protegido em `docs/atual.md`) |
| `vla_fase1_backbone_aleatorio.pt` | 12/08 23:46 | — | — | — | evidência do bug do backbone aleatório (`experimentos.md` §5); histórico, não usar |
| `vla_objetivo.pt` | 12/08 02:41 | — | — | — | do braço de imitação/retrospecto abandonado (`experimentos.md` §1.2); fora do caminho de RL atual |
| **`vla_fase1.pt`** | **13/08 01:24** | **100/100** | 53%→**87,5-100%** (últimas 7 it.) | — | **SAUDÁVEL. É a base correta para reiniciar a Fase 2.** |
| `vla_fase2_pulo_degenerado.pt` | 13/08 11:47 | 95/120 | 41,7%→48,3% (médias 15 primeiras/15 últimas) | 58,1%→**98,9%** | tentativa 1 (`v2`, "pulo sem preço"). Falha conhecida — guardado só como evidência |
| `vla_fase2_entropia_alta.pt` | 13/08 12:13 | 25/120 | oscila 0-50%, termina em 0% | 34-72%, instável | tentativa 2 (`v3`, "entropia somada crua"). Falha conhecida — guardado só como evidência |
| **`vla_fase2.pt`** | **13/08 12:33** | **15/120** | 53,6%→44,6% (médias 7 primeiras/7 últimas) | 46,8%→**69,9%** (médias 7/7) | tentativa 3 (`v5`, `CUSTO_PULO=0,05`). **Contaminado. NÃO retomar (§20).** |

**Regra prática:** o próximo treino de Fase 2 carrega `--ckpt-entrada
checkpoints_vla/vla_fase1.pt` (pesos limpos, cabeças recriadas), nunca
`--retomar checkpoints_vla/vla_fase2.pt` nem qualquer um dos dois arquivos
`_degenerado`/`_alta`. Isso já está certo em `docs/atual.md`, mas hoje é só
disciplina de quem digita o comando — não há nada no código que impeça o
erro (ver §7 na tabela abaixo).

### Cronologia reconstruída da Fase 2 (a partir dos `.pt`, não só do texto)

Os três arquivos acima são, na ordem em que foram salvos por último, as três
tentativas que `docs/atual.md` resume em uma tabela. Os números batem:

1. **`v2` — `pulo_degenerado.pt`** (95 iterações rodadas): pulo sobe até
   quase 99% nas últimas 15 iterações — bate com "pulo → 99%" do resumo.
2. **`v3` — `entropia_alta.pt`** (25 iterações, abandonada cedo): taxa cai a
   zero na última iteração salva, pulo oscila sem convergir — consistente
   com "entropia subindo, pulo travado".
3. **`v5` — `vla_fase2.pt`** (15 iterações, é o checkpoint pausado hoje):
   `CUSTO_PULO=0,05`, o valor ainda hardcoded como default em
   `treino/treinar_fase2.py:55`.

### Achado novo desta reconstrução: o "it 16" que motivou a pausa não é verificável

`docs/atual.md` cita três pontos — `it 13`, `it 15`, `it 16` — para justificar
a pausa e o plano de subir `CUSTO_PULO`. O checkpoint salvo (`iteracao=15`,
`len(hist)=15`) **não contém `it 16`** — checkpoint é salvo a cada 5
iterações (`it % 5 == 0`), então a corrida foi interrompida entre a 15ª e a
20ª, e o número de `it 16` só existia no console no momento da decisão. Sem
stdout redirecionado a arquivo (violação de §10, ver abaixo), esse ponto de
dado não pode mais ser reproduzido.

Além disso, o `hist` persistido mostra `taxa` em incrementos de 0,125 — ou
seja, **N=8 episódios por iteração**. Com N=8, o desvio-padrão de uma
proporção em torno de 50% é ≈17,7 pontos percentuais; uma "queda" de 38%
para 12% em duas iterações está dentro de ~1,5 desvio-padrão de ruído puro
de amostragem, não necessariamente um colapso limpo. O padrão de subida do
`pulo` (46,8%→69,9% de média early/late) é mais consistente que a queda da
`taxa` — mas a leitura "chegada caiu por causa do pulo" hoje se apoia em 15
pontos ruidosos de telemetria de treino, não numa avaliação formal
(`avaliar_fase2.py`, que usa `n` maior e é determinística por seed fixa).
**Antes de gastar outra corrida de treino, vale rodar uma avaliação formal
do `vla_fase2.pt` atual contra `geo_pulo`/`piloto` para confirmar que a
degradação é real e não ruído de lote pequeno.**

---

## 2. Bloqueadores — achados que tocam direto a decisão de retomar a Fase 2

Em ordem de peso, não de número de seção:

### 2.1 — `CUSTO_PULO=0,05` (o valor já contaminado) continua sendo o default no código
`treino/treinar_fase2.py:55`: `CUSTO_PULO = float(os.environ.get("CUSTO_PULO", "0.05"))`.
O plano de subir para `~0,15` existe só em `docs/atual.md`; rodar o comando
documentado sem exportar `CUSTO_PULO=0.15` manualmente reproduz a mesma
tentativa que já falhou. (§2, §7)

### 2.2 — O horizonte de crédito não cobre o episódio
`GAMMA=0,95` (`treino/treinar_fase1.py:46`, reusado na Fase 2) dá horizonte
de crédito `1/(1-γ) ≈ 20` passos. `PASSOS_MAX_F2=50` — o episódio é **2,5×**
mais longo que o horizonte. Isso é a condição exata que `docs/metodo.md` §15
descreve como "nenhum preço pequeno é seguro": não é so uma questão de achar
o valor certo de `CUSTO_PULO`, é que o desconto amassa qualquer preço antes
dele alcançar o passo onde o custo real (perda de autoridade de giro no ar)
se manifesta. **Subir para 0,15 pode não resolver — as alternativas de
desenho são `γ` maior (aumenta o horizonte) ou reformular o custo do pulo
como diferença de potencial em vez de preço fixo por passo.** (§15)

### 2.3 — A ação conjunta (giro, pulo) compartilha uma única vantagem
`politica/politica_fase2.py`: `log_prob = lp_y + lp_j`, e o REINFORCE usa uma
`advt` só para os dois. Isso significa que qualquer vantagem positiva média
reforça girar e pular **juntos**, mesmo quando só um dos dois estava certo.
É o mecanismo mais plausível para explicar por que `pulo` sobe rápido
enquanto a `taxa` bate teto de ruído — mais rápido do que "preço fraco
acumulando devagar" explicaria sozinho. (§22, §23)

### 2.4 — Os diagnósticos que as regras novas pedem não existem ainda
Nem `treinar_fase1.py` nem `treinar_fase2.py` logam:
- vantagem média condicionada à ação de pulo vs. à ação de giro (§23 pede
  isso para identificar qual ação está "puxando" o colapso);
- `g.std()` do lote por iteração, ao lado da entropia (§25 — a normalização
  `adv = (g - g.mean()) / (g.std() + 1e-6)`, linha `treinar_fase2.py:178`,
  pode estar inflando o gradiente de lotes de baixa variância sem que nada
  denuncie isso hoje).

Sem esses dois logs, é impossível decidir entre as hipóteses 2.2 e 2.3 (ou
uma terceira: normalização de vantagem amplificando um lote ruim) antes de
gastar outra corrida completa.

### 2.5 — Falta piso do mesmo tipo da política para a Fase 2
`avaliacao/avaliar_fase2.py` só compara contra `geo_pulo` (determinístico) e
`piloto` (determinístico). A política treinada é estocástica
(`pol.amostrar = True`). Não existe um piso "vaga e pula sem ler a entrada"
para a Fase 2 — o mesmo tipo de lacuna que custou 4h na Fase 1 com `so_W`.
(§1)

### 2.6 — O held-out de tarefas existe no código mas nunca é lido
`treino/treinar_fase2.py:141-144` separa 15% do banco para avaliação, mas
`avaliacao/avaliar_fase2.py` nunca importa esse corte — gera tarefas ao vivo
do mesmo arquivo de largadas usado no treino. Risco: ganho de visão medido
pode ser memorização de cena, não generalização. (§4)

---

## 3. Higiene barata — sem decisão de desenho, só consertar

Estes não exigem escolher entre alternativas; são defaults perigosos ou
lacunas de instrumentação que dá para fechar direto:

| item | onde | o que fazer |
|---|---|---|
| `PilotoBFS(raio=16)` e `montar_tarefas(..., raio=16, ...)` ainda são o default de classe/função | `ambiente/fase2.py:90,184` | trocar o default para `40` — o valor sabotado não deveria ser o que sobra quando alguém esquece de passar o argumento (§11) |
| `goal_encoder` (treinável) fora da lista à mão de `compactar_backbone` | `infra/gpu_utils.py:128` | incluir `vla.goal_encoder` na lista, ou trocar por `{n:p for n,p in vla.named_parameters() if p.requires_grad}` como já é feito no save/load (§12) |
| `DIST_MIN`/`DIST_MAX`/`PASSOS_MAX_F2` revertem para os defaults da Fase 1 (`3,0`/`8,0`/`20`) se a env var não for exportada | `ambiente/arena_plana.py:37-38`, `ambiente/fase2.py:49` | `assert` de regime no início de `treinar_fase2.py`/`avaliar_fase2.py` (ex.: `assert DIST_MIN >= 14`) (§3, §18) |
| Não existe utilitário para comparar soma de pesos entre processos, que §13 prescreve como diagnóstico padrão | — | um script curto em `infra/` que imprime `sum(p.sum() for p in vla.parameters())` dado um checkpoint |
| Corrida longa sem stdout redirecionado a arquivo | comandos documentados em `docs/atual.md`, `docs/fase2.md` | acrescentar `> logs/fase2_$(date).log` (ou equivalente) nos comandos documentados (§10) — é o que teria salvado o `it 16` |
| Nenhuma checagem visual (`/ver`) registrada antes da decisão de pausar | — | abrir `http://127.0.0.1:3002/ver` e observar o pulo degenerando antes de decidir o próximo preço (§9) |
| Falta o teste de "distribuição congelada" do §24 | — | antes do próximo treino, rodar `vla_fase1.pt` **sem nenhum passo de gradiente** contra `geo_pulo`/piso — se já bate perto do que a corrida pausada atingiu, o "ganho" observado é concentração de probabilidade, não aprendizado novo |

---

## 4. Tabela completa das 27 regras

| § | status | achado (uma linha) |
|---|---|---|
| 1 | VIOLAÇÃO | Fase 2 sem piso estocástico, só determinístico |
| 2 | VIOLAÇÃO ativa | `CUSTO_PULO=0,05` (já contaminado) é o default no código |
| 3 | OPORTUNIDADE | regime de distância/passos reverte pra Fase 1 se env var esquecida |
| 4 | VIOLAÇÃO parcial | held-out de tarefas existe mas não é lido na avaliação |
| 5 | CONFORME | quebra por faixa de desvio, com `reta` como controle |
| 6 | CONFORME (frágil) | `observar()` chamado nos 4 laços de rollout existentes, mas são 4 cópias |
| 7 | OPORTUNIDADE | 3ª tentativa de preço sem cálculo de horizonte de crédito |
| 8 | N/A | nada para reproduzir no momento |
| 9 | OPORTUNIDADE | decisão de pausar apoiada só em telemetria agregada, sem abrir `/ver` |
| 10 | VIOLAÇÃO | corrida pausada sem stdout redirecionado — `it 16` é irreproduzível |
| 11 | OPORTUNIDADE | `PilotoBFS`/`montar_tarefas` ainda têm `raio=16` como default |
| 12 | VIOLAÇÃO | `goal_encoder` fora da lista à mão de `compactar_backbone` |
| 13 | CONFORME + oportunidade | ordem da semente ok; falta utilitário de soma-de-pesos |
| 14 | CONFORME | baselines medidos antes das 3 corridas |
| 15 | VIOLAÇÃO (achado central) | preço sem conta de horizonte; γ×episódio incompatível |
| 16 | CONFORME | par lr/entropia consistente nos dois scripts |
| 17 | VIOLAÇÃO | decisão pendente desenhada para semente única; N=8/iteração é ruidoso |
| 18 | CONFORME na prática | mas defaults de fábrica reproduzem o bug se env var não setada |
| 19 | CONFORME | fonte única confirmada, sem duplicação em JS |
| 20 | CONFORME (no doc) | `docs/atual.md` já recomenda `--ckpt-entrada vla_fase1.pt`; falta trava no código |
| 21 | CONFORME | checagem de água/lava direta já está em `arena_plana.py` |
| 22 | ativo, não mais hipotético | fatoração independente giro×pulo é candidata real ao colapso atual |
| 23 | VIOLAÇÃO/achado central | mecanismo mais provável da degradação; diagnóstico prescrito não implementado |
| 24 | OPORTUNIDADE | falta testar distribuição congelada do checkpoint de entrada antes do retreino |
| 25 | VIOLAÇÃO direta | `g.std()` não logado; normalização pode inflar lote de baixa variância |
| 26 | CONFORME | mistura 50/50 é o caso previsto pela própria regra |
| 27 | N/A | só `Navegar` implementada; risco ainda não ativo |

---

## 5. Perguntas de desenho — não são "conserto", são escolha

Estas exigem decisão, não só aplicar patch:

1. **Custo do pulo:** subir `CUSTO_PULO` de novo (4ª tentativa da mesma
   classe de conserto) vs. aumentar `γ` vs. reformular como termo de
   potencial (§15). A auditoria sugere que só subir o valor é a aposta mais
   fraca das três, dado o descompasso horizonte×episódio medido em 2.2.
2. **Ação conjunta:** manter `log p(giro) + log p(pulo)` independentes vs.
   condicionar uma na outra (§22/§23). Vale testar depois de resolver 2.2 —
   se o colapso sumir só com o preço/horizonte corrigido, a fatoração
   independente não era o problema.
3. **Quantas sementes rodar antes de decidir** se a Fase 2 "funcionou" ou se
   deve pular para a Fase 3 (§17). O critério de decisão em `docs/atual.md`
   ("se vier ambíguo, não insista") hoje está definido para uma corrida só.

---

## 6. Ordem sugerida para o plano

1. Aplicar a higiene barata da seção 3 (nenhuma decisão de desenho embutida).
2. Rodar avaliação formal (`avaliar_fase2.py`, `n` grande) do `vla_fase2.pt`
   atual contra `geo_pulo`/`piloto`, para confirmar ou descartar que a
   degradação de 15 iterações é sinal e não ruído de N=8 (seção 1).
3. Rodar o teste de distribuição congelada do §24 a partir de `vla_fase1.pt`.
4. Decidir a pergunta 5.1 (custo vs. γ vs. potencial) com os logs de 2.4 já
   coletados de uma corrida curta de sondagem (10-15 iterações).
5. Só então treinar a corrida completa — sempre a partir de
   `checkpoints_vla/vla_fase1.pt` (`--ckpt-entrada`, nunca `--retomar` de um
   checkpoint de Fase 2).
