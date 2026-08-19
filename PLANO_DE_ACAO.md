# Plano de Ação e Roadmap — Qwen3Loop VLA

Documento mestre de decisões de engenharia, marcos concluídos e protocolos de avaliação.

---

## 🎯 Marco Atual: Fase 5.5 (PPO-BC Híbrido 70 it & Avaliação TopView 2D)

Após a conclusão das 70 iterações de treinamento com clipping PPO real ($\epsilon=0.2$), Critic GAE e espaço de ação fatorado 54D, o modelo alcançou novo recorde de recompensa ($\text{Rec} = -3.56$) e erro mínimo do Critic ($Value\_Loss = 0.026$).

### 1. Avaliação Espacial TopView 2D (Minecraft)
Mede a navegação sequencial com raio padrão de $1.5\text{m}$:
```bash
python fase5/avaliar_fase5_topview.py --ckpt checkpoints_vla/vla_fase5_ppo_bc_melhor.pt --lotes 3 --passos 100 --raio 1.5
```
- Relatório visual interativo: `fase5/relatorio_topview_ppo_bc_melhor.html`
- Gráfico de trajetórias 2D: `docs/imagens/topview_fase5_ppo_bc_melhor.png`

### 2. Checkpoints Oficiais Salvos
- **Melhor Modelo:** `checkpoints_vla/vla_fase5_ppo_bc_melhor.pt` (Iteração 66, Rec: -3.56, Submeta 1: 12.5%)
- **Modelo Final:** `checkpoints_vla/vla_fase5_ppo_bc.pt` (Iteração 70, Rec: -4.12, Entropia: 1.25)

---

## Histórico de Fases e Decisões de Arquitetura
declarar sucesso (`docs/metodo.md` §17).

## 2. Higiene aplicada no código antes de treinar

| item | arquivo | mudança |
|---|---|---|
| Diagnóstico que faltava (§23/§25) | `treino/treinar_fase2.py` | logs por iteração: `g_std`, vantagem média condicionada a `pulou`/`não pulou` — salvos no `hist` do checkpoint e impressos |
| Regime silencioso (§3/§18) | `treino/treinar_fase2.py` | `assert PASSOS_MAX_F2 >= 40` no início do `main()` — falha alto e explícito em vez de treinar silenciosamente no regime da Fase 1 |
| `raio=16` como default sabotado (§11) | `ambiente/fase2.py` | `montar_tarefas` e `PilotoBFS.__init__` agora default `raio=40`. O banco em disco (`dataset/banco_fase2.json`) já foi gerado com `raio=40` via `gerar_banco` — sem efeito retroativo, só fecha a armadilha para uma próxima chamada direta |
| `goal_encoder` fora da lista à mão (§12) | `infra/gpu_utils.py` | incluído em `compactar_backbone`. Era inofensivo (save/load já usa `named_parameters(requires_grad)`), mas era o mesmo padrão de lista-à-mão que já causou incidente |
| `.gpu_em_uso.lock` órfão | raiz | removido — confirmado antes, via `nvidia-smi`/`tasklist`, que nenhum processo Python/Node ainda segurava a GPU |

## 3. Itens Implementados e Saneados (Auditoria Concluída)

- §1 — **[CONCLUÍDO]** `AleatorioComPulo` implementado em `ambiente/fase2.py` e integrado a `avaliacao/avaliar_fase2.py`.
- §4 — **[CONCLUÍDO]** Split held-out (15%) do banco suportado nativamente com `--held-out` em `avaliacao/avaliar_fase2.py`.
- §13 — **[CONCLUÍDO]** Utilitário `infra/verificar_pesos.py` implementado e validado.
- §20 — **[CONCLUÍDO]** Trava explícita contra retomada de checkpoints contaminados adicionada em `treino/treinar_fase2.py`.
- §25 — **[CONCLUÍDO]** Piso `PISO_G_STD = 3.0` implementado em `treino/treinar_fase2.py`.
- Escala — **[CONCLUÍDO]** `ALCANCE_F2 = 30.0` corrigido em `politica/politica_fase2.py`.
- Otimização — **[CONCLUÍDO]** `minilote=12` confirmado como sweet spot máximo da RTX 3060 12GB (15.2 spl/s, 4.3 GB de folga).

## 4. A corrida

```
CUSTO_PULO=0.15 DIST_MIN=14 DIST_MAX=30 PASSOS_MAX_F2=50 ATRASOS_SIM=0,2,4 \
python treino/treinar_fase2.py --iteracoes 120 --minilote 12 --vram 0.88 \
       --gamma 0.98 --ckpt-entrada checkpoints_vla/vla_fase1.pt \
       > logs/fase2_2026-08-13.log 2>&1
```

Base: `checkpoints_vla/vla_fase1.pt` (saudável, iteração 100/100 — nunca
`vla_fase2.pt` nem os dois `_degenerado`/`_alta`, ver `AJUSTES.md` §1).
Saída: novo `checkpoints_vla/vla_fase2.pt`, sobrescrevendo o contaminado —
os três checkpoints de evidência (`vla_fase2_pulo_degenerado.pt`,
`vla_fase2_entropia_alta.pt`) ficam preservados para comparação.

## 5. Critério de leitura do resultado

Olhar `g_std`, `adv(pulou)`/`adv(não pulou)` e `entropia` por iteração no log,
não só `chegada`/`pulo` finais — são exatamente os números que faltavam para
distinguir as três hipóteses de `AJUSTES.md` §2.2-2.4:

- se `pulo` estabiliza (não satura perto de 100%) e `chegada` não cai junto
  → preço+horizonte resolveram; próximo passo é 2ª semente de confirmação.
- se `pulo` volta a subir e `adv(pulou)` fica consistentemente maior que
  `adv(não pulou)` mesmo com o preço corrigido → é a fatoração conjunta
  (pergunta 2), não o preço — próximo passo é condicionar as cabeças.
- se `g_std` cai muito em iterações específicas e a curva fica instável na
  mesma cadência → suspeitar da normalização de vantagem (§25), testar
  centralizar sem dividir pelo desvio.

---

## Resultado (2026-08-13, ~1h44 depois)

**Terceiro cenário confirmado, não o primeiro nem o segundo.** Avaliação
formal (`avaliar_fase2.py --episodios 80 --desvio-min 1.2 --raio 40`,
protocolo idêntico ao de `fase2.md`):

| política | geral | leve | obstruído |
|---|---|---|---|
| `geo_pulo` (piso) | 50,0% | 48% | 53% |
| `piloto` (teto) | 53,8% | 50% | 58% |
| **`modelo`** | **0,0%** | 0% | 0% |
| `modelo_cego` | 1,2% | 0% | 3% |

Confirmado com uma sonda extra em modo estocástico (mesmo modo do treino,
n=40, script ad-hoc reaproveitando `avaliar()`/`relatar()` de
`avaliar_fase2.py`): também 0%. **Não é divergência argmax-vs-amostragem —
a política não converge para nada que funcione, em nenhum dos dois modos.**

### O que a correção resolveu

`pulo` não saturou perto de 100% como nas 3 tentativas anteriores — pico de
67% (blocos it 31-45), caindo para ~30% na segunda metade. `CUSTO_PULO=0,15`
+ `γ=0,98` mudaram essa dinâmica de verdade; a hipótese de `AJUSTES.md` §2.2
(horizonte de crédito curto demais) tinha fundamento real.

### O que não resolveu, e o achado novo

- **Entropia colapsou** para 0,02-0,10 (normalizado, máximo 1,0) nas últimas
  10 iterações — convergência prematura para algo que não funciona.
- **`g_std` e `chegada` caíram juntos na mesma janela** (it 31-45: `g_std`
  1,99, o mais baixo do treino inteiro; `chegada` 0,8%, a pior). É a
  assinatura que `docs/metodo.md` §25 previu — evidência concreta a favor de
  suspeitar da normalização `(g - g.mean())/(g.std()+eps)` amplificando um
  lote de baixa variância, não coincidência.
- **A hipótese da fatoração conjunta (pergunta 2, §22/§23) NÃO se sustentou**
  na corrida inteira: `adv(pulou) > adv(não pulou)` na primeira metade,
  **inverteu** na segunda (`adv(não pulou)` maior nas últimas 10 iterações,
  coerente com `pulo` caindo para ~30%). O sinal de crédito por ação parece
  ter corrigido a direção sozinho — não é o suspeito principal desta vez.
- Achado lateral: os números de piso/teto desta avaliação (`geo_pulo` 50%,
  `piloto` 53,8%) divergem da calibração original em `fase2.md`
  (`geo_pulo` 36,2%, `piloto` 53,8% — o piloto bate exato, o piso não).
  `piloto` reproduziu, `geo_pulo` não — vale investigar antes de confiar
  cegamente na comparação piso/teto na próxima rodada.

### Checkpoint

`checkpoints_vla/vla_fase2.pt` foi sobrescrito por este resultado. Está tão
contaminado quanto os três anteriores — **não usar como base**. A próxima
tentativa parte de `checkpoints_vla/vla_fase1.pt` (intocado, saudável),
exatamente como esta partiu.

### Próximo passo, dado o que se sabe agora

Não é mais só "achar o preço certo do pulo" — isso já foi resolvido. O que
falta é: (1) testar a normalização de vantagem sem dividir por `g.std()`
(§25) dado o achado da janela it 31-45; (2) considerar `γ` menor que 0,98
(o aumento de horizonte de crédito também aumenta a variância do retorno
Monte Carlo, e pode estar brigando com a normalização instável em (1)); (3)
2ª semente antes de decidir qualquer coisa — uma corrida só, mesmo com esse
resultado ruim, não separa "esse fix não funciona" de "essa semente específica
degenerou" (`docs/metodo.md` §17). Fica para decisão do usuário antes de
gastar outra ~1h44 de treino.
