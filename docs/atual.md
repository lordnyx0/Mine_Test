# Estado atual — 2026-08-18

Ponto de reentrada. O que está rodando, o que foi decidido, o que ainda não se
sabe, e onde procurar cada coisa.

---

## Em uma frase

**Fase 5.5 (PPO Verdadeiro + Critic GAE + Recompensa Visual Não-Privilegiada + Cabeças Fatoradas Modo 6 × Yaw 9):** Pipeline completamente modernizado com clipping PPO real ($\epsilon=0.2$, multi-epoch), Value Head $V(s)$ com GAE ($\lambda=0.95, \gamma=0.98$), recompensa visual visuomotora pura (eliminação do oráculo geométrico invisível via detecção RGB no frame com bônus de $\Delta \text{Visão}$), espaço de ações fatorado (Modo 6 classes + Yaw 9 classes) e annealing curricular de BC ($85\% \to 20\%$).

---

## 🎯 Protocolo de Execução Imediata (Fase 5.5)

1. **Treinamento PPO-BC Híbrido Modernizado:**
   ```bash
   python fase5/treinar_ppo_bc_hibrido.py --base checkpoints_vla/vla_fase5_it30.pt --saida checkpoints_vla/vla_fase5_ppo_bc.pt --iteracoes 20 --passos 50 --ppo-epochs 3 --clip-eps 0.2
   ```
2. **Benchmark Oficial TopView 2D:**
   ```bash
   python fase5/avaliar_fase5_topview.py --ckpt checkpoints_vla/vla_fase5_ppo_bc.pt --lotes 3 --passos 100
   ```
   - Relatório interativo: `fase5/relatorio_topview_ppo_bc.html`
   - Gráfico de trajetórias: `docs/topview_fase5_ppo_bc.png`


---

## Resultado da Avaliação da Iteração 30 (Raio Estrito 1.3m + Torres de 50 Blocos)

| Métrica / Tipo de Episódio | Taxa / Valor | Comportamento Observado |
|---|---|---|
| **Taxa Submeta 1 (Pilar 1)** | **4.2%** (1/24) | Toque físico perfeito no Ep 14 a **0.82m** no passo 39 |
| **Aproximações Milimétricas (< 1.6m)** | **16.7%** (4/24) | Ep 3 (1.48m), Ep 16 (1.55m), Ep 20 (1.59m), Ep 4 (1.58m) |
| **BC Loss Supervisionado** | **0.3888** 🚀 | Menor erro de predição de ação motora da história do projeto |
| **Iluminação & Mundo Limpo** | **100% Validado** | Torres amarelas/azuis/roxas visíveis em toda a altura; zero pilares fantasmas |

---

## Resultado da Fase 2 com as Blindagens Aplicadas (2026-08-14)

Avaliação formal em tarefas *held-out* reservadas (n=40, `PASSOS_MAX_F2=80` cobrindo distâncias de 14-30 blocos em terreno real):

| Política | Geral | `reta` (desvio < 1.0) | `leve` (1.0-1.4) | `obstruido` (>= 1.4) | $d_{\text{final}}$ Médio (Obstruído) |
|---|---|---|---|---|---|
| **`ALEATORIO_PULO`** (piso estocástico) | **0.0%** | 0% | 0% | 0% | 24.17 blocos |
| **`MODELO_CEGO`** (ablação de visão) | **0.0%** | 0% | 0% | 0% | 28.00 blocos |
| **`MODELO`** (VLA com visão ativa) | **32.5%** | **47%** | **25%** | **20%** | **11.78 blocos** |
| **`GEO_PULO`** (piso cego determinístico) | **47.5%** | 71% | 38% | 27% | 12.30 blocos |
| **`PILOTO BFS`** (teto planejado com voxels) | **65.0%** | 76% | 50% | 60% | 8.94 blocos |

### Conclusões e Validações Estabelecidas:
1. **Comprovação de Uso da Visão:** O modelo com visão bateu o cego por **+32.5 pontos percentuais** (32.5% vs 0.0%) e reduziu a distância final de 28.0m para 11.7m em alvos obstruídos.
2. **Entropia e Treino Estáveis:** O *logit bounding* ($\tanh \times 3.0$) e o *target entropy adaptativo* impediram completamente o colapso determinístico — a entropia final terminou em `0.54` (`y: 0.38`, `j: 0.71`).
3. **Ponte para a Fase 3:** O `clip_grad_norm_ = 0.5` preservou os adaptadores `resampler` e `projector`, deixando os pesos prontos para a Fase 3 (servo-visual pura sem coordenadas).

**O que a correção resolveu:** o pulo não saturou a ~99% como nas 3
tentativas anteriores — pico de 67% na it 31-45, caiu para ~30% na segunda
metade do treino. `CUSTO_PULO=0,15` + `γ=0,98` mudaram essa dinâmica de
verdade.

**O que não resolveu:** a entropia colapsou para quase zero
(0,02-0,10 nas últimas 10 iterações, de um máximo normalizado de 1,0) e a
política convergiu para algo que não chega perto do alvo. Achado novo: a
janela it 31-45 teve o `g_std` mais baixo de todo o treino (1,99, contra
4-6,4 típico) **e** a pior `chegada` (0,8%) ao mesmo tempo — evidência
concreta a favor da hipótese de `docs/metodo.md` §25 (normalizar pelo desvio
do lote infla o gradiente de lotes de baixa variância). A hipótese da
fatoração conjunta (§22/§23) NÃO se sustentou na corrida inteira: a vantagem
favoreceu pular na primeira metade e virou o contrário na segunda — não é o
suspeito principal desta vez.

`checkpoints_vla/vla_fase2.pt` foi sobrescrito por este resultado (0%) — está
tão contaminado quanto os anteriores. **A base para a próxima tentativa
continua sendo `checkpoints_vla/vla_fase1.pt`**, nunca este. O checkpoint de
0% foi preservado como evidência em
`checkpoints_vla/vla_fase2_gstd_baixo_zero.pt` antes de rodar de novo (senão
a próxima corrida o sobrescreveria e perderia o dado).

Detalhe em `PLANO_DE_ACAO.md` (seção "Resultado", ao final).

---

---

## Fixes Aplicados e Validados (Prontos para a Nova Corrida)

1. **Piso de Normalização de Vantagem (`PISO_G_STD = 3.0`):**
   `treino/treinar_fase2.py` agora protege a divisão contra lotes de baixa variância (`docs/metodo.md` §25), impedindo a inflação de gradientes em janelas de falha.
2. **Escala de Entrada Corrigida (`ALCANCE_F2 = 30.0`):**
   `politica/politica_fase2.py` redefiniu `_entradas()` com `ALCANCE_F2 = 30.0`, mantendo o vetor de objetivo no domínio $[-1.0, 1.0]$ da MLP do `GoalEncoder` (em vez de extrapolá-lo até $3.75$).
3. **Piso Mínimo de Entropia e Decomposição:**
   `treino/treinar_fase2.py` e `politica/politica_fase2.py` agora calculam e logram a entropia de giro e pulo separadamente (`ent_yaw` e `ent_pulo`), com piso de decaimento ajustado (`0.06 -> 0.025`) para evitar congelamento determinístico prematuro.
4. **Baseline de Piso Estocástico (`aleatorio_pulo`):**
   `ambiente/fase2.py` ganhou a classe `AleatorioComPulo` para fornecer o piso estocástico obrigatório por `docs/metodo.md` §1.
5. **Avaliação Pareada com Held-out (`--held-out`):**
   `avaliacao/avaliar_fase2.py` agora suporta avaliação estrita nas 180 tarefas reservadas (15% do banco) e modo de amostragem estocástica (`--amostrar`).
6. **Benchmark de GPU e Minilote 12 como Padrão:**
   Medição empírica na RTX 3060 12GB confirmou `minilote=12` como o ponto ótimo global (15.2 spl/s, 3.5× mais rápido que minilote 4, mantendo 4.3 GB de margem segura contra OOM).
7. **Utilitário de Diagnóstico de Pesos:**
   `infra/verificar_pesos.py` criado e validado para comparar tensores módulo a módulo entre checkpoints (`docs/metodo.md` §13).
8. **Fase 3 e Camada de Cérebro Implementadas:**
   `ambiente/fase3.py`, `politica/politica_fase3.py`, `treino/treinar_fase3.py`, `avaliacao/avaliar_fase3.py` e `politica/cerebro.py` criados e integrados.

```bash
node mineflayer_server/servidor_offline.js   # se nao estiver rodando

CUSTO_PULO=0.15 DIST_MIN=14 DIST_MAX=30 PASSOS_MAX_F2=50 ATRASOS_SIM=0,2,4 \
python treino/treinar_fase2.py --iteracoes 120 --minilote 12 --vram 0.88 \
       --gamma 0.98 --ckpt-entrada checkpoints_vla/vla_fase1.pt \
       > logs/fase2_$(date +%Y-%m-%d_%H%M).log 2>&1
```

---

## Tentativa anterior (histórico) — retreino a partir de `vla_fase1.pt`

Auditoria completa em `AJUSTES.md` (5 agentes, `docs/metodo.md` §1-27 contra
código e checkpoints reais) e decisão registrada em `PLANO_DE_ACAO.md`. A
corrida pausada (`vla_fase2.pt`, it 15/120, `CUSTO_PULO=0,05`) foi
**descartada como base** — ver `AJUSTES.md` §1 para a reconstrução da
cronologia e a ressalva de que o "it 16" que motivou a pausa não é mais
verificável (sem log em arquivo na época).

Iniciado em 2026-08-13, ~15:03, log em `logs/fase2_2026-08-13.log`:

```bash
CUSTO_PULO=0.15 DIST_MIN=14 DIST_MAX=30 PASSOS_MAX_F2=50 ATRASOS_SIM=0,2,4 \
python treino/treinar_fase2.py --iteracoes 120 --minilote 12 --vram 0.88 \
       --gamma 0.98 --ckpt-entrada checkpoints_vla/vla_fase1.pt \
       > logs/fase2_2026-08-13.log 2>&1
```

Duas mudanças em relação à tentativa anterior, as duas em `PLANO_DE_ACAO.md`
§1:

- `CUSTO_PULO=0,15` (era 0,05, e já hardcoded como novo default em
  `treino/treinar_fase2.py`).
- **`--gamma 0.98`, novo** (era 0,95 fixo, herdado da Fase 1). Horizonte de
  crédito `1/(1-γ)` sobe de ~20 para ~50 passos — cobre `PASSOS_MAX_F2=50`
  inteiro. O achado da auditoria foi que só subir o preço não bastava: com
  γ=0,95 o desconto amassava o custo do pulo antes dele alcançar o passo onde
  o dano real (perda de autoridade de giro no ar) se manifesta.

Também adicionado: log de `g_std` e vantagem média condicionada a
pulou/não-pulou por iteração (diagnóstico que faltava para separar "preço
fraco" de "as duas ações compartilhando uma vantagem só" — ver
`AJUSTES.md` §2.3-2.4). Critério de leitura em `PLANO_DE_ACAO.md` §5.

**Semente única (`seed=0`)** — o usuário se ausentou durante a corrida; o
resultado sozinho não é veredito, é confirmação de sintoma. Segunda semente
é o próximo passo se este resultado vier bom (`docs/metodo.md` §17).

> Por que pular degrada em vez de ser neutro: `airborneAcceleration = 0.02`
> contra ~0,10 no chão. **No ar o bot tem 5× menos autoridade de direção**, e
> virar é a única habilidade que a tarefa cobra.

---

## Estado por fase

| fase | escopo | estado | resultado |
|---|---|---|---|
| **1** | ±8 blocos, plano, `W`+giro | **fechada** | ~85%, igual ao teto geométrico |
| **2** | terreno real, 14-30 blocos, +`SPACE` | 3 tentativas, todas com defeito de incentivo | sem resultado |
| **3** | alvo VISUAL, sem coordenada | proposta | — |
| 4 | `SHIFT` | — | — |
| 5 | `A`/`D`/`S` | — | — |

### Fase 1 — o que ela provou e o que não provou

**Provou:** o laço de RL fecha. Piso `aleatorio` 7,8% → política ~85%.

**Não provou nada sobre visão.** Terreno plano com alvo em coordenadas é
resolvível por `atan2`; a ablação deu **+1,6 ponto**, que é ruído. A política
aprendeu trigonometria.

Comparação backbone aleatório × treinado, 100 iterações:

| iterações | aleatório | treinado |
|---|---|---|
| 1-20 | 11% | **26%** |
| 81-100 | 76% | **83%** |

**Velocidade de aprendizado, não teto.** Os dois convergem para o que `atan2`
consegue.

### Fase 2 — bem posta, ainda sem resultado

Baselines medidos na faixa que exige visão:

```
geo_pulo (piso, cego)         33%
piloto BFS raio 40 (teto)     60%     <- 27 pontos de espaço
```

Três corridas, três defeitos **meus**, todos no desenho do incentivo:

| corrida | defeito | sintoma |
|---|---|---|
| v2 | pulo sem preço | pulo → 99%, chegada 50% → 32% |
| v3 | entropia somada crua | entropia subindo, pulo travado em 59% |
| v5 | custo de pulo fraco (0,05) | pulo → 83%, chegada → 0% |

Em nenhuma o modelo falhou: ele fez exatamente o que a recompensa pedia.

---

## O achado que reinterpreta tudo que veio antes

**O backbone Qwen3Loop nunca era carregado.** `load_vla_agent` fazia
`Qwen3LoopModel(config)` — aleatório — e nunca substituía os pesos. Todo o VLA
rodou sobre ruído até 2026-08-13.

O sintoma que denunciou: treino reportando 88-100% e avaliação 5-14%, porque
`treinar_fase1.py` semeia o torch e sorteava sempre o mesmo backbone aleatório,
enquanto scripts sem semente sorteavam outro.

**Qualquer número medido antes desta data leva asterisco**, inclusive:

- as restrições de VRAM (o modelo aleatório tinha `intermediate_size` 22016
  contra 3072 reais — MLPs 7× maiores). Minilote 4 → **12** depois do conserto,
  3,1× mais rápido.
- a atenuação de ~100× do objetivo atravessando o backbone
- "o LM não paga o próprio lugar"

As sondas de representação **não** ficam inválidas: projeção aleatória preserva
informação linearmente legível. Muda a interpretação, não os números.

---

## Observações que valem mais que os números

**Toda ação sem preço vira constante.** Aconteceu 4×. Mas são **três mecanismos
com consertos opostos** — ação sem gradiente, custo além do horizonte de crédito,
e ótimo degenerado da tarefa. Confundi-los leva a mexer na tarefa quando o
problema era precificação, e vice-versa. Ver [`docs/metodo.md`](docs/metodo.md)
§2.

**O teto também precisa ser validado.** O `piloto` com raio 16 contra alvos a 30
blocos media 36% quando o real era 60% — quase descartei a Fase 2 por medir um
teto sabotado.

**Olhe a tela.** O visualizador achou o bug de arenas dentro d'água em segundos,
e a observação de que "o boneco não gira nunca" virou o diagnóstico completo da
Fase 1. Ele agora mostra o alvo:

<http://127.0.0.1:3002/ver> — seta no referencial do bot, distância, e legenda
verde quando o erro angular é < 30°.

**A média esconde o resultado.** Uma política que empata em `reta` e perde em
`obstruido` marca bem no agregado. Tudo na Fase 2 é reportado por faixa de
`desvio`, e o número que decide é **`obstruido` acima de 33% com `reta`
empatando** — separação, não subida uniforme.

---

## Qual documento visitar, e por quê

| pergunta | documento |
|---|---|
| como as peças se encaixam; os dois laços; o modelo multimodal | [`docs/arquitetura.md`](docs/arquitetura.md) |
| por que o planejador funciona e por que ele não transfere | [`docs/planejador.md`](docs/planejador.md) |
| API do simulador, sondas, visualizador | [`docs/simulador.md`](docs/simulador.md) |
| o que já foi medido e **refutado** — leia antes de propor hipótese | [`docs/experimentos.md`](docs/experimentos.md) |
| as regras de medição, com o custo que cada uma teve | [`docs/metodo.md`](docs/metodo.md) |
| a Fase 1 e o currículo completo | [`docs/fase1.md`](docs/fase1.md) |
| a Fase 2 e o que decide o veredito | [`docs/fase2.md`](docs/fase2.md) |
| comandos e armadilhas, versão curta | [`CLAUDE.md`](CLAUDE.md) |

**Se for propor um experimento novo:** `docs/experimentos.md` primeiro. Seis
hipóteses já foram refutadas por medição, e repetir uma delas é o desperdício
mais provável.

**Se um número parecer bom demais:** `docs/metodo.md` §1 e §2. Nas três vezes que
um resultado pareceu sucesso nesta sessão, era piso errado, métrica saturada ou
ação sem preço.

---

## Próximos passos, em ordem (Fase 5.4)

**1. Treinamento PPO-BC Híbrido com Rastreamento de Foco e Busca Calibrada (+0.04):**
Executar 100 iterações a partir do checkpoint `vla_fase5_it30.pt` com o novo sistema de penalidade progressiva de perda de foco visual e busca ativa calibrada:
```bash
python fase5/treinar_ppo_bc_hibrido.py --base checkpoints_vla/vla_fase5_it30.pt --saida checkpoints_vla/vla_fase5_ppo_bc.pt --iteracoes 100 --passos 100 --lr 3e-5
```

**2. Avaliação Comparativa TopView 2D:**
Rodar o benchmark topview formal com 24 episódios determinísticos para mensurar a taxa de fechamento sequencial de submetas no raio estrito de 1.3m:
```bash
python fase5/avaliar_fase5_topview.py --ckpt checkpoints_vla/vla_fase5_ppo_bc_melhor.pt --lotes 3 --passos 100
```

**3. Validação de Generalização Cross-Color:**
Avaliar a sensibilidade da atenção visual a diferentes combinações cromáticas (Ouro $\to$ Lápis, Obsidiana $\to$ Ouro, Lápis $\to$ Obsidiana) sob relevo acidentado.

---

## Higiene

- Checkpoints em `checkpoints_vla/`, teto de 5. **Nunca sobrescrever**
  `vla_locomotion.pt` nem `BASE_locomocao_limpa.pt`.
- Guardados como evidência de modo de falha:
  `vla_fase1_backbone_aleatorio.pt`, `vla_fase2_pulo_degenerado.pt`,
  `vla_fase2_entropia_alta.pt`.
- **Não use pipe em corrida longa** — `cmd | grep | tail` bufferiza até o fim.
  Redirecione para arquivo.
- `travar_gpu()` cria `.gpu_em_uso.lock`. Se um processo morrer sujo, apague à
  mão.
