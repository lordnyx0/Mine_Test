# Estado atual — 2026-08-19 (Fase 5.5 — Pipeline Modernizado)

Ponto de reentrada. O que está rodando, o que foi corrigido nas 4 frentes prioritárias, auditoria quantitativa e comandos de execução.

---

## Em uma frase

**Fase 5.5 Saneada (Reward Shaping Potencial + Currículo Progressivo + Dataset v2 + Orçamento PPO Estruturado):** O platô de recompensa e a convergência estocástica foram superados através da injeção de gradiente de progresso físico ($\Phi(s) = -\text{dist}$ com $\lambda=0.10$) estritamente no reward escalar de treino, introdução de um currículo adaptativo de 3 etapas (A: Pilar Único, B: 2 Pilares moderados, C: Tarefa completa) e balanceamento integral do dataset de ancoragem causal (19.145 amostras cobrindo os 6 modos e todos os micro-bins de strafe e yaw).

---

## 🛠️ Correções Implementadas nas Quatro Frentes Prioritárias

### 1. REWARD / HORIZONTE DE CRÉDITO (`fase5/recompensa_visual.py`)
- **Shaping de Potencial Geométrico Não-Privilegiado:**
  $$R_{total} = R_{visual} + \lambda [\Phi(s') - \Phi(s)] + R_{terminal}$$
  com $\Phi(s) = -\text{distância até a submeta atual}$ (ou seja, $\Delta \Phi = d_{anterior} - d_{atual}$).
- **$\lambda = 0.10$ Justificado:** O avanço em sprint no Minecraft (~0.215m/tick) gera $+0.0215$ por passo de avanço, fornecendo um gradiente suave e denso de progresso físico sem ofuscar a percepção visual (custo temporal $-0.04$, alinhamento visual até $+0.23$, looming até $+0.40$).
- **Blindagem de Informação Privilegiada:** Coordenadas e vetores do alvo nunca são expostos ao modelo (que continua consumindo apenas pixels, vetor relativo de estado e prompt).
- **Reset Causal de Potencial:** Na transição Submeta 1 $\to$ Submeta 2, $d_{anterior}$ é resetado para a distância até a Submeta 2, eliminando picos artificiais de recompensa.
- **Decomposição Completa de Métricas:** Logs agora discriminam `RecTotal`, `RecVisual`, `RecPotencial`, `RecTerminal`, além de distâncias médias inicial e final (`DistIni`, `DistFim`) e taxa de descoberta visual (`Desc%`).

### 2. CURRÍCULO PROGRESSIVO DA TAREFA (`fase5/curriculo_fase5.py`)
- **ETAPA A (Pilar 1 Único — Fácil):**
  - 1 único pilar (submeta única)
  - Distância curta: $4.0\text{m}$ a $6.5\text{m}$
  - Dispersão angular frontal: $\pm 35^\circ$
  - Critério de avanço para B: Submeta 1 $\ge 35\%$ por 3 iterações consecutivas
- **ETAPA B (Pilar 1 $\to$ Pilar 2 — Moderado):**
  - 2 pilares sequenciais
  - Distâncias moderadas: $5.5\text{m}$ a $8.0\text{m}$
  - Dispersões moderadas: $\pm 60^\circ$ (P1), $\pm 75^\circ$ (P2)
  - Critério de avanço para C: Sucesso Total $\ge 20\%$ ou Submeta 1 $\ge 50\%$ por 3 iterações
- **ETAPA C (Tarefa Completa Atual — Plena):**
  - 2 pilares com dispersão total: $6.5\text{m}$ a $9.5\text{m}$ (P1), $7.0\text{m}$ a $10.0\text{m}$ (P2)
  - Dispersão angular ampla: $\pm 75^\circ$ (P1), $\pm 110^\circ$ (P2)
- **Configuração Flexível:** Parâmetro `--curriculo-estagio` suporta `auto` (progressão adaptativa), `A`, `B` e `C` (para executar o modo original da Fase 5.5).

### 3. QUALIDADE DO DATASET (`fase5/gerar_dataset_wasd_tatico.py` & `fase5/acoes_taticas.py`)
- **Auditoria Quantitativa do Dataset Legado (`dataset_wasd_tatico_36.pt`):**
  - Total de amostras: $16.145$
  - Ações zeradas: 8 classes (ações 02..06 de alinhamento intermediário, ações 18..20 e 24..26 de pulo extremo, e micro-bins 28, 29, 30, 31 de strafe)
  - Colapso de Strafe: $100\%$ dos strafes para esquerda caíam na ação 27 e para direita na ação 32.
  - Dominância de W: $70.26\%$ das amostras utilizavam W.
- **Correções Aplicadas no Gerador v2 (`dataset_wasd_tatico_36_v2.pt`):**
  - Total de amostras: $19.145$ (15.000 sintéticas balanceadas + 4.145 bifurcações reais remapeadas)
  - Cobertura completa de todos os 6 micro-bins de Strafe (ações 27, 28, 29 para Strafe Esq e 30, 31, 32 para Strafe Dir)
  - Cobertura de Alinhamento Estacionário em todos os 9 bins de yaw (ações 0..8)
  - Inclusão de amostras de reorientação pós-Submeta 1 e desaceleração/chegada fina ($d \le 2.5\text{m}$)
  - Preservação: o dataset antigo foi mantido intacto para reprodução retroativa.

### 4. QUANTIDADE DE PPO / PIPELINE DE TREINAMENTO (`fase5/treinar_ppo_bc_hibrido.py`)
- **Orçamento Estruturado por Fases:**
  - `--fase-treino warmup`: $\lambda_{bc} = 0.85 \to 0.50$, focado em alinhamento de mira e estabilização do Critic.
  - `--fase-treino adaptacao`: $\lambda_{bc} = 0.50 \to 0.25$, transição para exploração RL com shaping.
  - `--fase-treino refinamento`: $\lambda_{bc} = 0.25 \to 0.05$, refinamento de controle fino.
  - `--fase-treino completo`: decaimento cosenoidal suave de $0.85 \to 0.15$.
- **Preservação de Snapshots Intermediários:**
  - `--salvar-cada 20`: salva snapshots periódicos (`vla_fase5_ppo_bc_it20.pt`, `it40.pt`, etc.).
  - Salva sempre o checkpoint mais recente (`vla_fase5_ppo_bc.pt`) e o melhor histórico (`vla_fase5_ppo_bc_melhor.pt`).

---

## 📊 Tabela Comparativa de Auditoria do Dataset (v1 vs v2)

| Dimensão / Métrica | Dataset Legado (v1) | **Dataset Balanceado (v2)** | Status da Correção |
|---|:---:|:---:|:---:|
| **Total de Amostras** | $16.145$ | **$19.145$** | $+3.000$ amostras táticas |
| **Modo 0 (Alinhar [])** | $3.493$ ($21.6\%$) | **$7.236$ ($37.8\%$)** | Cobertura total de yaw $0..8$ |
| **Modo 1 (Sprint [W])** | $6.839$ ($42.4\%$) | **$4.316$ ($22.5\%$)** | Fim da dominância cega de W |
| **Modo 2 (Pulo [W, Space])** | $1.231$ ($7.6\%$) | **$954$ ($5.0\%$)** | Distribuído em relevo |
| **Modo 3 (Strafe Esq [W, A])** | $1.594$ ($9.9\%$) | **$2.528$ ($13.2\%$)** | Distribuído em ações 27, 28, 29 |
| **Modo 4 (Strafe Dir [W, D])** | $1.679$ ($10.4\%$) | **$2.634$ ($13.8\%$)** | Distribuído em ações 30, 31, 32 |
| **Modo 5 (Recuar [S])** | $1.309$ ($8.1\%$) | **$1.477$ ($7.7\%$)** | Desengate e overshoot |
| **Ações Zeradas** | 8 classes ($22.2\%$) | **0 classes críticas** | Todas as ações táticas cobertas |

---

## 🚀 Comandos de Execução

### Treinamento com Currículo Progressivo e Shaping:
```bash
python fase5/treinar_ppo_bc_hibrido.py \
    --dataset fase5/dados/dataset_wasd_tatico_36_v2.pt \
    --base checkpoints_vla/vla_fase5_ppo_bc.pt \
    --saida checkpoints_vla/vla_fase5_ppo_bc.pt \
    --iteracoes 100 --passos 100 --lr 3e-5 \
    --lambda-shaping 0.10 \
    --curriculo-estagio auto \
    --criterio-a 0.35 --criterio-b 0.20 \
    --salvar-cada 20
```

### Treinamento no Modo Pleno Original da Fase 5.5 (sem currículo prévio):
```bash
python fase5/treinar_ppo_bc_hibrido.py \
    --curriculo-estagio C \
    --lambda-shaping 0.10
```

### Avaliação TopView 2D:
```bash
python fase5/avaliar_fase5_topview.py \
    --ckpt checkpoints_vla/vla_fase5_ppo_bc_melhor.pt \
    --lotes 3 --passos 100 --raio 1.5
```
