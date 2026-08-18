# Fase 5 — Decisões Esparsas, Ancoragem Causal e Raciocínio Tático Holonômico (WASD)

A **Fase 5** expande a cognição do `Qwen3LoopVLA` ao introduzir **Seleção de Políticas Esparsas (*Sparse Policy Selection*)**, **Ancoragem de Buffer Causal** e **Controle Tático Holonômico (36 Ações WASD + Giro Parado + Pulo)**, permitindo ao agente resolver tarefas de navegação sequencial com fixação de olhar e raciocínio deliberado.

---

## 1. 🏗️ Diagrama da Arquitetura Completa

```mermaid
graph TD
    subgraph PERCEPCAO["1. PERCEPÇÃO MULTIMODAL & ESTADO"]
        V["Visão 1ª Pessoa<br><i>(Câmera 640x360 RGB)</i>"] --> VE["Vision Encoder (SigLIP) + Resampler"]
        S["Vetor de Estado 32D<br><i>[dx, dz, dist, cos, sin, vel, estagio_ativo]</i>"] --> SE["State Encoder (MLP 32D → 896D)"]
        T["Instrução Textual / Prompt<br><i>'Vá até o bloco amarelo e depois ao azul'</i>"] --> TE["Qwen Embedding Table"]
    end

    VE --> QWEN["QWEN3LOOP VLA BACKBONE<br><i>(28 Camadas, LoRA rank=16, alpha=32.0, Hidden=896)</i>"]
    SE --> QWEN
    TE --> QWEN

    subgraph COGNICAO["2. TRÊS LOOPS DE RACIOCÍNIO LATENTE (LoopSplit 7-20)"]
        QWEN --> L1["Loop 1: Ancoragem Semântica & Avaliação Espacial<br><i>(Onde estou em relação à Submeta 1 vs Submeta 2?)</i>"]
        L1 --> L2["Loop 2: Deliberação Tática & Simulação Mental<br><i>(Intenção: Giro Parado | Sprint | Strafe | Ré)</i>"]
        L2 --> L3["Loop 3: Refinamento Motor Fino & Controle de Inércia<br><i>(Cálculo preciso de Yaw Delta e Pulo)</i>"]
    end

    subgraph ATUACAO["3. CABEÇA DE DECISÃO TÁTICA (36 CLASSES)"]
        L3 --> CAB36["Cabeça Tática Holonômica (MLP 896D → 256D → 36D)"]
        CAB36 --> DEC["Decodificador de Ação Tática"]
        DEC --> MODOS["<b>Modos de Intenção:</b><br>• [0..8] Giro Parado (hold: [])<br>• [9..17] Sprint Frontal (hold: [W])<br>• [18..26] Sprint com Pulo (hold: [W, SPACE])<br>• [27..29] Strafe Esquerda (hold: [W, A])<br>• [30..32] Strafe Direita (hold: [W, D])<br>• [33..35] Recuo / Desengate (hold: [S])"]
        DEC --> FILTRO["Filtro Inercial Passa-Baixas<br><i>dx_t = 0.65·dx + 0.35·dx_{t-1}</i>"]
    end

    MODOS --> SIM["Simulador Minecraft (Mineflayer Server / 8 Envs)"]
    FILTRO --> SIM
```

---

## 2. 🧠 O Problema do "Efeito Beyblade" e a Solução Tática

### O Diagnóstico dos Traçados TopView:
Nas primeiras iterações da Fase 5 (onde o robô possuía apenas `W` e rotação de mouse), os traçados 2D revelaram 3 anomalias críticas:
1. **Efeito Sacarrolha / Mola (Ep 4, 16, 18):** O robô mantinha `W` e disparava giros de mouse alternados ($\pm 60^\circ$), avançando em espirais.
2. **Efeito Pião / Vortex Local (Ep 1, 9, 22):** Ao colidir ou errar o ângulo no spawn, o robô disparava giros máximos de $\pm 120^\circ$ e ficava preso girando em torno de si mesmo.
3. **Sprint Invertido (Ep 2, 5, 17):** Ao spawnar de costas ($180^\circ$), o robô corria de frente antes de virar a câmera, afastando-se do objetivo.

### A Solução: Espaço Tático Holonômico (36 Ações):
* **Giro Parado (`hold: []` + Mouse):** Para orientar a câmera no spawn ou em curvas de $>45^\circ$ sem avançar para frente às cegas.
* **Strafe com Fixação de Olhar (`hold: [W, A]` / `hold: [W, D]`):** Mantém o pilar alvo cravado na mira da câmera enquanto o corpo se desloca lateralmente para contornar o relevo.
* **Recuo / Desengate (`hold: [S]`):** Permite dar um passo para trás quando colide com uma parede ou passa do ponto do pilar, restaurando o campo de visão.
* **Filtro Inercial:** Amortece variações bruscas de velocidade angular ($\alpha=0.65$).

---

## 3. ⚖️ Estratégia de Ancoragem de Buffer (70% Densa / 30% Esparsa)

Para evitar o **esquecimento catastrófico** (*catastrophic forgetting*), o treinamento utiliza um dataset composto:

| Componente | Proporção | Função no Treinamento |
|---|---|---|
| **Buffer Denso de Locomoção** | **70%** (~11.000 amostras) | Preserva a capacidade motora contínua de sprint em linha reta, micro-ajustes ($\pm 5^\circ$) e saltos de relevo. |
| **Buffer de Decisões Calibradas** | **30%** (~4.500 amostras) | Foca nos momentos de alta incerteza (picos de entropia), spawn desalinhado, desengate de quinas e transição de submetas. |

---

## 4. 📊 Histórico Comparativo de Benchmarks TopView 2D

| Modelo / Iteração | Espaço de Ação | Submeta 1 | **Sucesso Total (1 $\rightarrow$ 2)** | Comportamento Observado |
|---|---|---|---|---|
| **Sparse Policy Isolada** | 18 Classes | 4.2% (1/24) | **0.0%** (0/24) | Perdeu locomoção densa; colapso em espirais. |
| **Calibrado Puro (Small)** | 18 Classes | 8.3% (2/24) | **0.0%** (0/24) | Overfitting em 793 amostras; memorização rígida. |
| **Sparse Ancorada (Small)** | 18 Classes | 20.8% (5/24) | **4.2%** (1/24) | Locomoção recuperada; 1 sucesso de ponta a ponta. |
| **Sparse Ancorada (Grande)** | 18 Classes | 20.8% (5/24) | **12.5%** (3/24) | 3 sucessos completos; gargalo residual no efeito pião. |
| **WASD Tático Holonômico (Ép 3)** | 36 Classes | 12.5% (3/24) | **4.2%** (1/24) | Trajetórias em arcos abertos (subtreinado). |
| **WASD Tático Holonômico (Ép 12)** | **36 Classes** | **37.5% (9/24)** 🚀 | **0.0%** (0/24) | **Míssil direto em 5–12 passos** para Submeta 1; gargalo de frenagem pós-meta. |
| **WASD + RL Refinement (Fase 5.3)** | **36 Classes** | **16.7% (4/24)** | **4.2% (1/24)** 🎯 | Primeiro sucesso de ponta a ponta (Ep 21), mas com perda de tiro direto. |
| **PPO-BC Híbrido 70/30 (15 it)** | **36 Classes** | **29.2% (7/24)** 🚀 | **8.3% (2/24)** 🎯🎯 | Ancoragem de 70% BC recuperou a precisão na Submeta 1 e PPO dobrou os sucessos de ponta a ponta (Ep 7 e 21). |
| **PPO-BC Híbrido 70/30 (100 it - Melhor)** | **36 Classes** | **37.5% (9/24)** 🚀 | **4.2% (1/24)** 🎯 | BC Loss convergiu para 0.3880; 9 acertos cravados na Submeta 1 e Sucesso Total no Ep 10 (0.41m / 2.14m). |
| **PPO-BC Híbrido 70/30 (IT 30 - Raio 1.3m)** | **36 Classes** | **4.2% (1/24)** 🎯 | **0.0% (0/24)** | Avaliado sob raio físico estrito ($1.3\text{m}$); chegadas milimétricas em Ep 14 ($0.82\text{m}$), Ep 3 ($1.48\text{m}$), Ep 16 ($1.55\text{m}$) e Ep 20 ($1.59\text{m}$); BC Loss atingiu recorde de **0.3888**. |

---

## 5. 🛠️ Correções Estruturais e Melhorias Implementadas (Fase 5.4)

1. **Correção de Iluminação no Renderizador Voxel ([`mineflayer_server/voxel_renderer.js`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/mineflayer_server/voxel_renderer.js)):**
   - Corrigido o cálculo de `skyLight` em seções superiores ($y \ge 80$). Seções de céu aberto alocadas dinamicamente agora herdam luz plena do sol (`skyLight = 15`), eliminando o bug visual de torres pretas no horizonte.
2. **Limpeza Global de Pilares Temporários ([`mineflayer_server/servidor_offline.js`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/mineflayer_server/servidor_offline.js)):**
   - Criada a função `limparTodosBlocosTemporarios()`, garantindo que todos os pilares colocados por qualquer ambiente sejam destruídos no reset e antes de novas inserções, impedindo acúmulo de pilares duplicados.
3. **Torres Farol de 50 Blocos e Raio de Chegada Estrito ($1.3\text{m}$):**
   - Torres de 50 blocos de altura fornecem visibilidade 3D de longo alcance sobre copas de árvores e montanhas.
   - O raio de chegada foi reduzido de $2.5\text{m}$ para $1.3\text{m}$ para exigir contato físico real com o bloco.
4. **Calibração da Função de Recompensas e Gaze Tracking:**
   - **Busca Ativa Calibrada (+0.04):** Elimina o risco de *reward farming* estático.
   - **Penalidade Progressiva por Perda de Foco:** $-0.05 \times (n - 5)$ após 5 loops sem contato visual após o alvo ter sido avistado pela primeira vez.

---

## 6. 📊 Tabela Oficial de Recompensas e Punições

| Componente | Categoria | Valor / Fórmula | Condição de Disparo | Objetivo / Efeito no Agente |
| :--- | :---: | :---: | :--- | :--- |
| **Sucesso Total** | 🏁 Meta | **`+15.0`** *(Fixo)* | Distância ao Pilar 2 $\le 1.3\text{m}$ (Estágio 2). | Recompensa máxima; encerra o episódio com vitória completa. |
| **Submeta 1** | 🏁 Meta | **`+5.0`** *(Fixo)* | Distância ao Pilar 1 $\le 1.3\text{m}$ (Estágio 1). | Transiciona o alvo para o Pilar 2, reseta perda de foco e ativa frenagem. |
| **Avanço Vetorial** | 🧭 Distância | **`+1.5 × Δd`** *(clip $[-0.5, +1.5]$)* | Baseado em $\Delta d = d_{\text{ant}} - d_{\text{atual}}$. | Recompensa encurtar a distância euclidiana até a torre alvo. |
| **Foco Visual** | 👀 Visão | **`+0.12 × cos(Δθ)`** *(até $+0.12$)* | Alvo no cone de mira frontal ($\Delta\theta \le 45^\circ$, $\cos > 0.70$). | Marca alvo avistado, reseta perda de foco e premia foco visual. |
| **Perda de Foco** | ⚠️ Penalidade | **`-0.05 × (n - 5)`** *(Progressivo)* | Mais de 5 loops ($1.25\text{s}$) sem foco após já ter avistado o alvo. | Força persistência de olhar; pune distração ou esquecimento da meta. |
| **Busca Ativa** | 🔄 Câmera | **`+0.04`** *(Por passo)* | Alvo fora de mira ($\cos < 0.50$), **sem segurar `W`**. | Estimula girar a câmera para buscar sem incentivar farming parado. |
| **Corrida Cega** | ⚠️ Penalidade | **`-0.35`** *(Por passo)* | Segurar `W` com alvo fora de mira ($\Delta\theta > 78^\circ$, $\cos < 0.20$). | Pune severamente sprint frontal descontrolado sem saber onde está o alvo. |
| **Frenagem Inercial** | 🛑 Transição | **`+0.25`** *(Por passo)* | Primeiros 4 passos pós-Submeta 1, **sem segurar `W`**. | Elimina a inércia do sprint logo após o primeiro toque. |
| **Giro Orientado** | 🛑 Transição | **`+0.20`** *(Por passo)* | Primeiros 4 passos pós-Submeta 1 girando para o Pilar 2 ($\cos > 0.25$). | Estimula apontar a mira para a segunda torre após frear. |
| **Custo de Tempo** | ⏱️ Regularização | **`-0.05`** *(Por passo)* | A cada passo em que o robô permanece vivo. | Anti-looping: penaliza trajetórias redundantes ou demora excessiva. |
| **Queda na Água/Lava** | ☠️ Penalidade | **`-3.0`** *(Fixo)* | Contato físico com blocos de água ou lava. | Pune e encerra imediatamente o episódio do robô no ato (`vivo = False`). |

---

## 7. 📁 Arquivos e Módulos Centrais da Fase 5

* **[`fase5/acoes_taticas.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/fase5/acoes_taticas.py):** Mapeador ortogonal das 36 classes e oráculo tático.
* **[`politica/politica_raciocinio.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/politica/politica_raciocinio.py):** Implementação do loop recursivo, cabeça 36D e filtro inercial.
* **[`fase5/treinar_ppo_bc_hibrido.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/fase5/treinar_ppo_bc_hibrido.py):** Loop de otimização conjunta PPO-BC com ancoragem causal contínua e rastreamento de foco.
* **[`fase5/avaliar_fase5_topview.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/fase5/avaliar_fase5_topview.py):** Benchmark 2D Top-Down oficial com geração de mapas de trajetórias e relatórios interativos.
* **[`checkpoints_vla/vla_fase5_it30.pt`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/checkpoints_vla/vla_fase5_it30.pt):** Checkpoint preservado da Iteração 30 com BC Loss recorde de 0.3888.
* **[`fase5/gerar_dataset_wasd_tatico.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/fase5/gerar_dataset_wasd_tatico.py):** Construtor do dataset tático balanceado.
* **[`fase5/treinar_wasd_tatico.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/fase5/treinar_wasd_tatico.py):** Pipeline de treino com salvamento de checkpoint por época.
* **[`fase5/avaliar_fase5_topview.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/fase5/avaliar_fase5_topview.py):** Benchmark oficial com geração de gráficos 2D e relatórios HTML.
