# 🧠 Fase 7: CoT-GRPO Autoregressive Reasoning VLA (VLA 2.0)

## 1. 📌 Visão Geral & Motivação

Nas fases 1 a 6, o agente VLA utilizou cabeças lineares (`cabeca_modo` e `cabeca_yaw`) acopladas ao último token do modelo de linguagem. Embora o agente tenha aprendido locomoção em 3D e navegação precisa entre pilares, **o modelo de linguagem não estava raciocinando autoregressivamente durante a tomada de decisão motora**. O gradiente do PPO otimizava apenas os pesos de classificação motora, sem retropropagar sobre a sintaxe e a lógica textual do LLM.

A **Fase 7** investigou a transição para o **CoT-GRPO Autoregressive Reasoning VLA (VLA 2.0)**:
1. **Geração Autoregressiva Pura:** O modelo gera no espaço de tokens:
   $$\text{Prefixo Multimodal} \longrightarrow \langle\text{think}\rangle \dots \text{raciocínio espacial} \dots \langle/\text{think}\rangle \langle\text{action}\rangle \text{modo\_yaw} \langle/\text{action}\rangle$$
2. **GRPO Token-Level (Group Relative Policy Optimization):** Para cada observação visual, o modelo amostra $G=4$ caminhos de pensamento alternativos sob temperaturas escalonadas $[0.6, 0.8, 1.0, 1.2]$. As ações são executadas em paralelo e a vantagem relativa de grupo atualiza **todos os tokens do $\langle\text{think}\rangle$**.
3. **Ambiente Natural Orgânico:** Testado no terreno procedural do Minecraft (árvores, colinas, desníveis de terra, praias e desvios de visão em 3D).

---

## 2. 📐 Formulação Matemática do GRPO Token-Level

Para cada estado $s$, o agente gera um grupo de $G$ conclusões textuais $\{y_1, y_2, \dots, y_G\}$.

### Vantagem Relativa de Grupo:
$$A_i = \frac{R(y_i) - \text{mean}(\{R(y_j)\}_{j=1}^G)}{\text{std}(\{R(y_j)\}_{j=1}^G) + \epsilon}$$

### Função Objetivo GRPO:
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min \left( \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})} A_i, \; \text{clip}\left(\frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$

---

## 3. 🔬 Achados Experimentais & Análise de Gargalos Computacionais

### A. Custo Computacional do Sampling Online
* Em contraste com políticas feedforward (como a da Fase 5, que executam em **$\sim 5\text{ms}$**), a amostragem autoregressiva de $500$ tokens em lotes de $32$ sequências simultâneas ($8\text{ robôs} \times 4\text{ amostras}$) demanda **$\sim 120 - 180\text{ segundos}$ por passo de ambiente**.
* Para loops de RL online (onde dezenas de milhares de passos são necessários para convergir), o tempo total de treinamento torna-se proibitivamente alto para GPUs de consumo (8GB - 12GB VRAM).

### B. O Problema da Memória Temporal e Horizonte Parcialmente Observável (POMDP)
* A navegação em mundos abertos com árvores e colinas não é estritamente markoviana: quando um obstáculo bloqueia o alvo, o robô precisa lembrar a posição visual em passos anteriores.
* Acumular histórico multimodal ($T$ frames visuais $\times 32$ tokens $+ \text{ações passadas} + 500\text{ tokens CoT}$) gera uma explosão de contexto para milhares de tokens a cada milissegundo, saturando a VRAM e a largura de banda de memória da GPU.

---

## 4. 💡 Recomendações e Caminhos Eficientes Futuros

1. **Arquitetura Hierárquica (Planejador Lento + Executor Rápido):**
   * LLM/VLM como planejador em baixa frequência ($0.2\text{ Hz} \sim 0.5\text{ Hz}$ / a cada 2-5 segundos) gerando metas de navegação vetorial e sub-objetivos.
   * Política motora leve feedforward (como a arquitetura da Fase 5) executando ações motoras em alta frequência ($20\text{ Hz}$ / $50\text{ms}$).
2. **Imitação Offline / SFT Puro (Fase 6):**
   * Treinamento supervisionado direto sobre demonstrações pré-gravadas, eliminando o custo de geração autoregressiva online no loop de RL.
3. **Política Fatorada Discreta PPO-BC (Fase 5):**
   * Alta eficiência em inferência e treino com baixo consumo de VRAM e convergência robusta em múltiplos pilares.

---

## 5. 📁 Estrutura de Arquivos da Fase 7

* [`fase7/politica_cot_autoregressiva.py`](file:///C:/Users/Nyx/Desktop/minecraft%20adapter/fase7/politica_cot_autoregressiva.py): Política de geração autoregressiva CoT multimodal com KV-cache.
* [`fase7/ambiente_cognitivo.py`](file:///C:/Users/Nyx/Desktop/minecraft%20adapter/fase7/ambiente_cognitivo.py): Gerador de tarefas em terreno procedural natural do Minecraft.
* [`fase7/treinar_grpo_cot_vla.py`](file:///C:/Users/Nyx/Desktop/minecraft%20adapter/fase7/treinar_grpo_cot_vla.py): Loop de treinamento GRPO com amostragem em grupo ($G=4$) e logging em tempo real.
* [`fase7/raciocinios_tempo_real.json`](file:///C:/Users/Nyx/Desktop/minecraft%20adapter/fase7/raciocinios_tempo_real.json): Stream estruturado dos raciocínios e vantagens GRPO gerados.
