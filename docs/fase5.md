# Fase 5 — Treinamento PPO-BC Multimodal Híbrido e Análise da Regressão Cognitiva

A Fase 5 teve como objetivo treinar o agente multimodal `Qwen3Loop` (0.6B) em tarefas de navegação com submetas sequenciais e múltiplos pilares no Minecraft através de uma combinação de PPO (Reinforcement Learning) e Behavioral Cloning (BC) com o Cérebro Supervisor.

---

## 1. Resultados no Ambiente Físico (Minecraft) — Sucesso Operacional

No ambiente 3D, a combinação de PPO + BC com o Cérebro Supervisor atingiu desempenho recorde de locomoção e controle espacial:
- **Submeta 1 ($P_1$):** **$93.8\%$** de taxa de chegada no TopView (15/16 episódios).
- **Missão Completa ($P_1 	o P_2$):** **$50.0\%$** de taxa de sucesso global no TopView (8/16 episódios).
- **Locomoção Ativa ($W\%$):** $> 78\%$ com supressão total do efeito espiral ($0.5\%$ de tempo estático).
- **Execução em Tempo Real:** 16 episódios paralelos em $112.1	ext{ s}$ no simulador.

---

## 2. A Falha do Experimento de Raciocínio (Regressão no Causal LM)

O grande objetivo científico da Fase 5 era verificar se o treinamento corporificado com causalidade física e submetas no Minecraft fortaleceria a capacidade de raciocínio simbólico, lógico e matemático do modelo.

### Resultados no Benchmark Oficial de 97 Itens (GGUF True Loop Q8_0):
$$\text{Baseline Fase 4: } \mathbf{86.4\%} \;\longrightarrow\; \text{Fase 5 (Direto): } \mathbf{54.8\%} \;\longrightarrow\; \text{Fase 5 (com <think> curto): } \mathbf{39.0\%}$$

| Categoria | Fase 4 (Baseline) | Fase 5 (PPO-BC) | Variação |
| :--- | :---: | :---: | :---: |
| 💻 **Programming** | **91.7%** | 73.3% | $-18.4\%$ |
| ✍️ **Writing** | **93.8%** | 75.0% | $-18.8\%$ |
| 🎨 **Creativity** | **100.0%** | **100.0%** | $0.0\%$ |
| 🧠 **Reasoning** | **80.0%** | 60.0% | $-20.0\%$ |
| 📑 **Summarization** | **95.2%** | 73.8% | $-21.4\%$ |
| 🎯 **Instruction Following** | **75.0%** | 65.0% | $-10.0\%$ |
| 🔍 **Context & Memory** | **85.7%** | 28.6% | $-57.1\%$ |
| 📐 **Mathematics** | **90.0%** | 40.0% | $-50.0\%$ |
| 🌐 **General Knowledge** | **80.0%** | 40.0% | $-40.0\%$ |
| **MÉDIA GERAL (97 ITENS)** | **86.4%** | **54.8%** | $\mathbf{-31.6\%}$ |

---

## 3. As Causas-Raiz da Falha da Fase 5

1. **Ausência Total de Perda de Linguagem ($\mathcal{L}_{\text{LM}} = 0$):**
   Durante as 50 iterações de PPO-BC, as 112 camadas de LoRA ($W_q, W_k, W_v, W_o$) foram atualizadas unicamente por gradientes da política motora (36 classes discretas de teclas/mouse) a partir do último token latente $h_{\text{last}}$. Nenhum token de texto foi retropropagado.
2. **O "Atalho Motor" (*The Motor Shortcut*):**
   O transformer aprendeu a atuar como um mero multiplicador geométrico de matriz (projetando pixels em teclas), desacoplando a auto-atenção causal da semântica dos 151k tokens do vocabulário.
3. **Degeneração Autoregressiva e Repetição (*Attention Drift*):**
   Sem auto-inibição atencional para texto, a geração livre entrou em ciclos periódicos (*limit cycles*), repetindo frases como *"Let me think. Let me try..."* dezenas de vezes.

---

## 4. A Solução: Transição para CoT-VLA (Raciocínio em Linguagem na Ação)

Para que o aprendizado no Minecraft transfira para o raciocínio formal, a Fase 6 adota o **CoT-VLA**:
- O modelo gera explicitamente o monólogo de raciocínio espacial em `<think>...</think>` antes de emitir a ação `<action>...</action>`.
- O treinamento utiliza perda multitarefa: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CoT-VLA}} + \gamma \mathcal{L}_{\text{Anchor}}$ (intercalando navegação e raciocínio lógico/matemático).
