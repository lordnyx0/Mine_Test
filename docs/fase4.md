# Fase 4 — Raciocínio Lógico, Submetas Sequenciais e Loops Recursivos

A Fase 4 introduz **raciocínio causal e planejamento de submetas** sobre o `Qwen3Loop`, superando a navegação reativa de alvo único e ensinando o modelo a resolver problemas em múltiplos passos com regras estritas.

---

## 1. Por que Raciocínio no Minecraft?

Ao contrário do texto web onde o modelo pode alucinar, o Minecraft impõe **verificação causal estrita**:
- Uma submeta não concluída bloqueia o avanço da cadeia lógica.
- Obstáculos físicos (muros) punem linhas retas ingênuas.
- Sequências de cores exigem memória persistente no estado interno do transformer.

---

## 2. As Famílias de Tarefas Lógicas (`ambiente/tarefas_logicas.py`)

1. **Sequenciamento Causal Multi-Etapas com Currículo Angular:**
   - Prompt Completo: `"Vá até o bloco amarelo e depois vá até o bloco azul."`
   - **Nível 1 (Cone Suave):** Pilar 2 a $\pm 25^\circ$ do vetor de avanço inicial.
   - **Nível 2 (Desvio Lateral):** Pilar 2 a $\pm 70^\circ$.
   - **Nível 3 (Varredura Completa 360°):** Pilar 2 em qualquer quadrante ($\pm 180^\circ$).
2. **Desvio de Barreira com Sub-objetivo Visual:**
   - Prompt: `"Contorne o muro de pedra pela abertura para alcançar o bloco roxo."`
   - Muro de $3\times3$ com abertura lateral única forçando planejamento espacial não-linear.

---

## 3. As Melhorias de Segunda Geração (Fase 4 v2)

Para resolver o gargalo de ~12% de sucesso onde o modelo perdia o segundo pilar de vista:

1. **Injeção Ativa do Estágio no Forward de `agir()`:**
   - O indicador de estágio ativo ($0 \to 1$) é injetado diretamente em `sv[:, 16]` **antes** da passagem pelo modelo, permitindo que as atenções do transformer saibam exatamente qual submeta está ativa.
2. **Prompts Dinâmicos por Estágio:**
   - Etapa 1: `"Objetivo: vá até o bloco amarelo [Etapa 1/2]"`
   - Etapa 2: `"Objetivo: vá até o bloco azul [Etapa 2/2]"`
3. **Mecanismo de Reorientação Visual:**
   - Recompensa de transição angular que estimula o giro de varredura assim que a Submeta 1 é concluída ($d \le 2.5\text{m}$).

---

## 4. A Arquitetura Nativa de 3 Loops Intermediários (`LoopSplit`)

O `Qwen3Loop` possui 28 camadas desenhadas no padrão `LoopSplit`, cujas propriedades funcionais foram medidas empiricamente nos relatórios de `Testes` e `Testes2`:
- **Camadas [0 a 6] (Entrada):** Executam $1\times$ — ancoragem e fusão de embeddings visuais e textuais.
- **Camadas [7 a 20] (Miolo / Loop Cognitivo):** Executam **$3\times$ recursivamente** — com papéis bem diferenciados (medidos no `RELATORIO_FASE2.md` de Testes2):
  - *Passagem 1 (Busca Associativa Dura):* Responsável por 98% da ancoragem semântica entre a instrução em texto e a cena visual.
  - *Passagens 2 e 3 (Refinamento Recorrente):* O erro relativo cai pela metade ($0.328 \to 0.164$), onde o modelo atua como um **simulador de mundo mental**, projetando submetas e antecipando o contorno de obstáculos antes de agir.
- **Camadas [21 a 27] (Saída):** Executam $1\times$ — projeção final para os logits de controle motor (`yaw`, `jump`).

---

## 5. Validação Cruzada: Benchmark de Lógica e Matemática (`avaliacao/bench_gguf.py`)

Para comprovar que o treino no Minecraft melhora o raciocínio formal do modelo em vez de degradá-lo, o agente é avaliado diretamente contra o dataset oficial consolidado em [`benchmarks/eval_benchmark.json`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/benchmarks/eval_benchmark.json):
- **Categorias Avaliadas:** `reasoning` (10 itens), `mathematics` (10 itens), `programming` (10 itens), etc.
- **Mecanismo de Alta Velocidade (GGUF True Loop Q8_0):** O modelo preserva 28 camadas físicas (604.15 MiB) e executa a 125+ tok/s via `llama-server.exe` com patches C++ (`TENSOR_DUPLICATED`).
- **Comando:** `python avaliacao/bench_gguf.py --modelo models_gguf/fase4_loop_q8_0.gguf --nome fase4_loop_q8`
- **Acurácia Geral Medida:** **86.4%** (80.0% em reasoning, 90.0% em mathematics).
