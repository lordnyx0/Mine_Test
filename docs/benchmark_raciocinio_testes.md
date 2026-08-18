# Benchmark de Raciocínio e Capacidade Geral (`Qwen3Loop`)

> **Objetivo:** Avaliar a retenção e o ganho de capacidade cognitiva, raciocínio lógico e matemática do modelo `Qwen3Loop` após o treinamento corporificado (*Embodied RL*) no Minecraft, de forma **100% auto-contida** neste repositório.

---

## 1. Por Que Avaliar o Raciocínio com o Benchmark Oficial?

No projeto, o backbone do agente é o `Qwen3Loop` (baseado em Qwen3-0.6B com camadas em loop $K=3$).
Durante o treinamento por reforço no Minecraft (Fases 1 a 4), o modelo aprendeu:
- Navegação espacial 3D
- Discriminação visual de cores
- Pulo neural com regularização de energia
- Sequenciamento lógico causal (*Pilar 1 ➔ Pilar 2*)

Para garantir que o modelo **não sofreu esquecimento catastrófico (*catastrophic forgetting*)** e que o treinamento espacial transferiu raciocínio causal para as categorias de **raciocínio lógico (`reasoning`)** e **matemática (`mathematics`)**, realizamos a validação cruzada contra o benchmark oficial [`benchmarks/eval_benchmark.json`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/benchmarks/eval_benchmark.json).

---

## 2. Modos de Avaliação Disponíveis

O pipeline de avaliação agora está consolidado diretamente no repositório:

### Modo A: Benchmark GGUF True Loop (Recomendado — Alta Velocidade & Baixa VRAM)
Executa o benchmark sobre o binário `llama-server.exe` com patches de C++/CUDA (`TENSOR_DUPLICATED`), preservando as **28 camadas físicas (604 MiB)** e gerando a **125+ tok/s**:

```powershell
python avaliacao/bench_gguf.py --modelo models_gguf/fase4_loop_q8_0.gguf --nome fase4_loop_q8
```
Ou filtrando por categorias:
```powershell
python avaliacao/bench_gguf.py --modelo models_gguf/fase4_loop_q8_0.gguf --nome fase4_raciocinio --categorias reasoning mathematics
```

### Modo B: Direto em PyTorch (Validação em bfloat16)
Carrega o checkpoint consolidado da Fase 4 ([`checkpoints_vla/vla_fase4_merged.pt`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/checkpoints_vla/vla_fase4_merged.pt)), sincroniza os pesos no `Qwen3LoopForCausalLM` e avalia na GPU em `bfloat16`:

```powershell
python avaliacao/avaliar_logica_testes.py --ckpt checkpoints_vla/vla_fase4_merged.pt --modo direto
```

---

## 3. Estrutura dos Arquivos Locais

| Componente | Caminho Local | Descrição |
| :--- | :--- | :--- |
| **Pacote Qwen3Loop** | [`qwen3loop/`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/qwen3loop) | Definição da arquitetura, config e modeling do `Qwen3Loop`. |
| **Módulo de Avaliação** | [`evaluation/`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/evaluation) | Scorer oficial, Jinja render, token limits e metric types. |
| **Dataset Oficial** | [`benchmarks/eval_benchmark.json`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/benchmarks/eval_benchmark.json) | 97 itens categorizados (reasoning, math, prog, etc.). |
| **Configuração de Avaliação** | [`eval_config.yaml`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/eval_config.yaml) | Tetos de tokens por categoria e stop tokens. |
| **Backbone Base** | [`checkpoints_vla/backbone_base/`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/checkpoints_vla/backbone_base) | Pesos base HF e tokenizer do Qwen3Loop. |
| **Modelo Quantizado Q8** | [`models_gguf/fase4_loop_q8_0.gguf`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/models_gguf/fase4_loop_q8_0.gguf) | Modelo True Loop Q8_0 (604.15 MiB). |
| **Harness GGUF** | [`avaliacao/bench_gguf.py`](file:///c:/Users/Nyx/Desktop/minecraft%20adapter/avaliacao/bench_gguf.py) | Executor do benchmark GGUF local. |
| **Resultados** | `avaliacao/results_gguf_bench/` | Respostas (`responses.jsonl`) e relatórios (`resumo.json`). |

---

## 4. Métricas e Interpretação

| Métrica | Significado | Meta Esperada | Resultado Medido (Fase 4 GGUF Q8) |
| :--- | :--- | :--- | :--- |
| **Taxa de Retenção Lógica** | % de acerto em `reasoning` comparado ao modelo base | $\ge 95\%$ da acurácia base | **80.0%** (8/10) |
| **Precisão Matemática** | % de respostas exatas em `mathematics` | Preservação das respostas de ponto fixo | **90.0%** (9/10) |
| **Programação & Código** | % de testes em `programming` | Preservação de sintaxe e lógica | **91.7%** (n=10) |
| **Overall (97 itens)** | Média geral em 11 categorias de benchmark | Manter $\ge 80\%$ | **86.4%** (83.8/97 pontos) |

---

## 5. Resultados Medidos — Fase 4 True Loop GGUF Q8_0

```
==============================================================================
 fase4_loop_q8 — 97 itens em 1.1 min (8183 tokens, 124.9 t/s medio)
==============================================================================
  context                   85.7%  (n=7)
  creativity               100.0%  (n=7)
  general_knowledge         80.0%  (n=10)
  instruction_following     75.0%  (n=10)
  mathematics               90.0%  (n=10)
  programming               91.7%  (n=10)
  reasoning                 80.0%  (n=10)
  robustness               100.0%  (n=8)
  summarization             95.2%  (n=7)
  translation               70.0%  (n=10)
  writing                   93.8%  (n=8)
------------------------------------------------------------------------------
  MEDIA GERAL               86.4%
==============================================================================
```
