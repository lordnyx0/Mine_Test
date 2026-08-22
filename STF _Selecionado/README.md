# 📦 STF _Selecionado — Pacote Autônomo de Conversão, Servidor & Benchmark Q8

Esta pasta reúne todos os recursos necessários para:
1. **Converter e Quantizar** checkpoints PyTorch / LoRA para o formato **GGUF Q8_0**.
2. **Subir o Servidor Local de Inferência** (`llama-server`) com aceleração GPU total.
3. **Executar a Avaliação de Raciocínio Lógico (Benchmark)** e comparar a retenção cognitiva com o **Modelo Professor**.
4. **Configurar e Rodar o Simulador Minecraft Offline** (8 instâncias paralelas).

---

## 📂 Estrutura da Pasta

```
STF _Selecionado/
├── README.md                                  # Este guia explicativo
├── modelos/
│   ├── base_professor_f16.gguf                # Modelo Base Professor em FP16 (1.2 GB)
│   ├── base_professor_q8_0.gguf               # Modelo Base Professor quantizado em Q8_0 (639 MB)
│   └── backbone_base_fp16/                    # Pesos HuggingFace base do Qwen3Loop
├── benchmark/
│   ├── eval_benchmark.json                    # Dataset de testes de raciocínio e lógica
│   ├── eval_config.yaml                       # Configuração dos parâmetros de avaliação
│   ├── bench_gguf.py                          # Client de inferência e coleta de respostas
│   └── avaliar_logica_testes.py               # Avaliador e calculador de notas
└── scripts/
    ├── exportar_para_hf.py                    # Funde adaptadores LoRA e exporta para formato HF
    ├── converter_e_quantizar.py               # Pipeline automático: Checkpoint .pt -> HF -> GGUF Q8
    ├── iniciar_servidor_llm.bat               # Inicia o servidor HTTP local na porta 8085
    ├── iniciar_servidor_minecraft.bat         # Inicia o simulador Mineflayer na porta 3002
    └── executar_benchmark.py                  # Dispara a bateria de testes e gera relatórios
```

---

## 🛠️ Passo a Passo de Utilização

### 1️⃣ Como Converter e Quantizar um Checkpoint para GGUF Q8

Para transformar qualquer checkpoint treinado (como `vla_fase6_cot_melhor.pt` ou outros) em GGUF Q8:

```bash
python "STF _Selecionado/scripts/converter_e_quantizar.py" --ckpt "checkpoints_vla/vla_fase6_cot_melhor.pt"
```

O script irá automaticamente:
1. Fundir os adaptadores LoRA no backbone base.
2. Salvar a pasta Hugging Face em `STF _Selecionado/modelos/modelo_hf_exportado`.
3. Converter para `STF _Selecionado/modelos/modelo_f16.gguf`.
4. Quantizar diretamente para `STF _Selecionado/modelos/modelo_q8_0.gguf`.

---

### 2️⃣ Como Iniciar o Servidor LLM Local (Inferência Rápida em GPU)

Para carregar o modelo no servidor local com aceleração CUDA:

* **Para rodar o Modelo Professor Base:**
  Dê um duplo clique no arquivo:
  ```
  STF _Selecionado/scripts/iniciar_servidor_llm.bat
  ```
* **Para rodar o modelo SFT recém-convertido:**
  Execute via terminal passando o caminho do modelo:
  ```bash
  "STF _Selecionado/scripts/iniciar_servidor_llm.bat" "STF _Selecionado/modelos/modelo_q8_0.gguf"
  ```

O servidor ficará ativo em `http://127.0.0.1:8085/v1` pronto para receber chamadas de inferência.

---

### 3️⃣ Como Rodar o Benchmark de Raciocínio Lógico

Com o servidor ativo no passo anterior, abra outro terminal e execute:

```bash
python "STF _Selecionado/scripts/executar_benchmark.py" --name "meu_modelo_sft_q8"
```

O script irá:
1. Enviar todas as questões do `eval_benchmark.json` para o modelo em execução.
2. Salvar as respostas em `avaliacao/results_gguf_bench/meu_modelo_sft_q8/responses.jsonl`.
3. Tabular e imprimir a nota final por categoria (Raciocínio, Matemática, Programação, etc.) e comparar com a referência do **Professor Base ($76.9\%$)**.

---

### 4️⃣ Como Rodar o Simulador Minecraft Offline

Para iniciar as 8 instâncias paralelas do ambiente de robótica:

* Dê um duplo clique em:
  ```
  STF _Selecionado/scripts/iniciar_servidor_minecraft.bat
  ```
* Ou via terminal:
  ```bash
  node servidor_offline.js
  ```
O simulador responderá no endpoint `http://127.0.0.1:3002/lote/info`.
