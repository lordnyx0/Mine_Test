# Estrutura do Projeto e Mapa de Diretórios

Este documento descreve a organização modular do repositório `lordnyx0/Mine_Test`.

```
minecraft adapter/
├── ambiente/               # Interface com o ambiente e simulação
├── avaliacao/              # Scripts de validação e avaliação específicos por fase
├── benchmarks/             # Benchmarks sintéticos e datasets de teste
├── checkpoints_vla/        # Pesos, backbones e checkpoints treinados do VLA
├── dataset/                # Datasets de demonstrações de locomoção e navegação
├── docs/                   # Documentação técnica, arquitetura e relatórios
│   ├── imagens/            # Gráficos de trajetórias (topview) e capturas de debug
│   └── referencias/        # Papers e artigos de pesquisa de referência
├── evaluation/             # Framework padronizado e modular de benchmarks
├── fase5/                  # Fase 5.5 atual: PPO + BC Híbrido, WASD tático e Recompensa Visual
├── ferramentas/            # Utilitários de apoio (plotagem de trajetórias, monitores)
├── infra/                  # Utilitários de GPU, locks de concorrência e launchers
├── legado/                 # Códigos e pipelines experimentais de fases anteriores
├── logs/                   # Logs de execução e treinamentos
├── mineflayer_server/      # Servidor de simulação física offline Node.js (8 bots paralelos)
├── modelo/                 # Definição do modelo VLA, LoRA adaptado e cabeças de ação
├── models_gguf/            # Modelos quantizados em formato GGUF para inferência leve
├── politica/               # Políticas de inferência e loop de raciocínio (Qwen3Loop)
├── qwen3loop/              # Arquitetura Qwen3Loop e definições de camadas de loop
├── scratch/                # Scripts temporários, diagnósticos e testes pontuais
├── scripts/                # Automações em batch (.bat) e scripts utilitários
└── treino/                 # Pipelines de treinamento das Fases 1 a 4
```

---

## Detalhamento dos Pacotes Principais

### 1. Núcleo VLA e Política
* **`modelo/`**: Contém `vla_model.py` (arquitetura multimodal SigLIP + Backbone + Heads) e `lora_vla.py` (aplicação de LoRA nos blocos de atenção).
* **`politica/`**: Contém `politica_raciocinio.py` (`PoliticaRaciocinioLoop`), responsável pelo loop recursivo de pensamento e amostragem de ações motoras.
* **`qwen3loop/`**: Implementação mínima e customizada do backbone recursivo `Qwen3LoopModel` (28 camadas executadas em loops virtuais).

### 2. Ambiente e Simulação
* **`ambiente/`**: Conexão HTTP com o simulador, codificação do vetor de estado proprioceptivo de 32 dimensões e gerenciamento de tarefas lógicas.
* **`mineflayer_server/`**: Servidor Node.js (`servidor_offline.js`) com física em JavaScript e suporte a N instâncias em lote via HTTP na porta 3002.

### 3. Fase 5.5 (Atual)
* **`fase5/treinar_ppo_bc_hibrido.py`**: Pipeline principal de aprendizado por reforço PPO com ancoragem comportamental (BC) e recompensa visual esparsa/densa.
* **`fase5/acoes_taticas.py`**: Espaço de ações fatorado (6 modos de locomoção $\times$ 9 resoluções de yaw = 36 ações).
* **`fase5/recompensa_visual.py`**: Rastreador visual por SigLIP e cálculo de densidade de recompensa por aproximação de alvo.

### 4. Infraestrutura e Avaliação
* **`infra/`**: `gpu_utils.py` (gestão de VRAM, precisão bfloat16 e controle de locks) e `run_vla_agent.py`.
* **`evaluation/`**: Suite padronizada com runners assíncronos, cálculo de métricas e persistência de relatórios.
* **`avaliacao/`**: Validadores específicos de cada etapa do desenvolvimento (Fase 1 a 5).
