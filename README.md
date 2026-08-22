# 🎮 Mine_Test: Agente VLA Multimodal Embodied no Minecraft

Um framework de ponta a ponta para pesquisa e desenvolvimento de agentes **Vision-Language-Action (VLA)** e **Raciocínio Espacial (Chain-of-Thought RL)** no ambiente 3D do Minecraft.

---

## 🌟 Destaques do Projeto

* **Backbone Multimodal:** Integração do modelo `Qwen2.5-0.5B-Instruct` com encoder de visão (CLIP/SigLIP) e encoder de estado físico (posição, velocidade, yaw, pitch e contato com o solo).
* **Simulador Paralelo de Alta Performance:** Servidor Mineflayer rodando $8$ instâncias em paralelo sem necessidade de interface gráfica aberta, processando lotes de ações e frames em milissegundos.
* **Currículo Evolutivo e Navegação Multi-Alvos:** Navegação autônoma contínua entre múltiplos pilares coloridos (Sub-metas A $\to$ B $\to$ C).
* **Exploração CoT-VLA & GRPO Token-Level:** Raciocínio explícito no espaço de linguagem (`<think>...</think>`) alinhado à tomada de decisão espacial motora.

---

## 📊 Linha do Tempo & Fases Desenvolvidas

```
[Fase 1: Baseline VLA] ──► [Fase 2: VLA + Estado Físico] ──► [Fase 3: LoRA + Telemetria]
                                                                     │
[Fase 6: CoT-VLA SFT] ◄── [Fase 5: PPO-BC 3 Pilares (87.5%)] ◄──────┘
         │
         ▼
[Fase 7: CoT-GRPO Autoregressivo em Terreno Natural]
```

### Resumo das Fases:
1. **Fase 1 a 3 (Fundações):** Desenvolvimento da infraestrutura de captura de visão, telemetria e calibração de ações motoras básicas.
2. **Fase 4 (Qwen3Loop Causal):** Benchmark de raciocínio causal com LLM quantizado em GGUF Q8.
3. **Fase 5 (PPO-BC Híbrido Multiestágios):** Alcançou **$87.5\%$ de sucesso** em navegação sequencial entre 3 pilares com espaço de ações fatorado (6 modos $\times$ 9 ângulos yaw).
4. **Fase 6 (CoT-VLA Imitation Learning):** Treinamento supervisionado multitarefa integrando raciocínio textual e navegação visual.
5. **Fase 7 (CoT-GRPO Autoregressivo):** Exploração de RL no nível de tokens ($500$ tokens CoT) em terreno procedural do Minecraft, analisando custos e trade-offs de contexto temporal em hardware local.

---

## 🚀 Como Executar

### 1. Iniciar o Simulador Offline (8 Ambientes Paralelos)
```bash
node servidor_offline.js
```

### 2. Avaliação da Melhor Política de Navegação (Fase 5 PPO-BC)
```bash
python fase5/avaliar_fase5_topview.py
```

### 3. Avaliação do Modelo CoT-VLA (Fase 6)
```bash
python fase6/avaliar_fase6_topview.py
```

---

## 📚 Documentação Completa
Consulte os relatórios detalhados na pasta [`docs/`](./docs):
* [`docs/fase5.md`](./docs/fase5.md): Relatório completo da política PPO-BC e navegação multi-pilares.
* [`docs/fase6.md`](./docs/fase6.md): Arquitetura CoT-VLA e benchmarks de preservação de raciocínio.
* [`docs/fase7.md`](./docs/fase7.md): Formulação matemática do GRPO Token-Level e análise de trade-offs computacionais.
* [`docs/atual.md`](./docs/atual.md): Resumo consolidado do estado do projeto.
