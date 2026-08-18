# Documentação — Agente Minecraft

Agente para Minecraft construído sobre o `Qwen3Loop` (Qwen3-0.6B adaptado a
arquitetura em loop). Objetivo de longo prazo: um agente que aprenda tarefas e
transponha para outros jogos.

O `CLAUDE.md` na raiz é o guia rápido — comandos, armadilhas e números que não
devem ser remedidos. Estes documentos são a referência de profundidade.

| documento | quando ler |
|---|---|
| [arquitetura.md](arquitetura.md) | como as peças se encaixam; os dois laços; o modelo multimodal |
| [planejador.md](planejador.md) | como a navegação clássica funciona e por que ela é o especialista |
| [simulador.md](simulador.md) | API HTTP, sondas de diagnóstico, visualizador |
| [experimentos.md](experimentos.md) | o que foi medido, o que foi refutado, e por quê |
| [fase1.md](fase1.md) | controle local em terreno plano, e o currículo completo |
| [fase2.md](fase2.md) | terreno real, obstáculos e `SPACE` — onde a visão foi comprovada causalmente |
| [fase3.md](fase3.md) | servo-visão pura sem coordenadas, multi-cores (Roxo, Amarelo, Azul) e Cérebro |
| [fase4.md](fase4.md) | raciocínio lógico sequencial, submetas e 3 loops recursivos no Qwen3Loop |
| [fase5.md](fase5.md) | sparse policy selection, ancoragem de buffer causal e raciocínio tático holonômico (WASD 36 ações) |
| [playbook_scaling_vla.md](playbook_scaling_vla.md) | **guia mestre de escalonamento para modelos maiores (1.5B/7B), passo a passo e armadilhas evitadas** |
| [roadmap_futuro.md](roadmap_futuro.md) | expansão de capacidade cognitiva com LoRA, World Model e transferência para Qwen3Loop v2 |
| [metodo.md](metodo.md) | as regras de medição e de treino de RL que este projeto aprendeu na marra, mais as leis da literatura que sustentam cada uma |
| [HANDOFF.md](HANDOFF.md) | histórico do `Qwen3Loop` base, antes de virar agente Minecraft |

> **Achado de 2026-08-13:** o backbone Qwen3Loop **nunca era carregado** —
> `Qwen3LoopModel(config)` nasce aleatório e ninguém substituía os pesos. Todo o
> VLA rodou sobre ruído até esta data. Corrigido; ver
> [experimentos.md §5](experimentos.md).

## O estado em um parágrafo

A navegação clássica funciona (85% de chegada a 100 blocos). Três tentativas
independentes de destilar isso numa rede por casamento de ação falharam, e a
causa foi medida: a ação do planejador não é inferível da observação do aluno
(19% contra 11% de acaso). O que passou a funcionar foi RL com recompensa do
ambiente, sem professor, num problema local decomposto — a Fase 1.

## Decisão de arquitetura

**Especialistas motores primeiro, LLM coordenando depois.** Não é MoE: MoE são
blocos FFN dentro do transformer com roteamento por token, para eficiência de
parâmetros, e não dão modularidade de comportamento nem permitem medir um
especialista isolado. O que serve aqui é registro de habilidades com uso de
ferramentas.
