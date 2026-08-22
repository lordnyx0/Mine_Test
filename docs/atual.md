# 📌 Estado Atual do Projeto — Fases 1 a 7 Consolidado

## 1. 🎯 Visão Executiva
O projeto desenvolveu um agente VLA (Vision-Language-Action) multimodal embodied para navegação e raciocínio 3D no Minecraft, integrando o **Qwen2.5-0.5B-Instruct**, encoders de visão e estado, e estratégias de alinhamento por reforço (PPO, BC, CoT-VLA e GRPO Token-Level).

---

## 2. 📊 Histórico Consolidado de Fases e Resultados

| Fase | Arquitetura | Espaço de Ação | Treinamento | Performance / Status |
| :---: | :---: | :---: | :---: | :---: |
| **Fase 1** | Baseline VLA | Contínuo 3D | Imitation Learning | Locomoção básica em linha reta |
| **Fase 2** | VLA com Estado Físico | Contínuo | PPO Online | Estabilização de saltos e velocidade |
| **Fase 3** | LoRA Adaptado | Discreto 12D | PPO + Memória | Navegação com telemetria |
| **Fase 4** | Qwen3Loop Causal | Autoregressivo | Pré-treino Causal | $76.9\%$ no benchmark de raciocínio |
| **Fase 5** | PPO-BC Híbrido | Fatorado Discreto 54D | PPO-BC + Currículo | **$87.5\%$ de sucesso** em navegação sequencial de 3 pilares |
| **Fase 6** | CoT-VLA Híbrido | Imitation + LoRA | SFT em Dataset CoT | $66.2\%$ benchmark + raciocínio motor fatorado |
| **Fase 7** | CoT-GRPO Autoregressivo | Autoregressivo ($500$ tokens) | GRPO Token-Level | Raciocínio espacial profundo em terreno natural orgânico |

---

## 3. 🧠 Principais Conclusões Técnicas

1. **Eficiência de Inferência vs. Capacidade Cognitiva:**
   * Políticas com cabeças fatoradas diretas (**Fase 5**) oferecem alta taxa de atualização ($20\text{ Hz}$, $5\text{ms}$ por ação), ideais para robótica em tempo real.
   * Modelos puramente autoregressivos (**Fase 7**) geram cadeias ricas de pensamento espacial, mas demandam arquitetura hierárquica (planejamento lento assíncrono a $0.5\text{ Hz}$) para viabilidade em hardware local.
2. **Navegação Embodied em Terreno Natural:**
   * O relevo procedural do Minecraft (árvores, colinas e desníveis) fornece desafios realistas de visão e locomoção 3D sem a necessidade de muros artificiais.
