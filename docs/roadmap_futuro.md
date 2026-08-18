# Roadmap Futuro — LoRA, World Model e Transferência para o Qwen3Loop v2

Este documento estabelece o protocolo de engenharia para expandir a capacidade cognitiva do modelo, amadurecer as camadas intermediárias de loop e transferir 100% do aprendizado para versões futuras do `Qwen3Loop`.

---

## 1. Módulo LoRA Desacoplado (`modelo/lora_vla.py`)

Para permitir que o modelo desenvolva raciocínio espacial profundo sem esquecimento catastrófico do conhecimento de linguagem pré-treinado:

```python
from modelo.lora_vla import aplicar_lora, obter_parametros_lora

# Injeta matrizes LoRA (r=16, alpha=32) nas atenções do Qwen
vla.qwen_model = aplicar_lora(vla.qwen_model, r=16, alpha=32.0)
treinaveis = [p for p in vla.parameters() if p.requires_grad]
# Parâmetros treináveis saltam de 10.6M para ~35M - 45M
```

### Benefícios:
- **Proteção dos Pesos Base:** Os 290M parâmetros originais permanecem congelados.
- **Fusão Direta (*LoRA Merging*):** O delta aprendido ($\Delta W_{\text{motor}}$) pode ser somado diretamente a qualquer novo checkpoint base através de `mesclar_todos_lora(modelo)`.

---

## 2. Perda Auxiliar de Modelo de Mundo (`modelo/world_model_loss.py`)

Para treinar as camadas intermediárias de loop recursivo a compreenderem física e dinâmica 3D:

$$L_{\text{total}} = L_{\text{PPO}} + \lambda_{\text{world}} \cdot L_{\text{world}}$$

Onde $L_{\text{world}}$ penaliza o erro entre a previsão latente do próximo frame $\hat{z}_{t+1}$ e a representação real $z_{t+1}$ extraída pelo SigLIP.

---

## 3. Protocolo de Warm-Start para o `Qwen3Loop v2`

Quando uma versão superior do `Qwen3Loop` for disponibilizada:

1. **Herança dos Adaptadores Visuais:**
   - O `Resampler`, `Projector`, `StateEncoder` e `ActionHeads` treinados no checkpoint `vla_fase3.pt` são carregados diretamente no novo modelo.
2. **Transferência de LoRA:**
   - As matrizes LoRA treinadas são somadas aos novos pesos do Qwen v2 via interpolação de posto:
     $$\mathbf{W}_{\text{v2\_embodied}} = \mathbf{W}_{\text{Qwen\_v2}} + 0.8 \cdot \Delta\mathbf{W}_{\text{motor}}$$
3. **Co-Treino com o Dataset Embodied (`infra/dataset_embodied.py`):**
   - Usar os shards gravados em `dataset_embodied/` para treinar conjuntamente:
     - 80% Dados de Texto / Raciocínio Geral
     - 20% Dados Embodied Multimodais do Simulador

---

## 4. Transferência Cross-Game (Unity / Godot / FPS)

Como o VLA opera em representações latentes generalizadas (pixels 640x360 $\to$ SigLIP $\to$ tokens de motor):
- Qualquer jogo que forneça um stream de frames e aceite comandos de teclado/mouse (`W`, `mouse_dx`, `SPACE`) pode ser plugado diretamente na mesma interface de inferência `agir(estado, obs, prompt)`.
