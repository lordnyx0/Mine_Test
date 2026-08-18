# Playbook de Escalonamento & Reprodução para Modelos VLA Maiores
> **Guia Passo a Passo Definitivo para Treinar Modelos VLA (Qwen 1.5B, 7B ou similares) em Raciocínio Espacial e Controle Tático no Minecraft — Evitando Todas as Armadilhas Diagnosticadas.**

---

## 🎯 Visão Geral do Pipeline de Escalonamento

Treinar um LLM de grande porte para controlar um agente no Minecraft ou qualquer ambiente 3D dinâmico não deve ser feito nem por **RL puro do zero** (*computacionalmente proibitivo*), nem por **Imitação Simples** (*gera volatilidade e ciclos limites*).

O método comprovado segue uma **Estrutura em 4 Estágios**:

```
[1. Backbone & LoRA] ──► [2. Espaço de Ação WASD] ──► [3. Dataset Ancorado 70/30] ──► [4. Inferência com Filtro Inercial]
```

---

## 📋 Passo a Passo para Reprodução em um Novo Modelo

### Passo 1: Configuração do Backbone e Adaptadores LoRA
1. **Seleção de Camadas para Adaptação:**
   * Em arquiteturas transformer padrão, aplique LoRA nas projeções de atenção (`q_proj`, `k_proj`, `v_proj`, `o_proj`) das **camadas intermediárias de raciocínio** (equivalente às camadas centrais do modelo).
   * **Hiperparâmetros recomendados:** $\text{rank} = 16$, $\alpha = 32.0$, $\text{dropout} = 0.05$.
2. **Projeção Multimodal (State Encoder + Vision Resampler):**
   * Projete o vetor de estado geométrico ($32\text{D}$) através de uma MLP de 2 camadas: $\text{Linear}(32 \to 256) \to \text{GELU} \to \text{Linear}(256 \to D_{\text{hidden}})$.
   * Concatene os embeddings de estado antes dos tokens de instrução textual.

---

### Passo 2: O Espaço de Ação Holonômico (36 Ações Ortogonais)
**NUNCA utilize apenas ações de avanço frontal (`W` + rotação de mouse).** 
O modelo deve possuir 36 classes ortogonais divididas em 4 intenções claras:

| Faixa de Classes | Intenção Tática | Teclas Pressionadas | Bins de Yaw |
|---|---|---|---|
| **0 a 8** | **Giro Parado (Alinhamento)** | `hold: []` (Sem W) | $\pm 120^\circ, \pm 60^\circ, \pm 25^\circ, \pm 5^\circ, 0^\circ$ |
| **9 a 17** | **Sprint Frontal (Avanço Reto)** | `hold: ["W"]` | $\pm 120^\circ, \pm 60^\circ, \pm 25^\circ, \pm 5^\circ, 0^\circ$ |
| **18 a 26** | **Sprint com Salto (Transposição)** | `hold: ["W", "SPACE"]` | $\pm 120^\circ, \pm 60^\circ, \pm 25^\circ, \pm 5^\circ, 0^\circ$ |
| **27 a 29** | **Strafe Esquerda (Fixação de Olhar)** | `hold: ["W", "A"]` | $-5^\circ, 0^\circ, +5^\circ$ |
| **30 a 32** | **Strafe Direita (Fixação de Olhar)** | `hold: ["W", "D"]` | $-5^\circ, 0^\circ, +5^\circ$ |
| **33 a 35** | **Marcha Ré (Desengate / Quinas)** | `hold: ["S"]` | $-5^\circ, 0^\circ, +5^\circ$ |

---

### Passo 3: Mineração e Ancoragem do Dataset (A Proporção de Ouro 70/30)
Para que o modelo não sofra **esquecimento catastrófico** nem decore trajetórias estáticas:

1. **70% do Buffer (Locomoção Densa Contínua — ~11.000 amostras):**
   * Trajetórias em linha reta em solo plano e irregular.
   * Ângulos pequenos ($|\theta| \le 12^\circ$) mapeados para **Sprint Puro (`W`)**.
2. **30% do Buffer (Bifurcações de Alta Incerteza / Spawn Desalinhado — ~5.000 amostras):**
   * Spawns com ângulo reverso ($|\theta| > 45^\circ$) mapeados para **Giro Parado (`[]`)**.
   * Alvos a média distância descentralizados ($15^\circ < |\theta| \le 45^\circ$) mapeados para **Strafe Lateral (`W+A` / `W+D`)**.
   * Proximidade de colisão frontal ($d < 1.5\text{m}$) mapeada para **Marcha Ré (`S`)**.

---

### Passo 4: Otimização Supervisionada
* **Otimizador:** `AdamW` com `weight_decay = 1e-4`.
* **Learning Rate:** $2.0 \times 10^{-4}$ com **Cosine Annealing Scheduler** decaindo até $2.0 \times 10^{-5}$.
* **Épocas:** 12 a 15 épocas completas.
* **Critério de Parada:** A Loss deve convergir para a faixa de **$0.18$ a $0.25$**. 
  * Se a Loss for $> 0.40 \rightarrow$ O modelo gerará órbitas elípticas abertas por falta de atração vetorial.
  * Se a Loss for $< 0.05 \rightarrow$ Alerta de overfitting/memorização em dataset pequeno.

---

### Passo 5: O Filtro Inercial Passa-Baixas na Inferência (OBRIGATÓRIO)
Mesmo com uma rede perfeitamente treinada, a taxa de controle discreto (passos de 250ms) pode entrar em ressonância angular.
No loop de inferência `agir()`, aplique amortecimento sobre a rotação do mouse:
$$\text{dx}_{\text{suavizado}} = 0.65 \cdot \text{dx}_{\text{predito}} + 0.35 \cdot \text{dx}_{t-1}$$

---

## 🚫 Lista de Armadilhas Fatais Diagnosticadas (O QUE NUNCA FAZER)

| # | Erro Comum | Consequência no Jogo | Como Evitar |
|---|---|---|---|
| ❌ **1** | Treinar em dataset pequeno de calibração pura (< 1.000 amostras). | **Overfitting Imediato:** Acurácia vai a 99%, mas o robô perde a capacidade de andar reto e trava. | Mantenha sempre um buffer grande (>15k amostras) com 70% de locomoção densa. |
| ❌ **2** | Usar apenas `W` + mouse sem `A/D` (Strafe) e sem `[]` (Giro Parado). | **Efeito Beyblade / Sacarrolha:** O robô avança em espirais oscilatórias de $\pm 60^\circ$ e rodopia no spawn. | Utilize o espaço holonômico de 36 ações. |
| ❌ **3** | Permitir que o robô corra para frente no spawn antes de se alinhar. | **Sprint Cego Invertido:** O robô nasce de costas para o objetivo e se afasta 15 metros antes de virar. | Forçar a classe **Giro Parado (0..8)** sempre que $|\theta| > 45^\circ$. |
| ❌ **4** | Não amortecer o yaw delta na inferência. | **Ressonância Harmônica:** Alternância rápida entre $+120^\circ$ e $-120^\circ$ no mouse. | Aplicar o filtro inercial passa-baixas ($\alpha=0.65$). |
| ❌ **5** | Ignorar a frenagem pós-submeta. | **Overshoot Cego:** O robô atinge a Submeta 1 a 100km/h e passa direto, errando o ângulo para a Submeta 2. | Injetar frenagem (`hold: []` ou `hold: ["S"]`) nos 3 primeiros passos após a conclusão da Submeta 1. |
| ❌ **6** | Treinar RL online isolado sem ancoragem supervisionada contínua. | **Degradação de Submeta:** O RL aprende a desacelerar na transição, mas perde acurácia de tiro direto na Submeta 1 (esquecimento catastrófico). | **PPO-BC Híbrido:** Durante o RL, mantenha sempre **70% do minilote amostrado do buffer supervisionado offline** ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda \mathcal{L}_{\text{BC}}$). |
