# FASE 3 — Alvo VISUAL (Servo-Visão Pura Sem Coordenadas)

A pergunta mais forte e definitiva da ponte multimodal:
> **O modelo consegue navegar até um objeto VISÍVEL no campo de visão, sem receber coordenadas externas?**

---

## 1. Especificação da Fase 3

Nas Fases 1 e 2, o vetor de objetivo continha `(frente, lado, dist, angulo)`, permitindo que a rede neural usasse trigonometria analítica (`atan2`) sem depender estritamente dos pixels da câmera.

Na **Fase 3**, o canal de coordenadas é **100% ZERADO** (`goal_vec = zeros(4)`). O modelo depende exclusivamente de:
1. **Frames da Câmera (640x360):** Processados pelo encoder visual SigLIP + adaptadores treináveis (`resampler` e `projector`).
2. **Instrução Textual em Linguagem Natural:** Instrução tokenizada pelo Qwen (`"Objetivo: va ate o bloco roxo."`, `"Objetivo: va ate o bloco amarelo."`, `"Objetivo: va ate o bloco azul."`).

---

## 2. Alvos Visuais e Paleta de Cores de Alto Contraste

Para evitar falsos positivos com troncos e folhagens comuns do mapa, o simulador insere pilares verticais de 2 blocos de altura com cores vívidas:

| Cor Alvo | Bloco do Minecraft (ID) | RGB de Renderização | Prompt de Instrução |
|---|---|---|---|
| 🟣 **Roxo** | `obsidian` (ID 49) | `[155, 38, 182]` | *"Objetivo: va ate o bloco roxo."* |
| 🟡 **Amarelo** | `gold_block` (ID 41) | `[245, 215, 20]` | *"Objetivo: va ate o bloco amarelo."* |
| 🔵 **Azul** | `lapis_block` (ID 22) | `[25, 110, 245]` | *"Objetivo: va ate o bloco azul."* |

---

## 3. Protocolo de Treinamento e Otimização

- **Orçamento de Passos:** `PASSOS_MAX_F3 = 80` ($20\text{ segundos}$ a 4 Hz).
- **Raio de Chegada:** $2.0\text{ metros}$ da base do pilar.
- **Recompensa Densa:** $r_t = d_{t-1} - d_t$ (recompensa por aproximação euclidiana) + bônus de $+10.0$ na chegada.
- **Blindagens de Estabilidade:**
  - *Logit Bounding:* `torch.tanh(logits / 3.0) * 3.0` (impede colapso determinístico da entropia).
  - *Piso de Desvio Padrão:* `PISO_G_STD = 3.0` (evita explosão de vantagens e gradientes).
  - *Proteção dos Adaptadores:* `clip_grad_norm_ = 0.5` e decaimento de peso $10^{-2}$.
  - *Minilote Ideal:* `minilote = 12` na RTX 3060 12GB (4.3 GB de folga de VRAM).

---

## 4. Integração com a Camada do Cérebro (`politica/cerebro.py`)

A arquitetura hierárquica divide o agente em dois níveis:
1. **VLA Reflexo (4 Hz):** Mapeia pixels + instrução de texto em rotações imediatas da câmera ($\Delta\text{yaw}$).
2. **Cérebro Supervisor (~1 Hz):** Monitora o deslocamento real do robô.
   - **Anti-Stuck:** Se $\Delta d < 0.12\text{m}$ por 2 passos seguidos, injeta manobra de evasão lateral (`W` + `SPACE` + mouse lateral).
   - **Parada Terminal:** Solta `W` instantaneamente (`hold: []`) ao penetrar no raio de $2.0\text{m}$.
   - **Varredura Ativa:** Executa varredura angular contínua se o alvo não estiver no cone de visão frontal.

---

## 5. Resultados Empíricos Obtidos

| Política | Chegada Geral | $d_{\text{final}}$ Médio | Cor Roxo | Cor Amarelo | Cor Azul |
|---|---|---|---|---|---|
| **`ALEATORIO`** | 8.3% | 17.53m | 5.0% | 0.0% | 18.8% |
| **`SO_W`** | 8.3% | 21.66m | 5.0% | 8.3% | 12.5% |
| **`MODELO VLA PURO`** | 12.5% | 9.97m | 15.0% | 16.7% | 6.2% |
| **`CEREBRO + VLA`** | **20.8%** | **9.89m** | 5.0% | 16.7% | **43.8%** |
| **`PILOTO BFS`** (Teto) | 87.5% | 2.99m | 90.0% | 83.3% | 87.5% |
