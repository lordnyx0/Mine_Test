# Guia de Treinamento Cloud no Modal ($30 de Crédito Gratuito)

O **Modal (modal.com)** é uma plataforma de computação em nuvem *serverless* que cobra por segundo de GPU, perfeita para treinar nosso modelo com velocidade máxima e custo quase zero.

Com os **$30 dólares gratuitos**:
- **GPU NVIDIA A10G (24 GB VRAM)**: $\sim \$1.10 / \text{hora} \to$ **$27.2\text{ horas}$ de treino grátis**.
- **GPU NVIDIA L4 (24 GB VRAM)**: $\sim \$0.80 / \text{hora} \to$ **$37.5\text{ horas}$ de treino grátis**.
- Um treino de 80 iterações (com 16 bots em paralelo) leva apenas **$\sim 25\text{ minutos}$** e custa **apenas $\sim \$0.45$**.

---

## 🚀 Passo a Passo (2 Minutos de Configuração)

### 1. Instalar o Modal no seu Python
Abra o terminal no seu computador e instale o pacote oficial do Modal:
```powershell
pip install modal
```

---

### 2. Autenticar sua Conta (Pegar os $30 de Crédito)
Execute o comando abaixo. Ele abrirá automaticamente o navegador para você fazer login com sua conta do Modal e vincular seu token:
```powershell
modal setup
```

---

### 3. Enviar o Save do Minecraft para o Volume da Nuvem (Apenas 1x)
Para que o simulador no Modal tenha o mesmo terreno 3D do seu mundo:
```powershell
modal volume put minecraft-world-save "C:\Users\Nyx\AppData\Roaming\.minecraft\saves\New World-" /
```

---

### 4. Disparar o Treinamento da Fase 4 na Nuvem
Execute o script cloud com 1 comando:
```powershell
modal run infra/modal_treino.py --iteracoes 80 --gpu A10G --n-envs 16
```

> **💡 O que acontece nos bastidores:**
> - O Modal sobe instantaneamente uma máquina com **NVIDIA A10G (24GB VRAM)**.
> - O servidor Node.js offline inicia com **16 bots paralelos** em memória.
> - O PPO treina o **Qwen3Loop + LoRA** em precisão mista `bfloat16` a velocidade máxima.
> - Ao término, o checkpoint é salvo no `modal.Volume` persistente.

---

### 5. Baixar o Checkpoint Treinado para seu PC
Quando o treino terminar, baixe o modelo treinado para sua pasta local:
```powershell
modal volume get minecraft-vla-checkpoints vla_fase4_logica.pt ./checkpoints_vla/
```

---

## 📈 Tabela de Comparação: Local vs Modal Cloud

| Recurso | Local (RTX 3060 12GB) | Modal Cloud (NVIDIA A10G / L4 24GB) |
|---|---|---|
| **VRAM Disponível** | 12 GB | **24 GB** (permite minilotes $4\times$ maiores) |
| **Ambientes Simultâneos** | 8 bots | **16 a 32 bots paralelos** |
| **Tempo da Fase 4 (80 its)** | $\sim 4\text{h }40\text{min}$ | **$\sim 25\text{ a }30\text{ minutos}$** |
| **Custo Real da Execução** | Energia/Tempo | **$\sim \$0.45$ (coberto pelos $30 de bônus)** |
| **Execuções Restantes com $30** | — | **Mais de 55 treinos completos!** |
