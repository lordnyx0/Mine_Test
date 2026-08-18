# Legado da Fase 5 — Geradores de Dataset e Treinamentos Anteriores

Este diretório contém os scripts experimentais das subfases 5.1, 5.2.1, 5.2.2 e 5.2.3 que foram consolidados na arquitetura oficial de **36 Classes WASD Tático da Fase 5.4**.

---

## 🗂️ Mapeamento Detalhado por Subfase

### 1. Subfase 5.1: Sparse Policy Selection (18 Classes Discretas)
* **`gerar_demonstracoes_esparsas.py`:** Gerava as 500 demonstrações sintéticas iniciais com 18 classes (apenas `[W]` e giros angulares).
* **`treinar_coldstart.py`:** Treinamento supervisionado inicial sobre as 18 classes.
* **`treinar_sparse_policy.py`:** Treino de RL puro da política esparsa de 18 classes.
* *Diagnóstico:* Demonstrou que sem strafe ou ré, o robô travava em árvores e cantos.

---

### 2. Subfase 5.2.1: Mineração por Entropia (Detecção de Incerteza)
* **`minerador_decisoes_entropia.py`:** Coletava estados de alta entropia/dúvida no momento da transição entre os pilares 1 e 2.
* **`analisar_dataset_entropia.py`:** Gerava histogramas de distribuição de incerteza do modelo.
* *Diagnóstico:* Minerar por entropia pura capturava decisões onde o modelo já acertava, gerando ruído sem sinal de correção.

---

### 3. Subfase 5.2.2: Filtro de Calibração Direcional (Erros de Bifurcação)
* **`gerar_dataset_calibrado.py`:** Filtrava **apenas** os momentos em que o robô estava incerto E errou a ação ($\text{ação\_executada} \neq \text{ação\_ótima}$).
* **`treinar_calibracao.py`:** Fine-tuning supervisionado de calibração focado exclusivamente nos erros de curva.
* *Diagnóstico:* Melhorou a curva na Submeta 1, mas gerou esquecimento da marcha direta (corrida em linha reta).

---

### 4. Subfase 5.2.3: Ancoragem Causal Densa (Dataset Expandido 18D)
* **`gerar_dataset_ancorado.py`:** Fundia amostras de navegação em linha reta com os pontos de erro de bifurcação (18 classes).
* **`construir_dataset_ancorado_grande.py`:** Expandia o dataset para mais de 10.000 amostras equilibradas.
* **`treinar_sparse_policy_ancorada.py`:** Treino de RL ancorado no dataset expandido de 18 classes.
* *Diagnóstico:* Provou que 18 classes eram matematicamente insuficientes para contornar relevos mantendo a mira na torre (exigiu a criação das 36 classes WASD).

---

## 🚀 Linha do Tempo e Versão Ativa:

* **Fase 5.3:** Introdução do espaço de ação **WASD Tático de 36 Classes** e filtro inercial ($\alpha=0.65$).
* **Fase 5.4 (ATUAL - PRODUÇÃO):** Regime **PPO-BC Híbrido (70/30)** + **Rastreamento de Foco Visual** + **Torres Farol 3D de 50 blocos** ([`fase5/gerar_dataset_wasd_tatico.py`](../../fase5/gerar_dataset_wasd_tatico.py) e [`fase5/treinar_ppo_bc_hibrido.py`](../../fase5/treinar_ppo_bc_hibrido.py)).

