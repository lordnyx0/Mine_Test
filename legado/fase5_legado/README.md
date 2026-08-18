# Legado da Fase 5 — Geradores de Dataset e Treinamentos Anteriores

Este diretório contém os scripts experimentais das subfases 5.1 e 5.2 que foram substituídos pela arquitetura oficial de **36 Classes WASD Tático da Fase 5.4**.

---

## 🗂️ Mapeamento por Fase e Função

### 1. Subfase 5.1: Sparse Policy Selection (18 Classes de Ação)
* **`gerar_demonstracoes_esparsas.py`:** Gerava as 500 demonstrações sintéticas iniciais com 18 classes (apenas `[W]` e giros). Substituído pelas 36 classes holonômicas.
* **`treinar_coldstart.py`:** Treinamento supervisionado inicial sobre as 18 classes.
* **`treinar_sparse_policy.py`:** Treino de RL sobre a política esparsa de 18 classes.

### 2. Subfase 5.2: Mineração de Entropia e Calibração Direcional
* **`minerador_decisoes_entropia.py`:** Coletava estados de alta entropia/dúvida no momento da curva entre os pilares 1 e 2.
* **`analisar_dataset_entropia.py`:** Plotava histogramas de incerteza do modelo.
* **`gerar_dataset_ancorado.py`:** Fundia amostras de navegação em linha reta com os pontos de bifurcação (18 classes).
* **`gerar_dataset_calibrado.py`:** Filtrava apenas os erros de bifurcação para ajuste supervisionado.
* **`construir_dataset_ancorado_grande.py`:** Expandia o dataset para 10.000 amostras com a cabeça de 18 ações.
* **`treinar_calibracao.py`:** Fine-tuning supervisionado de calibração sobre os erros de bifurcação.
* **`treinar_sparse_policy_ancorada.py`:** Treino de ancoragem da política esparsa de 18 classes.

---

## 🚀 Gerador Oficial Ativo (Fase 5.4):
* **[`fase5/gerar_dataset_wasd_tatico.py`](../../fase5/gerar_dataset_wasd_tatico.py):** Gerador oficial de 16.145 amostras com suporte completo a **36 classes WASD** (giro parado, sprint frontal, sprint com salto, strafes laterais e recuo).
