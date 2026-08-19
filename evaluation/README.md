# Evaluation Framework

Suite padronizada de benchmarking e avaliação modular para agentes de raciocínio VLA.

## Estrutura de Módulos

* **`runner.py`**: Executor de baterias de testes em lote e coleta assíncrona de resultados.
* **`scoring.py`**: Métricas de pontuação, cálculo de taxas de sucesso, submetas e tempos de conclusão.
* **`types.py`**: Dataclasses e tipos estruturados de configuração, episódios e relatórios.
* **`models.py`**: Interfaces de carregamento de políticas e adaptadores de modelo.
* **`storage.py`**: Serialização e salvamento estruturado dos resultados em JSON/HTML.
* **`generation.py`**: Geradores de cenários de teste e condições de avaliação.
* **`progress.py`**: Barras de progresso e rastreamento de execução.
