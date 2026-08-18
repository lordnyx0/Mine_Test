# Legado

Código de fases e protótipos superados pelo pipeline atual (ver
[`../CLAUDE.md`](../CLAUDE.md) e [`../docs/atual.md`](../docs/atual.md)).
Mantido como referência histórica — não faz parte do fluxo ativo e não é
mantido.

Nenhum arquivo em `legado/` é importado pelo pipeline ativo. O inverso não é
verdade: vários scripts aqui importam módulos do pipeline ativo por nome solto
(`from run_vla_agent import ...`, `from train_vla import ...`, `from
gpu_utils import ...`, `from estado_sim import ...`, `from vla_model import
...`, `from bot_vision_capture import ...`, `from position_reward_evaluator
import ...`, `from avaliar_no_sim import ...`, `from politica_fase1 import
...`, `from arena_plana import ...`). Desde a reorganização em pacotes
(`modelo/`, `ambiente/`, `politica/`, `treino/`, `avaliacao/`, `infra/`), esse
import solto não resolve mais — o módulo mudou de nome de pacote, não só de
pasta. Para reviver um script daqui, troque o import pelo caminho qualificado
(ex.: `from infra.run_vla_agent import load_vla_agent`) e rode com a raiz do
projeto no `PYTHONPATH`:

```bash
PYTHONPATH="$(pwd)/.." python legado/<subpasta>/<script>.py
```

## Subpastas

| pasta | o que é | por que foi superada |
|---|---|---|
| `prototipo_tempo_real/` | primeira versão do agente: lê tela do Minecraft ao vivo (`vision_encoder.py`, `qwen_vision.py`), injeta teclado (`direct_input.py`), lê chat (`chat_reader.py`), memória (`memory_system.py`), tudo orquestrado por `trainer_interface.py` + `run_agent.bat` | substituído pelo simulador offline (`mineflayer_server/servidor_offline.js`), que lê voxels direto do save sem precisar do jogo aberto |
| `fase_exploracao/` | ambiente de exploração livre com recompensa de cobertura (`exploration_env.py`, `treinar_exploracao.py`) e geração/sondagem de dataset ao vivo (`gerar_dataset_real.py`, `sondar_representacao.py`) | fase anterior à Fase 1/2 atuais; sondagem foi refeita em `sondar_offline.py`, que já é o padrão ativo |
| `imitacao_retrospecto/` | clonagem de ação do planejador com rotulagem em retrospecto (`treinar_objetivo.py`, `treinar_base_real.py`) e geração de dataset de locomoção (`generate_locomotion_dataset.py`, `gerar_dataset_sim.py`) | refutado por medição — ver tabela "O que já foi refutado" em `../CLAUDE.md`: 3 tentativas de clonagem falharam (5% contra 52% do baseline trivial). O que funcionou foi RL com recompensa do ambiente (`../treino/treinar_fase1.py`, `../treino/treinar_fase2.py`). `gate_retrospecto.py` (a função `objetivo_relativo`) ficou em `../infra/` porque `../avaliacao/avaliar_objetivo.py` ainda depende dela para reproduzir o ponto "VLA por retrospecto" da tabela de números estabelecidos |
| `diagnosticos_pontuais/` | scripts de diagnóstico de uma via só, já respondidos: teste de pooling (`testar_pooling.py`, ver [[vla-minecraft-representacao-ok]]), overfit de sanidade (`overfit_single_sample.py`, `overfit_small_dataset.py`, `sanity_check_vla.py`), auto-imitação (`retrain_self_imitation.py`, `diagnosticar_imitacao.py`), experimento de locomoção antigo (`experimento_locomocao.py`), avaliador de recompensa duplicado (`reward_evaluator.py` — a versão em uso é `../position_reward_evaluator.py`), teste avulso de CLIP (`scratch_test_clip.py`) | pergunta já respondida e documentada em `../docs/experimentos.md`; mantidos apenas como registro do teste |
| `servidor_llm_gguf/` | rota abandonada de servir o modelo via `llama.cpp`/GGUF (`fase1_loop_q8_0.gguf`, `start_server.bat`, `scratch_patch_gguf.py`, `PLANO_GGUF_FASE2.md`, `eval_config.yaml` órfão de um `run_evaluation.py` que não existe mais neste projeto) | `PLANO_GGUF_FASE2.md` documenta o veredito: GGUF não representa pesos compartilhados entre camadas do loop sem duplicá-los (56 camadas físicas), anulando o ganho de VRAM — não é adaptação de script, seria um port inteiro do `GatedDeltaNetModule` |
