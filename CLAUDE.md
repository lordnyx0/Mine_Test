# Agente Minecraft — Qwen3Loop VLA

> **Comece por [`docs/atual.md`](docs/atual.md)** — estado do treino, o que está pausado e
> como retomar, e por que cada próximo passo.
>
> Referência de profundidade em [`docs/`](docs/README.md): arquitetura,
> planejador, simulador, experimentos, fases e método.

Agente para Minecraft construído sobre o `Qwen3Loop` (Qwen3-0.6B adaptado a
arquitetura em loop, ver [`docs/HANDOFF.md`](docs/HANDOFF.md)). O objetivo de longo prazo é um agente
que aprenda tarefas e transponha para outros jogos.

**Arquitetura decidida:** especialistas motores primeiro, LLM coordenando
depois. Não é MoE — é registro de habilidades com uso de ferramentas.

---

## Comece por aqui

O simulador precisa estar de pé para quase tudo. Ele **não** precisa do
Minecraft aberto: lê os voxels direto dos arquivos de região do save.

```bash
node mineflayer_server/servidor_offline.js      # porta 3002, 8 ambientes
```

Visualizador ao vivo dos 8 ambientes: <http://127.0.0.1:3002/ver>

Use-o. Três hipóteses foram refutadas por medição agregada sem revelar a causa;
o usuário achou um bug de arena em água em segundos, olhando.

---

## Mapa do código

### Simulador (Node, `mineflayer_server/`)

| arquivo | o que é |
|---|---|
| `servidor_offline.js` | HTTP na 3002. `/lote/reset`, `/lote/passo`, `/lote/piloto`, `/lote/estado`, `/lote/frame`, `/ver` |
| `mundo_offline.js` | lê colunas dos `.mca` do save, com cache LRU |
| `bot_offline.js` | física (prismarine-physics), estado, sondas de água e relevo |
| `planejador.js` | **o planejador**: `buscar()` BFS + `Objetivos` + `Piloto` |
| `voxel_renderer.js` | render do frame 640x360 |

### Política e treino (Python)

Organizado pela arquitetura de [`docs/arquitetura.md`](docs/arquitetura.md): o
modelo multimodal, o ambiente que ele treina/avalia contra, a camada de
controle, os laços de treino e avaliação, e a infra que todos compartilham.
Cada pasta é um pacote Python (`__init__.py` vazio) — os imports entre elas são
qualificados (`from infra.gpu_utils import ...`), nunca `import gpu_utils`
solto.

| pasta | arquivo | o que é |
|---|---|---|
| `modelo/` | `vla_model.py` | SigLIP → Resampler → Projetor → (+ State, + Goal) → Qwen3Loop → cabeças |
| `modelo/` | `lora_vla.py` | adaptadores LoRA desacoplados para o Qwen3Loop (expande capacidade sem esquecimento) |
| `modelo/` | `world_model_loss.py` | perda de predição de latente futuro (treina as camadas em loop com física 3D) |
| `modelo/` | `estado_sim.py` | **a** definição do `state_vec` de 32 dims (`EstadoEpisodio`) |
| `ambiente/` | `arena_plana.py` | busca de arenas planas, amostragem de alvo, baselines da Fase 1 |
| `ambiente/` | `fase2.py` | largadas em terreno real, banco de tarefas, baselines da Fase 2 (`geo_pulo`, `aleatorio_pulo`, `piloto`) |
| `ambiente/` | `fase3.py` | ambiente de alvo visual multi-cores com ground-snapping no chão sólido |
| `ambiente/` | `tarefas_logicas.py` | gerador de sequenciamento multi-etapas (Pilar 1 ➔ Pilar 2) com cores e distâncias |
| `politica/` | `politica_fase1.py` | política `W + giro` da Fase 1, com ablação de visão |
| `politica/` | `politica_fase2.py` | política `giro × pulo`, ação conjunta, `ALCANCE_F2 = 30.0` |
| `politica/` | `politica_fase3.py` | política servo-visual pura (sem coordenadas) com tokenização multi-prompt |
| `politica/` | `politica_raciocinio.py` | política de raciocínio multi-loops (K=3) com cabeça de 18 ações e pulo neural |
| `politica/` | `habilidades.py` | contrato de habilidades e predicados (água, lava, chão) |
| `politica/` | `cerebro.py` | `Intencao` compartilhada, `CerebroRegra` e `PoliticaCerebroVLA` hierárquico |
| `treino/` | `treinar_fase1.py` | **RL (REINFORCE)** com recompensa do ambiente |
| `treino/` | `treinar_fase2.py` | RL da Fase 2, piso de `g_std=3.0`, `minilote=12`, telemetria decomposta |
| `treino/` | `treinar_fase3.py` | RL PPO com 2 épocas, ratio clipping, buffer de sucesso e currículo de 80 passos |
| `treino/` | `treinar_fase4_logica.py` | RL PPO da Fase 4 (Lógica Sequencial + Pulo Neural com Regularização de Energia) |
| `avaliacao/` | `avaliar_fase1.py` | avaliação pareada com quebra por setor |
| `avaliacao/` | `avaliar_fase2.py` | avaliação quebrada por grau de obstrução, suporte a `--held-out` e `--amostrar` |
| `avaliacao/` | `avaliar_fase3.py` | avaliação servo-visual multi-cores com ablação de visão e Cérebro |
| `avaliacao/` | `avaliar_fase4_topview.py` | avaliação oficial da Fase 4 com TopView 2D, perfil topológico (Y) e log estruturado |
| `avaliacao/` | `avaliar_logica_testes.py` | benchmark oficial de lógica/raciocínio/matemática (direto ou GGUF) |
| `avaliacao/` | `bench_gguf.py` | executor de benchmark GGUF Q8_0 True Loop (125+ tok/s) via `llama-server` |
| `avaliacao/` | `avaliar_objetivo.py` | avaliação de longo alcance (100 blocos) |
| `avaliacao/` | `avaliar_no_sim.py` | avaliação pareada contra `so_W` e `Piloto`, usada por `sondar_offline.py` |
| `avaliacao/` | `sondar_offline.py` | sondas de representação, offline |
| `scripts/` | `fundir_lora_fase4.py` | fusão permanente dos 112 tensores LoRA no checkpoint consolidado `vla_fase4_merged.pt` |
| `scripts/` | `exportar_fase4_hf.py` | extração dos pesos do backbone Fase 4 para formato Hugging Face |

### Pacotes e Módulos do Modelo & Avaliação

| pasta | o que é |
|---|---|
| `qwen3loop/` | implementação oficial do `Qwen3Loop` (modeling e config com `LoopSplit`) |
| `evaluation/` | harness oficial de avaliação (scorer, render Jinja, tipos e relatórios) |
| `benchmarks/` | dataset oficial de validação (`eval_benchmark.json`) |
| `models_gguf/` | modelos quantizados em GGUF True Loop (`fase4_loop_q8_0.gguf`) |

### Infra compartilhada (Python, `infra/`)

| arquivo | o que é |
|---|---|
| `dataset_embodied.py` | gravador e exportador de dataset multimodal para co-treino do Qwen3Loop v2 |
| `verificar_pesos.py` | auditoria de integridade e comparação pareada de tensores entre checkpoints |
| `run_vla_agent.py` | carrega o backbone (`load_vla_agent`) — usado por todo treino/avaliação |
| `gpu_utils.py` | trava de GPU, limite de VRAM, `compactar_backbone` |
| `bot_vision_capture.py`, `position_reward_evaluator.py`, `trajectory_logger.py` | usados por `run_vla_agent.py` |
| `train_vla.py` | tokenizer (`get_tokenizer`) — dependência de `sondar_offline.py` e `avaliar_no_sim.py` |
| `gate_retrospecto.py` | `objetivo_relativo()` — dependência de `avaliar_objetivo.py` para o ponto "VLA por retrospecto" da tabela de números estabelecidos |

### Organização da raiz

Só o pipeline ativo (tabelas acima) fica solto na raiz, organizado em pacotes.
O resto tem pasta:

| pasta | conteúdo |
|---|---|
| `legado/` | fases e protótipos superados — ver [`legado/LEIA-ME.md`](legado/LEIA-ME.md) antes de reviver algo daqui |
| `imagens_diagnostico/` | frames e amostras salvas por scripts de diagnóstico |
| `logs/` | saída redirecionada de corridas passadas |
| `scripts/` | lançadores e utilitários de exportação e rede |

---

## O planejador, em uma tela

Código puro, sem rede neural. `mineflayer_server/planejador.js`.

1. Lê voxels num raio de 16 ou 40 blocos.
2. `pisavel(x,y,z)`: pés vazios, cabeça vazia, chão sólido, nada perigoso.
3. BFS até 9000 células, maximizando `objetivo.pontuar(no)`.
4. `Piloto` converte caminho em teclas: mira ~6 blocos à frente, gira, `W`,
   `SPACE` em degrau, replaneja ao chegar ou travar.

```js
Objetivos.explorar(origem, memoria, λ)   // d(origem) − λ·visitas
Objetivos.ponto(alvo)                    // −d(alvo)              GULOSO
Objetivos.rumo(alvoFinal, memoria, λ)    // −d(alvo) − λ·visitas  (não ajuda, ver abaixo)
Objetivos.bloco(nomes, registry)         // achou ? 1e6−custo : −custo
```

**Limitação estrutural:** água e abismo não têm chão sólido, então **não
existem no grafo**. Um lago torna o outro lado inalcançável. E ele precisa de
acesso a voxels — por isso não transpõe para outro jogo.

---

## Números estabelecidos — não remedir

### Locomoção de longo alcance (~100 blocos, 400 passos, alvos pareados)

| política | chegada | travado |
|---|---|---|
| `reto` (aponta e anda) | 52-55% | 42% |
| `piloto` (`Objetivos.ponto`) | 80-90% | 12-15% |
| VLA clonado do piloto | 5% | 77% |
| VLA por retrospecto, com `W` forçado | 0% | 35% |

### Fase 1 — alvo local 3-8 blocos, terreno plano, `W + giro`, **14 passos**

| política | chegada |
|---|---|
| `so_W` (anda reto) | 3,1% |
| `aleatorio` (vaga sem ler a entrada) | **7,8%** — o piso certo |
| `geometrico` (trigonometria) | **85,9%** — o teto |

Com orçamento de 40 passos, `aleatorio` marcava ~90% e a primeira corrida de RL
aprendeu exatamente isso: distribuição de giro **idêntica em 13 ângulos de
alvo**, 14,1% na avaliação. O orçamento faz parte da métrica.

A sonda decidiu o caso: **R² 0,978/0,962 e erro angular mediano de 3 graus** —
o ângulo do alvo ESTÁ no 1024 e é linearmente legível. Caso B: problema de
otimização da política, não de representação.

### Representação

- Colapso visual **resolvido** pelo alvo auxiliar de rota: posto efetivo passou
  de 1,4/1024 para 3,2 no projetor e 47,2 no hidden.
- `route_head` fica **60% abaixo** do preditor cego em holdout.
- A representação **não** é o gargalo.

---

## O que já foi refutado — não repita

| hipótese | teste | resultado |
|---|---|---|
| clonar a ação do Piloto | inferibilidade com split por episódio | 19% contra 11% de acaso |
| rotulagem em retrospecto salva a clonagem | treino 4 épocas + avaliação | 5% contra 52% do baseline trivial |
| o pooling é o gargalo | reajuste das cabeças sobre 3 vetores | última posição é a MELHOR |
| memória de visitas solta do mínimo local | `Objetivos.rumo` | empata com o guloso |
| água bloqueia a locomoção | grupo de controle | 33% das falhas, 62% dos sucessos |
| elevação bloqueia | Δy alvo vs bot | sinal invertido |

**Três tentativas independentes de destilar controle por casamento de ação
falharam.** O que funcionou foi RL com recompensa do ambiente, sem professor.

---

## Armadilhas que já custaram caro

**O backbone PRECISA ser carregado.** `load_vla_agent` fazia
`Qwen3LoopModel(config)` — aleatório — e nunca substituía os pesos. O projeto
inteiro rodou sobre ruído até 2026-08-13. O sintoma: treino reportando 88-100% e
avaliação 5-14%, porque `treinar_fase1.py` semeia o torch e sorteava sempre o
mesmo backbone aleatório, enquanto scripts sem semente sorteavam outro. Quatro
diagnósticos meus caíram antes de achar. Ver `docs/experimentos.md` §5.

**Config duplicada à mão diverge.** A config do backbone era codificada no
`load_vla_agent` e omitia `intermediate_size`: padrão 22016 contra 3072 reais.
Hoje vem de `Qwen3LoopConfig.from_pretrained(BACKBONE_DIR)`.

**Salve todo parâmetro treinável, por nome.** A lista de submódulos deixava
`frame_time_embed` de fora e ele voltava aleatório. Use
`{n: p for n, p in vla.named_parameters() if p.requires_grad}`.

**O teto também precisa ser validado.** O raio do BFS tem que cobrir a distância
do alvo: com raio 16 contra alvos a 30 blocos o "planejador" faz subida de
encosta e o teto mede 36% quando o real é 60%.

**Toda ação sem preço vira constante.** Aconteceu 4×: `W` sempre (colapsou a
visão), passeio aleatório (gamejou o orçamento), `SPACE` sempre (matou o giro —
`airborneAcceleration` 0,02 contra 0,10 no chão, 5× menos direção no ar).

São TRÊS causas com consertos opostos: (a) ação sem gradiente → tirar do espaço;
(b) custo além do horizonte de crédito → **precificar**; (c) ótimo degenerado da
tarefa → mudar a **tarefa**. A varredura de ângulo mostra o sintoma e não
distingue a causa. Ver `docs/metodo.md` §2.

**Métrica saturada.** "Distância do respawn" satura no planejador (42,6 contra
15,6). Sempre medir contra piso E teto.

**Split que vaza.** Amostras vizinhas do mesmo episódio de 70 passos são quase
idênticas. Split por blocos deu 39,7%; **por episódio, 19%**. Sempre separar
por episódio.

**Duas verdades entre treino e execução.** O laço de avaliação precisa chamar
`politica.observar()` a cada passo, senão a pilha de frames e o `state_vec`
congelam no instante inicial. Isso derrubou a Fase 1 de 90% para 11% até ser
achado. Reuse o mesmo laço; não escreva um segundo.

**Água finge ser terreno plano.** Água não é bloco sólido, então o perfil de
relevo atravessa e mede o leito — fundo de lago raso é perfeitamente plano.
Sempre checar `in_water` e `aguaPerto()` junto da planura.

**Minilote 4, com teto de VRAM em 0,88.** Os módulos treináveis ficam ANTES do
backbone, então o backward atravessa as 56 execuções de camada do LoopSplit
guardando ativações — pico medido de 8,81 GB. Minilote 4 é **2× mais rápido**
que 2 pelo mesmo tempo por passo; 6 dá OOM. Se o teto estiver em 0,70 (8,4 GB),
até o 4 estoura — foi o que me fez usar 2 por engano.

**Piso do mesmo tipo da política.** `so_W` é determinístico e marca 3,1%;
parecia piso seguro. Mas política estocástica que ignora a entrada **vaga**, e
vagar encontra alvo próximo. O piso certo é `aleatorio`. Custou 4h de treino e
uma conclusão errada de "sucesso".

**Frames de episódio curto.** `ATRASOS_PASSOS=(0,16,60)` são 0s, 4s e 15s. Em
episódio de 14 passos, `pilha_frames` faz clamp e os frames 2 e 3 viram o MESMO
frame inicial — dois terços da visão num duplicado. Use `ATRASOS_SIM=0,2,4`.

**Colapso de entropia no RL.** Com `lr=1e-4` e bônus de entropia 0,01, a
entropia vai a zero na 5ª iteração e a política vira determinística. Use
`lr=3e-5` com recozimento de 0,05 → 0,01.

**Trava de GPU.** `travar_gpu()` cria `.gpu_em_uso.lock`. Carregue o modelo UMA
vez por processo; a segunda chamada aborta. Se um processo morrer sujo, apague
o lock à mão.

**Não use pipe em corrida longa.** `cmd | grep | tail` bufferiza até o fim e
você fica cego por horas. Redirecione para arquivo.

---

## Fluxo da Fase 1

```bash
node mineflayer_server/servidor_offline.js        # 1. simulador

python ambiente/arena_plana.py --arenas 90        # 2. arenas planas (acumula)
python avaliacao/avaliar_fase1.py --episodios 64 \
       --politicas so_W,geometrico                # 3. baselines OBRIGATÓRIOS

python treino/treinar_fase1.py --iteracoes 120 \
       --minilote 2 --vram 0.75                   # 4. RL, ~4h

python avaliacao/avaliar_fase1.py --episodios 64 \
       --politicas geometrico,modelo,modelo_cego  # 5. avaliação + ABLAÇÃO
```

O passo 3 vem antes do 4 por regra: sem régua, "o modelo ajuda?" é opinião.

O passo 5 inclui `modelo_cego` (pixels zerados) porque terreno plano com alvo
em coordenadas é resolvível por trigonometria. **Se cego e vidente empatam, a
visão não contribuiu**, por mais alta que seja a taxa.

E rode a **varredura de ângulo**: cena e distância fixas, ângulo do alvo de
−180° a +180°. Se a distribuição de giro não muda, a política ignora o objetivo
— e é possível marcar alto sem ter aprendido nada.

---

## Convenções

- Comentários em português, explicando **por quê**, não o quê. Números medidos
  no comentário quando justificam a escolha.
- Giro em **unidades de mouse**, não graus: `yaw += dx*0.003` rad.
  `GRAUS_POR_UNIDADE = 0.003*180/π`. O sinal é **invertido** — mouse positivo
  afasta do alvo. Medido em malha fechada, não deduzido.
- `estado_sim.EstadoEpisodio` é a **única** definição do `state_vec` de 32 dims.
  Duplicá-la em JS criaria duas verdades que divergem em silêncio.
- Checkpoints em `checkpoints_vla/`, teto de 5. Nunca sobrescrever
  `vla_locomotion.pt` (a base) nem `BASE_locomocao_limpa.pt`.

---

## Estado atual

Fase 2 em construção — ver `docs/fase2.md`. Baselines calibrados: piso
`geo_pulo` 33% e teto `piloto` (raio 40) 60% na faixa obstruída, 27 pontos de
espaço. Falta gerar o banco e treinar.

Fase 1 fechada. O RL funciona, mas a tarefa era resolvível por `atan2` sem olhar
pixel — a ponte multimodal segue **sem teste conclusivo**. Com o backbone real
contra o aleatório: 83% contra 76% em 100 iterações, ou seja, ganho de
velocidade de aprendizado e não de teto.

### Histórico da Fase 1

A primeira corrida aprendeu a trapacear: 90% no treino, 14,1% na avaliação, com
distribuição de giro constante em todos os ângulos — passeio aleatório
disfarçado. A sonda mostrou **caso B** (informação presente no 1024, política
não usa), então o conserto foi de tarefa e não de modelo: orçamento de 40 → 14
passos, e `aleatorio` como piso permanente.

### Currículo

```
Fase 1   ±8 blocos, plano, W + giro                       FEITA
Fase 2   terreno real + SPACE + 14-30 blocos              em construção
Fase 3   alvo VISUAL: sem coordenada, "vá até a madeira"  proposta
Fase 4   SHIFT (velocidade variável)
Fase 5   A / D / S
```

A Fase 2 fundiu três fases do plano original — `SPACE`, distância maior e
obstáculos — porque são a mesma condição: obstáculo só existe se houver
distância para o desvio caber, e degrau de 1 bloco só é transponível com pulo.

A Fase 3 é o teste mais forte da ponte multimodal, porque **remove o canal de
coordenada**: com obstáculos a visão precisa ser suficiente e o `atan2` ainda dá
33% de graça; sem coordenada ela precisa ser necessária, e o piso vai a ~0%.
