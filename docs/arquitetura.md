# Arquitetura

Duas coisas distintas se chamam "arquitetura" neste projeto, e confundi-las
custa tempo:

1. **A arquitetura de controle** — como cérebro, habilidades e planejador se
   encaixam em tempo de execução.
2. **A arquitetura do modelo** — o VLA multimodal em si.

---

## 1. Arquitetura de controle: dois laços concorrentes

```
CÉREBRO          escolhe O QUÊ e POR QUÊ     ~1 Hz, assíncrono
   │             "procurar madeira"
   │  escreve
   ▼
INTENÇÃO         alvo + porquê + revisões    estado compartilhado
   │  lê (nunca espera)
   ▼
HABILIDADE       gerencia QUANDO DESISTIR    por invocação
   │             critério de sucesso próprio
   ▼
PLANEJADOR       resolve O CAMINHO           4 Hz
                 BFS 16 blocos sobre voxels
```

O laço de reflexo roda a 4 Hz e **nunca para**. O cérebro roda ao lado, lê um
resumo em texto, e escreve numa `Intenção` compartilhada que a habilidade
consulta sem bloquear.

**O cérebro está fora do caminho crítico.** Ele nunca é chamado de dentro do
laço de controle. Um cérebro que demore 2 segundos não congela o agente — ele
só demora a mudar de ideia. É isso que torna viável colocar um LLM ali sem que
a latência destrua o controle.

### Por que concorrente e não por turnos

Na versão por turnos, o agente para de existir enquanto o cérebro pensa, e só
reavalia quando a habilidade termina sozinha — não há ponto de desistência no
meio. Concorrente, qualquer amostragem do cérebro pode cortar a habilidade em
curso: *"eu ia contornar o lago, mas dali vi uma vila"*.

O mecanismo é o `supervisor` de `habilidades.executar()`, chamado a cada passo;
devolver um texto interrompe.

### Contrato das habilidades

Em `habilidades.py`. Duas coisas distintas de propósito:

- **Predicado** — pergunta barata sobre o estado agora. Sem estado próprio, sem
  duração, nunca falha. Devolve `None` para *desconhecido*, nunca `False`, para
  não mentir para quem decide.
- **Habilidade** — política que corre no tempo, com critério de sucesso próprio,
  e que **pode falhar**.

O contrato que importa: **toda habilidade devolve sucesso OU o motivo da
falha**, com vocabulário fechado:

```
sucesso · travou · bloqueado · orcamento · inalcancavel · morreu ·
precondicao · interrompido
```

Fechado porque é a interface com o cérebro: se cada habilidade inventar sua
string, o LLM não tem como aprender a reagir. E `travou` genérico não aciona
escolha nenhuma — por isso `bloqueado` carrega o obstáculo.

Cada habilidade declara também `custo`, `riscos` e `requer`. Um lago se
atravessa contornando, nadando por cima, nadando fundo ou construindo ponte: as
quatro chegam do outro lado, e o que as separa são esses três campos.

---

## 2. Arquitetura do modelo (`vla_model.py` e `politica_raciocinio.py`)

```
frames (K×3×224×224)              state_vec (32)        objetivo (4)   instrução (16 tokens)
        │                              │                     │            │
   SigLIP (congelado)                  │                     │            │
        │                              │                     │            │
  PerceiverResampler  N→32 tokens      │                     │            │
        │                              │                     │            │
   VisualProjector  →896/1024     StateEncoder          GoalEncoder   Embedding Table
  (32 tokens visuais)              (4 tokens)           (2 tokens)      (16 tokens)
        │                              │                     │            │
        └──────────────┬───────────────┴─────────────────────┴────────────┘
                       ▼
              Sequência Multimodal Concatenada: 52 TOKENS
                       ▼
              Qwen3Loop 0.6B Backbone (LoRA rank=16, alpha=32)
              28 camadas, LoopSplit → 3 Loops Recursivos Latentes
                       │
              last_hidden_state[:, -1, :]   →  896/1024
                       │
         ┌─────────────┼──────────────┬──────────────┐
         ▼             ▼              ▼              ▼
   cabeca_modo(6) cabeca_yaw(9)  cabeca_valor(1)  cabeca_36 (legado)
   [Parado, W,    [-120°..+120°]  Critic V(s)     [Espaço Unificado]
    W+Space, W+A,                 para GAE
    W+D, S]
         │             │              │
         └──────┬──────┘              │
                ▼                     ▼
     Ação Canônica 54D            Alvo GAE G_t
     (Modo x Yaw)
```

### Decisões e seus porquês


**Backbone e visão congelados.** Só treinam resampler, projetor, state_encoder,
goal_encoder e as cabeças — 10,6 M parâmetros. O checkpoint não contém o
backbone; ele é carregado de `Testes/checkpoints/qwen3loop_v2/final_model`.

> **O backbone PRECISA ser carregado explicitamente.** Até 2026-08-13
> `load_vla_agent` fazia `Qwen3LoopModel(config)` — inicialização **aleatória** —
> e nunca substituía os pesos. Todo o VLA rodou sobre ruído. Ver
> [experimentos.md §5](experimentos.md). A config também vem do diretório do
> modelo: a versão codificada à mão omitia `intermediate_size` e pegava o padrão
> 22016 contra os 3072 reais.

**Consequência de VRAM:** os módulos treináveis ficam *antes* do backbone, então
o backward atravessa as 56 execuções de camada do LoopSplit guardando
ativações. Pico medido de 8,81 GB no minilote 4, então ele exige `--vram 0.88`;
com teto em 0,70 (8,4 GB) até o 4 estoura. Minilote 4 é **2× mais rápido** que 2
pelo mesmo tempo por passo; 6 dá OOM.

**Giro discreto, não regressão.** MSE contra alvos ruidosos de média ~0 converge
para 0 — o agente ficava matematicamente incapaz de girar. Classificação sobre
bins não tem esse colapso.

**Bins em unidades de mouse.** O servidor aplica `yaw += dx*0.003` rad. Os bins
`(-262, -116, -58, -17, 0, 17, 58, 116, 262)` correspondem a ±45°, ±20°, ±10°,
±3°, 0. Com bins menores, virar 90° levava 9 passos e travava qualquer professor
com alvo lateral.

**Viés inicial de "não girar".** Uma cabeça aleatória tem argmax **constante**:
na política determinística isso vira o mesmo giro todo passo e o agente anda em
círculo. Pesos zerados e viés 2,0 no bin zero fazem a política começar reta e
*aprender* a girar. É também a inicialização certa para RL.

**Previsão de rotas como tarefa auxiliar.** 12 setores de navegabilidade vindos
do `/rotas`. É o único termo que obriga a via visual a permanecer informativa —
sem ele, treinar com "sempre aperte W" fez o projetor colapsar para um vetor
constante (posto efetivo 1,4 de 1024). Com ele: 3,2 no projetor, 47,2 no hidden,
e a cabeça de rotas fica 60% abaixo do preditor cego em holdout.

**Objetivo relativo e egocêntrico.** `(frente, lado, distância, ângulo)`.
Coordenada de mundo não transfere entre episódios nem entre jogos; "8 blocos à
frente e 3 à esquerda" transfere. Normalizado pelo alcance da fase (8 blocos na
Fase 1), não por 30 — dividir por 30 espremeria tudo em [0,1; 0,27].

**Embedding temporal por frame.** Sem ele os frames entram sem ordem e "agora"
fica indistinguível de "15s atrás", que é justamente a informação que dá noção
de movimento e de travado.

**Todo parâmetro treinável precisa ser salvo, por nome.** A lista de submódulos
escolhida à mão deixava `frame_time_embed` (4096 params) de fora, e ele voltava
aleatório no recarregamento. Hoje o checkpoint guarda
`{n: p for n, p in vla.named_parameters() if p.requires_grad}`.

### O que foi testado e NÃO mudar

**A leitura da última posição.** Uma sonda linear mostrava as rotas mais
legíveis nas posições de imagem (58,5% abaixo do cego) que na última posição
(31,3%), sugerindo desperdício. Reajustando as cabeças sobre cada vetor, a
última posição carrega **menos** geometria e **decide melhor** (39,7% contra
24,7% de recall). Geometria legível e ação inferível são coisas diferentes.

---

## Fluxo de dados em um passo (4 Hz)

```
servidor → observação (frame, estado, rotas)
         → EstadoEpisodio.passo()      atualiza pilha de frames e state_vec
         → política.agir()             uma inferência para os 8 ambientes
         → ação {hold, mouse, duration_ms}
         → servidor
```

`EstadoEpisodio` (`estado_sim.py`) é a **única** definição do `state_vec` de 32
dims, usada tanto para gerar dataset quanto para executar. Duplicá-la em JS
criaria duas verdades que divergiriam em silêncio — e a versão do bug em que o
laço de avaliação não chamava `observar()` derrubou a Fase 1 de 90% para 11%.
