# coding=utf-8
"""
Registro de HABILIDADES motoras — as primitivas que o LLM vai coordenar.

Duas coisas distintas, de proposito:

  PREDICADO   — pergunta barata sobre o estado agora. Sem estado proprio, sem
                duracao, nao falha. "estou na agua?", "tem bloco a frente?"
  HABILIDADE  — politica que corre no tempo, tem criterio de sucesso proprio e
                PODE FALHAR. "va ate (x,z)", "colete madeira".

Misturar os dois foi o que fez cada politica reimplementar as mesmas perguntas.
Predicado dispara habilidade; habilidade nao pergunta duas vezes a mesma coisa.

O contrato que importa: **toda habilidade devolve sucesso OU o motivo da
falha.** Navegar acerta 80% a 100 blocos (medido, n=40, contra 55% de andar
reto). Um cerebro que assumir 100% vai montar planos que quebram em silencio.
Por isso Resultado carrega `motivo`, e o vocabulario de motivos e fechado — e
o que o LLM vai ler para decidir se tenta de novo, se contorna ou se desiste.

Este modulo NAO decide nada. Ele expoe primitivas e as descreve; a escolha e
de quem orquestra.
"""
import math
import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

BASE = "http://127.0.0.1:3002"
GRAUS_POR_UNIDADE = 0.003 * 180 / math.pi
YAW_BINS = (-262, -116, -58, -17, 0, 17, 58, 116, 262)


def post(caminho, corpo, timeout=300):
    d = json.dumps(corpo).encode()
    r = urllib.request.Request(BASE + caminho, data=d, method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read())


# ── Resultado ─────────────────────────────────────────────────────────────────
# Vocabulario FECHADO de motivos. Fechado porque e a interface com o cerebro:
# se cada habilidade inventar sua string, o LLM nao consegue aprender a reagir.
SUCESSO = "sucesso"
TRAVOU = "travou"              # parou de progredir e nao recuperou
ORCAMENTO = "orcamento"        # progredia, acabaram os passos
INALCANCAVEL = "inalcancavel"  # a busca nao acha caminho nenhum
BLOQUEADO = "bloqueado"        # travou E sabemos POR QUE (agua, parede, abismo)
MORREU = "morreu"
PRECONDICAO = "precondicao"    # nem comecou: predicado de entrada falso
INTERROMPIDO = "interrompido"  # o supervisor cortou no meio

# Medido: das 4 falhas de navegar em 40 alvos, 4 de 4 ESTACIONARAM e 0 foram
# orcamento. Uma delas com o alvo a 5,3 blocos. Ou seja, "travou" e o motivo
# real, e um `travou` generico nao aciona escolha nenhuma — por isso BLOQUEADO
# carrega o obstaculo. E o que faz o cerebro decidir entre contornar, nadar
# ou construir ponte em vez de tentar de novo o mesmo.


@dataclass
class Resultado:
    status: str                       # SUCESSO ou um dos motivos de falha
    passos: int = 0
    dados: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == SUCESSO

    def para_llm(self):
        """Uma linha que o cerebro le. Sem jargao de implementacao."""
        if self.ok:
            return "ok (%d passos)" % self.passos
        return "falhou: %s (%d passos)%s" % (
            self.status, self.passos,
            (" — " + self.dados["detalhe"]) if "detalhe" in self.dados else "")


# ── Predicados ────────────────────────────────────────────────────────────────
class Predicados:
    """Perguntas sobre a observacao. Baratas, sem estado, nunca falham.

    `in_water` e `in_lava` sao expostos diretamente pelo estado do bot no simulador.
    """

    @staticmethod
    def na_agua(obs) -> bool:
        return bool(obs.get("estado", {}).get("in_water", False))

    @staticmethod
    def na_lava(obs) -> bool:
        return bool(obs.get("estado", {}).get("in_lava", False))

    @staticmethod
    def no_chao(obs) -> bool:
        return bool(obs.get("estado", {}).get("on_ground", False))

    @staticmethod
    def vida(obs) -> float:
        return float(obs.get("estado", {}).get("health", 20))

    @staticmethod
    def caindo(obs) -> bool:
        return float(obs.get("estado", {}).get("vy", 0)) < -0.5

    @staticmethod
    def frente_livre(obs) -> Optional[bool]:
        """Setor frontal do mapa de rotas. None se o servidor nao mandou rotas."""
        r = obs.get("rotas")
        if not r:
            return None
        return r[0] > 0.5

    @staticmethod
    def descreva(obs) -> str:
        """Estado em linguagem, para o cerebro ler junto com o resultado."""
        e = obs.get("estado", {})
        partes = ["y=%.0f" % e.get("y", 0), "vida=%.0f" % e.get("health", 20)]
        if not Predicados.no_chao(obs):
            partes.append("no ar")
        if Predicados.na_agua(obs):
            partes.append("na agua")
        if Predicados.na_lava(obs):
            partes.append("na lava")
        return ", ".join(partes)


# ── Habilidade ────────────────────────────────────────────────────────────────
class Habilidade:
    """Base. Uma habilidade e uma politica com criterio de parada proprio.

    Passo a passo em vez de bloqueante porque o cerebro precisa poder
    interromper: uma habilidade que so devolve no fim nao da como reagir a
    um mob que apareceu no meio da travessia.
    """
    nome = "abstrata"
    descricao = ""
    argumentos: Dict[str, str] = {}

    # Um lago se atravessa contornando, nadando por cima, nadando fundo ou
    # construindo ponte. As quatro CHEGAM do outro lado — o que as separa e
    # custo, risco e pre-requisito. Sem esses tres campos o cerebro escolheria
    # no escuro, e "prefira nadar fundo quando tiver folego e pressa" nao teria
    # como ser expresso.
    custo = "medio"          # baixo | medio | alto | variavel
    riscos: List[str] = []   # ex.: ["afogamento"], ["queda"], []
    requer: List[str] = []   # pre-requisitos materiais, ex.: ["blocos"]

    def aplicavel(self, obs) -> bool:
        return True

    def iniciar(self, obs, **kw):
        self.passos = 0

    def passo(self, obs) -> dict:
        raise NotImplementedError

    def terminou(self, obs) -> Optional[Resultado]:
        raise NotImplementedError

    def esquema(self) -> dict:
        """Descricao que vira ferramenta para o LLM."""
        return {"nome": self.nome, "descricao": self.descricao,
                "argumentos": self.argumentos, "custo": self.custo,
                "riscos": self.riscos, "requer": self.requer}


class Navegar(Habilidade):
    """Ir ate um ponto (x, z). A primeira habilidade, e a unica medida.

    Delega o controle ao planejador do servidor (`Objetivos.ponto`), que a 100
    blocos chega em 80% contra 55% de apontar-e-andar. `rumo` (o mesmo com
    memoria de visitas) empatou em 80% — a memoria nao muda o argmax da busca,
    entao nao vale o parametro extra ate haver evidencia.

    O criterio de falha distingue TRAVOU de ORCAMENTO porque os consertos sao
    opostos: travar e problema de planejamento, orcamento e de velocidade.
    """
    nome = "navegar"
    descricao = ("Vai ate a coordenada (x, z) por terra. Nao atravessa agua nem "
                 "abismo: a busca so anda sobre chao pisavel. Acerta ~80-90% a "
                 "80-100 blocos; quando falha e por bloqueio, nao por lentidao.")
    argumentos = {"x": "coordenada X do destino", "z": "coordenada Z do destino",
                  "raio_chegada": "distancia que conta como chegou (padrao 2.5)"}
    custo = "variavel"
    riscos = ["queda"]
    requer = []

    def __init__(self, env=0, raio_busca=16):
        self.env = env
        self.raio_busca = raio_busca

    def iniciar(self, obs, x=None, z=None, raio_chegada=2.5,
                paciencia=60, orcamento=400):
        self.alvo = (float(x), float(z))
        self.raio_chegada = raio_chegada
        self.paciencia = paciencia        # passos sem recorde antes de desistir
        self.orcamento = orcamento
        self.passos = 0
        self.dmin = self._dist(obs)
        self.sem_recorde = 0

    def _dist(self, obs):
        e = obs["estado"]
        return math.hypot(self.alvo[0] - e["x"], self.alvo[1] - e["z"])

    def passo(self, obs) -> dict:
        e = obs["estado"]
        extras = [None] * (self.env + 1)
        extras[self.env] = {"alvo": {"x": self.alvo[0], "y": e.get("y", 64),
                                     "z": self.alvo[1]}, "raio": self.raio_busca}
        r = post("/lote/piloto", {"objetivo": "ponto", "extras": extras})
        self.passos += 1
        return r["acoes"][self.env]

    def terminou(self, obs) -> Optional[Resultado]:
        if obs.get("morreu"):
            return Resultado(MORREU, self.passos)
        d = self._dist(obs)
        if d < self.dmin - 0.5:
            self.dmin = d
            self.sem_recorde = 0
        else:
            self.sem_recorde += 1
        if d <= self.raio_chegada:
            return Resultado(SUCESSO, self.passos, {"distancia_final": round(d, 1)})
        # TRAVOU antes de ORCAMENTO: se parou de progredir ha muito tempo, o
        # motivo verdadeiro e travamento, mesmo que o orcamento acabe junto.
        if self.sem_recorde >= self.paciencia:
            return Resultado(TRAVOU, self.passos,
                             {"detalhe": "sem progresso ha %d passos, a %.0f blocos do alvo"
                                         % (self.sem_recorde, d)})
        if self.passos >= self.orcamento:
            return Resultado(ORCAMENTO, self.passos,
                             {"detalhe": "parou a %.0f blocos do alvo, ainda progredindo" % d})
        return None


# ── Registro e execucao ───────────────────────────────────────────────────────
class Registro:
    """As habilidades disponiveis. E daqui que sai a lista de ferramentas
    que o LLM recebe."""

    def __init__(self):
        self._h: Dict[str, Habilidade] = {}

    def registrar(self, h: Habilidade):
        self._h[h.nome] = h
        return self

    def get(self, nome) -> Habilidade:
        return self._h[nome]

    def esquemas(self) -> List[dict]:
        return [h.esquema() for h in self._h.values()]

    def para_prompt(self) -> str:
        linhas = []
        for h in self._h.values():
            args = ", ".join("%s: %s" % (k, v) for k, v in h.argumentos.items())
            linhas.append("- %s(%s) — %s" % (h.nome, args, h.descricao))
        return "\n".join(linhas)


def executar(hab: Habilidade, obs_inicial, passo_env: Callable,
             supervisor: Optional[Callable] = None, **kw) -> Resultado:
    """Roda uma habilidade ate ela terminar.

    passo_env(acao) -> nova observacao. Injetado para o mesmo codigo servir ao
    simulador em lote, ao bot online e a um teste sem servidor nenhum.

    supervisor(obs, hab) -> str | None. Chamado a CADA passo; devolver um texto
    interrompe a habilidade com esse motivo. E o que permite "mudar o percurso
    no meio": o cerebro nao precisa esperar a travessia terminar para desistir
    dela porque apareceu um mob, a vida caiu ou ele mudou de ideia sobre o
    destino. Sem isso, habilidade longa vira ponto cego.
    """
    if not hab.aplicavel(obs_inicial):
        return Resultado(PRECONDICAO, 0)
    hab.iniciar(obs_inicial, **kw)
    obs = obs_inicial
    while True:
        if supervisor is not None:
            motivo = supervisor(obs, hab)
            if motivo:
                return Resultado(INTERROMPIDO, getattr(hab, "passos", 0),
                                 {"detalhe": motivo,
                                  "estado_final": Predicados.descreva(obs)})
        acao = hab.passo(obs)
        obs = passo_env(acao)
        r = hab.terminou(obs)
        if r is not None:
            r.dados["estado_final"] = Predicados.descreva(obs)
            return r


if __name__ == "__main__":
    reg = Registro().registrar(Navegar())
    print("habilidades disponiveis para o cerebro:\n")
    print(reg.para_prompt())
    print("\nesquema JSON:\n", json.dumps(reg.esquemas(), ensure_ascii=False, indent=2))
