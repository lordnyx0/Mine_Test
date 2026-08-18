# coding=utf-8
"""
Camada de Cérebro e Coordenação de Habilidades.

Conforme docs/arquitetura.md §1:
  "O cérebro está FORA do caminho crítico. Ele nunca é chamado de dentro do
   laço de controle (4 Hz). Um cérebro que demore 2 segundos não congela o
   agente — ele só demora a mudar de ideia. É isso que torna viável colocar
   um LLM ali sem que a latência destrua o controle."

Componentes:
  - Intencao: estado compartilhado assíncrono entre cérebro e habilidades.
  - CerebroBase / CerebroRegra: orquestrador baseline analítico antes de LLMs.
  - MotorConcorrente: orquestrador do loop de reflexo com supervisão não-bloqueante.
  - PoliticaCerebroVLA: acoplador hierárquico (Cérebro + VLA Motor) com desengate de colisão e varredura.
"""
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List

from politica.habilidades import (Habilidade, Navegar, Registro, Resultado,
                                 Predicados, SUCESSO, INTERROMPIDO)


@dataclass
class Intencao:
    """Estado compartilhado que o cérebro escreve e a habilidade consulta."""
    alvo: Any = None
    motivo: str = ""
    revisoes: int = 0
    criado_em: float = field(default_factory=time.time)
    atualizado_em: float = field(default_factory=time.time)
    cancelar: bool = False

    def atualizar(self, novo_alvo, motivo: str):
        self.alvo = novo_alvo
        self.motivo = motivo
        self.revisoes += 1
        self.atualizado_em = time.time()
        self.cancelar = False

    def pedir_cancelamento(self, motivo: str = "cancelado pelo cerebro"):
        self.motivo = motivo
        self.cancelar = True
        self.atualizado_em = time.time()


class CerebroBase:
    """Base para orquestradores de alto nível."""
    def decidir(self, obs: dict, intencao_atual: Intencao, reg: Registro) -> Optional[dict]:
        """Devolve dict com {'habilidade': str, 'args': dict, 'motivo': str} ou None."""
        raise NotImplementedError


class CerebroRegra(CerebroBase):
    """Baseline clássico de regras. O número a bater antes de LLMs entrarem."""
    
    def __init__(self, lista_alvos: Optional[List[tuple]] = None):
        self.lista_alvos = list(lista_alvos or [])
        self.indice = 0

    def decidir(self, obs: dict, intencao_atual: Intencao, reg: Registro) -> Optional[dict]:
        # Regra de segurança: se vida baixa ou caindo, interrompe
        if Predicados.vida(obs) <= 4:
            return {"habilidade": None, "motivo": "vida criticamente baixa"}
        
        # Se ainda há alvos na fila e a intenção atual terminou ou não existe
        if self.indice < len(self.lista_alvos):
            alvo = self.lista_alvos[self.indice]
            self.indice += 1
            return {
                "habilidade": "navegar",
                "args": {"x": alvo[0], "z": alvo[1]},
                "motivo": "seguir para próximo waypoint (%d/%d)" % (self.indice, len(self.lista_alvos))
            }
        return None


class MotorConcorrente:
    """Executa o laço de controle motor a 4 Hz enquanto o cérebro opera em paralelo."""

    def __init__(self, cerebro: CerebroBase, registro: Registro, passo_env: Callable):
        self.cerebro = cerebro
        self.registro = registro
        self.passo_env = passo_env
        self.intencao = Intencao()
        self._rodando = False
        self._lock = threading.Lock()
        self._obs_mais_recente = None

    def supervisor(self, obs, hab: Habilidade) -> Optional[str]:
        """Chamado a cada passo do laço de controle motor (4 Hz)."""
        with self._lock:
            self._obs_mais_recente = obs
            if self.intencao.cancelar:
                return self.intencao.motivo
        return None

    def _loop_cerebro(self, freq_hz: float = 1.0):
        """Thread assíncrona do cérebro (~1 Hz)."""
        intervalo = 1.0 / freq_hz
        while self._rodando:
            time.sleep(intervalo)
            with self._lock:
                obs = self._obs_mais_recente
            if obs is None:
                continue
            decisao = self.cerebro.decidir(obs, self.intencao, self.registro)
            if decisao and decisao.get("habilidade"):
                with self._lock:
                    self.intencao.atualizar(decisao.get("args"), decisao.get("motivo", ""))

    def executar(self, obs_inicial, passos_max: int = 1000):
        """Inicia a execução concorrente."""
        self._rodando = True
        self._obs_mais_recente = obs_inicial
        t_cerebro = threading.Thread(target=self._loop_cerebro, daemon=True)
        t_cerebro.start()

        obs = obs_inicial
        passos_totais = 0
        try:
            while self._rodando and passos_totais < passos_max:
                decisao = self.cerebro.decidir(obs, self.intencao, self.registro)
                if not decisao or not decisao.get("habilidade"):
                    break
                
                nome_hab = decisao["habilidade"]
                args_hab = decisao.get("args", {})
                hab = self.registro.get(nome_hab)
                
                hab.iniciar(obs, **args_hab)
                while True:
                    motivo = self.supervisor(obs, hab)
                    if motivo:
                        r = Resultado(INTERROMPIDO, hab.passos, {"detalhe": motivo})
                        break
                    
                    acao = hab.passo(obs)
                    obs = self.passo_env(acao)
                    passos_totais += 1
                    
                    r = hab.terminou(obs)
                    if r is not None:
                        break
                
                print(f"[Motor] Habilidade '{nome_hab}' finalizou: {r.para_llm()}")
        finally:
            self._rodando = False
            t_cerebro.join(timeout=1.0)


class PoliticaCerebroVLA:
    """Política hierárquica completa: Cérebro supervisor (~1 Hz) + VLA reflexo (4 Hz).

    Responsabilidades do Cérebro:
      - Desengate automático de colisão/travamento (se posição estagnar < 0.15m por 2 passos).
      - Transposição inteligente de degraus (acionamento de SPACE ao detectar subida).
      - Varredura angular ativa pós-submeta para localizar o próximo alvo.
      - Parada imediata ao atingir a meta.
    """
    def __init__(self, vla_pol):
        self.vla = vla_pol
        self.nome = "cerebro_vla"
        self.travado = {}
        self.ultima_pos = {}
        self.em_desengate = {}
        self.em_varredura = {}
        self.chegou = {}

    @property
    def cego(self):
        return getattr(self.vla, "cego", False)

    @cego.setter
    def cego(self, val):
        self.vla.cego = val

    @property
    def prompts_atuais(self):
        return getattr(self.vla, "prompts_atuais", [])

    @prompts_atuais.setter
    def prompts_atuais(self, val):
        self.vla.prompts_atuais = val

    @property
    def ultimo(self):
        return getattr(self.vla, "ultimo", {})

    @property
    def amostrar(self):
        return getattr(self.vla, "amostrar", False)

    @amostrar.setter
    def amostrar(self, val):
        self.vla.amostrar = val

    @property
    def temperatura(self):
        return getattr(self.vla, "temperatura", 0.8)

    @temperatura.setter
    def temperatura(self, val):
        self.vla.temperatura = val

    def normalizar(self, u8):
        return self.vla.normalizar(u8)

    def log_prob(self, px, sv, gv, a_idx, ids=None):
        return self.vla.log_prob(px, sv, gv, a_idx, ids=ids)

    def reiniciar(self, obs):
        self.vla.reiniciar(obs)
        n = len(obs)
        self.travado = {i: 0 for i in range(n)}
        self.ultima_pos = {i: (o["estado"]["x"], o["estado"]["z"]) for i, o in enumerate(obs)}
        self.em_desengate = {i: 0 for i in range(n)}
        self.em_varredura = {i: 0 for i in range(n)}
        self.chegou = {i: False for i in range(n)}

    def observar(self, obs):
        self.vla.observar(obs)

    def ativar_varredura(self, env_idx: int, passos_varredura: int = 3):
        """Comando do Cérebro para girar a câmera em busca do novo pilar."""
        self.em_varredura[env_idx] = passos_varredura

    def agir(self, ests, alvos_abs, obs, prompts=None, estagios=None):
        n = len(ests)
        acoes_vla = self.vla.agir(ests, alvos_abs, obs, prompts=prompts, estagios=estagios)
        acoes_finais = []

        for i in range(n):
            e = ests[i]
            pos_atual = (e["x"], e["z"])
            pos_ant = self.ultima_pos.get(i, pos_atual)
            dist_andou = math.hypot(pos_atual[0] - pos_ant[0], pos_atual[1] - pos_ant[1])
            self.ultima_pos[i] = pos_atual

            # 1. Checagem de parada terminal (se chegou ao raio, solta W)
            if alvos_abs and i < len(alvos_abs) and alvos_abs[i]:
                dist_alvo = math.hypot(alvos_abs[i][0] - e["x"], alvos_abs[i][1] - e["z"])
                if dist_alvo <= 2.0:
                    self.chegou[i] = True
                    acoes_finais.append({"hold": [], "mouse": [0, 0], "duration_ms": 250})
                    continue

            # 2. Comando de Varredura Visual pós-submeta (Cérebro reorienta câmera)
            if self.em_varredura.get(i, 0) > 0:
                self.em_varredura[i] -= 1
                acoes_finais.append({"hold": ["W"], "mouse": [58, 0], "duration_ms": 250})
                continue

            # 3. Gestão de manobra de desengate ativa (Cérebro desvia de obstáculo)
            if self.em_desengate.get(i, 0) > 0:
                self.em_desengate[i] -= 1
                acoes_finais.append({"hold": ["W", "SPACE"], "mouse": [116, 0], "duration_ms": 250})
                continue

            # 4. Detector de colisão/travamento (se andou menos de 0.15m com intenção de avanço)
            if dist_andou < 0.15:
                self.travado[i] = self.travado.get(i, 0) + 1
            else:
                self.travado[i] = 0

            # Dispara manobra de evasão se travou por 2 passos seguidos
            if self.travado[i] >= 2:
                self.em_desengate[i] = 2
                self.travado[i] = 0
                acoes_finais.append({"hold": ["W", "SPACE"], "mouse": [-116, 0], "duration_ms": 250})
                continue

            # 5. Caso normal: executa o comando refinado do VLA com pulo automático em degraus
            acao = dict(acoes_vla[i])
            if dist_andou < 0.35 and e.get("on_ground") and not e.get("in_water"):
                acao["hold"] = ["W", "SPACE"]
            acoes_finais.append(acao)

        return acoes_finais
