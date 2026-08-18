/**
 * planejador.js — busca no grid de voxels com OBJETIVO PLUGAVEL e memoria.
 *
 * Substitui o planejar() embutido no state_server, que tinha tres limitacoes
 * estruturais medidas na pratica:
 *
 *   1. Replanejava do zero a cada passo -> o alvo oscilava entre caminhos
 *      equivalentes e o bot gingava no lugar. Agora ha COMPROMISSO: o alvo
 *      vale por N passos ou ate ser alcancado/invalidado.
 *   2. Maximizava distancia da posicao ATUAL, o que e guloso: ele reexplora
 *      o que ja andou. Agora ha MEMORIA DE VISITAS, e o objetivo de exploracao
 *      penaliza regiao ja pisada — isso e busca de fronteira, nao de raio.
 *   3. O objetivo era fixo ("va longe"). Agora e uma funcao de pontuacao, e
 *      "ache madeira" ou "va ate (x,z)" entram sem tocar na busca.
 *
 * Os tres horizontes que o agente precisa considerar aparecem assim:
 *   curto  — estado atual e colisao imediata (o proprio grid pisavel)
 *   medio  — MemoriaVisitas do episodio, que empurra para fronteira
 *   longo  — MemoriaVisitas persistente entre episodios + alvos conhecidos
 */
'use strict'

const PASSAVEL = new Set([
  'air', 'tallgrass', 'double_plant', 'yellow_flower', 'red_flower', 'deadbush',
  'sapling', 'reeds', 'vine', 'snow_layer', 'torch', 'redstone_torch',
  'rail', 'golden_rail', 'detector_rail', 'activator_rail', 'redstone_wire',
  'wheat', 'carrots', 'potatoes', 'pumpkin_stem', 'melon_stem', 'nether_wart',
  'standing_sign', 'wall_sign', 'lever', 'stone_button', 'wooden_button',
  'stone_pressure_plate', 'wooden_pressure_plate', 'tripwire', 'tripwire_hook',
  'water', 'flowing_water', 'brown_mushroom', 'red_mushroom', 'web', 'fire'
])

// Perigosos: atravessaveis mas nunca desejaveis como destino
const PERIGO = new Set(['lava', 'flowing_lava', 'fire', 'cactus'])

const VIZINHOS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

let _passavelPorId = null
let _perigoPorId = null

function _tabelas (registry) {
  if (_passavelPorId) return
  _passavelPorId = new Uint8Array(256)
  _perigoPorId = new Uint8Array(256)
  for (const b of registry.blocksArray) {
    if (b.id >= 256) continue
    if (PASSAVEL.has(b.name)) _passavelPorId[b.id] = 1
    if (PERIGO.has(b.name)) _perigoPorId[b.id] = 1
  }
  _passavelPorId[0] = 1
}

function _chave (x, y, z) {
  return ((x + 8192) * 33554432) + ((z + 8192) * 256) + (y & 255)
}

// ── Memoria de visitas ────────────────────────────────────────────────────────
class MemoriaVisitas {
  /**
   * @param {number} celula  lado da celula em blocos
   * @param {number} meiaVida  passos ate uma visita valer metade (0 = nunca esquece)
   */
  constructor (celula = 8, meiaVida = 400) {
    this.celula = celula
    this.decaimento = meiaVida > 0 ? Math.pow(0.5, 1 / meiaVida) : 1
    this.mapa = new Map()
  }

  _k (x, z) {
    return Math.floor(x / this.celula) * 100000 + Math.floor(z / this.celula)
  }

  registrar (x, z, peso = 1) {
    const k = this._k(x, z)
    this.mapa.set(k, (this.mapa.get(k) || 0) + peso)
  }

  /** Decai tudo um passo. Chamar uma vez por passo do agente. */
  envelhecer () {
    if (this.decaimento >= 1) return
    for (const [k, v] of this.mapa) {
      const novo = v * this.decaimento
      if (novo < 0.02) this.mapa.delete(k)
      else this.mapa.set(k, novo)
    }
  }

  contagem (x, z) { return this.mapa.get(this._k(x, z)) || 0 }
  limpar () { this.mapa.clear() }
  get tamanho () { return this.mapa.size }
}

// ── Objetivos ─────────────────────────────────────────────────────────────────
/**
 * Um objetivo e {pontuar(no) -> number, parar?(no) -> bool}. A busca maximiza
 * pontuar; se parar() devolver true, encerra ali (alvo encontrado).
 */
const Objetivos = {
  /** Fronteira: longe da origem E pouco visitado. */
  explorar (origem, memoria, lambda = 6.0) {
    return {
      pontuar (no) {
        const d = Math.hypot(no.x - origem.x, no.z - origem.z)
        const visto = memoria ? memoria.contagem(no.x, no.z) : 0
        return d - lambda * visto
      }
    }
  },

  /** Ir ate um ponto especifico. GULOSO: sem memoria. */
  ponto (alvo) {
    const ax = (alvo && alvo.x !== undefined) ? alvo.x : 0
    const az = (alvo && alvo.z !== undefined) ? alvo.z : 0
    return {
      pontuar (no) { return -Math.hypot(no.x - ax, no.z - az) },
      parar (no) { return Math.hypot(no.x - ax, no.z - az) < 1.5 }
    }
  },

  /**
   * Viagem longa ate um alvo final: `ponto` com a memoria de visitas do
   * `explorar`. E a composicao das duas metades que existiam separadas.
   *
   * Por que a memoria: com raio de busca 16 e alvo a 100 blocos, `ponto` e
   * subida de encosta gulosa — toda celula que se afasta do alvo pontua pior,
   * entao qualquer concavidade (lago, ravina, encosta em U) prende o bot, que
   * oscila na borda. Medido: 79% de chegada a 93 blocos, contra 50% de andar
   * reto. Os 21% que faltam sao esse minimo local.
   *
   * A penalidade de visita e o que solta: depois de pisar no beco, aquelas
   * celulas custam mais e a busca prefere o ramo que contorna. lambda e a
   * tolerancia de desvio — 6.0 significa "uma visita vale 6 blocos a mais de
   * caminho". O envelhecer() da memoria evita que um corredor legitimamente
   * cruzado duas vezes fique proibido para sempre.
   */
  rumo (alvoFinal, memoria, lambda = 6.0) {
    return {
      pontuar (no) {
        const d = Math.hypot(no.x - alvoFinal.x, no.z - alvoFinal.z)
        const visto = memoria ? memoria.contagem(no.x, no.z) : 0
        return -d - lambda * visto
      },
      parar (no) { return Math.hypot(no.x - alvoFinal.x, no.z - alvoFinal.z) < 1.5 }
    }
  },

  /**
   * Achar um bloco pelo nome (madeira, pedra...). Pontua pela proximidade ao
   * primeiro encontrado; para assim que encosta nele. E este objetivo que faz
   * a mesma busca servir para "colete madeira" sem reescrever nada.
   */
  bloco (nomes, registry) {
    const ids = new Set()
    for (const b of registry.blocksArray) if (nomes.includes(b.name)) ids.add(b.id)
    return {
      ids,
      procuraBloco: true,
      pontuar (no) { return no.achou ? 1e6 - no.custo : -no.custo }
    }
  }
}

// ── Busca ─────────────────────────────────────────────────────────────────────
/**
 * Busca em largura sobre celulas onde o bot consegue ficar de pe.
 * @returns {{caminho, alvo, custo, nos, achou}}
 */
function buscar (bot, objetivo, opcoes = {}) {
  const raio = opcoes.raio || 40
  const maxNos = opcoes.maxNos || 9000
  _tabelas(bot.registry)

  const p = bot.entity.position
  const ox = Math.floor(p.x), oy = Math.floor(p.y), oz = Math.floor(p.z)

  const colCache = new Map()
  const pisavelCache = new Map()

  const tipoEm = (x, y, z) => {
    if (y < 0 || y > 255) return 0
    const cx = x >> 4, cz = z >> 4
    const ck = cx * 100000 + cz
    let col = colCache.get(ck)
    if (col === undefined) {
      try { col = bot.world.getColumn(cx, cz) || null } catch (_) { col = null }
      colCache.set(ck, col)
    }
    if (col === null || !col.sections) return -1
    const sec = col.sections[y >> 4]
    if (!sec || !sec.data) return 0
    const n = ((x & 15) | ((z & 15) << 4) | ((y & 15) << 8)) << 1
    return (sec.data[n] | (sec.data[n + 1] << 8)) >> 4
  }

  const pisavel = (x, y, z) => {
    const k = _chave(x, y, z)
    const memo = pisavelCache.get(k)
    if (memo !== undefined) return memo
    const pes = tipoEm(x, y, z)
    const cabeca = tipoEm(x, y + 1, z)
    const chao = tipoEm(x, y - 1, z)
    const ok = pes >= 0 && cabeca >= 0 && chao >= 0 &&
      _passavelPorId[pes] === 1 && _passavelPorId[cabeca] === 1 &&
      _passavelPorId[chao] === 0 && _perigoPorId[chao] === 0 &&
      _perigoPorId[pes] === 0
    pisavelCache.set(k, ok)
    return ok
  }

  /** O bloco procurado esta encostado nesta celula? */
  const temAlvo = (x, y, z, ids) => {
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
      [0, 1, 0], [0, -1, 0], [0, 2, 0]]) {
      if (ids.has(tipoEm(x + dx, y + dy, z + dz))) return true
    }
    return false
  }

  const pai = new Map()
  const inicio = _chave(ox, oy, oz)
  pai.set(inicio, null)

  let fila = [{ x: ox, y: oy, z: oz, custo: 0 }]
  let melhor = { no: { x: ox, y: oy, z: oz, custo: 0, achou: false }, pontos: -Infinity }
  let nos = 0
  let achou = false

  busca:
  while (fila.length && nos < maxNos) {
    const proxima = []
    for (const atual of fila) {
      nos++
      if (nos > maxNos) break

      for (const [dx, dz] of VIZINHOS) {
        const nx = atual.x + dx, nz = atual.z + dz
        if (Math.abs(nx - ox) > raio || Math.abs(nz - oz) > raio) continue

        // Diagonal nao pode cortar quina: um corpo de 0.6 de largura nao passa
        // entre dois blocos solidos. Sem isto a busca devolve caminhos que a
        // fisica recusa, e o bot fica empurrando a quina para sempre.
        if (dx !== 0 && dz !== 0) {
          const livreA = pisavel(atual.x + dx, atual.y, atual.z) ||
                         pisavel(atual.x + dx, atual.y + 1, atual.z)
          const livreB = pisavel(atual.x, atual.y, atual.z + dz) ||
                         pisavel(atual.x, atual.y + 1, atual.z + dz)
          if (!livreA || !livreB) continue
        }

        for (const dy of [0, 1, -1]) {
          const ny = atual.y + dy
          const k = _chave(nx, ny, nz)
          if (pai.has(k)) continue
          if (!pisavel(nx, ny, nz)) continue

          pai.set(k, atual)
          // custo maior para subir: pular e mais lento e mais arriscado
          const no = { x: nx, y: ny, z: nz, custo: atual.custo + (dy > 0 ? 2 : 1), sobe: dy > 0 }
          if (objetivo.procuraBloco) no.achou = temAlvo(nx, ny, nz, objetivo.ids)
          proxima.push(no)

          const pontos = objetivo.pontuar(no)
          if (pontos > melhor.pontos) melhor = { no, pontos }
          if (no.achou) achou = true
          if (objetivo.parar && objetivo.parar(no)) { melhor = { no, pontos }; break busca }
          if (achou && objetivo.procuraBloco) break busca
          break
        }
      }
    }
    fila = proxima
  }

  // Reconstroi o caminho
  const caminho = []
  let atual = melhor.no
  let guarda = 0
  while (atual && guarda++ < 5000) {
    caminho.push([atual.x, atual.y, atual.z])
    atual = pai.get(_chave(atual.x, atual.y, atual.z))
  }
  caminho.reverse()

  return {
    caminho,
    alvo: { x: melhor.no.x, y: melhor.no.y, z: melhor.no.z },
    custo: melhor.no.custo,
    pontos: melhor.pontos,
    achou,
    nos,
    alcance: +Math.hypot(melhor.no.x - ox, melhor.no.z - oz).toFixed(2)
  }
}

// ── Piloto: compromisso + controle ────────────────────────────────────────────
const GRAUS_POR_UNIDADE = 0.003 * 180 / Math.PI
const YAW_BINS = [-262, -116, -58, -17, 0, 17, 58, 116, 262]

/**
 * Segue um alvo por varios passos em vez de replanejar sempre.
 *
 * Sem compromisso, dois caminhos de pontuacao parecida se alternam a cada
 * passo e o bot fica gingando: foi a oscilacao que derrubou a primeira versao
 * do planejador para 0.9 blocos por episodio.
 */
class Piloto {
  constructor (opcoes = {}) {
    this.passosCompromisso = opcoes.passosCompromisso || 12
    this.raioChegada = opcoes.raioChegada || 3
    this.memoria = new MemoriaVisitas(opcoes.celula || 8, opcoes.meiaVida || 400)
    this.alvo = null
    this.caminho = []
    this.idx = 0
    this.restam = 0
    this.origem = null
    this.replanejamentos = 0
    this.distMira = opcoes.distMira || 6      // quantos blocos a frente mirar
    this.travado = 0
    this.ultimaPos = null
  }

  reiniciar (bot) {
    const p = bot.entity.position
    this.origem = { x: p.x, z: p.z }
    this.memoria.limpar()
    this.alvo = null
    this.caminho = []
    this.idx = 0
    this.restam = 0
    this.replanejamentos = 0
    this.travado = 0
    this.ultimaPos = null
  }

  /**
   * Ponto do caminho a ~distMira blocos a frente do bot. Avanca o indice
   * conforme o bot progride, para nao mirar em trecho ja percorrido.
   */
  _proximoPonto (p) {
    if (!this.caminho.length) return this.alvo
    // avanca o indice enquanto o ponto atual ja ficou para tras
    while (this.idx < this.caminho.length - 1) {
      const c = this.caminho[this.idx]
      if (Math.hypot(c[0] - p.x, c[2] - p.z) > 2.0) break
      this.idx++
    }
    let alvo = this.caminho[this.caminho.length - 1]
    for (let i = this.idx; i < this.caminho.length; i++) {
      const c = this.caminho[i]
      if (Math.hypot(c[0] - p.x, c[2] - p.z) >= this.distMira) { alvo = c; break }
    }
    return { x: alvo[0], y: alvo[1], z: alvo[2] }
  }

  /** Um passo: devolve a acao no formato do /action. */
  passo (bot, objetivoTipo = 'explorar', extra = {}) {
    const p = bot.entity.position
    if (!this.origem) this.reiniciar(bot)

    this.memoria.registrar(p.x, p.z)
    this.memoria.envelhecer()

    // Detecta travamento: replanejar quando parou de progredir e mais util
    // do que insistir no mesmo caminho por todo o compromisso.
    if (this.ultimaPos) {
      const andou = Math.hypot(p.x - this.ultimaPos.x, p.z - this.ultimaPos.z)
      this.travado = andou < 0.15 ? this.travado + 1 : 0
    }
    this.ultimaPos = { x: p.x, z: p.z }

    const chegou = this.alvo &&
      Math.hypot(this.alvo.x - p.x, this.alvo.z - p.z) < this.raioChegada

    if (!this.alvo || chegou || this.restam <= 0 || this.travado >= 3) {
      const obj = objetivoTipo === 'bloco'
        ? Objetivos.bloco(extra.blocos || ['log', 'log2'], bot.registry)
        : objetivoTipo === 'ponto'
          ? Objetivos.ponto(extra.alvo)
          : objetivoTipo === 'rumo'
            ? Objetivos.rumo(extra.alvo, this.memoria, extra.lambda ?? 6.0)
            : Objetivos.explorar(this.origem, this.memoria, extra.lambda ?? 6.0)

      const r = buscar(bot, obj, extra)
      this.alvo = r.alcance > 2 ? r.alvo : null
      this.caminho = r.caminho || []
      this.idx = 0
      this.ultimo = r
      this.restam = this.passosCompromisso
      this.travado = 0
      this.replanejamentos++
    }
    this.restam--

    if (!this.alvo) return { hold: ['W'], mouse: [0, 0], duration_ms: 250 }

    // Mira no PROXIMO PONTO DO CAMINHO, nao no destino final. Mirar no destino
    // faz o bot andar direto contra o obstaculo que o caminho contorna — foi o
    // que travou a versao anterior aos 14 blocos, apontada para um alvo a 38.
    const mira = this._proximoPonto(p)

    // Controle: gira para o ponto de mira e anda
    const yaw = bot.entity.yaw
    const fx = -Math.sin(yaw), fz = -Math.cos(yaw)
    const relX = mira.x + 0.5 - p.x, relZ = mira.z + 0.5 - p.z
    const frente = relX * fx + relZ * fz
    const lado = relX * (-fz) + relZ * fx
    const graus = Math.atan2(lado, frente) * 180 / Math.PI

    // sinal invertido: medido em malha fechada (mouse=+graus afasta do alvo)
    const desejado = -graus / GRAUS_POR_UNIDADE
    let bin = YAW_BINS[0]
    for (const b of YAW_BINS) if (Math.abs(b - desejado) < Math.abs(bin - desejado)) bin = b

    // Pula quando o proximo ponto do caminho esta acima, ou quando travou:
    // a busca autoriza degrau de 1 bloco, mas no Minecraft subir exige SPACE.
    // Sem isto qualquer caminho com desnivel empacava indefinidamente.
    const precisaSubir = mira.y > Math.floor(p.y)
    const hold = (precisaSubir || this.travado >= 1) ? ['W', 'SPACE'] : ['W']

    return {
      hold,
      mouse: [bin, 0],
      duration_ms: 250,
      _debug: { graus: +graus.toFixed(1), alvo: this.alvo, mira, sobe: precisaSubir, restam: this.restam,
                travado: this.travado, idx: this.idx, caminho: this.caminho.length }
    }
  }
}

module.exports = { buscar, Objetivos, MemoriaVisitas, Piloto, YAW_BINS }
