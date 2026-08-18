/**
 * bot_offline.js — um "bot" simulado, sem Minecraft e sem rede.
 *
 * Usa prismarine-physics, o MESMO motor que o mineflayer roda internamente,
 * entao o movimento (velocidade, inercia, gravidade, degrau, colisao, agua)
 * bate com o jogo em vez de ser uma aproximacao.
 *
 * Expoe exatamente a interface que voxel_renderer, mapaDeRotas e planejar ja
 * consomem — bot.entity, bot.world, bot.registry, bot.blockAt, bot.time — de
 * modo que essas tres pecas funcionam sem alteracao.
 */
'use strict'

const Vec3 = require('vec3')
const { Physics, PlayerState } = require('prismarine-physics')

const TICK_MS = 50          // 20 ticks por segundo, como no servidor real
const ALTURA_OLHOS = 1.62

function posNoChunk (pos) {
  return new Vec3(Math.floor(pos.x) & 15, Math.floor(pos.y), Math.floor(pos.z) & 15)
}

class BotOffline {
  /**
   * @param {MundoOffline} mundo
   * @param {object} registry  minecraft-data
   * @param {Vec3} posicao
   */
  constructor (mundo, registry, posicao, versao = '1.8.9') {
    this.version = versao
    this.registry = registry
    this.world = mundo
    this.entities = {}
    this.time = { timeOfDay: 6000, isDay: true, age: 0 }

    this.entity = {
      id: 1,
      type: 'player',
      username: 'SimAgent',
      position: posicao.clone(),
      velocity: new Vec3(0, 0, 0),
      yaw: 0,
      pitch: 0,
      onGround: false,
      height: 1.8,
      width: 0.6,
      isInWater: false,
      isInLava: false,
      isInWeb: false,
      isCollidedHorizontally: false,
      isCollidedVertically: false,
      elytraFlying: false,
      attributes: {},
      effects: {},
      eyeHeight: ALTURA_OLHOS
    }

    // Campos que o PlayerState le direto do bot
    this.jumpTicks = 0
    this.jumpQueued = false
    this.fireworkRocketDuration = 0
    this.inventory = { slots: new Array(46).fill(null) }

    this.controlState = {
      forward: false, back: false, left: false, right: false,
      jump: false, sprint: false, sneak: false
    }

    // Adaptador de mundo no formato que a fisica espera
    this._mundoFisica = { getBlock: (pos) => this.blockAt(pos) }
    this._fisica = Physics(registry, this._mundoFisica)

    this.ticks = 0
    this.morreu = false
    this.health = 20
    this.food = 20
  }

  // ── Mundo ─────────────────────────────────────────────────────────────────
  blockAt (pos) {
    if (!pos) return null
    const y = Math.floor(pos.y)
    if (y < 0 || y > 255) return null
    const col = this.world.getColumn(Math.floor(pos.x) >> 4, Math.floor(pos.z) >> 4)
    if (!col) return null
    let b
    try { b = col.getBlock(posNoChunk(pos)) } catch (_) { return null }
    if (!b) return null
    b.position = new Vec3(Math.floor(pos.x), y, Math.floor(pos.z))
    return b
  }

  setBlock (pos, blockId, blockData = 0) {
    if (!pos) return false
    const y = Math.floor(pos.y)
    if (y < 0 || y > 255) return false
    const col = this.world.getColumn(Math.floor(pos.x) >> 4, Math.floor(pos.z) >> 4)
    if (!col) return false
    const p = posNoChunk(pos)
    try {
      col.setBlockType(p, blockId)
      if (blockData) col.setBlockData(p, blockData)
      return true
    } catch (_) { return false }
  }

  // ── Controles (mesma API do mineflayer) ───────────────────────────────────
  setControlState (nome, valor) {
    if (nome in this.controlState) this.controlState[nome] = !!valor
  }

  getControlState (nome) { return !!this.controlState[nome] }

  soltarTudo () {
    for (const k of Object.keys(this.controlState)) this.controlState[k] = false
  }

  // ── Simulacao ─────────────────────────────────────────────────────────────
  /** Avanca um tick de fisica (50ms). */
  tick () {
    const estado = new PlayerState(this, this.controlState)
    this._fisica.simulatePlayer(estado, this._mundoFisica).apply(this)
    this.ticks++
    this.time.age++
    // Queda no vazio conta como morte, como no jogo
    if (this.entity.position.y < -8) this.morreu = true
    return this.entity.position
  }

  /** Avanca N milissegundos de simulacao. */
  avancar (ms) {
    const n = Math.max(1, Math.round(ms / TICK_MS))
    for (let i = 0; i < n; i++) this.tick()
    return n
  }

  /** Gira a camera em unidades de mouse (yaw += dx*0.003 rad), como o /action. */
  girar (dx, dy) {
    const sens = 0.003
    this.entity.yaw += dx * sens
    this.entity.pitch = Math.max(-Math.PI / 2,
      Math.min(Math.PI / 2, this.entity.pitch + dy * sens))
  }

  /** Reposiciona instantaneamente (equivale ao /tp do respawn). */
  teleportar (x, y, z) {
    this.entity.position = new Vec3(x, y, z)
    this.entity.velocity = new Vec3(0, 0, 0)
    this.entity.onGround = false
    this.morreu = false
    this.soltarTudo()
  }

  get estado () {
    const p = this.entity.position
    const v = this.entity.velocity
    return {
      x: +p.x.toFixed(4), y: +p.y.toFixed(4), z: +p.z.toFixed(4),
      yaw: +(this.entity.yaw * 180 / Math.PI).toFixed(2),
      pitch: +(this.entity.pitch * 180 / Math.PI).toFixed(2),
      vx: +v.x.toFixed(4), vy: +v.y.toFixed(4), vz: +v.z.toFixed(4),
      on_ground: !!this.entity.onGround,
      is_collided_horizontally: !!this.entity.isCollidedHorizontally,
      // A fisica JA mantem estes (prismarine-physics linha ~712), mas o estado
      // nunca os expunha — entao afogamento era literalmente indetectavel do
      // lado de fora, e o predicado "estou na agua?" nao tinha como existir.
      in_water: !!this.entity.isInWater,
      in_lava: !!this.entity.isInLava,
      health: this.health, food: this.food,
      connected: true,
      timestamp: Date.now(),
      ticks: this.ticks
    }
  }

  /**
   * Distancia horizontal ate o bloco de agua mais proximo, ou null.
   *
   * Serve a duas coisas: o predicado de natacao, e diagnosticar por que o
   * planejador empaca. O BFS so anda sobre celulas pisaveis, entao um lago
   * nao existe no grafo dele e o outro lado fica inalcancavel — a suspeita
   * para as falhas em que a distancia ao alvo estaciona no passo 3.
   *
   * CARO: ~676 consultas de bloco. Opcional de proposito, nunca no caminho
   * quente do passo.
   */
  aguaPerto (raio = 6) {
    const p = this.entity.position
    const bx = Math.floor(p.x), by = Math.floor(p.y), bz = Math.floor(p.z)
    let melhor = null
    for (let dx = -raio; dx <= raio; dx++) {
      for (let dz = -raio; dz <= raio; dz++) {
        for (let dy = -2; dy <= 1; dy++) {
          const b = this.blockAt(new Vec3(bx + dx, by + dy, bz + dz))
          if (b && b.name && b.name.indexOf('water') !== -1) {
            const d = Math.hypot(dx, dz)
            if (melhor === null || d < melhor) melhor = d
          }
        }
      }
    }
    return melhor === null ? null : +melhor.toFixed(1)
  }

  /**
   * Perfil de altura do chao ao longo de uma direcao, em blocos relativos ao
   * bot. Diz O QUE bloqueia, nao so QUE bloqueou:
   *   valores subindo forte  -> parede ou encosta ingreme (o BFS so aceita
   *                             degrau de 1, entao 2+ e intransponivel)
   *   valores despencando    -> abismo ou ravina
   *   valores planos         -> o bloqueio nao e relevo, procurar outra causa
   *
   * `null` numa posicao = coluna nao carregada (fora do raio de chunks), que
   * tambem torna a celula intransponivel para a busca.
   */
  perfilNaDirecao (dx, dz, alcance = 12) {
    const p = this.entity.position
    const n = Math.hypot(dx, dz)
    // Direcao degenerada: com |dir| ~ 0 a normalizacao daria (0,0) e as 12
    // amostras cairiam TODAS na propria celula do bot, o que faz qualquer
    // relevo proximo parecer parede. Foi o que invalidou o grupo de controle
    // (bots que CHEGARAM estao em cima do alvo, entao dir = alvo - pos ~ 0).
    // Devolver null e honesto: nao ha direcao a perfilar.
    if (n < 0.5) return null
    const ux = dx / n, uz = dz / n
    const by = Math.floor(p.y)
    const perfil = []
    for (let i = 1; i <= alcance; i++) {
      const x = Math.floor(p.x + ux * i)
      const z = Math.floor(p.z + uz * i)
      let alturaRel = null
      // Procura o topo solido numa janela vertical em torno da altura do bot
      for (let y = by + 3; y >= by - 6; y--) {
        const b = this.blockAt(new Vec3(x, y, z))
        if (b === null) break
        if (b.name && b.name !== 'air' && b.boundingBox === 'block') {
          alturaRel = y - by + 1
          break
        }
      }
      perfil.push(alturaRel)
    }
    return perfil
  }

  /** Nome do bloco numa posicao relativa aos pes. */
  blocoRel (dx, dy, dz) {
    const p = this.entity.position
    const b = this.blockAt(new Vec3(Math.floor(p.x) + dx, Math.floor(p.y) + dy,
                                    Math.floor(p.z) + dz))
    return b ? b.name : null
  }

  /** Modifica o tipo e dados de um bloco no mundo. */
  setBlock (pos, id, data = 0) {
    const bx = Math.floor(pos.x), by = Math.floor(pos.y), bz = Math.floor(pos.z)
    const col = this.world.getColumn(bx >> 4, bz >> 4)
    if (!col) return false
    const pInCol = new Vec3(bx & 15, by, bz & 15)
    col.setBlockType(pInCol, id)
    if (data) col.setBlockData(pInCol, data)
    return true
  }
}

module.exports = { BotOffline, TICK_MS }
