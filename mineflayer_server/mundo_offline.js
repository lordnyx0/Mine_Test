/**
 * mundo_offline.js — mundo lido dos arquivos de regiao do save, sem o jogo.
 *
 * Os .mca do save ja contem o terreno gerado (199 regioes neste mundo). Carregar
 * dali da um mundo real, com o MESMO layout de section que o voxel_renderer le
 * direto do buffer — entao renderer, mapa de rotas e planejador funcionam sem
 * uma linha de mudanca.
 *
 * getColumn() e SINCRONO porque e assim que o renderer consome; o carregamento
 * do disco e assincrono, entao a area precisa ser pre-carregada com preparar().
 */
'use strict'

const path = require('path')
const { Anvil } = require('prismarine-provider-anvil')

const SAVE_PADRAO = 'C:/Users/Nyx/AppData/Roaming/.minecraft/saves/New World-'

class MundoOffline {
  constructor (savePath = SAVE_PADRAO, versao = '1.8.9') {
    this.save = savePath
    this.anvil = new (Anvil(versao))(path.join(savePath, 'region'))
    this.colunas = new Map()      // chave -> Chunk | null (null = regiao ausente)
    this.carregando = new Map()
    this.acertos = 0
    this.faltas = 0
  }

  static _chave (cx, cz) { return cx * 100000 + cz }

  /** Sincrono, como o renderer espera. Retorna null se ainda nao carregado. */
  getColumn (cx, cz) {
    const k = MundoOffline._chave(cx, cz)
    const c = this.colunas.get(k)
    if (c === undefined) { this.faltas++; return null }
    this.acertos++
    return c
  }

  /** Marca uso recente (chamado no preparar, nao no caminho quente do render). */
  tocar (cx, cz) {
    const k = MundoOffline._chave(cx, cz)
    if (this.colunas.has(k)) {
      const v = this.colunas.get(k)
      this.colunas.delete(k)
      this.colunas.set(k, v)
    }
  }

  getColumns () { return [...this.colunas.values()].filter(Boolean) }

  async _carregarUma (cx, cz) {
    const k = MundoOffline._chave(cx, cz)
    if (this.colunas.has(k)) return this.colunas.get(k)
    if (this.carregando.has(k)) return this.carregando.get(k)

    const p = this.anvil.load(cx, cz)
      .then(col => { this.colunas.set(k, col || null); this.carregando.delete(k); return col })
      .catch(() => { this.colunas.set(k, null); this.carregando.delete(k); return null })
    this.carregando.set(k, p)
    return p
  }

  /**
   * Garante que os chunks num raio (em chunks) ao redor de (x,z) em blocos
   * estejam em memoria. Retorna quantos existem de fato.
   */
  async preparar (x, z, raioChunks = 5) {
    const cx0 = Math.floor(x) >> 4
    const cz0 = Math.floor(z) >> 4
    const tarefas = []
    for (let i = -raioChunks; i <= raioChunks; i++) {
      for (let j = -raioChunks; j <= raioChunks; j++) {
        tarefas.push(this._carregarUma(cx0 + i, cz0 + j))
      }
    }
    const res = await Promise.all(tarefas)
    for (let i = -raioChunks; i <= raioChunks; i++) {
      for (let j = -raioChunks; j <= raioChunks; j++) this.tocar(cx0 + i, cz0 + j)
    }
    return res.filter(Boolean).length
  }

  /**
   * Teto de memoria por LRU. Com N ambientes espalhados pelo mundo, podar por
   * distancia a UM ponto nao serve — cada um precisa da sua vizinhanca. Aqui
   * descarta o menos recentemente usado ate caber no limite.
   * Cada coluna ocupa ~196KB, entao 2600 colunas ~ 510MB.
   */
  podarLRU (maxColunas = 2600) {
    if (this.colunas.size <= maxColunas) return 0
    let remover = this.colunas.size - maxColunas
    let n = 0
    // Map preserva ordem de insercao; getColumn reinsere para marcar uso
    for (const k of this.colunas.keys()) {
      if (remover-- <= 0) break
      this.colunas.delete(k); n++
    }
    return n
  }

  get tamanho () { return this.colunas.size }
  get regioesAbertas () { return Object.keys(this.anvil.regions || {}).length }
}

module.exports = { MundoOffline, SAVE_PADRAO }
