/**
 * voxel_renderer.js — Renderizador de visao em 1a pessoa direto em Node.js
 *
 * Substitui o pipeline prismarine-viewer + Puppeteer.screenshot (80-200ms/frame)
 * por um raycaster DDA que le os blocos direto da world do Mineflayer e
 * escreve os pixels num canvas nativo (<10ms/frame).
 *
 * Sem browser, sem WebGL, sem janela. Puro CPU.
 *
 * Uso:
 *   const { VoxelRenderer } = require('./voxel_renderer')
 *   const renderer = new VoxelRenderer(bot)
 *   const jpegBuffer = renderer.render()
 */

'use strict'

const { createCanvas } = require('canvas')

// ── Constantes de face ────────────────────────────────────────────────────────
const FACE_X = 0
const FACE_Y = 1
const FACE_Z = 2

// Sombreamento por face (mesma convencao do Minecraft vanilla)
const SHADE_TOP = 1.0
const SHADE_BOTTOM = 0.45
const SHADE_Z = 0.8   // norte/sul
const SHADE_X = 0.62  // leste/oeste

// ── Paleta ────────────────────────────────────────────────────────────────────
// Cor unica  → [r, g, b]
// Cor p/face → { top: [...], side: [...], bottom: [...] }
const BLOCK_COLORS = {
  stone: [125, 125, 125],
  grass: { top: [106, 155, 65], side: [122, 130, 74], bottom: [134, 96, 67] },
  dirt: [134, 96, 67],
  cobblestone: [122, 122, 122],
  planks: [162, 130, 78],
  bedrock: [85, 85, 85],
  flowing_water: [50, 90, 200],
  water: [50, 90, 200],
  flowing_lava: [214, 96, 22],
  lava: [214, 96, 22],
  sand: [219, 207, 163],
  gravel: [136, 130, 127],
  gold_ore: [143, 139, 124],
  iron_ore: [136, 130, 126],
  coal_ore: [105, 105, 105],
  log: { top: [162, 130, 78], side: [102, 81, 50] },
  log2: { top: [162, 130, 78], side: [102, 81, 50] },
  leaves: [62, 129, 43],
  leaves2: [62, 129, 43],
  sponge: [195, 192, 74],
  glass: [175, 213, 219],
  lapis_ore: [102, 112, 134],
  lapis_block: [25, 110, 245],
  gold_block: [245, 215, 20],
  dispenser: [107, 107, 107],
  sandstone: [216, 208, 164],
  noteblock: [100, 67, 50],
  bed: [165, 44, 44],
  sticky_piston: [122, 122, 122],
  web: [222, 222, 222],
  piston: [122, 122, 122],
  piston_head: [162, 130, 78],
  wool: [222, 222, 222],
  double_stone_slab: [125, 125, 125],
  stone_slab: [125, 125, 125],
  brick_block: [151, 92, 76],
  tnt: [186, 60, 44],
  bookshelf: { top: [162, 130, 78], side: [124, 100, 66] },
  mossy_cobblestone: [104, 122, 100],
  obsidian: [155, 38, 182],
  mob_spawner: [50, 62, 74],
  oak_stairs: [162, 130, 78],
  chest: [138, 106, 55],
  diamond_ore: [110, 140, 140],
  diamond_block: [98, 219, 214],
  crafting_table: { top: [151, 118, 74], side: [124, 100, 66] },
  farmland: [110, 72, 40],
  furnace: [107, 107, 107],
  lit_furnace: [131, 111, 96],
  wooden_door: [143, 116, 72],
  ladder: [138, 110, 66],
  stone_stairs: [122, 122, 122],
  iron_door: [166, 166, 166],
  redstone_ore: [133, 107, 107],
  lit_redstone_ore: [151, 92, 92],
  snow_layer: [240, 245, 250],
  ice: [140, 180, 245],
  snow: [240, 245, 250],
  cactus: { top: [85, 127, 43], side: [60, 116, 39] },
  clay: [160, 166, 179],
  jukebox: [100, 67, 50],
  fence: [162, 130, 78],
  pumpkin: { top: [199, 128, 30], side: [193, 118, 21] },
  netherrack: [111, 54, 52],
  soul_sand: [84, 64, 51],
  glowstone: [231, 190, 122],
  portal: [88, 39, 158],
  lit_pumpkin: [216, 148, 44],
  stained_glass: [200, 200, 210],
  trapdoor: [131, 104, 62],
  monster_egg: [125, 125, 125],
  stonebrick: [122, 122, 122],
  brown_mushroom_block: [149, 111, 86],
  red_mushroom_block: [186, 60, 44],
  iron_bars: [130, 130, 130],
  glass_pane: [175, 213, 219],
  melon_block: { top: [111, 145, 30], side: [125, 158, 34] },
  fence_gate: [162, 130, 78],
  brick_stairs: [151, 92, 76],
  stone_brick_stairs: [122, 122, 122],
  mycelium: { top: [111, 100, 105], side: [124, 108, 106], bottom: [134, 96, 67] },
  waterlily: [32, 128, 48],
  nether_brick: [44, 22, 27],
  nether_brick_fence: [44, 22, 27],
  nether_brick_stairs: [44, 22, 27],
  enchanting_table: [128, 68, 66],
  brewing_stand: [124, 103, 81],
  cauldron: [76, 76, 76],
  end_portal_frame: [98, 122, 100],
  end_stone: [221, 223, 165],
  dragon_egg: [12, 9, 15],
  redstone_lamp: [95, 62, 35],
  lit_redstone_lamp: [227, 174, 110],
  double_wooden_slab: [162, 130, 78],
  wooden_slab: [162, 130, 78],
  cocoa: [152, 92, 30],
  sandstone_stairs: [216, 208, 164],
  emerald_ore: [108, 136, 111],
  ender_chest: [40, 60, 62],
  emerald_block: [42, 203, 87],
  spruce_stairs: [107, 81, 47],
  birch_stairs: [196, 179, 123],
  jungle_stairs: [153, 111, 76],
  command_block: [186, 146, 111],
  beacon: [117, 216, 209],
  cobblestone_wall: [122, 122, 122],
  flower_pot: [126, 66, 50],
  skull: [178, 172, 166],
  anvil: [70, 70, 70],
  trapped_chest: [138, 106, 55],
  unpowered_comparator: [178, 172, 166],
  powered_comparator: [188, 152, 146],
  daylight_detector: [147, 130, 100],
  daylight_detector_inverted: [107, 95, 74],
  redstone_block: [175, 24, 5],
  quartz_ore: [124, 90, 86],
  hopper: [70, 70, 70],
  quartz_block: [235, 229, 222],
  quartz_stairs: [235, 229, 222],
  dropper: [107, 107, 107],
  stained_hardened_clay: [150, 92, 66],
  stained_glass_pane: [200, 200, 210],
  acacia_stairs: [168, 90, 50],
  dark_oak_stairs: [67, 43, 20],
  slime: [111, 191, 96],
  barrier: [0, 0, 0],
  iron_trapdoor: [166, 166, 166],
  prismarine: [99, 156, 151],
  sea_lantern: [214, 227, 220],
  hay_block: { top: [166, 138, 20], side: [165, 139, 12] },
  carpet: [222, 222, 222],
  hardened_clay: [150, 92, 66],
  coal_block: [16, 16, 16],
  packed_ice: [141, 180, 250],
  red_sandstone: [186, 99, 40],
  red_sandstone_stairs: [186, 99, 40],
  double_stone_slab2: [186, 99, 40],
  stone_slab2: [186, 99, 40],
  spruce_fence_gate: [107, 81, 47],
  birch_fence_gate: [196, 179, 123],
  jungle_fence_gate: [153, 111, 76],
  dark_oak_fence_gate: [67, 43, 20],
  acacia_fence_gate: [168, 90, 50],
  spruce_fence: [107, 81, 47],
  birch_fence: [196, 179, 123],
  jungle_fence: [153, 111, 76],
  dark_oak_fence: [67, 43, 20],
  acacia_fence: [168, 90, 50],
  spruce_door: [107, 81, 47],
  birch_door: [196, 179, 123],
  jungle_door: [153, 111, 76],
  acacia_door: [168, 90, 50],
  dark_oak_door: [67, 43, 20],
  // "sprites" (boundingBox empty) — tint parcial
  sapling: [79, 130, 47],
  tallgrass: [98, 148, 60],
  deadbush: [148, 106, 43],
  yellow_flower: [222, 210, 60],
  red_flower: [200, 60, 60],
  brown_mushroom: [151, 118, 96],
  red_mushroom: [200, 60, 60],
  torch: [255, 200, 90],
  fire: [230, 140, 40],
  wheat: [180, 175, 80],
  reeds: [128, 175, 90],
  vine: [62, 129, 43],
  double_plant: [98, 148, 60],
  nether_wart: [150, 40, 50],
  carrots: [90, 150, 60],
  potatoes: [90, 150, 60],
  pumpkin_stem: [110, 150, 60],
  melon_stem: [110, 150, 60],
  redstone_torch: [255, 80, 60],
  unlit_redstone_torch: [120, 40, 30],
  cake: [230, 220, 210],
  lever: [124, 100, 66],
  standing_sign: [143, 116, 72],
  wall_sign: [143, 116, 72],
  standing_banner: [143, 116, 72],
  wall_banner: [143, 116, 72],
  stone_button: [125, 125, 125],
  wooden_button: [162, 130, 78],
  end_portal: [10, 10, 20]
}

// Blocos que emitem luz propria (nunca escurecem)
const EMISSIVE = new Set([
  'lava', 'flowing_lava', 'glowstone', 'torch', 'fire', 'sea_lantern',
  'lit_redstone_lamp', 'lit_furnace', 'lit_pumpkin', 'end_portal', 'portal',
  'redstone_torch', 'beacon'
])

// Blocos "empty" que sao planos no chao — nao devem tingir o ar acima deles
const SKIP_FLAT = new Set([
  'rail', 'golden_rail', 'detector_rail', 'activator_rail', 'redstone_wire',
  'stone_pressure_plate', 'wooden_pressure_plate', 'light_weighted_pressure_plate',
  'heavy_weighted_pressure_plate', 'tripwire', 'tripwire_hook', 'snow_layer',
  'piston_extension'
])

// Sprites (grama alta, flores, tochas) sao cruzes finas no centro do voxel.
// SPRITE_RADIUS_SQ = raio^2 do "cilindro" que o raio precisa acertar.
// SPRITE_MAX_ALPHA limita quanto a vegetacao pode escurecer a cena: sem teto,
// um raio rasante sobre um campo de grama satura e vira uma parede verde.
// Vegetacao (boundingBox 'empty') NAO e obstaculo: o bot atravessa. Ela nao
// informa nada para locomocao e so tapa o chao, entao entra fraca e com teto.
const SPRITE_RADIUS_SQ = 0.10   // raio ~0.32 -> ~31% de cobertura do voxel
const SPRITE_MAX_ALPHA = 0.28

// Blocos translucidos que o raio atravessa acumulando tinta
const GLASS = new Set(['glass', 'glass_pane', 'stained_glass', 'stained_glass_pane', 'ice', 'barrier'])

// Cores de tintura (metadata 0-15) para wool / stained_glass / stained clay / carpet
const DYE = [
  [234, 236, 237], [240, 118, 19], [189, 68, 179], [58, 175, 217],
  [248, 198, 39], [112, 185, 25], [237, 141, 172], [62, 68, 71],
  [142, 142, 134], [21, 137, 145], [121, 42, 172], [53, 57, 157],
  [114, 71, 40], [84, 109, 27], [161, 39, 34], [20, 21, 25]
]
const DYE_CLAY = [
  [209, 178, 161], [161, 83, 37], [149, 88, 108], [113, 108, 137],
  [186, 133, 35], [103, 117, 52], [161, 78, 78], [57, 42, 35],
  [135, 107, 98], [87, 91, 91], [118, 70, 86], [74, 60, 91],
  [77, 51, 35], [76, 83, 42], [143, 61, 46], [37, 22, 16]
]
const LOG_SIDE = [[102, 81, 50], [58, 38, 18], [216, 214, 208], [86, 67, 39]]
const LOG2_SIDE = [[104, 62, 32], [58, 38, 18]]
const LEAF_COLOR = [[62, 129, 43], [48, 106, 48], [70, 140, 45], [58, 132, 34]]
const LEAF2_COLOR = [[86, 129, 62], [50, 100, 40]]

// Cores de entidades por nome
const ENTITY_COLORS = {
  player: [214, 168, 130],
  zombie: [55, 122, 82],
  skeleton: [199, 199, 199],
  creeper: [76, 190, 76],
  spider: [78, 55, 44],
  cave_spider: [30, 82, 84],
  enderman: [22, 22, 30],
  cow: [96, 68, 48],
  mooshroom: [154, 44, 40],
  pig: [222, 140, 140],
  sheep: [232, 232, 232],
  chicken: [222, 222, 222],
  horse: [138, 106, 70],
  wolf: [190, 190, 190],
  villager: [148, 106, 78],
  witch: [58, 62, 80],
  slime: [111, 191, 96],
  blaze: [235, 160, 40],
  ghast: [222, 222, 222],
  squid: [40, 60, 90],
  bat: [70, 55, 44],
  item: [230, 200, 90],
  arrow: [190, 190, 190],
  xp_orb: [140, 230, 60]
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function clamp01 (v) { return v < 0 ? 0 : (v > 1 ? 1 : v) }

// Offsets dentro do Buffer de uma section 1.8 (16x16x16)
const BLOCK_BASE = 0
const LIGHT_BASE = 8192
const SKY_BASE = 10240

/** Testa se a section inteira e ar (stateId 0 em todos os 4096 blocos). */
function sectionIsEmpty (d) {
  for (let i = 0; i < 8192; i += 8) {
    if (d[i] | d[i + 1] | d[i + 2] | d[i + 3] | d[i + 4] | d[i + 5] | d[i + 6] | d[i + 7]) return false
  }
  return true
}

/** Hash inteiro deterministico → [0, 1). Usado para o "grão" de textura. */
function hash3 (x, y, z) {
  let h = (x * 374761393 + y * 668265263 + z * 1274126177) | 0
  h = (h ^ (h >>> 13)) | 0
  h = Math.imul(h, 1274126177)
  h = (h ^ (h >>> 16)) >>> 0
  return h / 4294967296
}

// ── Renderer ──────────────────────────────────────────────────────────────────
class VoxelRenderer {
  /**
   * @param {object} bot                    bot do mineflayer
   * @param {object} [opts]
   * @param {number} [opts.width=640]       largura de saida
   * @param {number} [opts.height=360]      altura de saida
   * @param {number} [opts.scale=4]         fator de downscale interno do raycast
   * @param {number} [opts.fov=70]          FOV vertical em graus
   * @param {number} [opts.maxDistance=64]  alcance maximo dos raios (blocos)
   * @param {number} [opts.quality=0.8]     qualidade JPEG
   * @param {boolean}[opts.crosshair=true]  desenha mira no centro
   * @param {boolean}[opts.entities=true]   desenha entidades (mobs/players/itens)
   * @param {boolean}[opts.sprites=true]    desenha vegetacao atravessavel (grama, flores)
   * @param {boolean}[opts.smooth=false]    interpolacao bilinear no upscale (custa ~5ms)
   */
  constructor (bot, opts = {}) {
    this.bot = bot

    this.width = opts.width || 640
    this.height = opts.height || 360
    this.scale = Math.max(1, opts.scale || 4)
    this.fov = (opts.fov || 70) * Math.PI / 180
    this.maxDistance = opts.maxDistance || 64
    this.quality = opts.quality != null ? opts.quality : 0.8
    this.drawCrosshair = opts.crosshair !== false
    this.drawEntities = opts.entities !== false
    this.drawSprites = opts.sprites !== false
    // Cairo faz o upscale bilinear em ~5ms; nearest custa ~0.7ms. Default: nearest.
    this.smooth = opts.smooth === true

    this.rw = Math.max(2, Math.round(this.width / this.scale))
    this.rh = Math.max(2, Math.round(this.height / this.scale))

    // Canvas pequeno (raycast) e canvas grande (saida)
    this.small = createCanvas(this.rw, this.rh)
    this.smallCtx = this.small.getContext('2d')
    this.imageData = this.smallCtx.createImageData(this.rw, this.rh)
    this.pixels = this.imageData.data

    this.canvas = createCanvas(this.width, this.height)
    this.ctx = this.canvas.getContext('2d')
    this.ctx.imageSmoothingEnabled = this.smooth

    // Z-buffer (distancia ao longo do eixo da camera) para as entidades
    this.depth = new Float32Array(this.rw * this.rh)

    // Offsets do plano de imagem (constantes por resolucao)
    const tanHalf = Math.tan(this.fov / 2)
    const aspect = this.rw / this.rh
    this.offU = new Float32Array(this.rw)
    this.offV = new Float32Array(this.rh)
    for (let i = 0; i < this.rw; i++) this.offU[i] = (2 * (i + 0.5) / this.rw - 1) * tanHalf * aspect
    for (let j = 0; j < this.rh; j++) this.offV[j] = (1 - 2 * (j + 0.5) / this.rh) * tanHalf
    this.tanHalf = tanHalf
    this.aspect = aspect

    // ── Cache de chunks: grade plana centrada no bot, reconstruida por frame ──
    this._chunkRadius = Math.ceil(this.maxDistance / 16) + 1
    this._span = this._chunkRadius * 2 + 1
    this._cols = new Array(this._span * this._span).fill(null)
    this._baseCx = 0
    this._baseCz = 0

    this._gen = 1            // geracao do cache de sections (ver _sectionData)
    this._layoutChecked = false

    // Lookup table de iluminacao (16 skyLight x 16 blockLight), refeita por frame
    this._litLut = new Float32Array(256)
    this._lutDay = -1

    this.lastRenderMs = 0
    this.frameCount = 0

    this._buildPalette()
    this._hookWorldEvents()
  }

  /**
   * Invalida o cache "section vazia" quando blocos mudam. Sem isso, um bloco
   * colocado numa section marcada como vazia ficaria invisivel.
   *
   * A invalidacao e por section (nao global) para nao forcar rescan de tudo
   * quando a agua flui ou um mob quebra um bloco longe.
   */
  _hookWorldEvents () {
    const bot = this.bot
    if (!bot || typeof bot.on !== 'function') return

    const invalidate = (pos) => {
      if (!pos) return
      const col = this._rawColumn(pos.x >> 4, pos.z >> 4)
      if (!col || !col.sections) return
      const sec = col.sections[pos.y >> 4]
      if (sec) sec.__vrGen = -1
    }

    try {
      bot.on('blockUpdate', (oldB, newB) => {
        invalidate((newB && newB.position) || (oldB && oldB.position))
      })
    } catch (_) {}
    try {
      bot.on('chunkColumnLoad', (corner) => {
        const col = this._rawColumn(corner.x >> 4, corner.z >> 4)
        if (col && col.sections) for (const s of col.sections) { if (s) s.__vrGen = -1 }
      })
    } catch (_) {}
  }

  _rawColumn (cx, cz) {
    try { return this.bot.world.getColumn(cx, cz) || null } catch (_) { return null }
  }

  // ── Paleta indexada por id ──────────────────────────────────────────────────
  _buildPalette () {
    const registry = this.bot.registry || require('minecraft-data')(this.bot.version || '1.8.9')
    this.registry = registry

    const N = 256
    this.colTop = new Uint8Array(N * 3)
    this.colSide = new Uint8Array(N * 3)
    this.colBottom = new Uint8Array(N * 3)
    this.solid = new Uint8Array(N)      // 1 = para o raio
    this.isWater = new Uint8Array(N)
    this.isGlass = new Uint8Array(N)
    this.emissive = new Uint8Array(N)
    this.spriteAlpha = new Float32Array(N)  // >0 = tint parcial, raio continua
    this.variantSide = new Array(N).fill(null)
    this.variantTop = new Array(N).fill(null)

    const put = (arr, id, c) => {
      arr[id * 3] = c[0]; arr[id * 3 + 1] = c[1]; arr[id * 3 + 2] = c[2]
    }

    for (const b of registry.blocksArray) {
      const id = b.id
      if (id >= N) continue
      if (id === 0) continue // ar

      let def = BLOCK_COLORS[b.name]
      if (!def) {
        // Fallback por material + hash estavel do nome
        const base = ({
          rock: [125, 125, 125], wood: [150, 120, 72], dirt: [134, 96, 67],
          plant: [90, 140, 55], leaves: [62, 129, 43], wool: [222, 222, 222],
          web: [222, 222, 222]
        })[b.material] || [140, 120, 130]
        const j = hash3(id, 7, 13)
        def = [
          Math.min(255, Math.round(base[0] * (0.8 + j * 0.4))),
          Math.min(255, Math.round(base[1] * (0.8 + j * 0.4))),
          Math.min(255, Math.round(base[2] * (0.8 + j * 0.4)))
        ]
      }

      const top = Array.isArray(def) ? def : (def.top || def.side)
      const side = Array.isArray(def) ? def : (def.side || def.top)
      const bottom = Array.isArray(def) ? def : (def.bottom || def.side || def.top)
      put(this.colTop, id, top)
      put(this.colSide, id, side)
      put(this.colBottom, id, bottom)

      if (EMISSIVE.has(b.name)) this.emissive[id] = 1

      if (b.name === 'water' || b.name === 'flowing_water') {
        this.isWater[id] = 1
      } else if (GLASS.has(b.name)) {
        this.isGlass[id] = 1
        this.solid[id] = b.name === 'ice' ? 0 : 0
      } else if (b.name === 'lava' || b.name === 'flowing_lava') {
        this.solid[id] = 1
      } else if (b.boundingBox === 'empty') {
        this.spriteAlpha[id] = (SKIP_FLAT.has(b.name) || !this.drawSprites) ? 0 : 0.30
      } else {
        this.solid[id] = 1
      }
    }

    // Variantes por metadata
    const dyed = [35, 95, 160, 171] // wool, stained_glass, stained_glass_pane, carpet
    for (const id of dyed) { this.variantSide[id] = DYE; this.variantTop[id] = DYE }
    this.variantSide[159] = DYE_CLAY; this.variantTop[159] = DYE_CLAY

    this.variantSide[17] = LOG_SIDE
    this.variantSide[162] = LOG2_SIDE
    this.variantTop[18] = LEAF_COLOR; this.variantSide[18] = LEAF_COLOR
    this.variantTop[161] = LEAF2_COLOR; this.variantSide[161] = LEAF2_COLOR
  }

  // ── Acesso rapido ao mundo ──────────────────────────────────────────────────
  /** Recarrega a grade de chunks ao redor do bot. Chamado 1x por frame. */
  _refreshChunkGrid (px, pz) {
    const span = this._span
    const r = this._chunkRadius
    const baseCx = (Math.floor(px) >> 4) - r
    const baseCz = (Math.floor(pz) >> 4) - r
    this._baseCx = baseCx
    this._baseCz = baseCz
    const cols = this._cols
    const world = this.bot.world
    for (let i = 0; i < span; i++) {
      const row = i * span
      const cx = baseCx + i
      for (let j = 0; j < span; j++) {
        let c = null
        try { c = world.getColumn(cx, baseCz + j) || null } catch (_) { c = null }
        cols[row + j] = c
      }
    }
    if (!this._layoutChecked) this._checkChunkLayout(cols)
  }

  /**
   * O raycast le os bytes da section direto (formato 1.8: 4096 blocos em
   * uint16 LE, depois blockLight e skyLight em nibbles). Se o mundo vier em
   * outro formato, isso renderizaria lixo em silencio — entao aborta alto.
   */
  _checkChunkLayout (cols) {
    for (const c of cols) {
      if (!c || !c.sections) continue
      for (const sec of c.sections) {
        if (!sec || !sec.data) continue
        this._layoutChecked = true
        const n = sec.data.length
        // 4096*3 = 12288 com skyLight, 4096*2.5 = 10240 sem
        if (n !== 12288 && n !== 10240) {
          throw new Error(
            `voxel_renderer: layout de chunk inesperado (section de ${n} bytes, ` +
            `esperado 12288 ou 10240). Este renderer le o formato do Minecraft 1.8 ` +
            `direto do buffer. Versao do bot: ${this.bot.version}. ` +
            `Use RENDERER=puppeteer para versoes mais novas.`)
        }
        return
      }
    }
  }

  _colAt (cx, cz) {
    const i = cx - this._baseCx
    const j = cz - this._baseCz
    const span = this._span
    if (i < 0 || j < 0 || i >= span || j >= span) return null
    return this._cols[i * span + j]
  }

  /**
   * Buffer da section que contem (x,y,z), ou null se a section e ar puro,
   * inexistente ou fora dos chunks carregados. null = "pule o bloco inteiro".
   */
  _sectionData (x, y, z) {
    const col = this._colAt(x >> 4, z >> 4)
    if (col === null) return null
    const secs = col.sections
    if (secs === undefined) return null
    const sec = secs[y >> 4]
    if (!sec) return null

    // Caminho quente: uma comparacao de inteiros. Revalidar lendo sec.data.buffer
    // custa dois getters de TypedArray e domina o custo do raycast.
    if (sec.__vrGen !== this._gen) this._initSection(sec, this._gen)
    return sec.__vrEmpty ? null : sec.__vrU8
  }

  /** Prepara o cache de uma section: view Uint8Array pura + flag "so ar". */
  _initSection (sec, gen) {
    const d = sec.data
    if (!d) {
      sec.__vrU8 = null
      sec.__vrEmpty = true
    } else {
      // Buffer e subclasse de Uint8Array; a view "pura" indexa mais rapido
      sec.__vrU8 = new Uint8Array(d.buffer, d.byteOffset, d.length)
      sec.__vrEmpty = sectionIsEmpty(sec.__vrU8)
    }
    sec.__vrGen = gen
  }

  /** Constroi a LUT de iluminacao final (0..1) indexada por sky*16+block. */
  _buildLitLut (dayFactor) {
    if (this._lutDay === dayFactor) return
    this._lutDay = dayFactor
    const lut = this._litLut
    for (let sky = 0; sky < 16; sky++) {
      const s = (sky / 15) * dayFactor
      for (let blk = 0; blk < 16; blk++) {
        const b = blk / 15
        const l = s > b ? s : b
        lut[sky * 16 + blk] = 0.13 + 0.87 * Math.pow(l, 0.85)
      }
    }
  }

  /** Iluminacao final (0..1) do voxel, ja passada pela LUT. */
  _litAt (x, y, z) {
    if (y < 0) return this._litLut[0]
    if (y > 255) return this._litLut[15 * 16]
    const col = this._colAt(x >> 4, z >> 4)
    if (col === null || !col.sections) return this._litLut[15 * 16]
    const sec = col.sections[y >> 4]
    if (!sec || !sec.data) return this._litLut[15 * 16]

    const d = sec.data
    const n = (x & 15) | ((z & 15) << 4) | ((y & 15) << 8)
    const half = n >> 1
    const odd = n & 1
    const bl = d[LIGHT_BASE + half]
    const blk = odd ? (bl >> 4) : (bl & 15)
    let sky = 15
    if (col.skyLightSent !== false) {
      const sl = d[SKY_BASE + half]
      sky = odd ? (sl >> 4) : (sl & 15)
      // Se a section não tiver luz solar gravada (seções superiores de céu), assume luz do dia plena (15)
      if (sky === 0 && y >= 60) sky = 15
    }
    return this._litLut[sky * 16 + blk]
  }

  /** Leitura pontual de stateId (usada fora do loop quente). */
  _stateAt (x, y, z) {
    if (y < 0 || y > 255) return 0
    const d = this._sectionData(x, y, z)
    if (d === null) return 0
    const o = ((x & 15) | ((z & 15) << 4) | ((y & 15) << 8)) << 1
    return d[o] | (d[o + 1] << 8)
  }

  // ── Ciclo dia/noite ─────────────────────────────────────────────────────────
  _dayFactor () {
    const t = (this.bot.time && this.bot.time.timeOfDay != null) ? this.bot.time.timeOfDay : 6000
    const raw = 0.5 + 0.5 * Math.cos(((t - 6000) / 24000) * Math.PI * 2)
    // Curva mais "quadrada": amanhecer/anoitecer rapidos como no jogo
    const f = clamp01((raw - 0.22) / 0.36)
    return 0.16 + 0.84 * (f * f * (3 - 2 * f))
  }

  // ── Render principal ────────────────────────────────────────────────────────
  /** @returns {Buffer|null} JPEG da visao em 1a pessoa, ou null se o bot nao esta pronto */
  render () {
    const bot = this.bot
    if (!bot || !bot.entity || !bot.world) return null

    const t0 = process.hrtime.bigint()

    const rw = this.rw
    const rh = this.rh
    const px = this.pixels
    const depth = this.depth
    const maxDist = this.maxDistance
    const dayFactor = this._dayFactor()
    this._buildLitLut(dayFactor)
    this._refreshChunkGrid(bot.entity.position.x, bot.entity.position.z)

    // ── Camera ────────────────────────────────────────────────────────────────
    const pos = bot.entity.position
    const eyeH = (bot.entity.height ? bot.entity.height * 0.9 : 1.62)
    const ox = pos.x
    const oy = pos.y + eyeH
    const oz = pos.z
    const yaw = bot.entity.yaw
    const pitch = bot.entity.pitch

    const cp = Math.cos(pitch)
    const sp = Math.sin(pitch)
    const cy = Math.cos(yaw)
    const sy = Math.sin(yaw)

    // forward / right / up (mesma convencao de bot.blockAtCursor)
    const fx = -sy * cp, fy = sp, fz = -cy * cp
    const rx = cy, ry = 0, rz = -sy
    // up = right × forward  (a ordem importa: forward × right aponta para baixo)
    const ux = ry * fz - rz * fy
    const uy = rz * fx - rx * fz
    const uz = rx * fy - ry * fx

    this.camera = { ox, oy, oz, fx, fy, fz, rx, ry, rz, ux, uy, uz }

    // ── Ceu ───────────────────────────────────────────────────────────────────
    const skyTop = [
      Math.round(9 + 106 * dayFactor),
      Math.round(12 + 145 * dayFactor),
      Math.round(30 + 205 * dayFactor)
    ]
    const skyHorizon = [
      Math.round(20 + 165 * dayFactor),
      Math.round(24 + 180 * dayFactor),
      Math.round(42 + 200 * dayFactor)
    ]
    this._sky = skyHorizon

    // Camera dentro d'agua?
    const camState = this._stateAt(Math.floor(ox), Math.floor(oy), Math.floor(oz))
    const camType = camState >> 4
    const underwater = camType < 256 && this.isWater[camType] === 1

    // ── Loop de raycast ───────────────────────────────────────────────────────
    this._skyTop = skyTop
    this._skyHor = skyHorizon
    this._dayF = dayFactor
    this._underwater = underwater

    let p = 0
    for (let j = 0; j < rh; j++) {
      const v = this.offV[j]
      for (let i = 0; i < rw; i++, p += 4) {
        const u = this.offU[i]

        let dx = fx + rx * u + ux * v
        let dy = fy + ry * u + uy * v
        let dz = fz + rz * u + uz * v
        const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz)
        dx *= inv; dy *= inv; dz *= inv

        this._castRay(ox, oy, oz, dx, dy, dz, p, p >> 2)
      }
    }

    // ── Entidades ─────────────────────────────────────────────────────────────
    if (this.drawEntities) this._renderEntities(dayFactor)

    // ── Upscale + HUD ─────────────────────────────────────────────────────────
    this.smallCtx.putImageData(this.imageData, 0, 0)
    this.ctx.imageSmoothingEnabled = this.smooth
    this.ctx.drawImage(this.small, 0, 0, this.width, this.height)

    if (this.drawCrosshair) this._drawCrosshair()

    const buf = this.canvas.toBuffer('image/jpeg', { quality: this.quality })

    this.lastRenderMs = Number(process.hrtime.bigint() - t0) / 1e6
    this.frameCount++
    return buf
  }

  // ── DDA hierarquico (Amanatides & Woo + salto de sections vazias) ───────────
  /**
   * Lanca um raio e escreve o pixel resultante direto em this.pixels[p..p+3]
   * e a profundidade em this.depth[di].
   */
  _castRay (ox, oy, oz, dx, dy, dz, p, di) {
    const maxDist = this.maxDistance
    const dayFactor = this._dayF
    const skyTop = this._skyTop
    const skyHorizon = this._skyHor
    const underwater = this._underwater
    const isWater = this.isWater
    const isGlass = this.isGlass
    const spriteAlpha = this.spriteAlpha
    const solid = this.solid
    const emissive = this.emissive

    // Componente exatamente zero vira epsilon: elimina Infinity/NaN do caminho
    // quente e permite trocar todas as divisoes por multiplicacoes.
    if (dx === 0) dx = 1e-30
    if (dy === 0) dy = 1e-30
    if (dz === 0) dz = 1e-30
    const invX = 1 / dx
    const invY = 1 / dy
    const invZ = 1 / dz

    let x = Math.floor(ox)
    let y = Math.floor(oy)
    let z = Math.floor(oz)

    const stepX = dx > 0 ? 1 : -1
    const stepY = dy > 0 ? 1 : -1
    const stepZ = dz > 0 ? 1 : -1

    const tDeltaX = invX < 0 ? -invX : invX
    const tDeltaY = invY < 0 ? -invY : invY
    const tDeltaZ = invZ < 0 ? -invZ : invZ

    let tMaxX = (dx > 0 ? (x + 1 - ox) : (x - ox)) * invX
    let tMaxY = (dy > 0 ? (y + 1 - oy) : (y - oy)) * invY
    let tMaxZ = (dz > 0 ? (z + 1 - oz) : (z - oz)) * invZ

    // Acumuladores de meios translucidos
    let waterVoxels = underwater ? 3 : 0
    let tintR = 0, tintG = 0, tintB = 0, tintA = 0
    let spriteA = 0   // parcela de tintA vinda de sprites (teto em SPRITE_MAX_ALPHA)

    let t = 0
    let face = FACE_Y
    let first = true

    // Grade de chunks + cache de coluna/section correntes (inline: este e o
    // caminho mais quente do renderer, cada chamada de metodo aqui pesa)
    const cols = this._cols
    const span = this._span
    const baseCx = this._baseCx
    const baseCz = this._baseCz
    const gen = this._gen

    let colSecs = undefined
    let colCx = 0x7fffffff, colCz = 0x7fffffff
    let secData = null
    let secSx = 0x7fffffff, secSy = 0x7fffffff, secSz = 0x7fffffff

    const maxSteps = 4096

    for (let s = 0; s < maxSteps; s++) {
      const sx = x >> 4, sy = y >> 4, sz = z >> 4
      if (sx !== secSx || sy !== secSy || sz !== secSz) {
        secSx = sx; secSy = sy; secSz = sz

        if (sx !== colCx || sz !== colCz) {
          colCx = sx; colCz = sz
          const ci = sx - baseCx
          const cj = sz - baseCz
          if (ci < 0 || cj < 0 || ci >= span || cj >= span) colSecs = undefined
          else {
            const col = cols[ci * span + cj]
            colSecs = col === null ? undefined : col.sections
          }
        }

        if (colSecs === undefined || y < 0 || y > 255) {
          secData = null
        } else {
          const sec = colSecs[sy]
          if (!sec) secData = null
          else {
            if (sec.__vrGen !== gen) this._initSection(sec, gen)
            secData = sec.__vrEmpty ? null : sec.__vrU8
          }
        }
      }

      if (secData === null) {
        // ── Section vazia/ausente: salta direto para a saida dela ───────────
        const bx = sx << 4, by = sy << 4, bz = sz << 4
        const tex = (dx > 0 ? (bx + 16 - ox) : (bx - ox)) * invX
        const tey = (dy > 0 ? (by + 16 - oy) : (by - oy)) * invY
        const tez = (dz > 0 ? (bz + 16 - oz) : (bz - oz)) * invZ
        let te = tex
        face = FACE_X
        if (tey < te) { te = tey; face = FACE_Y }
        if (tez < te) { te = tez; face = FACE_Z }

        t = te + 1e-4
        if (t > maxDist) break
        first = false

        x = Math.floor(ox + dx * t)
        y = Math.floor(oy + dy * t)
        z = Math.floor(oz + dz * t)
        if (y < 0 || y > 255) break

        tMaxX = (dx > 0 ? (x + 1 - ox) : (x - ox)) * invX
        tMaxY = (dy > 0 ? (y + 1 - oy) : (y - oy)) * invY
        tMaxZ = (dz > 0 ? (z + 1 - oz) : (z - oz)) * invZ
        continue
      }

      // ── Le o bloco atual ───────────────────────────────────────────────────
      if (!first) {
        const o = ((x & 15) | ((z & 15) << 4) | ((y & 15) << 8)) << 1
        const state = secData[o] | (secData[o + 1] << 8)

        if (state !== 0) {
          const id = state >> 4
          if (id > 0 && id < 256) {
            const meta = state & 15

            if (isWater[id] === 1) {
              waterVoxels++
            } else if (isGlass[id] === 1) {
              const a = 0.22 * (1 - tintA)
              const c = this._faceColor(id, meta, face, dy)
              tintR += c[0] * a; tintG += c[1] * a; tintB += c[2] * a; tintA += a
              // Meio ficou opaco: a cor É a tinta acumulada. Sair pelo caminho
              // do ceu aqui pintaria o pixel de ceu esverdeado.
              if (tintA > 0.9) {
                this._writeFogged(p, di, tintR / tintA, tintG / tintA, tintB / tintA,
                  t, underwater, dayFactor, skyHorizon)
                return
              }
            } else if (spriteAlpha[id] > 0 && spriteA < SPRITE_MAX_ALPHA) {
              // Sprites sao cruzes finas no meio do voxel, nao cubos cheios.
              // Sem este teste um raio rasante atravessa dezenas de moitas de
              // grama e o campo inteiro vira uma parede verde opaca.
              const tExit = tMaxX < tMaxY
                ? (tMaxX < tMaxZ ? tMaxX : tMaxZ)
                : (tMaxY < tMaxZ ? tMaxY : tMaxZ)
              const tm = (t + tExit) * 0.5
              const hx = ox + dx * tm - x - 0.5
              const hz = oz + dz * tm - z - 0.5
              if (hx * hx + hz * hz < SPRITE_RADIUS_SQ) {
                const lit = emissive[id] ? 1 : this._litAt(x, y, z)
                let a = spriteAlpha[id] * (1 - tintA)
                const room = SPRITE_MAX_ALPHA - spriteA
                if (a > room) a = room
                const c = this._faceColor(id, meta, FACE_Y, -1)
                tintR += c[0] * lit * a; tintG += c[1] * lit * a; tintB += c[2] * lit * a
                tintA += a
                spriteA += a
              }
            } else if (solid[id] === 1) {
              // ── Superficie opaca atingida ─────────────────────────────────
              const base = this._faceColor(id, meta, face, dy)

              const shade = face === FACE_Y
                ? (dy < 0 ? SHADE_TOP : SHADE_BOTTOM)
                : (face === FACE_X ? SHADE_X : SHADE_Z)

              // Ruido determinista por voxel+face → "grao" que o encoder le
              const grain = 0.90 + 0.20 * hash3(x * 3 + face, y * 5 + face, z * 7 + face)

              // Voxel imediatamente anterior ao acerto = um passo atras pela
              // face de entrada (vale tambem apos um salto de section)
              const lit = emissive[id] ? 1 : this._litAt(
                face === FACE_X ? x - stepX : x,
                face === FACE_Y ? y - stepY : y,
                face === FACE_Z ? z - stepZ : z)
              const k = shade * grain * lit

              let r = base[0] * k
              let g = base[1] * k
              let b = base[2] * k

              if (waterVoxels > 0) {
                const wa = Math.min(0.82, 1 - Math.exp(-0.20 * waterVoxels))
                const wd = 0.3 + 0.7 * dayFactor
                r = r * (1 - wa) + 38 * wa * wd
                g = g * (1 - wa) + 80 * wa * wd
                b = b * (1 - wa) + 165 * wa * wd
              }
              if (tintA > 0) {
                r = r * (1 - tintA) + tintR
                g = g * (1 - tintA) + tintG
                b = b * (1 - tintA) + tintB
              }

              // Neblina por distancia
              const fogStart = underwater ? 2 : maxDist * 0.65
              const fogEnd = underwater ? 16 : maxDist
              const fog = clamp01((t - fogStart) / (fogEnd - fogStart))
              if (fog > 0) {
                const fr = underwater ? 30 * dayFactor : skyHorizon[0]
                const fg = underwater ? 70 * dayFactor : skyHorizon[1]
                const fb = underwater ? 140 * dayFactor : skyHorizon[2]
                r = r * (1 - fog) + fr * fog
                g = g * (1 - fog) + fg * fog
                b = b * (1 - fog) + fb * fog
              }

              this._write(p, di, r, g, b, t)
              return
            }
          }
        }
      }
      first = false

      // ── Avanca um voxel ────────────────────────────────────────────────────
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) { t = tMaxX; x += stepX; tMaxX += tDeltaX; face = FACE_X }
        else { t = tMaxZ; z += stepZ; tMaxZ += tDeltaZ; face = FACE_Z }
      } else {
        if (tMaxY < tMaxZ) {
          t = tMaxY; y += stepY; tMaxY += tDeltaY; face = FACE_Y
          // stepY e constante, entao sair de [0,255] e definitivo
          if (y < 0 || y > 255) break
        } else { t = tMaxZ; z += stepZ; tMaxZ += tDeltaZ; face = FACE_Z }
      }
      if (t > maxDist) break
    }

    // ── Nada atingido: ceu ──────────────────────────────────────────────────
    const h = clamp01(dy * 1.6 + 0.12)
    let r = skyHorizon[0] + (skyTop[0] - skyHorizon[0]) * h
    let g = skyHorizon[1] + (skyTop[1] - skyHorizon[1]) * h
    let b = skyHorizon[2] + (skyTop[2] - skyHorizon[2]) * h

    // Olhando para baixo sem chao carregado → void escuro
    if (dy < -0.25) {
      const k = clamp01((-dy - 0.25) / 0.5)
      r = r * (1 - k) + 12 * k
      g = g * (1 - k) + 12 * k
      b = b * (1 - k) + 16 * k
    }

    if (waterVoxels > 0) {
      const wa = Math.min(0.9, 1 - Math.exp(-0.20 * waterVoxels))
      const wd = 0.3 + 0.7 * dayFactor
      r = r * (1 - wa) + 38 * wa * wd
      g = g * (1 - wa) + 80 * wa * wd
      b = b * (1 - wa) + 165 * wa * wd
    }
    if (tintA > 0) {
      r = r * (1 - tintA) + tintR
      g = g * (1 - tintA) + tintG
      b = b * (1 - tintA) + tintB
    }

    this._write(p, di, r, g, b, Infinity)
  }

  /** Escreve o pixel aplicando a neblina por distancia. */
  _writeFogged (p, di, r, g, b, t, underwater, dayFactor, skyHorizon) {
    const maxDist = this.maxDistance
    const fogStart = underwater ? 2 : maxDist * 0.65
    const fogEnd = underwater ? 16 : maxDist
    const fog = clamp01((t - fogStart) / (fogEnd - fogStart))
    if (fog > 0) {
      const fr = underwater ? 30 * dayFactor : skyHorizon[0]
      const fg = underwater ? 70 * dayFactor : skyHorizon[1]
      const fb = underwater ? 140 * dayFactor : skyHorizon[2]
      r = r * (1 - fog) + fr * fog
      g = g * (1 - fog) + fg * fog
      b = b * (1 - fog) + fb * fog
    }
    this._write(p, di, r, g, b, t)
  }

  _write (p, di, r, g, b, t) {
    const px = this.pixels
    px[p] = r < 0 ? 0 : (r > 255 ? 255 : r)
    px[p + 1] = g < 0 ? 0 : (g > 255 ? 255 : g)
    px[p + 2] = b < 0 ? 0 : (b > 255 ? 255 : b)
    px[p + 3] = 255
    this.depth[di] = t
  }

  _faceColor (id, meta, face, dy) {
    const c = this._colBuf || (this._colBuf = new Float64Array(3))

    const variants = (face === FACE_Y && dy < 0) ? this.variantTop[id] : this.variantSide[id]
    if (variants) {
      const idx = (id === 17 || id === 162) ? (meta & 3) : meta
      const vc = variants[idx % variants.length]
      if (vc) { c[0] = vc[0]; c[1] = vc[1]; c[2] = vc[2]; return c }
    }

    let arr
    if (face === FACE_Y) arr = dy < 0 ? this.colTop : this.colBottom
    else arr = this.colSide

    const o = id * 3
    c[0] = arr[o]; c[1] = arr[o + 1]; c[2] = arr[o + 2]
    return c
  }

  // ── Entidades (billboards com teste de profundidade) ────────────────────────
  _renderEntities (dayFactor) {
    const bot = this.bot
    if (!bot.entities) return

    const cam = this.camera
    const rw = this.rw, rh = this.rh
    const px = this.pixels, depth = this.depth
    const maxDist = this.maxDistance
    const skyH = this._sky

    const list = []
    for (const id in bot.entities) {
      const e = bot.entities[id]
      if (!e || e === bot.entity || !e.position) continue
      const relX = e.position.x - cam.ox
      const relZ = e.position.z - cam.oz
      const eh = e.height || 1.8
      const relY = (e.position.y + eh * 0.5) - cam.oy
      const cz = relX * cam.fx + relY * cam.fy + relZ * cam.fz
      if (cz <= 0.25 || cz > maxDist) continue
      list.push({ e, relX, relY, relZ, cz })
    }
    // Desenha do mais distante para o mais proximo
    list.sort((a, b) => b.cz - a.cz)

    for (const it of list) {
      const { e, relX, relY, relZ, cz } = it
      const cx = relX * cam.rx + relY * cam.ry + relZ * cam.rz
      const cyv = relX * cam.ux + relY * cam.uy + relZ * cam.uz

      const sx = (cx / (cz * this.tanHalf * this.aspect) * 0.5 + 0.5) * rw
      const sy = (0.5 - cyv / (cz * this.tanHalf)) * rh

      const ew = e.width || 0.6
      const eh = e.height || 1.8
      const hw = (ew * 0.5) / (cz * this.tanHalf * this.aspect) * 0.5 * rw
      const hh = (eh * 0.5) / (cz * this.tanHalf) * 0.5 * rh
      if (hw < 0.4 || hh < 0.4) continue

      const x0 = Math.max(0, Math.floor(sx - hw))
      const x1 = Math.min(rw - 1, Math.ceil(sx + hw))
      const y0 = Math.max(0, Math.floor(sy - hh))
      const y1 = Math.min(rh - 1, Math.ceil(sy + hh))
      if (x1 < x0 || y1 < y0) continue

      const name = (e.name || e.displayName || e.type || 'unknown').toLowerCase()
      const col = ENTITY_COLORS[name] ||
        ENTITY_COLORS[e.type] ||
        (e.type === 'player' ? ENTITY_COLORS.player : [200, 60, 160])

      const lit = this._litAt(
        Math.floor(e.position.x), Math.floor(e.position.y + 0.5), Math.floor(e.position.z))

      const fogStart = maxDist * 0.45
      const fog = clamp01((cz - fogStart) / (maxDist - fogStart))

      for (let y = y0; y <= y1; y++) {
        const edgeY = (y === y0 || y === y1)
        for (let x = x0; x <= x1; x++) {
          const idx = y * rw + x
          if (cz >= depth[idx]) continue
          const edge = edgeY || x === x0 || x === x1
          const k = edge ? 0.45 : 1.0
          // leve variacao vertical (cabeca mais clara que o corpo)
          const bodyK = (y - y0) / Math.max(1, y1 - y0) < 0.28 ? 1.12 : 0.95
          let r = col[0] * lit * k * bodyK
          let g = col[1] * lit * k * bodyK
          let b = col[2] * lit * k * bodyK
          r = r * (1 - fog) + skyH[0] * fog
          g = g * (1 - fog) + skyH[1] * fog
          b = b * (1 - fog) + skyH[2] * fog
          const p = idx * 4
          px[p] = r > 255 ? 255 : r
          px[p + 1] = g > 255 ? 255 : g
          px[p + 2] = b > 255 ? 255 : b
          depth[idx] = cz
        }
      }
    }
  }

  _drawCrosshair () {
    const ctx = this.ctx
    const cx = this.width / 2
    const cy = this.height / 2
    const len = Math.round(this.height * 0.022)
    ctx.save()
    ctx.globalCompositeOperation = 'difference'
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(cx - len, cy); ctx.lineTo(cx + len, cy)
    ctx.moveTo(cx, cy - len); ctx.lineTo(cx, cy + len)
    ctx.stroke()
    ctx.restore()
  }

  /**
   * Diagnostico: refaz o raio de um pixel da imagem de SAIDA e lista todos os
   * blocos nao-ar que ele atravessa. Usado para conferir o que o renderer
   * desenhou contra a verdade do mundo.
   * @returns {object} direcao do raio e blocos encontrados
   */
  trace (outX, outY) {
    const bot = this.bot
    if (!bot || !bot.entity) return { error: 'bot nao pronto' }

    this._buildLitLut(this._dayFactor())
    this._refreshChunkGrid(bot.entity.position.x, bot.entity.position.z)

    const i = Math.max(0, Math.min(this.rw - 1, Math.round(outX / this.width * this.rw)))
    const j = Math.max(0, Math.min(this.rh - 1, Math.round(outY / this.height * this.rh)))

    const pos = bot.entity.position
    const eyeH = (bot.entity.height ? bot.entity.height * 0.9 : 1.62)
    const ox = pos.x, oy = pos.y + eyeH, oz = pos.z
    const cp = Math.cos(bot.entity.pitch), sp = Math.sin(bot.entity.pitch)
    const cy = Math.cos(bot.entity.yaw), sy = Math.sin(bot.entity.yaw)
    const fx = -sy * cp, fy = sp, fz = -cy * cp
    const rx = cy, rz = -sy
    const ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy

    const u = this.offU[i], v = this.offV[j]
    let dx = fx + rx * u + ux * v
    let dy = fy + uy * v
    let dz = fz + rz * u + uz * v
    const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz)
    dx *= inv; dy *= inv; dz *= inv

    const names = this.registry.blocks
    const hits = []
    let x = Math.floor(ox), y = Math.floor(oy), z = Math.floor(oz)
    const stepX = dx > 0 ? 1 : -1, stepY = dy > 0 ? 1 : -1, stepZ = dz > 0 ? 1 : -1
    const iX = 1 / (dx || 1e-30), iY = 1 / (dy || 1e-30), iZ = 1 / (dz || 1e-30)
    let tMaxX = (dx > 0 ? x + 1 - ox : x - ox) * iX
    let tMaxY = (dy > 0 ? y + 1 - oy : y - oy) * iY
    let tMaxZ = (dz > 0 ? z + 1 - oz : z - oz) * iZ
    const tdX = Math.abs(iX), tdY = Math.abs(iY), tdZ = Math.abs(iZ)
    let t = 0

    for (let s = 0; s < 4096 && t <= this.maxDistance; s++) {
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) { t = tMaxX; x += stepX; tMaxX += tdX } else { t = tMaxZ; z += stepZ; tMaxZ += tdZ }
      } else {
        if (tMaxY < tMaxZ) { t = tMaxY; y += stepY; tMaxY += tdY } else { t = tMaxZ; z += stepZ; tMaxZ += tdZ }
      }
      if (t > this.maxDistance || y < 0 || y > 255) break
      const state = this._stateAt(x, y, z)
      if (state === 0) continue
      const id = state >> 4
      if (id === 0 || id >= 256) continue
      const kind = this.isWater[id] ? 'agua'
        : this.isGlass[id] ? 'vidro'
          : this.spriteAlpha[id] > 0 ? 'sprite'
            : this.solid[id] ? 'SOLIDO' : 'ignorado'
      hits.push({
        d: +t.toFixed(1),
        pos: [x, y, z],
        block: names[id] ? names[id].name : `id${id}`,
        tratado_como: kind
      })
      if (kind === 'SOLIDO') break
      if (hits.length >= 40) break
    }

    return {
      pixel_saida: [outX, outY],
      pixel_interno: [i, j],
      direcao: { dx: +dx.toFixed(3), dy: +dy.toFixed(3), dz: +dz.toFixed(3) },
      olho: { x: +ox.toFixed(2), y: +oy.toFixed(2), z: +oz.toFixed(2) },
      blocos_no_caminho: hits
    }
  }

  /** Pronto quando o chunk sob o bot ja chegou. */
  isReady () {
    const bot = this.bot
    if (!bot || !bot.entity || !bot.world) return false
    const p = bot.entity.position
    return !!bot.world.getColumn(Math.floor(p.x) >> 4, Math.floor(p.z) >> 4)
  }

  get stats () {
    return {
      renderer: 'voxel-raycast',
      internal: `${this.rw}x${this.rh}`,
      output: `${this.width}x${this.height}`,
      last_render_ms: Number(this.lastRenderMs.toFixed(2)),
      frames: this.frameCount
    }
  }
}

module.exports = { VoxelRenderer }
