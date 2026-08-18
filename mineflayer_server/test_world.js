/**
 * test_world.js — Mundo sintetico + bot falso para testar o VoxelRenderer
 * sem precisar de um servidor Minecraft.
 *
 *   const { bot, columns, heightAt } = require('./test_world')
 */
'use strict'

const Vec3 = require('vec3')
const registry = require('minecraft-data')('1.8.9')
const Chunk = require('prismarine-chunk')('1.8.9')


// ── Mundo sintetico ───────────────────────────────────────────────────────────
const columns = new Map()
const key = (cx, cz) => cx * 100000 + cz

function getOrCreate (cx, cz) {
  const k = key(cx, cz)
  let c = columns.get(k)
  if (!c) {
    c = new Chunk()
    c.skyLightSent = true
    columns.set(k, c)
  }
  return c
}

const _v = new Vec3(0, 0, 0)
function setBlock (x, y, z, type, meta = 0) {
  if (y < 0 || y > 255) return
  const col = getOrCreate(x >> 4, z >> 4)
  _v.x = x & 15; _v.y = y; _v.z = z & 15
  col.setBlockStateId(_v, (type << 4) | meta)
}
function setLight (x, y, z, sky, blk) {
  const col = getOrCreate(x >> 4, z >> 4)
  _v.x = x & 15; _v.y = y; _v.z = z & 15
  col.setSkyLight(_v, sky)
  col.setBlockLight(_v, blk)
}

const R = 96          // raio do mundo em blocos
const SEA = 62

function heightAt (x, z) {
  return Math.round(
    64 +
    5 * Math.sin(x * 0.07) + 4 * Math.cos(z * 0.06) +
    2.5 * Math.sin((x + z) * 0.14)
  )
}

for (let x = -R; x <= R; x++) {
  for (let z = -R; z <= R; z++) {
    const h = heightAt(x, z)
    for (let y = 0; y <= h; y++) {
      let id
      if (y === 0) id = 7                      // bedrock
      else if (y > h - 1) id = h < SEA ? 12 : 2 // areia sob a agua, senao grama
      else if (y > h - 4) id = 3               // dirt
      else id = 1                              // stone
      setBlock(x, y, z, id)
    }
    // Lago
    for (let y = h + 1; y <= SEA; y++) setBlock(x, y, z, 9)
    // Luz do ceu acima da superficie
    const top = Math.max(h, SEA)
    for (let y = top + 1; y < Math.min(255, top + 24); y++) setLight(x, y, z, 15, 0)
    setLight(x, top, z, 15, 0)
  }
}

// Arvores
let seed = 12345
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff }
for (let i = 0; i < 90; i++) {
  const x = Math.round((rnd() * 2 - 1) * (R - 10))
  const z = Math.round((rnd() * 2 - 1) * (R - 10))
  const h = heightAt(x, z)
  if (h < SEA) continue
  const trunk = 4 + Math.floor(rnd() * 3)
  for (let y = 1; y <= trunk; y++) setBlock(x, h + y, z, 17, 0)
  for (let dx = -2; dx <= 2; dx++) {
    for (let dz = -2; dz <= 2; dz++) {
      for (let dy = 0; dy <= 2; dy++) {
        if (Math.abs(dx) + Math.abs(dz) + dy > 4) continue
        setBlock(x + dx, h + trunk + dy, z + dz, 18, 0)
      }
    }
  }
}

// Grama alta / flores
for (let i = 0; i < 500; i++) {
  const x = Math.round((rnd() * 2 - 1) * (R - 5))
  const z = Math.round((rnd() * 2 - 1) * (R - 5))
  const h = heightAt(x, z)
  if (h < SEA) continue
  setBlock(x, h + 1, z, rnd() < 0.8 ? 31 : (rnd() < 0.5 ? 37 : 38), 1)
}

// Torre de pedra + lã colorida para checar variantes de metadata
for (let y = 1; y <= 14; y++) {
  for (let dx = 0; dx < 3; dx++) {
    for (let dz = 0; dz < 3; dz++) {
      if (dx === 1 && dz === 1) continue
      setBlock(18 + dx, heightAt(18, 24) + y, 24 + dz, y > 9 ? 35 : 98, (y * 2) % 16)
    }
  }
}
// Bloco de glowstone para testar emissividade
setBlock(6, heightAt(6, 6) + 2, 6, 89)
for (let dx = -3; dx <= 3; dx++) {
  for (let dz = -3; dz <= 3; dz++) {
    for (let dy = -2; dy <= 3; dy++) {
      const l = Math.max(0, 15 - (Math.abs(dx) + Math.abs(dz) + Math.abs(dy)) * 2)
      setLight(6 + dx, heightAt(6, 6) + 2 + dy, 6 + dz, 15, l)
    }
  }
}


// ── Bot falso ─────────────────────────────────────────────────────────────────
const startY = heightAt(0, 0) + 1
const bot = {
  version: '1.8.9',
  registry,
  time: { timeOfDay: 6000, isDay: true },
  entity: {
    position: new Vec3(0.5, startY, 0.5),
    yaw: 0,
    pitch: 0,
    height: 1.8,
    width: 0.6
  },
  entities: {
    1: { id: 1, type: 'mob', name: 'zombie', position: new Vec3(3.5, startY, -6.5), height: 1.95, width: 0.6 },
    2: { id: 2, type: 'mob', name: 'cow', position: new Vec3(-4.5, heightAt(-4, -9) + 1, -9.5), height: 1.4, width: 0.9 },
    3: { id: 3, type: 'player', name: 'player', username: 'Steve', position: new Vec3(1.5, heightAt(1, -12) + 1, -12.5), height: 1.8, width: 0.6 }
  },
  world: {
    getColumn (cx, cz) { return columns.get(key(cx, cz)) || null }
  }
}

module.exports = { bot, columns, heightAt, SEA }
