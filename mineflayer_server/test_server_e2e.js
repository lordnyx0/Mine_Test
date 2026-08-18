/**
 * test_server_e2e.js — Testa o state_server.js inteiro sem servidor Minecraft.
 *
 * Injeta um stub de 'mineflayer' no require cache com um bot falso ligado a um
 * mundo sintetico, sobe o state_server e bate nos endpoints reais.
 *
 *   node test_server_e2e.js
 */
'use strict'

const http = require('http')
const fs = require('fs')
const path = require('path')
const { EventEmitter } = require('events')
const Vec3 = require('vec3')
const Chunk = require('prismarine-chunk')('1.8.9')
const registry = require('minecraft-data')('1.8.9')

const PORT = 3591

// ── Mundo sintetico compacto ──────────────────────────────────────────────────
const columns = new Map()
const ckey = (cx, cz) => cx * 100000 + cz
const _v = new Vec3(0, 0, 0)

function setBlock (x, y, z, type, meta = 0) {
  if (y < 0 || y > 255) return
  const k = ckey(x >> 4, z >> 4)
  let c = columns.get(k)
  if (!c) { c = new Chunk(); c.skyLightSent = true; columns.set(k, c) }
  _v.x = x & 15; _v.y = y; _v.z = z & 15
  c.setBlockStateId(_v, (type << 4) | meta)
}
function setSky (x, y, z, v) {
  const c = columns.get(ckey(x >> 4, z >> 4))
  if (!c) return
  _v.x = x & 15; _v.y = y; _v.z = z & 15
  c.setSkyLight(_v, v)
}

const R = 48
const H = 64
for (let x = -R; x <= R; x++) {
  for (let z = -R; z <= R; z++) {
    const h = H + Math.round(3 * Math.sin(x * 0.1) + 3 * Math.cos(z * 0.1))
    for (let y = 0; y <= h; y++) setBlock(x, y, z, y === h ? 2 : (y > h - 4 ? 3 : 1))
    for (let y = h + 1; y < h + 20; y++) setSky(x, y, z, 15)
    setSky(x, h, z, 15)
  }
}
// Uma parede de pedra pra ter algo vertical no enquadramento
for (let dx = -6; dx <= 6; dx++) {
  for (let dy = 1; dy <= 5; dy++) setBlock(dx, H + 3 + dy, -12, 98)
}

// ── Stub do mineflayer ────────────────────────────────────────────────────────
const controls = {}
const fakeBot = new EventEmitter()
Object.assign(fakeBot, {
  version: '1.8.9',
  registry,
  health: 20,
  food: 20,
  time: { timeOfDay: 6000, isDay: true },
  entity: {
    position: new Vec3(0.5, H + 4, 0.5),
    velocity: new Vec3(0, 0, 0),
    yaw: 0,
    pitch: 0,
    onGround: true,
    height: 1.8,
    width: 0.6
  },
  entities: {
    7: { id: 7, type: 'mob', name: 'zombie', position: new Vec3(2.5, H + 4, -7.5), height: 1.95, width: 0.6 }
  },
  world: { getColumn: (cx, cz) => columns.get(ckey(cx, cz)) || null },
  setControlState (k, v) { controls[k] = v }
})

const mineflayerPath = require.resolve('mineflayer', { paths: [__dirname] })
require.cache[mineflayerPath] = {
  id: mineflayerPath,
  filename: mineflayerPath,
  loaded: true,
  exports: {
    createBot () {
      setImmediate(() => fakeBot.emit('spawn'))
      return fakeBot
    }
  }
}

// ── Sobe o servidor real ──────────────────────────────────────────────────────
process.env.HTTP_PORT = String(PORT)
process.env.RENDERER = 'voxel'
process.env.FRAME_SCALE = process.env.FRAME_SCALE || '4'
require('./state_server.js')

// ── Cliente ───────────────────────────────────────────────────────────────────
function get (url) {
  return new Promise((resolve, reject) => {
    http.get(`http://127.0.0.1:${PORT}${url}`, res => {
      const chunks = []
      res.on('data', c => chunks.push(c))
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }))
    }).on('error', reject)
  })
}
function post (url, obj) {
  return new Promise((resolve, reject) => {
    const body = Buffer.from(JSON.stringify(obj))
    const req = http.request({
      host: '127.0.0.1', port: PORT, path: url, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': body.length }
    }, res => {
      const chunks = []
      res.on('data', c => chunks.push(c))
      res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks) }))
    })
    req.on('error', reject)
    req.end(body)
  })
}

let failures = 0
function check (name, cond, extra = '') {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  — ' + extra : ''}`)
  if (!cond) failures++
}

const sleep = ms => new Promise(r => setTimeout(r, ms))

;(async () => {
  await sleep(400)
  console.log('\n[e2e] Endpoints:')

  // /health
  let r = await get('/health')
  let h = JSON.parse(r.body)
  check('/health responde 200', r.status === 200)
  check('/health ok=true', h.ok === true)
  check('/health renderer=voxel', h.renderer === 'voxel')

  // Espera o renderer ficar pronto
  for (let i = 0; i < 60 && !h.viewer_ready; i++) {
    await sleep(150)
    h = JSON.parse((await get('/health')).body)
  }
  check('/health viewer_ready=true', h.viewer_ready === true)

  // /frame
  r = await get('/frame')
  check('/frame responde 200', r.status === 200, `status=${r.status}`)
  check('/frame Content-Type image/jpeg', r.headers['content-type'] === 'image/jpeg')
  check('/frame e um JPEG valido', r.body[0] === 0xFF && r.body[1] === 0xD8 && r.body[r.body.length - 2] === 0xFF && r.body[r.body.length - 1] === 0xD9,
    `${(r.body.length / 1024).toFixed(1)}KB`)
  check('/frame traz X-Render-Ms', r.headers['x-render-ms'] != null, `${r.headers['x-render-ms']}ms`)
  fs.writeFileSync(path.join(__dirname, 'render_test', 'e2e_frame.jpg'), r.body)

  // Latencia real ponta a ponta (HTTP incluso)
  const N = 30
  const lat = []
  for (let i = 0; i < N; i++) {
    fakeBot.entity.yaw = (i / N) * Math.PI * 2
    const t0 = process.hrtime.bigint()
    const rr = await get('/frame')
    lat.push(Number(process.hrtime.bigint() - t0) / 1e6)
    if (rr.status !== 200) { check('frame do loop OK', false, `status ${rr.status}`); break }
  }
  lat.sort((a, b) => a - b)
  const avg = lat.reduce((a, b) => a + b, 0) / lat.length
  console.log(`\n[e2e] Latencia HTTP ponta a ponta (${N} frames):`)
  console.log(`      avg=${avg.toFixed(2)}ms  p50=${lat[N >> 1].toFixed(2)}ms  p95=${lat[Math.floor(N * 0.95)].toFixed(2)}ms  max=${lat[N - 1].toFixed(2)}ms  (${(1000 / avg).toFixed(0)} fps)`)
  check('latencia media < 30ms', avg < 30, `${avg.toFixed(2)}ms`)

  // Frames diferentes conforme o bot gira (nao e frame congelado)
  fakeBot.entity.yaw = 0
  const a = (await get('/frame')).body
  fakeBot.entity.yaw = Math.PI / 2
  const b = (await get('/frame')).body
  check('frame muda quando o yaw muda', !a.equals(b))

  // /stats
  r = await get('/stats')
  const st = JSON.parse(r.body)
  check('/stats responde 200', r.status === 200)
  check('/stats conta frames', st.frames > N, `frames=${st.frames}`)
  console.log(`      /stats: avg=${st.avg_render_ms}ms max=${st.max_render_ms}ms detail=${JSON.stringify(st.detail)}`)

  // /state e /delta
  const s = JSON.parse((await get('/state')).body)
  check('/state traz posicao', s.x === 0.5 && s.z === 0.5, `x=${s.x} y=${s.y} z=${s.z}`)
  check('/delta responde 200', (await get('/delta')).status === 200)

  // POST /action
  const pr = await post('/action', { hold: ['W', 'SPACE'], mouse: [40, 0], duration_ms: 50 })
  check('POST /action responde 200', pr.status === 200)
  check('/action ativa forward', controls.forward === true)
  check('/action ativa jump', controls.jump === true)
  check('/action gira a camera', Math.abs(fakeBot.entity.yaw - (Math.PI / 2 + 40 * 0.003)) < 1e-6,
    `yaw=${fakeBot.entity.yaw.toFixed(4)}`)
  await sleep(120)
  check('/action solta as teclas apos duration_ms', controls.forward === false && controls.jump === false)

  // 404
  check('rota desconhecida da 404', (await get('/nope')).status === 404)

  console.log(`\n[e2e] ${failures === 0 ? 'TODOS OS TESTES PASSARAM' : failures + ' TESTE(S) FALHARAM'}`)
  process.exit(failures === 0 ? 0 : 1)
})().catch(e => { console.error('[e2e] erro:', e); process.exit(1) })
