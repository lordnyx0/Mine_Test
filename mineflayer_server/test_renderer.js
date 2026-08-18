/**
 * test_renderer.js — Benchmark + smoke test do VoxelRenderer.
 *
 *   node test_renderer.js
 */
'use strict'

const fs = require('fs')
const path = require('path')
const { VoxelRenderer } = require('./voxel_renderer')
const { bot, columns, heightAt } = require('./test_world')

const OUT_DIR = path.join(__dirname, 'render_test')
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR)

console.log(`[test] Mundo sintetico: ${columns.size} chunks.`)

// ── Render ────────────────────────────────────────────────────────────────────
function bench (label, opts, frames = 40) {
  const r = new VoxelRenderer(bot, opts)
  // aquecimento
  for (let i = 0; i < 5; i++) { bot.entity.yaw = i * 0.1; r.render() }

  const times = []
  let buf = null
  for (let i = 0; i < frames; i++) {
    bot.entity.yaw = (i / frames) * Math.PI * 2
    bot.entity.pitch = Math.sin(i * 0.2) * 0.25
    const t0 = process.hrtime.bigint()
    buf = r.render()
    times.push(Number(process.hrtime.bigint() - t0) / 1e6)
  }
  times.sort((a, b) => a - b)
  const avg = times.reduce((a, b) => a + b, 0) / times.length
  console.log(
    `  ${label.padEnd(26)} avg=${avg.toFixed(2)}ms  p50=${times[frames >> 1].toFixed(2)}ms  ` +
    `p95=${times[Math.floor(frames * 0.95)].toFixed(2)}ms  max=${times[frames - 1].toFixed(2)}ms  ` +
    `${(buf.length / 1024).toFixed(1)}KB  (${(1000 / avg).toFixed(0)} fps)`
  )
  return { r, buf, avg }
}

console.log('\n[test] Benchmark (640x360 de saida):')
bench('scale=8  (80x45)', { scale: 8 })
bench('scale=6  (107x60)', { scale: 6 })
const mid = bench('scale=4  (160x90)', { scale: 4 })
bench('scale=3  (213x120)', { scale: 3 })
bench('scale=2  (320x180)', { scale: 2 })
bench('scale=1  (640x360)', { scale: 1 }, 15)

// ── Amostras visuais ──────────────────────────────────────────────────────────
console.log('\n[test] Salvando amostras em ./render_test ...')
const shots = [
  ['01_horizonte', { yaw: 0, pitch: 0, time: 6000 }],
  ['02_olhando_baixo', { yaw: 0.6, pitch: -0.7, time: 6000 }],
  ['03_olhando_cima', { yaw: 0.6, pitch: 0.8, time: 6000 }],
  ['04_torre', { yaw: -Math.atan2(19, 25), pitch: 0.15, time: 6000 }],
  ['05_noite', { yaw: 0, pitch: 0, time: 18000 }],
  ['06_entardecer', { yaw: 0, pitch: 0, time: 12800 }],
  ['07_glowstone_noite', { yaw: -Math.atan2(6, 6) - Math.PI, pitch: 0, time: 18000 }]
]
const shotR = new VoxelRenderer(bot, { scale: 4 })
for (const [name, s] of shots) {
  bot.entity.yaw = s.yaw
  bot.entity.pitch = s.pitch
  bot.time.timeOfDay = s.time
  const buf = shotR.render()
  fs.writeFileSync(path.join(OUT_DIR, name + '.jpg'), buf)
  console.log(`  ${name}.jpg  ${(buf.length / 1024).toFixed(1)}KB  ${shotR.lastRenderMs.toFixed(2)}ms`)
}

// Comparativo de escalas na mesma cena
bot.entity.yaw = 0; bot.entity.pitch = 0; bot.time.timeOfDay = 6000
for (const sc of [1, 2, 4, 8]) {
  const r = new VoxelRenderer(bot, { scale: sc })
  fs.writeFileSync(path.join(OUT_DIR, `scale_${sc}.jpg`), r.render())
}

console.log('\n[test] OK. Recomendado: scale=4 →', mid.avg.toFixed(2), 'ms/frame')
