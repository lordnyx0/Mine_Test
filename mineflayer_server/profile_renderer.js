/**
 * profile_renderer.js — Quebra o custo do frame por estagio.
 *
 *   node profile_renderer.js
 */
'use strict'

const { VoxelRenderer } = require('./voxel_renderer')
const { bot } = require('./test_world')

const N = 40

function timeIt (r) {
  const t = process.hrtime.bigint()
  for (let i = 0; i < N; i++) { bot.entity.yaw = (i / N) * 6.283; r.render() }
  return Number(process.hrtime.bigint() - t) / 1e6 / N
}

function stages (label, opts) {
  const r = new VoxelRenderer(bot, opts)
  for (let i = 0; i < 8; i++) { bot.entity.yaw = i * 0.3; r.render() }

  const total = timeIt(r)

  const origToBuffer = r.canvas.toBuffer
  r.canvas.toBuffer = () => Buffer.alloc(0)
  const noJpeg = timeIt(r)

  const origDraw = r.ctx.drawImage
  r.ctx.drawImage = () => {}
  const origCross = r.drawCrosshair
  r.drawCrosshair = false
  const rayOnly = timeIt(r)

  r.canvas.toBuffer = origToBuffer
  r.ctx.drawImage = origDraw
  r.drawCrosshair = origCross

  const rays = r.rw * r.rh
  console.log(
    `  ${label.padEnd(20)} total=${total.toFixed(2)}ms  ` +
    `[raycast ${rayOnly.toFixed(2)} | upscale+hud ${(noJpeg - rayOnly).toFixed(2)} | jpeg ${(total - noJpeg).toFixed(2)}]  ` +
    `${(rayOnly * 1000 / rays).toFixed(3)}us/raio`
  )
}

console.log('\n[prof] Saida 640x360:')
for (const s of [3, 4, 5, 6, 8]) stages(`scale=${s}`, { scale: s })

console.log('\n[prof] Saida 320x180 (SigLIP redimensiona pra 224 de qualquer jeito):')
for (const s of [2, 3, 4]) stages(`scale=${s}`, { width: 320, height: 180, scale: s })

console.log('\n[prof] Efeito do alcance (scale=4, 640x360):')
for (const d of [32, 48, 64, 96]) stages(`dist=${d}`, { scale: 4, maxDistance: d })

console.log('\n[prof] Extras (scale=4, 640x360):')
stages('sem entidades', { scale: 4, entities: false })
stages('sem crosshair', { scale: 4, crosshair: false })
stages('upscale bilinear', { scale: 4, smooth: true })
stages('jpeg q=0.6', { scale: 4, quality: 0.6 })
console.log()
