/**
 * servidor_offline.js — N ambientes Minecraft simulados, sem o jogo aberto.
 *
 * Mundo lido dos arquivos de regiao do save; movimento por prismarine-physics
 * (o mesmo motor do mineflayer, medido em 4.06 blocos/s contra ~4.3 do jogo);
 * visao pelo voxel_renderer; navegacao pelo navegacao.js. Nada disso precisou
 * mudar — os tres consomem a mesma interface de bot.
 *
 * A API e em LOTE de proposito. Um passo custa ~15ms de ambiente contra ~300ms
 * de inferencia na GPU: serializar N ambientes em N requisicoes desperdicaria
 * o ganho. Um POST devolve os N frames de uma vez, e o Python bateleia.
 *
 *   POST /lote/reset  {n?, envs?}          -> respawna
 *   POST /lote/passo  {acoes:[{hold,mouse,duration_ms}]}
 *   GET  /lote/info
 *
 *   node servidor_offline.js            (porta 3002, 8 ambientes)
 */
'use strict'

const http = require('http')
const Vec3 = require('vec3')

const { MundoOffline, SAVE_PADRAO } = require('./mundo_offline')
const { BotOffline, TICK_MS } = require('./bot_offline')
const { VoxelRenderer } = require('./voxel_renderer')
const { mapaDeRotas, planejar } = require('./navegacao')
const { Piloto, buscar, Objetivos } = require('./planejador')

const PORTA = parseInt(process.env.PORTA_OFFLINE || '3002')
const N_ENVS = parseInt(process.env.N_ENVS || '8')
const SAVE = process.env.SAVE_PATH || SAVE_PADRAO
const RAIO_CHUNKS = parseInt(process.env.RAIO_CHUNKS || '5')
const MAX_COLUNAS = parseInt(process.env.MAX_COLUNAS || '2600')

const FRAME_W = parseInt(process.env.FRAME_W || '640')
const FRAME_H = parseInt(process.env.FRAME_H || '360')
const FRAME_SCALE = parseInt(process.env.FRAME_SCALE || '4')
const FRAME_DIST = parseFloat(process.env.FRAME_DIST || '64')

// Area de respawn: regioes ja geradas no save
const RESPAWN_RAIO = parseFloat(process.env.RESPAWN_RAIO || '1200')
const SPAWN_X = parseFloat(process.env.SPAWN_X || '143')
const SPAWN_Z = parseFloat(process.env.SPAWN_Z || '240')

const registry = require('minecraft-data')('1.8.9')

const NAO_SOLIDO = new Set([
  'air', 'water', 'flowing_water', 'lava', 'flowing_lava', 'leaves', 'leaves2',
  'tallgrass', 'double_plant', 'yellow_flower', 'red_flower', 'snow_layer',
  'deadbush', 'sapling', 'reeds', 'vine', 'fire', 'web', 'torch'
])

const mundo = new MundoOffline(SAVE)
const ambientes = []

// ── Ambiente ──────────────────────────────────────────────────────────────────
class Ambiente {
  constructor (id) {
    this.id = id
    this.bot = new BotOffline(mundo, registry, new Vec3(SPAWN_X + 0.5, 80, SPAWN_Z + 0.5))
    this.render = new VoxelRenderer(this.bot, {
      width: FRAME_W, height: FRAME_H, scale: FRAME_SCALE, maxDistance: FRAME_DIST
    })
    this.passos = 0
    this.pronto = false
    this.piloto = new Piloto()      // memoria e compromisso por ambiente
    this.alvo = null                // publicado pelo treino, so para o /ver
  }

  /** Procura um ponto com chao solido e espaco livre acima. */
  async respawnar (tentativas = 24, fixo = null) {
    for (let t = 0; t < tentativas; t++) {
      let x, z
      if (fixo) {
        // Suporta [x, z] ou [x, y, z] ou [x, y, z, yaw]
        x = Math.floor(fixo[0])
        z = fixo.length >= 3 ? Math.floor(fixo[2]) : Math.floor(fixo[1])
        if (t > 0) return false          // sem sorteio: uma tentativa so
      } else {
        const ang = Math.random() * Math.PI * 2
        const dist = 60 + Math.random() * RESPAWN_RAIO
        x = Math.floor(SPAWN_X + Math.cos(ang) * dist)
        z = Math.floor(SPAWN_Z + Math.sin(ang) * dist)
      }

      await mundo.preparar(x, z, RAIO_CHUNKS)
      const col = mundo.getColumn(x >> 4, z >> 4)
      if (!col) continue

const LIQUIDOS = new Set(['water', 'flowing_water', 'lava', 'flowing_lava'])

      // topo solido
      let y = -1
      for (let yy = 140; yy > 4; yy--) {
        let b
        try { b = col.getBlock(new Vec3(x & 15, yy, z & 15)) } catch (_) { break }
        if (b && !NAO_SOLIDO.has(b.name) && !LIQUIDOS.has(b.name)) { y = yy + 1; break }
      }
      if (y < 5) continue

      // espaco para o corpo: livre e sem liquidos
      const livreESeco = [0, 1].every(d => {
        try {
          const b = col.getBlock(new Vec3(x & 15, y + d, z & 15))
          return b && NAO_SOLIDO.has(b.name) && !LIQUIDOS.has(b.name)
        } catch (_) { return false }
      })
      if (!livreESeco) continue

      // Entorno de 6 blocos livre de liquidos (garante solo firme longe de rios)
      let temLiquidoPerto = false
      for (let dx = -6; dx <= 6; dx++) {
        for (let dz = -6; dz <= 6; dz++) {
          try {
            const b = col.getBlock(new Vec3((x + dx) & 15, y, (z + dz) & 15))
            if (b && LIQUIDOS.has(b.name)) { temLiquidoPerto = true; break }
          } catch (_) {}
        }
        if (temLiquidoPerto) break
      }
      if (temLiquidoPerto) continue

      this.bot.teleportar(x + 0.5, y + 0.2, z + 0.5)
      this.bot.entity.yaw = Math.random() * Math.PI * 2
      this.bot.entity.pitch = 0
      for (let i = 0; i < 10; i++) this.bot.tick()      // assenta
      this.render._gen++                                // invalida cache de sections
      this.piloto.reiniciar(this.bot)                   // memoria e por episodio
      this.passos = 0
      this.pronto = true
      return true
    }
    return false
  }

  /** Aplica a acao e avanca a fisica. */
  agir (acao) {
    const b = this.bot
    b.soltarTudo()
    const hold = (acao.hold || []).map(k => String(k).toUpperCase())
    if (hold.includes('W')) b.setControlState('forward', true)
    if (hold.includes('S')) b.setControlState('back', true)
    if (hold.includes('A')) b.setControlState('left', true)
    if (hold.includes('D')) b.setControlState('right', true)
    if (hold.includes('SPACE')) b.setControlState('jump', true)
    if (hold.includes('SHIFT')) b.setControlState('sneak', true)

    const [dx, dy] = acao.mouse || [0, 0]
    if (dx || dy) b.girar(dx, dy)

    b.avancar(acao.duration_ms || 250)
    b.soltarTudo()
    this.passos++
  }

  async manterChunks () {
    const p = this.bot.entity.position
    await mundo.preparar(p.x, p.z, RAIO_CHUNKS)
  }

  observar (comFrame = true, comRotas = true, comDiag = false, dir = null) {
    const o = { env: this.id, estado: this.bot.estado, passos: this.passos,
                morreu: this.bot.morreu }
    // Diagnostico opcional: caro (~676 consultas de bloco), so quando pedido.
    if (comDiag) {
      o.diag = {
        agua_perto: this.bot.aguaPerto(6),
        bloco_pes: this.bot.blocoRel(0, 0, 0),
        bloco_abaixo: this.bot.blocoRel(0, -1, 0)
      }
      // Perfil de relevo na direcao pedida (tipicamente rumo ao alvo).
      if (dir) o.diag.perfil = this.bot.perfilNaDirecao(dir[0], dir[1], 12)
    }
    if (comRotas) {
      const m = mapaDeRotas(this.bot)
      o.rotas = m ? m.livre : null
    }
    if (comFrame) {
      const buf = this.render.render()
      o.frame_b64 = buf ? buf.toString('base64') : null
      o.render_ms = +this.render.lastRenderMs.toFixed(1)
    }
    return o
  }
}

// ── HTTP ──────────────────────────────────────────────────────────────────────
function corpo (req) {
  return new Promise(resolve => {
    let s = ''
    req.on('data', c => { s += c })
    req.on('end', () => { try { resolve(JSON.parse(s || '{}')) } catch (_) { resolve({}) } })
  })
}

let blocosTemporariosMundo = []

function limparTodosBlocosTemporarios () {
  if (blocosTemporariosMundo.length === 0) return
  for (const b of blocosTemporariosMundo) {
    const col = mundo.getColumn(b.bx >> 4, b.bz >> 4)
    if (col) {
      for (let h = 0; h < b.altura; h++) {
        const pInCol = new Vec3(b.bx & 15, b.yChao + h, b.bz & 15)
        col.setBlockType(pInCol, 0)
        try { col.setSkyLight(pInCol, 15) } catch (_) {}
      }
    }
  }
  blocosTemporariosMundo = []
  for (const a of ambientes) a.render._gen++
}

const servidor = http.createServer(async (req, res) => {
  res.setHeader('Content-Type', 'application/json')

  try {
    if (req.method === 'GET' && req.url.startsWith('/lote/info')) {
      res.writeHead(200)
      return res.end(JSON.stringify({
        envs: ambientes.length,
        colunas_em_memoria: mundo.tamanho,
        prontos: ambientes.filter(a => a.pronto).length,
        frame: `${FRAME_W}x${FRAME_H}`, tick_ms: TICK_MS
      }))
    }

    if (req.method === 'POST' && req.url.startsWith('/lote/reset')) {
      const b = await corpo(req)
      const alvos = b.envs || ambientes.map(a => a.id)
      const pos = b.posicoes || null
      const t0 = Date.now()
      limparTodosBlocosTemporarios()
      await Promise.all(alvos.map((id, k) =>
        ambientes[id].respawnar(24, pos ? pos[k] : null)))
      mundo.podarLRU(MAX_COLUNAS)
      const obs = alvos.map(i => ambientes[i].observar(true, true))
      res.writeHead(200)
      return res.end(JSON.stringify({ ms: Date.now() - t0, obs }))
    }

    if (req.method === 'POST' && req.url.startsWith('/lote/passo')) {
      const b = await corpo(req)
      const acoes = b.acoes || []
      const t0 = Date.now()

      for (let i = 0; i < ambientes.length && i < acoes.length; i++) {
        if (acoes[i]) ambientes[i].agir(acoes[i])
      }
      await Promise.all(ambientes.map(a => a.manterChunks()))
      mundo.podarLRU(MAX_COLUNAS)
      if (mundo.regioesAbertas > 48) await mundo.limitarRegioes(40)

      const tObs = Date.now()
      const obs = ambientes.map((a, i) => a.observar(b.frames !== false, b.rotas !== false,
                                                    b.diag === true,
                                                    b.dirs && b.dirs[i]))
      res.writeHead(200)
      return res.end(JSON.stringify({
        ms_total: Date.now() - t0, ms_obs: Date.now() - tObs, obs
      }))
    }

    // Acao sugerida pelo PILOTO (planejador com compromisso e memoria).
    // Serve como professor: um POST devolve a acao de todos os ambientes.
    if (req.method === 'POST' && req.url.startsWith('/lote/piloto')) {
      const b = await corpo(req)
      const t0 = Date.now()
      // permite variar o compromisso por requisicao (experimentos)
      if (b.compromisso) for (const a of ambientes) a.piloto.passosCompromisso = b.compromisso
      // `extras` (array) da um alvo POR ambiente. Necessario para avaliar
      // "va ate B": cada ambiente nasce num lugar e tem o seu proprio B, e o
      // `extra` unico aplicaria o mesmo ponto a todos.
      const acoes = ambientes.map((a, i) =>
        a.piloto.passo(a.bot, b.objetivo || 'explorar',
                       (b.extras && b.extras[i]) || b.extra || {}))
      res.writeHead(200)
      return res.end(JSON.stringify({ ms: Date.now() - t0, acoes }))
    }

    // Alvo alcancavel? E QUAO obstruido esta o caminho ate ele?
    //
    // `desvio` = custo do caminho da busca / distancia em linha reta. 1.0 e reta
    // livre; 1.5+ significa que ha algo no meio que obriga a contornar. E a
    // medida que separa "a visao ajudou" de "trigonometria bastou": num alvo
    // com desvio 1.0 o baseline geometrico e otimo, e so em desvio alto existe
    // algo para a visao contribuir.
    if (req.method === 'POST' && req.url.startsWith('/lote/alcancavel')) {
      const b = await corpo(req)
      const raio = b.raio || 16
      const out = ambientes.map((a, i) => {
        const alvo = b.alvos && b.alvos[i]
        if (!alvo) return null
        const p = a.bot.entity.position
        const r = buscar(a.bot, Objetivos.ponto({ x: alvo.x, y: p.y, z: alvo.z }), { raio })
        const resta = Math.hypot(r.alvo.x - alvo.x, r.alvo.z - alvo.z)
        const reta = Math.hypot(alvo.x - p.x, alvo.z - p.z)
        return {
          alcancavel: resta < 1.6,
          resta: +resta.toFixed(2),
          custo: r.custo,
          reta: +reta.toFixed(2),
          desvio: reta > 0.5 ? +(r.custo / reta).toFixed(2) : null
        }
      })
      res.writeHead(200)
      return res.end(JSON.stringify({ alvos: out }))
    }

    // O treino publica os alvos aqui para que o /ver possa mostrar se o bot
    // esta indo na direcao certa. Sem isso o visualizador mostra um boneco
    // andando e nao ha como julgar, a olho, se ele navega ou so se move.
    if (req.method === 'POST' && req.url.startsWith('/lote/alvos')) {
      const b = await corpo(req)
      ambientes.forEach((a, i) => { a.alvo = (b.alvos && b.alvos[i]) || null })
      res.writeHead(200)
      return res.end(JSON.stringify({ ok: true }))
    }

    // Colocar blocos em posicoes especificas (ex: pilar de bloco colorido no CHAO SOLIDO real)
    if (req.method === 'POST' && req.url.startsWith('/lote/colocar_bloco')) {
      const b = await corpo(req)
      const alvos = b.blocos || []
      const resultados = []

      // 1. Limpa TODOS os pilares temporários anteriores do mundo inteiro
      limparTodosBlocosTemporarios()

      // 2. Coloca os novos pilares da tarefa atual
      for (const item of alvos) {
        const envId = item.env || 0
        const a = ambientes[envId]
        if (!a) continue

        const id = item.id || 49
        const data = item.data || 0
        const bx = Math.floor(item.x)
        const bz = Math.floor(item.z)

        await mundo.preparar(bx, bz, 2)
        const col = a.bot.world.getColumn(bx >> 4, bz >> 4)

        let yChao = item.y !== undefined ? Math.floor(item.y) : Math.floor(a.bot.entity.position.y)
        if (col) {
          for (let yy = 130; yy > 4; yy--) {
            let blk
            try { blk = col.getBlock(new Vec3(bx & 15, yy, bz & 15)) } catch (_) { break }
            if (blk && !NAO_SOLIDO.has(blk.name)) {
              yChao = yy + 1
              break
            }
          }

          const alturaPilar = item.altura || 50
          for (let h = 0; h < alturaPilar; h++) {
            const pInCol = new Vec3(bx & 15, yChao + h, bz & 15)
            col.setBlockType(pInCol, id)
            if (data) col.setBlockData(pInCol, data)
            try { col.setSkyLight(pInCol, 15) } catch (_) {}
          }
          blocosTemporariosMundo.push({ bx, bz, yChao, altura: alturaPilar })
          resultados.push({ env: envId, x: bx, y: yChao, z: bz, id, altura: alturaPilar })
        }
      }
      for (const a of ambientes) a.render._gen++
      res.writeHead(200)
      return res.end(JSON.stringify({ ok: true, blocos: resultados }))
    }

    // ── Visualizacao ────────────────────────────────────────────────────────
    // Depurar locomocao por numero agregado e caro: tres hipoteses (memoria de
    // visitas, agua, elevacao) foram refutadas por medicao sem que nenhuma
    // revelasse o bloqueio real. Ver o bot empacar responde em segundos.
    if (req.method === 'GET' && req.url.startsWith('/lote/frame')) {
      const q = new URL(req.url, 'http://x').searchParams
      const i = Math.min(ambientes.length - 1, Math.max(0, parseInt(q.get('env') || '0')))
      const buf = ambientes[i].render.render()
      if (!buf) { res.writeHead(503); return res.end(JSON.stringify({ erro: 'sem frame' })) }
      res.writeHead(200, { 'Content-Type': 'image/jpeg', 'Cache-Control': 'no-store' })
      return res.end(buf)
    }

    if (req.method === 'GET' && req.url.startsWith('/lote/estado')) {
      res.writeHead(200)
      return res.end(JSON.stringify({
        envs: ambientes.map(a => {
          const o = {
            env: a.id, passos: a.passos, morreu: a.bot.morreu,
            estado: a.bot.estado, agua_perto: a.bot.aguaPerto(4)
          }
          if (a.alvo) {
            const p = a.bot.entity.position
            const yaw = a.bot.entity.yaw
            const fx = -Math.sin(yaw), fz = -Math.cos(yaw)
            const rx = a.alvo.x - p.x, rz = a.alvo.z - p.z
            o.alvo = {
              dist: +Math.hypot(rx, rz).toFixed(1),
              // Positivo = alvo a ESQUERDA, mesma convencao de erro_angular
              graus: +(Math.atan2(rx * (-fz) + rz * fx, rx * fx + rz * fz)
                       * 180 / Math.PI).toFixed(0)
            }
          }
          return o
        })
      }))
    }

    // Painel: recarrega os frames sozinho. Um <img> por ambiente, sem
    // dependencia externa (o navegador ja faz tudo que e preciso).
    if (req.method === 'GET' && (req.url === '/' || req.url.startsWith('/ver'))) {
      const n = ambientes.length
      const cels = Array.from({ length: n }, (_, i) =>
        `<figure><img id="f${i}" src="/lote/frame?env=${i}"><figcaption id="c${i}">env ${i}</figcaption></figure>`).join('')
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
      return res.end(`<!doctype html><meta charset=utf-8><title>Simulador</title>
<style>body{background:#111;color:#ddd;font:13px system-ui;margin:12px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}
figure{margin:0}img{width:100%;border-radius:6px;display:block;background:#000}
figcaption{padding:4px 2px;font-variant-numeric:tabular-nums;color:#9bd}</style>
<h3>Simulador offline — ${n} ambientes</h3><main>${cels}</main><script>
async function tick(){
  try{
    const r = await fetch('/lote/estado'); const d = await r.json();
    for(const e of d.envs){
      const s = e.estado;
      let txt = \`env \${e.env} | y \${s.y.toFixed(0)} | passos \${e.passos}\`;
      if (e.alvo) {
        const g = e.alvo.graus, a = Math.abs(g);
        // Seta no referencial do BOT: para cima = alvo a frente.
        const seta = a < 22 ? '\u2191' : a > 158 ? '\u2193'
                   : g > 0 ? (a < 68 ? '\u2196' : '\u2190')
                           : (a < 68 ? '\u2197' : '\u2192');
        txt += \` | ALVO \${seta} \${e.alvo.dist}m \${g>0?'+':''}\${g}\u00b0\`;
      }
      txt += (s.in_water ? ' | NA AGUA' : '') + (e.morreu ? ' | MORREU' : '');
      const cap = document.getElementById('c'+e.env);
      cap.textContent = txt;
      // Verde quando aponta para o alvo, ambar quando desalinhado
      cap.style.color = e.alvo ? (Math.abs(e.alvo.graus) < 30 ? '#8fd18f' : '#e0a052') : '';
      document.getElementById('f'+e.env).src = '/lote/frame?env='+e.env+'&t='+Date.now();
    }
  }catch(_){}
  setTimeout(tick, 700);
}
tick();
</script>`)
    }

    // ── Radar 2D Top-Down com Histórico de Todas as Runs ────────────────────
    if (req.method === 'GET' && (req.url.startsWith('/topview') || req.url.startsWith('/mapa'))) {
      const n = ambientes.length
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
      return res.end(`<!doctype html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Top-Down Radar — Histórico de Trajetórias</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0b0f19; color: #f8fafc; margin: 0; padding: 16px; }
        header { text-align: center; margin-bottom: 16px; }
        h1 { color: #38bdf8; margin: 0 0 4px 0; font-size: 22px; }
        p.subtitle { color: #94a3b8; margin: 0; font-size: 13px; }
        .toolbar { display: flex; justify-content: center; align-items: center; gap: 15px; margin: 12px 0; flex-wrap: wrap; background: #1e293b; padding: 8px 16px; border-radius: 8px; border: 1px solid #334155; }
        .btn { background: #0284c7; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 12px; }
        .btn:hover { background: #0369a1; }
        label { font-size: 13px; color: #cbd5e1; cursor: pointer; display: flex; align-items: center; gap: 5px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; max-width: 1500px; margin: 0 auto; }
        .card { background: #1e293b; border-radius: 10px; padding: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 14px; font-weight: bold; }
        .badge { background: #0369a1; color: #bae6fd; font-size: 11px; padding: 2px 7px; border-radius: 4px; }
        canvas { background: #050811; border-radius: 6px; width: 100%; height: 260px; display: block; border: 1px solid #1e293b; }
        .stats { margin-top: 8px; font-size: 12px; color: #94a3b8; font-family: monospace; line-height: 1.4; }
        .target-txt { color: #f59e0b; font-weight: bold; }
    </style>
</head>
<body>
    <header>
        <h1>🗺️ Top-Down Radar — Todas as Runs e Trajetórias</h1>
        <p class="subtitle">Histórico acumulativo das rotas navegadas pelos 8 robôs no mundo aberto</p>
        <div class="toolbar">
            <label><input type="checkbox" id="chk_historico" checked> Mostrar Runs Anteriores</label>
            <label><input type="checkbox" id="chk_atual" checked> Mostrar Run Atual</label>
            <button class="btn" onclick="limparHistorico()">Limpar Histórico de Runs</button>
            <span id="run_count" style="font-size:12px; color:#38bdf8; font-weight:bold;">Runs Gravadas: 0</span>
        </div>
    </header>

    <div class="grid" id="grid"></div>

    <script>
        const NUM_ENVS = ${n};
        const CORES_RUNS = ['#818cf8', '#6366f1', '#a855f7', '#ec4899', '#f43f5e', '#3b82f6', '#06b6d4', '#10b981', '#eab308'];
        
        let historico = Array.from({ length: NUM_ENVS }, () => ({
            runsPassadas: [],
            runAtual: { pontos: [], largada: null },
            ultimoPasso: -1
        }));

        try {
            const salvo = localStorage.getItem('topview_runs_v2');
            if (salvo) {
                const parsed = JSON.parse(salvo);
                parsed.forEach((h, i) => { if (historico[i]) historico[i].runsPassadas = h.runsPassadas || []; });
            }
        } catch (_) {}

        const container = document.getElementById('grid');
        for (let i = 0; i < NUM_ENVS; i++) {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = \`
                <div class="card-head">
                    <span>Ambiente \${i}</span>
                    <span class="badge" id="b_\${i}">Passo 0</span>
                </div>
                <canvas id="cv_\${i}" width="340" height="260"></canvas>
                <div class="stats" id="st_\${i}">Conectando...</div>
            \`;
            container.appendChild(card);
        }

        function limparHistorico() {
            historico.forEach(h => { h.runsPassadas = []; h.runAtual = { pontos: [], largada: null }; h.ultimoPasso = -1; });
            localStorage.removeItem('topview_runs_v2');
            atualizarContador();
        }

        function atualizarContador() {
            let tot = historico.reduce((acc, h) => acc + h.runsPassadas.length, 0);
            document.getElementById('run_count').textContent = \`Runs Gravadas: \${tot}\`;
        }
        atualizarContador();

        async function tickRadar() {
            try {
                const r = await fetch('/lote/estado');
                const d = await r.json();

                let houveReset = false;
                for (const e of d.envs) {
                    const id = e.env;
                    const s = e.estado;
                    const h = historico[id];

                    if (e.passos < h.ultimoPasso && h.runAtual.pontos.length > 3) {
                        h.runsPassadas.push({
                            largada: h.runAtual.largada,
                            pontos: [...h.runAtual.pontos]
                        });
                        if (h.runsPassadas.length > 60) h.runsPassadas.shift();
                        h.runAtual = { pontos: [], largada: { x: s.x, z: s.z } };
                        houveReset = true;
                    }

                    if (!h.runAtual.largada) h.runAtual.largada = { x: s.x, z: s.z };
                    h.ultimoPasso = e.passos;

                    const pts = h.runAtual.pontos;
                    if (pts.length === 0 || Math.hypot(s.x - pts[pts.length - 1].x, s.z - pts[pts.length - 1].z) > 0.05) {
                        pts.push({ x: s.x, z: s.z, yaw: s.yaw, y: s.y });
                    }

                    desenhar(id, s, e, h);
                }

                if (houveReset) {
                    atualizarContador();
                    try {
                        localStorage.setItem('topview_runs_v2', JSON.stringify(
                            historico.map(h => ({ runsPassadas: h.runsPassadas }))
                        ));
                    } catch (_) {}
                }
            } catch (_) {}
            setTimeout(tickRadar, 300);
        }

        function desenhar(id, s, envData, h) {
            const cv = document.getElementById('cv_' + id);
            if (!cv) return;
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height;

            ctx.clearRect(0, 0, W, H);

            const x0 = h.runAtual.largada ? h.runAtual.largada.x : s.x;
            const z0 = h.runAtual.largada ? h.runAtual.largada.z : s.z;

            let minX = Math.min(x0, s.x) - 10, maxX = Math.max(x0, s.x) + 10;
            let minZ = Math.min(z0, s.z) - 10, maxZ = Math.max(z0, s.z) + 10;

            const chkHist = document.getElementById('chk_historico').checked;
            const chkAtual = document.getElementById('chk_atual').checked;

            if (chkHist) {
                h.runsPassadas.forEach(run => {
                    run.pontos.forEach(p => {
                        if (p.x < minX) minX = p.x - 4;
                        if (p.x > maxX) maxX = p.x + 4;
                        if (p.z < minZ) minZ = p.z - 4;
                        if (p.z > maxZ) maxZ = p.z + 4;
                    });
                });
            }

            const span = Math.max(maxX - minX, maxZ - minZ, 20);
            const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;

            function toX(x) { return W / 2 + ((x - cx) / span) * (W * 0.85); }
            function toY(z) { return H / 2 + ((z - cz) / span) * (H * 0.85); }

            // Grid radar
            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 1;
            for (let g = -50; g <= 50; g += 10) {
                ctx.beginPath(); ctx.moveTo(toX(cx + g), 0); ctx.lineTo(toX(cx + g), H); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, toY(cz + g)); ctx.lineTo(W, toY(cz + g)); ctx.stroke();
            }

            // Desenha Runs Anteriores (Linhas Semitransparentes com gradiente de tempo)
            if (chkHist && h.runsPassadas.length > 0) {
                const totalRuns = h.runsPassadas.length;
                h.runsPassadas.forEach((run, idx) => {
                    if (run.pontos.length < 2) return;
                    const cor = CORES_RUNS[idx % CORES_RUNS.length];
                    const alpha = 0.25 + 0.45 * (idx / totalRuns);
                    ctx.strokeStyle = cor;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    for (let k = 0; k < run.pontos.length; k++) {
                        const px = toX(run.pontos[k].x), py = toY(run.pontos[k].z);
                        if (k === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;
                });
            }

            // Desenha Run Atual (Linha Laser Ciano Brilhante)
            if (chkAtual && h.runAtual.pontos.length > 1) {
                ctx.strokeStyle = '#38bdf8';
                ctx.lineWidth = 3;
                ctx.beginPath();
                for (let k = 0; k < h.runAtual.pontos.length; k++) {
                    const px = toX(h.runAtual.pontos[k].x), py = toY(h.runAtual.pontos[k].z);
                    if (k === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.stroke();
            }

            // Ponto de Partida da Run Atual (Verde)
            ctx.fillStyle = '#22c55e';
            ctx.beginPath();
            ctx.arc(toX(x0), toY(z0), 5, 0, Math.PI * 2);
            ctx.fill();

            // Robô Atual (Branco com contorno ciano)
            const curX = toX(s.x), curY = toY(s.z);
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(curX, curY, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Direção de Yaw
            const rad = (s.yaw * Math.PI) / 180;
            ctx.strokeStyle = '#f8fafc';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(curX, curY);
            ctx.lineTo(curX - Math.sin(rad) * 16, curY - Math.cos(rad) * 16);
            ctx.stroke();

            const dLiq = Math.hypot(s.x - x0, s.z - z0);
            document.getElementById('b_' + id).textContent = \`Passo \${envData.passos} | \${dLiq.toFixed(1)}m\`;
            let stTxt = \`Pos: (\${s.x.toFixed(1)}, \${s.y.toFixed(0)}, \${s.z.toFixed(1)}) | Yaw: \${s.yaw.toFixed(1)}°<br>\`;
            if (envData.alvo) {
                stTxt += \`<span class="target-txt">Alvo: \${envData.alvo.dist}m | \${envData.alvo.graus}°</span>\`;
            } else {
                stTxt += \`Runs neste card: \${h.runsPassadas.length}\`;
            }
            document.getElementById('st_' + id).innerHTML = stTxt;
        }

        tickRadar();
    </script>
</body>
</html>`)
    }

    if (req.method === 'GET' && req.url.startsWith('/lote/plano')) {
      const q = new URL(req.url, 'http://x').searchParams
      const i = parseInt(q.get('env') || '0')
      const t0 = Date.now()
      const p = planejar(ambientes[i].bot, undefined, undefined,
                         parseFloat(q.get('wp') || '10'))
      res.writeHead(200)
      return res.end(JSON.stringify(Object.assign({ ms: Date.now() - t0 }, p)))
    }

    res.writeHead(404)
    res.end(JSON.stringify({ erro: 'rotas: POST /lote/reset /lote/passo | GET /lote/info /lote/plano' }))
  } catch (e) {
    res.writeHead(500)
    res.end(JSON.stringify({ erro: e.message, pilha: (e.stack || '').split('\n').slice(0, 3) }))
  }
})

;(async () => {
  console.log('='.repeat(62))
  console.log(` Simulador offline — ${N_ENVS} ambientes, sem Minecraft aberto`)
  console.log('='.repeat(62))
  console.log(`[save]  ${SAVE}`)

  for (let i = 0; i < N_ENVS; i++) ambientes.push(new Ambiente(i))

  const t0 = Date.now()
  let ok = 0
  for (const a of ambientes) if (await a.respawnar()) ok++
  console.log(`[env]   ${ok}/${N_ENVS} ambientes posicionados em ` +
    `${((Date.now() - t0) / 1000).toFixed(1)}s | ${mundo.tamanho} colunas em memoria`)

  // Encerramento limpo: sem isto o GC reclama dos handles abertos e o
  // processo morre com ERR_INVALID_STATE em vez de sair normalmente.
  let encerrando = false
  const encerrar = async (sinal) => {
    if (encerrando) return
    encerrando = true
    console.log(`
[sair]  ${sinal} — fechando ${mundo.regioesAbertas} regioes...`)
    await mundo.fechar()
    process.exit(0)
  }
  for (const s of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.on(s, () => { encerrar(s) })
  }
  process.on('beforeExit', () => { encerrar('beforeExit') })

  servidor.listen(PORTA, '127.0.0.1', () => {
    console.log(`[http]  http://127.0.0.1:${PORTA}/lote/info`)
    console.log('='.repeat(62))
  })
})()
