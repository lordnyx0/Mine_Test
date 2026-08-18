/**
 * Mineflayer VLA Agent Server v4 — Minecraft 1.8.9
 *
 * O bot Mineflayer É o agente. Sem DirectInput, sem captura de tela do Windows.
 *
 * Renderização: raycaster voxel em Node puro (voxel_renderer.js), ~10-15ms/frame.
 * O caminho antigo (prismarine-viewer + Puppeteer.screenshot, 80-200ms/frame)
 * continua disponível via RENDERER=puppeteer.
 *
 * Endpoints:
 *   GET  /health  → verifica se bot está conectado e se o renderer está pronto
 *   GET  /frame   → retorna JPEG da visão em 1ª pessoa do bot
 *   GET  /state   → posição/velocidade/saúde do bot
 *   GET  /delta   → deslocamento do bot desde o último tick (sinal de recompensa)
 *   GET  /stats   → métricas de renderização
 *   POST /action  → executa WASD + câmera diretamente no jogo via setControlState
 */

const mineflayer = require("mineflayer");
const http       = require("http");

// ── Config ────────────────────────────────────────────────────────────────────
const MC_HOST      = process.env.MC_HOST      || "127.0.0.1";
const MC_PORT      = parseInt(process.env.MC_PORT      || "25565");
const MC_VERSION   = process.env.MC_VERSION   || "1.8.9";
const BOT_USERNAME = process.env.BOT_USERNAME || "VLAAgent";
const HTTP_PORT    = parseInt(process.env.HTTP_PORT    || "3001");
const VIEW_PORT    = parseInt(process.env.VIEW_PORT    || "3002");

// "voxel" (padrão, rápido) | "puppeteer" (legado, lento) | "none"
const RENDERER     = (process.env.RENDERER || "voxel").toLowerCase();

// Parâmetros do renderer voxel
const FRAME_W         = parseInt(process.env.FRAME_W        || "640");
const FRAME_H         = parseInt(process.env.FRAME_H        || "360");
const FRAME_SCALE     = parseInt(process.env.FRAME_SCALE    || "4");   // downscale do raycast
const FRAME_FOV       = parseFloat(process.env.FRAME_FOV    || "70");
const FRAME_DIST      = parseFloat(process.env.FRAME_DIST   || "64");  // alcance em blocos
const FRAME_QUALITY   = parseFloat(process.env.FRAME_QUALITY|| "0.8");
const FRAME_CROSSHAIR = process.env.FRAME_CROSSHAIR !== "0";
const FRAME_ENTITIES  = process.env.FRAME_ENTITIES  !== "0";
// Vegetacao atravessavel (grama alta, flores). "0" limpa a visao do agente.
const FRAME_SPRITES   = process.env.FRAME_SPRITES   !== "0";
const FRAME_SMOOTH    = process.env.FRAME_SMOOTH    === "1";
// Reaproveita o último frame se ele tem menos de N ms (0 = sempre renderiza)
const FRAME_MIN_MS    = parseFloat(process.env.FRAME_MIN_MS || "0");
// ─────────────────────────────────────────────────────────────────────────────

// ── State ─────────────────────────────────────────────────────────────────────
let botState = {
  connected: false,
  x: 0, y: 0, z: 0,
  yaw: 0, pitch: 0,
  vx: 0, vy: 0, vz: 0,
  on_ground: true,
  health: 20, food: 20,
  game_mode: "survival",
  timestamp: Date.now()
};

let prevPos = { x: 0, y: 0, z: 0, timestamp: Date.now() };
let delta   = { dx: 0, dy: 0, dz: 0, speed: 0, moved: false, dt_ms: 0 };

let bot        = null;
let viewerPage = null;  // Puppeteer page for frame capture (modo legado)
let viewerReady = false;

let renderer      = null;   // VoxelRenderer
let lastFrame     = null;   // Buffer do último JPEG
let lastFrameAt   = 0;
let frameCount    = 0;
let frameMsTotal  = 0;
let frameMsMax    = 0;
// ─────────────────────────────────────────────────────────────────────────────

// ── Voxel Renderer (padrão) ───────────────────────────────────────────────────
function initVoxelRenderer() {
  const { VoxelRenderer } = require("./voxel_renderer");
  renderer = new VoxelRenderer(bot, {
    width:       FRAME_W,
    height:      FRAME_H,
    scale:       FRAME_SCALE,
    fov:         FRAME_FOV,
    maxDistance: FRAME_DIST,
    quality:     FRAME_QUALITY,
    crosshair:   FRAME_CROSSHAIR,
    entities:    FRAME_ENTITIES,
    sprites:     FRAME_SPRITES,
    smooth:      FRAME_SMOOTH,
  });

  console.log(`[Frame] Renderer voxel: ${renderer.rw}x${renderer.rh} raycast → ${FRAME_W}x${FRAME_H} JPEG`);

  // Espera os chunks ao redor do bot chegarem
  const t0 = Date.now();
  const poll = setInterval(() => {
    if (renderer.isReady()) {
      clearInterval(poll);
      const buf = renderVoxelFrame();
      if (!buf) return;
      viewerReady = true;
      console.log(
        `[Frame] Pronto em ${((Date.now() - t0) / 1000).toFixed(1)}s — ` +
        `primeiro frame: ${renderer.lastRenderMs.toFixed(1)}ms, ${(buf.length / 1024).toFixed(1)}KB`
      );
      console.log(`[Frame] GET http://127.0.0.1:${HTTP_PORT}/frame`);
    } else if (Date.now() - t0 > 60000) {
      clearInterval(poll);
      console.log("[Frame] AVISO: chunks nao chegaram em 60s. Liberando /frame mesmo assim.");
      viewerReady = true;
    }
  }, 250);
}

/** Renderiza (ou reaproveita) um frame. Retorna Buffer JPEG ou null. */
function renderVoxelFrame() {
  if (!renderer) return null;
  const now = Date.now();
  if (FRAME_MIN_MS > 0 && lastFrame && (now - lastFrameAt) < FRAME_MIN_MS) return lastFrame;

  let buf;
  try {
    buf = renderer.render();
  } catch (e) {
    // Ex.: layout de chunk incompatível. Desliga o renderer em vez de derrubar
    // o servidor — /action e /delta continuam funcionando.
    console.error("[Frame] Renderer desativado apos erro:", e.message);
    renderer = null;
    viewerReady = false;
    return null;
  }
  if (!buf) return lastFrame;

  lastFrame    = buf;
  lastFrameAt  = now;
  frameCount++;
  frameMsTotal += renderer.lastRenderMs;
  if (renderer.lastRenderMs > frameMsMax) frameMsMax = renderer.lastRenderMs;
  return buf;
}
// ─────────────────────────────────────────────────────────────────────────────

// ── Puppeteer Frame Capture (legado) ──────────────────────────────────────────
async function initPuppeteerViewer() {
  let puppeteer;
  try {
    puppeteer = require("puppeteer");
  } catch (e) {
    console.log("[Frame] puppeteer nao instalado — endpoint /frame indisponivel.");
    return;
  }

  try {
    console.log("[Frame] Iniciando browser com WebGL real (janela minimizada)...");
    const browser = await puppeteer.launch({
      headless: false,   // janela real = WebGL funciona de verdade
      defaultViewport: null,
      args: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--window-size=640,400",
        "--window-position=9999,9999",  // joga a janela fora da tela visível
      ]
    });

    viewerPage = await browser.newPage();
    await viewerPage.setViewport({ width: 640, height: 360 });

    console.log(`[Frame] Abrindo viewer em http://127.0.0.1:${VIEW_PORT} ...`);
    await viewerPage.goto(`http://127.0.0.1:${VIEW_PORT}`, {
      waitUntil: "domcontentloaded",
      timeout: 30000
    });

    // Aguardar Three.js inicializar e chunks carregarem
    console.log("[Frame] Aguardando Three.js + chunks carregarem (8s)...");
    await new Promise(r => setTimeout(r, 8000));

    viewerReady = true;
    console.log("[Frame] Pronto! GET /frame disponivel em http://127.0.0.1:" + HTTP_PORT + "/frame");
  } catch (e) {
    console.error("[Frame] Erro ao inicializar Puppeteer:", e.message);
  }
}

async function captureFrame() {
  if (!viewerPage || !viewerReady) return null;
  try {
    return await viewerPage.screenshot({
      type: "jpeg",
      quality: 85,
      clip: { x: 0, y: 0, width: 640, height: 360 }
    });
  } catch (e) {
    console.error("[Frame] Erro na captura:", e.message);
    return null;
  }
}
// ─────────────────────────────────────────────────────────────────────────────

// ── Bot ───────────────────────────────────────────────────────────────────────
function createBot() {
  console.log(`[Bot] Conectando a ${MC_HOST}:${MC_PORT} v${MC_VERSION} como '${BOT_USERNAME}'...`);

  bot = mineflayer.createBot({
    host:     MC_HOST,
    port:     MC_PORT,
    version:  MC_VERSION,
    username: BOT_USERNAME,
  });

  bot.once("spawn", () => {
    botState.connected = true;
    console.log("[Bot] Spawnado com sucesso no mundo!");

    if (RENDERER === "voxel") {
      try {
        initVoxelRenderer();
      } catch (e) {
        console.error("[Frame] Falha ao iniciar o renderer voxel:", e.message);
      }
    } else if (RENDERER === "puppeteer") {
      // Caminho legado: prismarine-viewer (WebGL) + Puppeteer.screenshot
      try {
        const { mineflayer: viewer } = require("prismarine-viewer");
        viewer(bot, { firstPerson: true, port: VIEW_PORT });
        console.log(`[Viewer] Visao 1a pessoa em http://127.0.0.1:${VIEW_PORT}`);
        setTimeout(initPuppeteerViewer, 3000);
      } catch (e) {
        console.log("[Viewer] prismarine-viewer nao disponivel:", e.message);
      }
    } else {
      console.log("[Frame] RENDERER=none — /frame desabilitado.");
    }

    updateState();
  });

  bot.on("physicsTick", () => {
    if (!bot.entity) return;
    updateState();
    computeDelta();
  });

  bot.on("message", (msg) => {
    try {
      lastChat.push(msg.toString());
      if (lastChat.length > 20) lastChat.shift();
    } catch (_) {}
  });

  bot.on("health", () => {
    botState.health = bot.health;
    botState.food   = bot.food;
  });

  bot.on("death", () => {
    console.log("[Bot] Morreu. Soltando teclas...");
    botState.health = 0;
    releaseAllKeys();
  });

  bot.on("error", err => {
    console.error("[Bot] Erro:", err.message);
    botState.connected = false;
  });

  bot.on("end", () => {
    console.log("[Bot] Desconectado. Reconectando em 5s...");
    botState.connected = false;
    viewerReady = false;
    renderer = null;   // será recriado no próximo spawn (novo objeto bot)
    releaseAllKeys();
    setTimeout(createBot, 5000);
  });
}

function updateState() {
  const pos = bot.entity.position;
  const vel = bot.entity.velocity;
  botState = {
    ...botState,
    x:         parseFloat(pos.x.toFixed(4)),
    y:         parseFloat(pos.y.toFixed(4)),
    z:         parseFloat(pos.z.toFixed(4)),
    yaw:       parseFloat((bot.entity.yaw   * 180 / Math.PI).toFixed(2)),
    pitch:     parseFloat((bot.entity.pitch * 180 / Math.PI).toFixed(2)),
    vx:        parseFloat(vel.x.toFixed(4)),
    vy:        parseFloat(vel.y.toFixed(4)),
    vz:        parseFloat(vel.z.toFixed(4)),
    on_ground: bot.entity.onGround,
    timestamp: Date.now()
  };
}

function computeDelta() {
  const now   = Date.now();
  const dt    = now - prevPos.timestamp;
  const dx    = botState.x - prevPos.x;
  const dz    = botState.z - prevPos.z;
  const dy    = botState.y - prevPos.y;
  const speed = Math.sqrt(dx * dx + dz * dz);

  delta = {
    dx:    parseFloat(dx.toFixed(5)),
    dy:    parseFloat(dy.toFixed(5)),
    dz:    parseFloat(dz.toFixed(5)),
    speed: parseFloat(speed.toFixed(5)),
    moved: speed > 0.005,
    dt_ms: dt
  };

  prevPos = { x: botState.x, y: botState.y, z: botState.z, timestamp: now };
}

const KEY_TO_CONTROL = {
  W: "forward", S: "back", A: "left", D: "right",
  SPACE: "jump", SHIFT: "sneak", CTRL: "sprint"
};

let releaseTimer = null;

// ── Mapa de rotas ─────────────────────────────────────────────────────────────
// Para K direções ao redor do bot, quanto dá para andar sem bater.
// É a verdade-fundamental para o modelo APRENDER A PREVER rotas: ele recebe
// só a imagem e tem que reproduzir este vetor. Sinal denso e dependente da
// cena, que é justamente o que faltava (o alvo constante "sempre W" fez a via
// visual colapsar).
const ROTA_K = parseInt(process.env.ROTA_K || "12");   // setores ao redor
const ROTA_MAX = parseFloat(process.env.ROTA_MAX || "16");  // alcance em blocos

// Atravessavel a pe. Note que FOLHAS NAO entram: em Minecraft elas bloqueiam.
// (SOLID_SKIP existe para outro fim — achar chao de pouso — e inclui folhas.)
const ROTA_PASSAVEL = new Set([
  "air", "tallgrass", "double_plant", "yellow_flower", "red_flower", "deadbush",
  "sapling", "reeds", "vine", "snow_layer", "torch", "redstone_torch",
  "rail", "golden_rail", "detector_rail", "activator_rail", "redstone_wire",
  "wheat", "carrots", "potatoes", "pumpkin_stem", "melon_stem", "nether_wart",
  "standing_sign", "wall_sign", "lever", "stone_button", "wooden_button",
  "stone_pressure_plate", "wooden_pressure_plate", "tripwire", "tripwire_hook",
  "water", "flowing_water", "brown_mushroom", "red_mushroom", "web", "fire"
]);

function _passavel(nome) {
  if (!nome) return true;
  return ROTA_PASSAVEL.has(nome);
}

/**
 * Distância livre numa direção horizontal, medida na altura do peito
 * (feet+1): uma parede bloqueia, um degrau de 1 bloco não.
 */
function distanciaLivre(ox, oy, oz, dx, dz, alcance) {
  const Vec3 = require("vec3");
  const passo = 0.25;
  const yPeito = Math.floor(oy) + 1;
  for (let d = passo; d <= alcance; d += passo) {
    const x = Math.floor(ox + dx * d);
    const z = Math.floor(oz + dz * d);
    let b;
    try { b = bot.blockAt(new Vec3(x, yPeito, z)); } catch (_) { return d; }
    if (b === null) return d;                    // chunk ausente: trata como fim
    if (!_passavel(b.name)) return d;
  }
  return alcance;
}

/**
 * Vetor de navegabilidade ao redor do bot, começando na direção que ele
 * encara e girando no sentido horário. Normalizado em [0,1].
 */
function mapaDeRotas() {
  if (!bot || !bot.entity) return null;
  const p = bot.entity.position;
  const yaw = bot.entity.yaw;
  const livres = [];
  for (let i = 0; i < ROTA_K; i++) {
    const ang = yaw + (i / ROTA_K) * Math.PI * 2;
    const dx = -Math.sin(ang);
    const dz = -Math.cos(ang);
    livres.push(distanciaLivre(p.x, p.y, p.z, dx, dz, ROTA_MAX));
  }
  const norm = livres.map(d => +(d / ROTA_MAX).toFixed(4));
  let melhor = 0;
  for (let i = 1; i < norm.length; i++) if (norm[i] > norm[melhor]) melhor = i;
  return {
    k: ROTA_K,
    alcance: ROTA_MAX,
    livre: norm,                               // 0 = parede colada, 1 = livre
    melhor_setor: melhor,                      // 0 = para onde ja olha
    melhor_angulo_graus: +((melhor / ROTA_K) * 360).toFixed(1),
    frente_livre: norm[0]
  };
}
// ─────────────────────────────────────────────────────────────────────────────

// ── Planejador (BFS sobre o grid de voxels) ───────────────────────────────────
// O professor guloso, que so olhava os 12 setores a 16 blocos, empacava em
// minimo local: media de 10-16 blocos em 70 passos, com episodios zerados.
// Uma busca em largura de verdade enxerga a saida ao redor do obstaculo, e e
// ela que fornece o rotulo de WAYPOINT — o alvo que o modelo deve aprender a
// tracar, em vez de uma sequencia de botoes que divergiria em malha aberta.

const PLANO_RAIO = parseInt(process.env.PLANO_RAIO || "40");
const PLANO_MAX_NOS = parseInt(process.env.PLANO_MAX_NOS || "9000");

/** Empacota (x,y,z) num inteiro para chave de visitados. */
function _chave(x, y, z) {
  return ((x + 4096) << 20) ^ ((z + 4096) << 8) ^ (y & 255);
}

// Leitura direta do buffer da section, como no voxel_renderer. bot.blockAt()
// aloca um Vec3 e um objeto Block por consulta; o BFS faz ~500 mil consultas e
// levava 51 SEGUNDOS. Aqui sao tres leituras de array.
let _passavelPorId = null;

function _initPassavelPorId() {
  _passavelPorId = new Uint8Array(256);
  try {
    for (const b of bot.registry.blocksArray) {
      if (b.id < 256 && ROTA_PASSAVEL.has(b.name)) _passavelPorId[b.id] = 1;
    }
  } catch (_) {}
  _passavelPorId[0] = 1;   // ar
}

/** Id do bloco em (x,y,z). -1 = chunk ausente. */
function _tipoEm(x, y, z, colCache) {
  if (y < 0 || y > 255) return 0;
  const cx = x >> 4, cz = z >> 4;
  const ck = cx * 100000 + cz;
  let col = colCache.get(ck);
  if (col === undefined) {
    try { col = bot.world.getColumn(cx, cz) || null; } catch (_) { col = null; }
    colCache.set(ck, col);
  }
  if (col === null || !col.sections) return -1;
  const sec = col.sections[y >> 4];
  if (!sec || !sec.data) return 0;
  const n = ((x & 15) | ((z & 15) << 4) | ((y & 15) << 8)) << 1;
  return (sec.data[n] | (sec.data[n + 1] << 8)) >> 4;
}

/**
 * (x,y,z) e uma celula onde o bot consegue FICAR DE PE:
 * corpo livre em y e y+1, chao solido em y-1.
 */
function _pisavel(x, y, z, cache, colCache) {
  const k = _chave(x, y, z);
  const memo = cache.get(k);
  if (memo !== undefined) return memo;

  if (_passavelPorId === null) _initPassavelPorId();

  const pes = _tipoEm(x, y, z, colCache);
  const cabeca = _tipoEm(x, y + 1, z, colCache);
  const chao = _tipoEm(x, y - 1, z, colCache);

  const ok = pes >= 0 && cabeca >= 0 && chao >= 0 &&
             _passavelPorId[pes] === 1 && _passavelPorId[cabeca] === 1 &&
             _passavelPorId[chao] === 0;
  cache.set(k, ok);
  return ok;
}

const _VIZINHOS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];

/**
 * BFS a partir do bot. Retorna o caminho ate a celula alcancavel mais distante
 * (ou ate um alvo dado), e o waypoint a ~distWaypoint blocos ao longo dele.
 */
function planejar(alvoX, alvoZ, distWaypoint = 10) {
  if (!bot || !bot.entity) return null;
  const p = bot.entity.position;
  const ox = Math.floor(p.x), oy = Math.floor(p.y), oz = Math.floor(p.z);

  const cache = new Map();
  const colCache = new Map();
  const pai = new Map();
  const inicio = _chave(ox, oy, oz);
  pai.set(inicio, null);

  let fila = [[ox, oy, oz]];
  let melhor = { x: ox, y: oy, z: oz, d: 0, chave: inicio };
  let nos = 0;

  while (fila.length && nos < PLANO_MAX_NOS) {
    const proxima = [];
    for (const [x, y, z] of fila) {
      nos++;
      if (nos > PLANO_MAX_NOS) break;

      for (const [dx, dz] of _VIZINHOS) {
        const nx = x + dx, nz = z + dz;
        if (Math.abs(nx - ox) > PLANO_RAIO || Math.abs(nz - oz) > PLANO_RAIO) continue;

        // degrau de ate 1 bloco para cima ou para baixo
        for (const dy of [0, 1, -1]) {
          const ny = y + dy;
          const k = _chave(nx, ny, nz);
          if (pai.has(k)) continue;
          if (!_pisavel(nx, ny, nz, cache, colCache)) continue;

          pai.set(k, [x, y, z]);
          proxima.push([nx, ny, nz]);

          const d = alvoX === undefined
            ? Math.hypot(nx - ox, nz - oz)                 // explorar: mais longe
            : -Math.hypot(nx - alvoX, nz - alvoZ);         // ir ate o alvo
          if (d > melhor.d) melhor = { x: nx, y: ny, z: nz, d, chave: k };
          break;   // um nivel de y por vizinho horizontal
        }
      }
    }
    fila = proxima;
  }

  // Reconstroi o caminho do bot ate o melhor no
  const caminho = [];
  let atual = [melhor.x, melhor.y, melhor.z];
  let guarda = 0;
  while (atual && guarda++ < 4000) {
    caminho.push(atual);
    atual = pai.get(_chave(atual[0], atual[1], atual[2]));
  }
  caminho.reverse();

  if (caminho.length < 2) return { alcance: 0, caminho: [], waypoint: null, nos };

  // Waypoint: ponto do caminho a ~distWaypoint blocos do bot
  let wp = caminho[caminho.length - 1];
  for (const c of caminho) {
    if (Math.hypot(c[0] - ox, c[2] - oz) >= distWaypoint) { wp = c; break; }
  }

  // Converte para o referencial do bot: frente e lado
  const yaw = bot.entity.yaw;
  const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
  const relX = wp[0] + 0.5 - p.x, relZ = wp[2] + 0.5 - p.z;
  const frente = relX * fx + relZ * fz;
  const lado = relX * (-fz) + relZ * fx;
  const graus = Math.atan2(lado, frente) * 180 / Math.PI;   // + = direita

  return {
    alcance: +Math.hypot(melhor.x - ox, melhor.z - oz).toFixed(2),
    nos,
    caminho: caminho.filter((_, i) => i % 3 === 0 || i === caminho.length - 1),
    waypoint: { x: wp[0], y: wp[1], z: wp[2] },
    waypoint_rel: { frente: +frente.toFixed(2), lado: +lado.toFixed(2),
                    distancia: +Math.hypot(frente, lado).toFixed(2),
                    graus: +graus.toFixed(1) },
  };
}
// ─────────────────────────────────────────────────────────────────────────────

// ── Respawn aleatório ─────────────────────────────────────────────────────────
// Precisa de "Permitir cheats" ligado no mundo aberto para LAN.

let lastChat = [];
const SOLID_SKIP = new Set([
  "air", "water", "flowing_water", "lava", "flowing_lava", "leaves", "leaves2",
  "tallgrass", "double_plant", "yellow_flower", "red_flower", "snow_layer",
  "deadbush", "sapling", "reeds", "vine", "fire", "web"
]);

/** Altura da primeira superfície sólida em (x,z), ou null se a coluna não chegou. */
function alturaDaSuperficie(x, z) {
  const Vec3 = require("vec3");
  let col;
  try { col = bot.world.getColumn(x >> 4, z >> 4); } catch (_) { return null; }
  if (!col) return null;
  for (let y = 200; y >= 1; y--) {
    let b;
    try { b = bot.blockAt(new Vec3(x, y, z)); } catch (_) { return null; }
    if (!b) continue;
    if (!SOLID_SKIP.has(b.name)) {
      // Recusa pouso em líquido logo acima
      const acima = bot.blockAt(new Vec3(x, y + 1, z));
      if (acima && (acima.name.includes("water") || acima.name.includes("lava"))) return null;
      return y + 1;
    }
  }
  return null;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

/**
 * Teleporta o bot para um ponto aleatório com superfície sólida.
 * Estratégia: tp para y=200 (o tp zera a queda), espera a coluna carregar,
 * calcula a superfície e tp de novo para ela.
 */
async function respawnAleatorio(raio = 1500, tentativas = 12, fixo = null) {
  if (!bot || !bot.entity) return { ok: false, erro: "bot nao pronto" };

  const base = { x: Math.floor(bot.entity.position.x), z: Math.floor(bot.entity.position.z) };

  for (let t = 0; t < tentativas; t++) {
    let x, z;
    if (fixo) {
      // Coordenada dirigida, para comparar com o simulador no mesmo terreno
      x = Math.floor(fixo[0]); z = Math.floor(fixo[1]);
      if (t > 2) return { ok: false, erro: "ponto fixo sem superficie valida" };
    } else {
      const ang = Math.random() * Math.PI * 2;
      const dist = 200 + Math.random() * raio;
      x = Math.floor(base.x + Math.cos(ang) * dist);
      z = Math.floor(base.z + Math.sin(ang) * dist);
    }

    lastChat = [];
    bot.chat(`/tp ${BOT_USERNAME} ${x + 0.5} 200 ${z + 0.5}`);

    // Espera a coluna chegar, re-teleportando para zerar a queda
    let surf = null;
    for (let i = 0; i < 24; i++) {
      await sleep(250);
      surf = alturaDaSuperficie(x, z);
      if (surf !== null) break;
      if (i % 4 === 3) bot.chat(`/tp ${BOT_USERNAME} ${x + 0.5} 200 ${z + 0.5}`);
      if (lastChat.some(m => /permission|permiss|Unknown command|desconhecid/i.test(m))) {
        return {
          ok: false,
          erro: "Servidor recusou /tp. Abra o mundo para LAN com 'Permitir cheats' ligado.",
          chat: lastChat.slice(-3)
        };
      }
    }

    if (surf === null || surf < 2 || surf > 200) continue;

    bot.chat(`/tp ${BOT_USERNAME} ${x + 0.5} ${surf} ${z + 0.5}`);
    await sleep(700);

    const p = bot.entity.position;
    const chegou = Math.abs(p.x - (x + 0.5)) < 3 && Math.abs(p.z - (z + 0.5)) < 3;
    if (chegou) {
      releaseAllKeys();
      if (renderer) renderer._gen++;   // invalida o cache de sections do mundo novo
      return {
        ok: true, x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2),
        tentativas: t + 1
      };
    }
  }
  return { ok: false, erro: `nenhum ponto valido em ${tentativas} tentativas` };
}
// ─────────────────────────────────────────────────────────────────────────────

function releaseAllKeys() {
  if (!bot) return;
  for (const k of ["forward", "back", "left", "right", "jump", "sneak", "sprint"]) {
    try { bot.setControlState(k, false); } catch (_) {}
  }
}

function applyAction(action) {
  if (!bot || !bot.entity) return;

  const hold = new Set((action.hold || []).map(k => String(k).toUpperCase()));
  const [dx, dy] = action.mouse || [0, 0];

  // Cancela o release agendado pela ação anterior. Sem isso, um timer antigo
  // solta as teclas no meio da ação atual — o agente perde ~35% do movimento
  // quando envia ações mais rápido que duration_ms.
  if (releaseTimer !== null) {
    clearTimeout(releaseTimer);
    releaseTimer = null;
  }

  // Aplica só a diferença: manter W apertado entre duas ações não deve gerar
  // um release+press (que a tick de física pode observar como "solto").
  for (const key in KEY_TO_CONTROL) {
    const ctrl = KEY_TO_CONTROL[key];
    const want = hold.has(key);
    try {
      if (bot.getControlState(ctrl) !== want) bot.setControlState(ctrl, want);
    } catch (_) {
      bot.setControlState(ctrl, want);
    }
  }

  // Rotação de câmera
  if (dx !== 0 || dy !== 0) {
    const sens = 0.003;
    bot.entity.yaw   += dx * sens;
    bot.entity.pitch  = Math.max(-Math.PI / 2,
                          Math.min(Math.PI / 2, bot.entity.pitch + dy * sens));
  }

  const dur = action.duration_ms || 100;
  releaseTimer = setTimeout(() => { releaseTimer = null; releaseAllKeys(); }, dur);
}
// ─────────────────────────────────────────────────────────────────────────────

// ── HTTP Server ───────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");

  // ── GET ──────────────────────────────────────────────────────────────────
  if (req.method === "GET") {
    res.setHeader("Content-Type", "application/json");

    if (req.url === "/health") {
      res.writeHead(200);
      return res.end(JSON.stringify({
        ok: botState.connected,
        viewer_ready: viewerReady,
        renderer: RENDERER,
        timestamp: Date.now()
      }));
    }

    // Verdade-fundamental vinda do próprio mineflayer, para conferir o renderer
    if (req.url === "/debug") {
      if (!bot || !bot.entity) { res.writeHead(503); return res.end(JSON.stringify({ error: "bot nao pronto" })); }

      const p = bot.entity.position;
      const eyeY = p.y + (bot.entity.height ? bot.entity.height * 0.9 : 1.62);
      const nameAt = (x, y, z) => {
        try {
          const b = bot.blockAt(new (require("vec3"))(x, y, z));
          return b ? b.name : "?";
        } catch (_) { return "?"; }
      };

      // Coluna vertical na posição do bot
      const column = {};
      for (let dy = -2; dy <= 4; dy++) {
        const y = Math.floor(p.y) + dy;
        column[`y${dy >= 0 ? "+" : ""}${dy} (${y})`] = nameAt(Math.floor(p.x), y, Math.floor(p.z));
      }

      // O que está na frente, na altura dos olhos
      const fx = -Math.sin(bot.entity.yaw) * Math.cos(bot.entity.pitch);
      const fz = -Math.cos(bot.entity.yaw) * Math.cos(bot.entity.pitch);
      const fy = Math.sin(bot.entity.pitch);
      const ahead = [];
      for (let d = 1; d <= 8; d++) {
        ahead.push(`${d}: ` + nameAt(
          Math.floor(p.x + fx * d), Math.floor(eyeY + fy * d), Math.floor(p.z + fz * d)));
      }

      // Bloco mirado, via raycast do próprio mineflayer
      let cursor = null;
      try {
        const b = bot.blockAtCursor(64);
        if (b) cursor = { name: b.name, pos: b.position, dist: +b.position.distanceTo(p).toFixed(2) };
      } catch (e) { cursor = { error: e.message }; }

      res.writeHead(200);
      return res.end(JSON.stringify({
        position: { x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2) },
        eye_y: +eyeY.toFixed(2),
        entity_height: bot.entity.height,
        yaw_deg: +(bot.entity.yaw * 180 / Math.PI).toFixed(1),
        pitch_deg: +(bot.entity.pitch * 180 / Math.PI).toFixed(1),
        block_at_eye: nameAt(Math.floor(p.x), Math.floor(eyeY), Math.floor(p.z)),
        column_at_bot: column,
        blocks_ahead_at_eye_level: ahead,
        block_at_cursor: cursor,
        entities_near: Object.values(bot.entities || {})
          .filter(e => e && e !== bot.entity && e.position)
          .map(e => ({
            name: e.name, displayName: e.displayName, type: e.type, kind: e.kind,
            username: e.username, w: e.width, h: e.height,
            dist: +e.position.distanceTo(p).toFixed(1)
          }))
          .filter(e => e.dist < 64)
          .sort((a, b) => a.dist - b.dist)
          .slice(0, 15),
        time_of_day: bot.time ? bot.time.timeOfDay : null,
        loaded_columns: (() => { try { return bot.world.getColumns().length; } catch (_) { return null; } })()
      }, null, 2));
    }

    // /probe?x=445&y=80 → o que o raio daquele pixel realmente atravessa
    if (req.url.startsWith("/probe")) {
      if (!renderer) { res.writeHead(503); return res.end(JSON.stringify({ error: "renderer nao ativo" })); }
      const q = new URL(req.url, "http://x").searchParams;
      const px = parseInt(q.get("x") || String(FRAME_W >> 1));
      const py = parseInt(q.get("y") || String(FRAME_H >> 1));
      res.writeHead(200);
      return res.end(JSON.stringify(renderer.trace(px, py), null, 2));
    }

    // Navegabilidade ao redor: verdade-fundamental para o modelo prever rotas
    if (req.url === "/rotas") {
      const m = mapaDeRotas();
      if (!m) { res.writeHead(503); return res.end(JSON.stringify({ error: "bot nao pronto" })); }
      res.writeHead(200);
      return res.end(JSON.stringify(m));
    }

    // Plano global: caminho BFS e o waypoint que o modelo deve aprender a tracar
    if (req.url.startsWith("/plano")) {
      const q = new URL(req.url, "http://x").searchParams;
      const ax = q.get("x") !== null ? parseFloat(q.get("x")) : undefined;
      const az = q.get("z") !== null ? parseFloat(q.get("z")) : undefined;
      const dw = parseFloat(q.get("wp") || "10");
      const t0 = Date.now();
      const r = planejar(ax, az, dw);
      if (!r) { res.writeHead(503); return res.end(JSON.stringify({ error: "bot nao pronto" })); }
      r.ms = Date.now() - t0;
      res.writeHead(200);
      return res.end(JSON.stringify(r));
    }

    if (req.url === "/stats") {
      res.writeHead(200);
      return res.end(JSON.stringify({
        renderer: RENDERER,
        ready: viewerReady,
        frames: frameCount,
        avg_render_ms: frameCount ? parseFloat((frameMsTotal / frameCount).toFixed(2)) : null,
        max_render_ms: parseFloat(frameMsMax.toFixed(2)),
        last_render_ms: renderer ? parseFloat(renderer.lastRenderMs.toFixed(2)) : null,
        last_frame_kb: lastFrame ? parseFloat((lastFrame.length / 1024).toFixed(1)) : null,
        detail: renderer ? renderer.stats : null
      }, null, 2));
    }

    if (req.url === "/state") {
      res.writeHead(200);
      return res.end(JSON.stringify(botState, null, 2));
    }

    if (req.url === "/delta") {
      res.writeHead(200);
      return res.end(JSON.stringify(delta, null, 2));
    }

    // Visão em 1ª pessoa do bot como JPEG
    if (req.url === "/frame" || req.url.startsWith("/frame?")) {
      const frame = RENDERER === "voxel" ? renderVoxelFrame() : await captureFrame();
      if (!frame) {
        res.setHeader("Content-Type", "application/json");
        res.writeHead(503);
        return res.end(JSON.stringify({
          error: "Renderer nao pronto. Aguarde os chunks chegarem apos o bot spawnar."
        }));
      }
      res.setHeader("Content-Type", "image/jpeg");
      res.setHeader("Cache-Control", "no-store");
      if (renderer) res.setHeader("X-Render-Ms", renderer.lastRenderMs.toFixed(2));
      res.writeHead(200);
      return res.end(frame);
    }

    res.writeHead(404);
    return res.end(JSON.stringify({
      error: "Endpoints: GET /health /state /delta /frame /stats | POST /action"
    }));
  }

  // ── POST /respawn ─────────────────────────────────────────────────────────
  // Teleporta o bot para um ponto aleatório com superfície sólida.
  if (req.method === "POST" && req.url.startsWith("/respawn")) {
    res.setHeader("Content-Type", "application/json");
    const q = new URL(req.url, "http://x").searchParams;
    const raio = parseFloat(q.get("raio") || "1500");
    const fx = q.get("x"), fz = q.get("z");
    const r = await respawnAleatorio(raio, 12,
      (fx !== null && fz !== null) ? [parseFloat(fx), parseFloat(fz)] : null);
    res.writeHead(r.ok ? 200 : 500);
    return res.end(JSON.stringify(r));
  }

  // ── POST /action ──────────────────────────────────────────────────────────
  if (req.method === "POST" && req.url === "/action") {
    res.setHeader("Content-Type", "application/json");
    let body = "";
    req.on("data", chunk => { body += chunk; });
    req.on("end", () => {
      try {
        const action = JSON.parse(body);
        applyAction(action);
        res.writeHead(200);
        res.end(JSON.stringify({ ok: true, applied: action }));
      } catch (e) {
        res.writeHead(400);
        res.end(JSON.stringify({ error: "JSON invalido: " + e.message }));
      }
    });
    return;
  }

  res.setHeader("Content-Type", "application/json");
  res.writeHead(405);
  res.end(JSON.stringify({ error: "Metodo nao suportado" }));
});

server.listen(HTTP_PORT, "127.0.0.1", () => {
  console.log("=".repeat(60));
  console.log(" Mineflayer VLA Agent Server v4");
  console.log("=".repeat(60));
  console.log(`[HTTP]     http://127.0.0.1:${HTTP_PORT}`);
  console.log(`[Renderer] ${RENDERER}` +
    (RENDERER === "voxel" ? `  ${FRAME_W}x${FRAME_H} (scale ${FRAME_SCALE}, dist ${FRAME_DIST})` : ""));
  if (RENDERER === "puppeteer") console.log(`[Viewer]   http://127.0.0.1:${VIEW_PORT}  (apos bot spawnar)`);
  console.log(`[Frame]    http://127.0.0.1:${HTTP_PORT}/frame`);
  console.log(`[Stats]    http://127.0.0.1:${HTTP_PORT}/stats`);
  console.log("=".repeat(60));
  createBot();
});
