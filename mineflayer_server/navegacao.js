/**
 * navegacao.js — mapa de rotas e planejador BFS, desacoplados do bot global.
 *
 * Extraidos do state_server.js para poder rodar tambem sobre os bots
 * SIMULADOS do servidor offline. Recebem `bot` como parametro; dependem
 * apenas de bot.world.getColumn, bot.blockAt, bot.entity e bot.registry.
 */
'use strict'

const ROTA_K = parseInt(process.env.ROTA_K || "12");
const ROTA_MAX = parseFloat(process.env.ROTA_MAX || "16");
const PLANO_RAIO = parseInt(process.env.PLANO_RAIO || "40");
const PLANO_MAX_NOS = parseInt(process.env.PLANO_MAX_NOS || "9000");

// ── Mapa de rotas ─────────────────────────────────────────────────────────────
// Para K direções ao redor do bot, quanto dá para andar sem bater.
// É a verdade-fundamental para o modelo APRENDER A PREVER rotas: ele recebe
// só a imagem e tem que reproduzir este vetor. Sinal denso e dependente da
// cena, que é justamente o que faltava (o alvo constante "sempre W" fez a via
// visual colapsar).

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
function distanciaLivre(bot, ox, oy, oz, dx, dz, alcance) {
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
function mapaDeRotas(bot) {
  if (!bot || !bot.entity) return null;
  const p = bot.entity.position;
  const yaw = bot.entity.yaw;
  const livres = [];
  for (let i = 0; i < ROTA_K; i++) {
    const ang = yaw + (i / ROTA_K) * Math.PI * 2;
    const dx = -Math.sin(ang);
    const dz = -Math.cos(ang);
    livres.push(distanciaLivre(bot, p.x, p.y, p.z, dx, dz, ROTA_MAX));
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


/** Empacota (x,y,z) num inteiro para chave de visitados. */
function _chave(x, y, z) {
  return ((x + 4096) << 20) ^ ((z + 4096) << 8) ^ (y & 255);
}

// Leitura direta do buffer da section, como no voxel_renderer. bot.blockAt()
// aloca um Vec3 e um objeto Block por consulta; o BFS faz ~500 mil consultas e
// levava 51 SEGUNDOS. Aqui sao tres leituras de array.
let _passavelPorId = null;

function _initPassavelPorId(bot) {
  _passavelPorId = new Uint8Array(256);
  try {
    for (const b of bot.registry.blocksArray) {
      if (b.id < 256 && ROTA_PASSAVEL.has(b.name)) _passavelPorId[b.id] = 1;
    }
  } catch (_) {}
  _passavelPorId[0] = 1;   // ar
}

/** Id do bloco em (x,y,z). -1 = chunk ausente. */
function _tipoEm(bot, x, y, z, colCache) {
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
function _pisavel(bot, x, y, z, cache, colCache) {
  const k = _chave(x, y, z);
  const memo = cache.get(k);
  if (memo !== undefined) return memo;

  if (_passavelPorId === null) _initPassavelPorId(bot);

  const pes = _tipoEm(bot, x, y, z, colCache);
  const cabeca = _tipoEm(bot, x, y + 1, z, colCache);
  const chao = _tipoEm(bot, x, y - 1, z, colCache);

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
function planejar(bot, alvoX, alvoZ, distWaypoint = 10) {
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
          if (!_pisavel(bot, nx, ny, nz, cache, colCache)) continue;

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


module.exports = { mapaDeRotas, planejar, ROTA_K, ROTA_MAX };
