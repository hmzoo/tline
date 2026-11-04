'use strict';

/* ============================================================
   tline — capture du pointeur + courbes de vitesse (fenêtre 3 s)
   ============================================================ */

const WINDOW_MS = 3000; // intervalle affiché
const TRAIL_MS = 700; // longueur de la traînée dans le cadre de capture
const SAMPLE_MS = 16; // pas d'échantillonnage (~60 Hz)
const SMOOTH_MS = 55; // horizon de lissage exponentiel de la vitesse
const ACCEL_SMOOTH_MS = 85; // horizon de lissage de la dérivée
const IDLE_MS = 90; // au-delà, la vitesse décroît vers 0
const DECAY_MS = 170; // constante de décroissance au repos
const NFFT = 256; // taille de la FFT (puissance de 2)
const SPECTRUM_MAX_HZ = 20; // bande de fréquences affichée
const SIM_MARGIN = 42; // marge latérale des trajectoires simulées (px)
const SIM_SPEED = 420; // mode « constant » : vitesse en px/s
const SIM_ACCEL = 480; // mode « accel » : montée en vitesse en px/s²
const SIM_ACCEL_MAX = 1400; // plafond de vitesse du mode « accel »
const SIM_HZ = 1; // fréquence des modes « cyclic » et « circle »
const ML_TABLE_MAX = 8000; // transitions conservées pour la recherche
const ML_MAX = 40000; // échantillons max en mémoire
const ML_MATCH_K = 6; // candidats tirés au hasard à chaque recalage
const CAM_GAIN = 0.12; // poursuite de la caméra (0 = fixe, 1 = centrée)
const CAM_GRID = 120; // pas de la grille monde (px)

const COLORS = {
  v: '#5eead4',
  a: '#fbbf24',
  s: '#a78bfa',
  grid: 'rgba(148, 163, 184, 0.13)',
  axis: 'rgba(148, 163, 184, 0.30)',
  text: 'rgba(148, 163, 184, 0.72)',
  empty: 'rgba(148, 163, 184, 0.42)',
};

const FONT = '10px ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace';

const captureEl = document.getElementById('capture');
const hintEl = document.getElementById('capture-hint');
const speedEl = document.getElementById('speed-chart');
const accelEl = document.getElementById('accel-chart');
const spectrumEl = document.getElementById('spectrum-chart');

const cctx = captureEl.getContext('2d');
const sctx = speedEl.getContext('2d');
const kctx = accelEl.getContext('2d');
const pctx = spectrumEl.getContext('2d');

const statNow = document.getElementById('stat-now');
const statPeak = document.getElementById('stat-peak');
const statDeriv = document.getElementById('stat-deriv');
const statFreq = document.getElementById('stat-freq');
const captureLabel = document.getElementById('capture-label');
const modeButtons = Array.from(document.querySelectorAll('[data-mode]'));
const btnTrain = document.getElementById('btn-train');
const btnSimulate = document.getElementById('btn-simulate');
const btnClear = document.getElementById('btn-clear');
const kbdTrain = document.getElementById('kbd-train');
const mlStatus = document.getElementById('ml-status');
const mlContext = document.getElementById('ml-context');
const mlContextValue = document.getElementById('ml-context-value');
const modelSelect = document.getElementById('model-select');
const modelName = document.getElementById('model-name');
const btnSaveModel = document.getElementById('btn-save-model');
const btnDeleteModel = document.getElementById('btn-delete-model');
const modelStatus = document.getElementById('model-status');

const MODE_LABELS = {
  manuel: 'manuel',
  constant: 'constant',
  accel: 'accélération',
  cyclic: 'cyclique',
  circle: 'cercle',
  chaotic: 'chaotique',
  ia: 'modèle IA · caméra',
};

/* ---------------- état ---------------- */

const samples = []; // { t, v, vx, vy, a } — positions glissantes
const trail = []; // { x, y, t } — traînée locale au cadre de capture
let pointer = null; // dernier point survolé { x, y, t }
let prev = null; // point précédent (pour le calcul de vitesse)
let smoothV = 0; // vitesse lissée (px/s)
let smoothVx = 0; // composante horizontale lissée (px/s)
let smoothVy = 0; // composante verticale lissée (px/s)
let prevV = 0; // vitesse lissée à l'image précédente
let smoothA = 0; // dérivée lissée de la vitesse (px/s²)
let lastMoveT = -Infinity;
let lastSampleT = 0;
let lastStatT = 0;
let lastStatusT = 0;
let touched = false;
let mode = 'manuel'; // manuel | constant | accel | cyclic | circle | chaotic | ia
let sim = null; // état du générateur de trajectoire { x, y, phase, t }
const cam = { x: 0, y: 0 }; // caméra virtuelle (coin haut-gauche de la vue, en coordonnées monde)

/* modèle de transitions appris sur les vitesses relevées */
const ML = {
  samples: [], // [[vx, vy], ...] relevés à ~60 Hz
  savedUpTo: 0, // index déjà envoyé au serveur
  recording: false,
  table: null, // { vx, vy, idx, H } : historique de vitesses → vitesse suivante
  ready: false,
  error: 0, // écart moyen de reconstruction (px/s)
  context: 8, // nombre de points de vitesse composant l'état
  segment: 24, // images rejouées avant de se recaler (~400 ms)
  segmentJitter: 0.5, // variation aléatoire de la longueur (± 50 %)
  continuity: 0.5, // pénalité de discontinuité au recalage
  blend: 8, // images de raccord après un recalage
  velSmooth: 0, // lissage court de la sortie (ms, 0 = aucun)
  blendLeft: 0,
  blendX: 0,
  blendY: 0,
  cursor: -1, // position dans la séquence enregistrée
  framesLeft: 0, // images restantes avant recalage
  lastPos: null, // dernière position vue (pour la vitesse image par image)
  histX: null, // historique courant du déroulé (index 0 = le plus récent)
  histY: null,
  started: false,
  x: 0,
  y: 0,
};

const speedState = { max: 200 };
const accelState = { max: 200 };
const specState = { max: 50 };

/* spectre : df = 1 / fenêtre = 0,33 Hz */
const DF = 1000 / WINDOW_MS;
const BINS = Math.min(Math.floor(SPECTRUM_MAX_HZ / DF), NFFT / 2);
const specMag = new Float64Array(BINS);
const specRe = new Float64Array(NFFT);
const specIm = new Float64Array(NFFT);
const specVals = new Float64Array(NFFT);
let specPeakHz = 0;

const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

/* ---------------- dimensionnement ---------------- */

function fit(canvas) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const scale = dpr();
  const w = Math.round(rect.width * scale);
  const h = Math.round(rect.height * scale);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  canvas._w = rect.width;
  canvas._h = rect.height;
}

const ro = new ResizeObserver((entries) => {
  for (const e of entries) fit(e.target);
});
[captureEl, speedEl, accelEl, spectrumEl].forEach((c) => {
  fit(c);
  ro.observe(c);
});

/* ---------------- capture : point courant ---------------- */

function feedPointer(x, y, t) {
  if (prev) {
    const dt = t - prev.t;
    if (dt > 0) {
      const vx = ((x - prev.x) / dt) * 1000;
      const vy = ((y - prev.y) / dt) * 1000;
      const v = Math.hypot(vx, vy);
      const k = 1 - Math.exp(-dt / SMOOTH_MS);
      smoothV += (v - smoothV) * k;
      smoothVx += (vx - smoothVx) * k;
      smoothVy += (vy - smoothVy) * k;
      touched = true;
    }
  }

  prev = { x, y, t };
  pointer = { x, y, t };
  lastMoveT = t;
  trail.push({ x, y, t });
  hintEl.classList.add('is-hidden');
}

captureEl.addEventListener('pointermove', (event) => {
  if (mode !== 'manuel') return;
  const rect = captureEl.getBoundingClientRect();
  feedPointer(event.clientX - rect.left, event.clientY - rect.top, performance.now());
});

captureEl.addEventListener('pointerleave', () => {
  if (mode !== 'manuel') return;
  pointer = null;
  prev = null;
});

/* ---------------- générateurs de trajectoire ---------------- */

function resetSim() {
  sim = null;
  prev = null;
}

function stepSimulation(now, dt) {
  if (mode === 'manuel') return;

  const w = captureEl._w || 0;
  const h = captureEl._h || 0;
  if (w < 60 || h < 60) return;

  if (mode === 'ia') {
    stepModel(now, dt, w, h);
    return;
  }

  const dtS = Math.max(0, Math.min(dt, 64)) / 1000;
  const cy = h / 2;
  const spanX = Math.max(10, w - 2 * SIM_MARGIN);

  if (!sim) sim = { x: SIM_MARGIN, y: cy, phase: 0, t: 0 };
  sim.t += dtS;

  let x = sim.x;
  let y = sim.y;

  switch (mode) {
    case 'constant': {
      // gauche → droite à vitesse constante, retour instantané
      x = sim.x + SIM_SPEED * dtS;
      if (x > w - SIM_MARGIN) {
        x = SIM_MARGIN;
        prev = null; // pas de pic de vitesse au retour
      }
      y = cy;
      break;
    }
    case 'accel': {
      // gauche → droite, vitesse croissant linéairement
      const v = Math.min(SIM_ACCEL_MAX, SIM_ACCEL * sim.t);
      x = sim.x + v * dtS;
      if (x > w - SIM_MARGIN) {
        x = SIM_MARGIN;
        sim.t = 0;
        prev = null;
      }
      y = cy;
      break;
    }
    case 'cyclic': {
      // aller-retour horizontal sinusoïdal
      sim.phase += 2 * Math.PI * SIM_HZ * dtS;
      const amp = Math.min(spanX / 2, 280);
      x = SIM_MARGIN + spanX / 2 + amp * Math.sin(sim.phase);
      y = cy;
      break;
    }
    case 'circle': {
      // trajectoire circulaire (vitesse constante)
      sim.phase += 2 * Math.PI * SIM_HZ * dtS;
      const radius = Math.min(w, h) / 3.2;
      x = w / 2 + radius * Math.cos(sim.phase);
      y = cy + radius * Math.sin(sim.phase);
      break;
    }
    case 'chaotic': {
      // somme de sinusoïdes incommensurables : spectre large, non périodique
      const t = sim.t;
      x =
        w / 2 +
        spanX *
          0.32 *
          (0.62 * Math.sin(0.73 * t) +
            0.3 * Math.sin(1.93 * t + 1.1) +
            0.16 * Math.sin(5.31 * t + 2.3));
      y =
        cy +
        h *
          0.3 *
          (0.62 * Math.sin(1.13 * t + 0.5) +
            0.3 * Math.sin(2.71 * t + 2.0) +
            0.16 * Math.sin(4.77 * t + 4.1));
      break;
    }
  }

  x = Math.max(4, Math.min(w - 4, x));
  y = Math.max(4, Math.min(h - 4, y));
  sim.x = x;
  sim.y = y;
  feedPointer(x, y, now);
}

/* ---------------- changement de mode ---------------- */

function resetMeasurement() {
  samples.length = 0;
  trail.length = 0;
  pointer = null;
  prev = null;
  smoothV = 0;
  prevV = 0;
  smoothA = 0;
  smoothVx = 0;
  smoothVy = 0;
  lastMoveT = -Infinity;
  lastSampleT = 0;
  lastStatT = 0;
  touched = false;
  specMag.fill(0);
  specPeakHz = 0;
  speedState.max = 200;
  accelState.max = 200;
  specState.max = 50;
  statNow.textContent = '—';
  statPeak.textContent = '—';
  statDeriv.textContent = '—';
  statFreq.textContent = '—';
}

function setMode(next) {
  mode = next;
  for (const btn of modeButtons) {
    const on = btn.dataset.mode === next;
    btn.classList.toggle('is-active', on);
    btn.setAttribute('aria-pressed', String(on));
  }
  captureLabel.textContent = `capture · ${MODE_LABELS[next] ?? next}`;
  hintEl.classList.toggle('is-hidden', next !== 'manuel');
  if (next !== 'manuel' && ML.recording) ML.recording = false;
  cam.x = 0;
  cam.y = 0;
  if (next === 'ia') {
    ML.started = false;
    ML.histX = null;
    ML.histY = null;
  }
  resetSim();
  resetMeasurement();
  updateStatus();
}

modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

/* ---------------- apprentissage : transitions par plus proches voisins ---------------- */

/**
 * Construit la table de transitions à partir des relevés. Un état n'est plus la
 * vitesse instantanée mais un vecteur de ML.context vitesses consécutives :
 * le modèle compare donc la *forme* du mouvement sur quelques centaines de ms,
 * pas seulement sa valeur courante.
 */
function trainModel() {
  const data = ML.samples;
  const n = data.length;
  ML.ready = false;
  ML.table = null;
  if (n < 400) return false;

  const H = Math.max(1, Math.min(ML.context, n - 2));

  const vx = new Float32Array(n);
  const vy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    vx[i] = data[i][0];
    vy[i] = data[i][1];
  }

  // points candidats : historique complet (H) et successeur (i + 1) disponibles
  const stride = Math.max(1, Math.ceil(n / ML_TABLE_MAX));
  const idx = [];
  for (let i = H; i < n - 1; i += stride) idx.push(i);
  if (idx.length < 50) return false;

  const table = { vx, vy, idx: Int32Array.from(idx), H };

  // erreur : on compare le successeur du voisin le plus proche au successeur réel
  let se = 0;
  let cnt = 0;
  const probes = Math.min(120, table.idx.length);
  for (let p = 0; p < probes; p++) {
    const i = table.idx[Math.floor(Math.random() * table.idx.length)];
    let best = -1;
    let bestD = Infinity;
    for (let c = 0; c < table.idx.length; c++) {
      const j = table.idx[c];
      if (j === i) continue;
      let d2 = 0;
      for (let q = 0; q < H; q++) {
        const dx = vx[i - q] - vx[j - q];
        const dy = vy[i - q] - vy[j - q];
        d2 += dx * dx + dy * dy;
      }
      if (d2 < bestD) {
        bestD = d2;
        best = j;
      }
    }
    if (best >= 0) {
      // vitesse prédite = vitesse courante + incrément du voisin
      const px = vx[i] + (vx[best + 1] - vx[best]);
      const py = vy[i] + (vy[best + 1] - vy[best]);
      const ex = px - vx[i + 1];
      const ey = py - vy[i + 1];
      se += ex * ex + ey * ey;
      cnt++;
    }
  }

  ML.table = table;
  ML.error = cnt ? Math.sqrt(se / cnt) : 0;
  ML.ready = true;
  return true;
}

/** Distance entre l'historique courant et l'état qui se termine à l'indice i. */
function historyDistance(table, i) {
  const { vx, vy, H } = table;
  let d = 0;
  for (let q = 0; q < H; q++) {
    const dx = vx[i - q] - ML.histX[q];
    const dy = vy[i - q] - ML.histY[q];
    d += dx * dx + dy * dy;
  }
  return d;
}

/**
 * Choisit un fragment à rejouer : les ML_MATCH_K meilleures situations sont
 * d'abord sélectionnées (historique proche + successeur cohérent), puis l'une
 * d'elles est tirée au hasard. C'est ce tirage qui rend chaque simulation
 * différente tout en restant composée de morceaux réels du geste.
 */
function pickMatch() {
  const t = ML.table;
  if (!t || !ML.histX) return -1;

  const { idx, vx: VX, vy: VY } = t;
  const cx = ML.histX[0];
  const cy = ML.histY[0];
  const k = ML_MATCH_K;
  const bestScore = new Float64Array(k).fill(Infinity);
  const bestIdx = new Int32Array(k).fill(-1);

  for (let c = 0; c < idx.length; c++) {
    const i = idx[c];
    const d2 = historyDistance(t, i);
    const ex = VX[i + 1] - cx;
    const ey = VY[i + 1] - cy;
    const score = d2 + ML.continuity * (ex * ex + ey * ey);
    if (score >= bestScore[k - 1]) continue;
    let p = k - 1;
    while (p > 0 && bestScore[p - 1] > score) {
      bestScore[p] = bestScore[p - 1];
      bestIdx[p] = bestIdx[p - 1];
      p--;
    }
    bestScore[p] = score;
    bestIdx[p] = i;
  }

  let total = 0;
  const weight = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    if (bestIdx[i] < 0) continue;
    weight[i] = 1 / (Math.sqrt(bestScore[i]) + 1);
    total += weight[i];
  }
  if (!total) return -1;

  let r = Math.random() * total;
  for (let i = 0; i < k; i++) {
    if (bestIdx[i] < 0) continue;
    r -= weight[i];
    if (r <= 0) return bestIdx[i];
  }
  return bestIdx[0];
}

/** Rejoue un fragment enregistré, recalé sur la situation courante. */
function stepModel(now, dt, w, h) {
  if (!ML.ready || !ML.table) return;

  const dtS = Math.max(0, Math.min(dt, 64)) / 1000;
  if (!ML.started) {
    ML.started = true;
    const t = ML.table;
    const H = t.H;
    const n = t.vx.length;
    // départ tiré au hasard dans l'enregistrement : deux simulations ne
    // commencent donc jamais par le même fragment
    const start = H + Math.floor(Math.random() * Math.max(1, n - H - 2));
    ML.histX = new Float32Array(H);
    ML.histY = new Float32Array(H);
    for (let q = 0; q < H; q++) {
      const i = Math.max(0, start - q);
      ML.histX[q] = t.vx[i];
      ML.histY[q] = t.vy[i];
    }
    ML.cursor = start;
    ML.framesLeft = ML.segment;
    ML.blendLeft = 0;
    ML.x = w / 2;
    ML.y = h / 2;
    prev = null;
  }

  const { vx: VX, vy: VY } = ML.table;
  const last = VX.length - 2;

  if (ML.cursor < 0 || ML.framesLeft <= 0 || ML.cursor >= last) {
    ML.cursor = pickMatch();
    if (ML.cursor < 0) return;
    const jitter = 1 + ML.segmentJitter * (Math.random() * 2 - 1);
    ML.framesLeft = Math.max(4, Math.round(ML.segment * jitter));
    ML.blendLeft = ML.blend;
    ML.blendX = ML.histX[0];
    ML.blendY = ML.histY[0];
  }

  let vx = VX[ML.cursor + 1];
  let vy = VY[ML.cursor + 1];
  ML.cursor++;
  ML.framesLeft--;

  // raccord progressif : les premières images après un recalage partent de la
  // vitesse courante pour éviter un saut visible (et un pic d'accélération)
  if (ML.blendLeft > 0) {
    const w = 1 - ML.blendLeft / (ML.blend + 1);
    vx = ML.blendX + (vx - ML.blendX) * w;
    vy = ML.blendY + (vy - ML.blendY) * w;
    ML.blendLeft--;
  }

  // lissage très court : atténue les à-coups d'une image sans effacer le style
  if (ML.velSmooth > 0) {
    const k = 1 - Math.exp(-dt / ML.velSmooth);
    vx = ML.histX[0] + (vx - ML.histX[0]) * k;
    vy = ML.histY[0] + (vy - ML.histY[0]) * k;
  }

  // aucune borne : la trajectoire n'est jamais modifiée, c'est la caméra qui
  // suit le point (contrairement à un rebond sur les bords du cadre)
  ML.x += vx * dtS;
  ML.y += vy * dtS;

  cam.x += (ML.x - w / 2 - cam.x) * CAM_GAIN;
  cam.y += (ML.y - h / 2 - cam.y) * CAM_GAIN;

  // garde-fou : la caméra rattrape le point s'il menace de sortir du cadre.
  // Seule la caméra bouge, la trajectoire n'est jamais modifiée.
  const margin = 26;
  const sx = ML.x - cam.x;
  const sy = ML.y - cam.y;
  if (sx < margin) cam.x = ML.x - margin;
  else if (sx > w - margin) cam.x = ML.x - (w - margin);
  if (sy < margin) cam.y = ML.y - margin;
  else if (sy > h - margin) cam.y = ML.y - (h - margin);

  // l'historique glisse d'un cran : la nouvelle vitesse devient la plus récente
  if (ML.histX.length > 1) {
    ML.histX.copyWithin(1, 0, ML.histX.length - 1);
    ML.histY.copyWithin(1, 0, ML.histY.length - 1);
  }
  ML.histX[0] = vx;
  ML.histY[0] = vy;

  feedPointer(ML.x, ML.y, now);
}

/* ---------------- interface : boutons et persistance ---------------- */

function setStatusText(text) {
  if (mlStatus.textContent !== text) mlStatus.textContent = text;
}

function updateStatus() {
  const n = ML.samples.length;
  if (ML.recording) setStatusText(`enregistrement… ${n} échantillons`);
  else if (ML.ready)
    setStatusText(
      `${n} échantillons · modèle prêt (contexte ${ML.table ? ML.table.H : ML.context}) · écart ${Math.round(ML.error)} px/s`,
    );
  else if (n) setStatusText(`${n} échantillons · modèle non entraîné`);
  else setStatusText('aucune donnée');

  btnTrain.classList.toggle('is-rec', ML.recording);
  btnTrain.textContent = ML.recording ? 'arrêter' : 'apprentissage';
  kbdTrain.classList.toggle('is-rec', ML.recording);
  btnSimulate.disabled = !ML.ready;
  btnSimulate.classList.toggle('is-active', mode === 'ia');
  btnClear.disabled = n === 0;
  btnSaveModel.disabled = n === 0;
}

/* ---------------- signature du geste ---------------- */

const SIG_KEYS = ['v', 'p90', 'a', 'jerk', 'turns', 'stops'];
const sigCells = {};
for (const who of ['geste', 'modele']) {
  for (const key of SIG_KEYS) {
    sigCells[`${who}-${key}`] = document.getElementById(`sig-${who}-${key}`);
  }
}

function fmtCompact(v) {
  const abs = Math.abs(v);
  if (abs >= 10000) return `${Math.round(v / 1000)}k`;
  if (abs >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return String(Math.round(v));
}

function renderSignature(who, sig) {
  for (const key of SIG_KEYS) {
    const el = sigCells[`${who}-${key}`];
    if (!el) continue;
    let text = '—';
    if (sig) {
      if (key === 'v') text = String(Math.round(sig.v));
      else if (key === 'p90') text = String(Math.round(sig.p90));
      else if (key === 'a') text = fmtCompact(sig.a);
      else if (key === 'jerk') text = fmtCompact(sig.jerk);
      else if (key === 'turns') text = sig.turns.toFixed(1);
      else text = `${Math.round(sig.stops)} %`;
    }
    if (el.textContent !== text) el.textContent = text;
  }
}

/**
 * Caractéristiques du geste sur la fenêtre affichée : niveau et dispersion de
 * vitesse, accélération moyenne, secousses (jerk), changements de direction et
 * part du temps à l'arrêt.
 */
function signatureOf(list) {
  const n = list.length;
  if (n < 12) return null;

  const speeds = new Float64Array(n);
  let sumV = 0;
  let sumA = 0;
  let stops = 0;
  let jerk2 = 0;
  let jerkN = 0;
  let turns = 0;
  let prevSign = 0;

  for (let i = 0; i < n; i++) {
    const s = list[i];
    speeds[i] = s.v;
    sumV += s.v;
    sumA += Math.abs(s.a);
    if (s.v < 25) stops++;

    if (i > 0) {
      const p = list[i - 1];
      const dt = (s.t - p.t) / 1000;
      if (dt > 0) {
        const j = (s.a - p.a) / dt;
        jerk2 += j * j;
        jerkN++;
      }
      // produit vectoriel : signe = sens de rotation de la vitesse
      const cross = s.vx * p.vy - s.vy * p.vx;
      if (Math.abs(cross) > 2000) {
        const sign = Math.sign(cross);
        if (prevSign && sign !== prevSign) turns++;
        prevSign = sign;
      }
    }
  }

  speeds.sort();
  const duration = Math.max(0.001, (list[n - 1].t - list[0].t) / 1000);

  return {
    v: sumV / n,
    p90: speeds[Math.min(n - 1, Math.floor(n * 0.9))],
    a: sumA / n,
    jerk: jerkN ? Math.sqrt(jerk2 / jerkN) : 0,
    turns: turns / duration,
    stops: (stops / n) * 100,
  };
}

async function saveDataset() {
  const chunk = ML.samples.slice(ML.savedUpTo);
  if (!chunk.length) return;
  ML.savedUpTo = ML.samples.length;
  try {
    await fetch('/api/dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: chunk }),
    });
  } catch {
    /* serveur indisponible : les données restent en mémoire */
  }
}

async function loadDataset() {
  try {
    const res = await fetch('/api/dataset');
    if (res.ok) {
      const body = await res.json();
      if (Array.isArray(body.samples) && body.samples.length) {
        ML.samples = body.samples.map((p) => [Number(p[0]), Number(p[1])]);
        ML.savedUpTo = ML.samples.length;
        trainModel();
      }
    }
  } catch {
    /* pas de serveur : mode hors-ligne */
  }
  updateStatus();
}

/* ---------------- modèles sauvegardés ---------------- */

function defaultModelName() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `modele-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
}

/** Signature calculée sur l'ensemble des relevés (dt = pas d'échantillonnage). */
function signatureOfSamples() {
  const dtS = SAMPLE_MS / 1000;
  const list = [];
  let prevV = 0;
  for (let i = 0; i < ML.samples.length; i++) {
    const [vx, vy] = ML.samples[i];
    const v = Math.hypot(vx, vy);
    list.push({ t: i * SAMPLE_MS, v, vx, vy, a: i ? (v - prevV) / dtS : 0 });
    prevV = v;
  }
  return signatureOf(list);
}

async function refreshModelList(selectName) {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error(String(res.status));
    const body = await res.json();
    const models = Array.isArray(body.models) ? body.models : [];
    const wanted = selectName ?? modelSelect.value;

    modelSelect.innerHTML = '<option value="">— aucun modèle —</option>';
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name} · ${m.samples} pts`;
      opt.title = [m.createdAt, m.error != null ? `écart ${m.error} px/s` : '']
        .filter(Boolean)
        .join(' · ');
      modelSelect.append(opt);
    }
    modelSelect.value = models.some((m) => m.name === wanted) ? wanted : '';
    btnDeleteModel.disabled = !modelSelect.value;
  } catch {
    /* serveur absent : liste vide */
  }
}

async function saveModel() {
  if (!ML.samples.length) return;

  const raw = modelName.value.trim() || defaultModelName();
  const name = raw
    .replace(/[^A-Za-z0-9._-]/g, '-')
    .replace(/^[^A-Za-z0-9]+/, '')
    .slice(0, 64);
  if (!name) {
    modelStatus.textContent = 'nom invalide';
    return;
  }

  const sig = signatureOfSamples();
  const velocities = ML.samples.map(([vx, vy]) => [
    Math.round(vx * 100) / 100,
    Math.round(vy * 100) / 100,
  ]);

  const payload = {
    version: 1,
    name,
    createdAt: new Date().toISOString(),
    samplingHz: Math.round((1000 / SAMPLE_MS) * 100) / 100,
    window: {
      context: ML.context,
      segment: ML.segment,
      blend: ML.blend,
      continuity: ML.continuity,
    },
    error: Math.round(ML.error),
    signature: sig
      ? {
          v: Math.round(sig.v),
          p90: Math.round(sig.p90),
          a: Math.round(sig.a),
          jerk: Math.round(sig.jerk),
          turns: Number(sig.turns.toFixed(2)),
          stops: Math.round(sig.stops),
        }
      : null,
    velocities,
  };

  modelStatus.textContent = 'enregistrement…';
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(name)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(String(res.status));
    modelName.value = '';
    await refreshModelList(name);
    modelStatus.textContent = `${name} · ${velocities.length} points enregistrés`;
  } catch {
    modelStatus.textContent = 'échec de l’enregistrement (serveur ?)';
  }
}

async function loadModel(name) {
  if (!name) return;
  modelStatus.textContent = 'chargement…';
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(String(res.status));
    const body = await res.json();
    const velocities = Array.isArray(body.velocities) ? body.velocities : [];
    if (velocities.length < 400) throw new Error('trop peu de points');

    ML.samples = velocities.map((p) => [Number(p[0]), Number(p[1])]);
    ML.savedUpTo = ML.samples.length;

    const win = body.window || {};
    ML.context = Math.max(1, Math.min(12, Number(win.context) || 8));
    ML.segment = Number(win.segment) || 24;
    ML.blend = Number(win.blend) || 8;
    ML.continuity = Number.isFinite(Number(win.continuity)) ? Number(win.continuity) : 0.5;
    mlContext.value = String(ML.context);
    mlContextValue.textContent = String(ML.context);

    ML.started = false;
    ML.histX = null;
    ML.histY = null;
    trainModel();
    if (mode === 'ia') setMode('manuel');
    renderSignature('modele', null);
    updateStatus();
    modelStatus.textContent = `${body.name || name} · ${ML.samples.length} points · écart ${Math.round(ML.error)} px/s`;
  } catch {
    modelStatus.textContent = 'modèle illisible';
  }
}

async function deleteModel(name) {
  if (!name) return;
  modelStatus.textContent = 'suppression…';
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(String(res.status));
    await refreshModelList('');
    modelStatus.textContent = `${name} supprimé`;
  } catch {
    modelStatus.textContent = 'échec de la suppression';
  }
}

modelSelect.addEventListener('change', () => {
  btnDeleteModel.disabled = !modelSelect.value;
  if (modelSelect.value) loadModel(modelSelect.value);
});

btnSaveModel.addEventListener('click', () => saveModel());

btnDeleteModel.addEventListener('click', () => {
  const name = modelSelect.value;
  if (name && window.confirm(`Supprimer le modèle « ${name} » ?`)) deleteModel(name);
});

modelName.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    saveModel();
  }
});

btnTrain.addEventListener('click', () => {
  if (ML.recording) {
    ML.recording = false;
    trainModel();
    saveDataset();
    updateStatus();
    return;
  }
  if (mode !== 'manuel') setMode('manuel');
  ML.recording = true;
  ML.savedUpTo = ML.samples.length;
  ML.lastPos = null;
  updateStatus();
});

btnSimulate.addEventListener('click', () => {
  if (!ML.ready) return;
  setMode(mode === 'ia' ? 'manuel' : 'ia');
});

btnClear.addEventListener('click', () => {
  ML.samples = [];
  ML.savedUpTo = 0;
  ML.table = null;
  ML.ready = false;
  ML.error = 0;
  ML.started = false;
  ML.histX = null;
  ML.histY = null;
  fetch('/api/dataset', { method: 'DELETE' }).catch(() => {});
  renderSignature('geste', null);
  renderSignature('modele', null);
  if (mode === 'ia') setMode('manuel');
  updateStatus();
});

mlContext.addEventListener('input', () => {
  ML.context = Number(mlContext.value);
  mlContextValue.textContent = String(ML.context);
  if (mode === 'ia') {
    ML.started = false;
    ML.histX = null;
    ML.histY = null;
    prev = null;
  }
  if (ML.samples.length >= 400) trainModel();
  updateStatus();
});

/* raccourci : barre d'espace = démarrer / arrêter l'apprentissage */
window.addEventListener('keydown', (event) => {
  if (event.key !== ' ' && event.code !== 'Space') return;
  if (event.repeat) return;

  const target = event.target;
  const tag = target && target.tagName ? target.tagName.toLowerCase() : '';
  if (
    tag === 'button' ||
    tag === 'input' ||
    tag === 'select' ||
    tag === 'textarea' ||
    (target && target.isContentEditable)
  ) {
    return; // la touche active déjà l'élément qui a le focus
  }

  event.preventDefault();
  btnTrain.click();
});

loadDataset();
refreshModelList();

/* ---------------- échantillonnage ---------------- */

function update(now, dt) {
  if (touched && now - lastMoveT > IDLE_MS) {
    smoothV *= Math.exp(-dt / DECAY_MS);
    if (smoothV < 0.5) smoothV = 0;
  }

  // dérivée de la vitesse : (v(t) − v(t−Δt)) / Δt
  const secs = Math.max(dt, 0.5) / 1000;
  const aRaw = (smoothV - prevV) / secs;
  prevV = smoothV;
  smoothA += (aRaw - smoothA) * (1 - Math.exp(-dt / ACCEL_SMOOTH_MS));
  if (Math.abs(smoothA) < 0.5) smoothA = 0;

  if (touched && now - lastSampleT >= SAMPLE_MS) {
    samples.push({ t: now, v: Math.max(0, smoothV), vx: smoothVx, vy: smoothVy, a: smoothA });
    lastSampleT = now;
  }

  const cutoff = now - WINDOW_MS - 250;
  while (samples.length && samples[0].t < cutoff) samples.shift();
}

/* ---------------- utilitaires ---------------- */

function niceMax(value) {
  if (!Number.isFinite(value) || value <= 0) return 100;
  const pow = Math.pow(10, Math.floor(Math.log10(value)));
  const n = value / pow;
  const mult = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return mult * pow;
}

function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function fmtSpeed(v) {
  if (!Number.isFinite(v)) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 10000) return `${sign}${Math.round(abs / 1000)}k`;
  if (abs >= 1000) return `${sign}${(abs / 1000).toFixed(1)}k`;
  return `${sign}${Math.round(abs)}`;
}

function fmtAxis(v, max) {
  const abs = Math.abs(v);
  if (max >= 8000) return `${Math.round(v / 1000)}k`;
  if (abs >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (max <= 20) return v.toFixed(1);
  return String(Math.round(v));
}

/* ---------------- rendu : cadre de capture ---------------- */

function drawCapture(now) {
  const w = captureEl._w || 0;
  const h = captureEl._h || 0;
  if (!w || !h) return;

  const s = dpr();
  cctx.setTransform(s, 0, 0, s, 0, 0);
  cctx.clearRect(0, 0, w, h);

  // grille « monde » : rend visible le déplacement de la caméra en simulation
  if (mode === 'ia') {
    const offX = ((-cam.x % CAM_GRID) + CAM_GRID) % CAM_GRID;
    const offY = ((-cam.y % CAM_GRID) + CAM_GRID) % CAM_GRID;
    cctx.lineWidth = 1;
    cctx.strokeStyle = 'rgba(148, 163, 184, 0.07)';
    cctx.beginPath();
    for (let gx = offX; gx < w; gx += CAM_GRID) {
      cctx.moveTo(Math.round(gx) + 0.5, 0);
      cctx.lineTo(Math.round(gx) + 0.5, h);
    }
    for (let gy = offY; gy < h; gy += CAM_GRID) {
      cctx.moveTo(0, Math.round(gy) + 0.5);
      cctx.lineTo(w, Math.round(gy) + 0.5);
    }
    cctx.stroke();
  }

  const camX = cam.x;
  const camY = cam.y;

  while (trail.length && now - trail[0].t > TRAIL_MS) trail.shift();

  cctx.lineCap = 'round';
  for (let i = 1; i < trail.length; i++) {
    const a = 1 - (now - trail[i].t) / TRAIL_MS;
    if (a <= 0) continue;
    cctx.strokeStyle = withAlpha(COLORS.v, a * 0.5);
    cctx.lineWidth = 0.8 + a * 1.8;
    cctx.beginPath();
    cctx.moveTo(trail[i - 1].x - camX, trail[i - 1].y - camY);
    cctx.lineTo(trail[i].x - camX, trail[i].y - camY);
    cctx.stroke();
  }

  if (!pointer) return;

  // coordonnées écran = coordonnées monde − caméra
  const x = pointer.x - camX;
  const y = pointer.y - camY;

  cctx.setLineDash([3, 5]);
  cctx.lineWidth = 1;
  cctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
  cctx.beginPath();
  cctx.moveTo(0, Math.round(y) + 0.5);
  cctx.lineTo(w, Math.round(y) + 0.5);
  cctx.moveTo(Math.round(x) + 0.5, 0);
  cctx.lineTo(Math.round(x) + 0.5, h);
  cctx.stroke();
  cctx.setLineDash([]);

  cctx.strokeStyle = withAlpha(COLORS.v, 0.4);
  cctx.beginPath();
  cctx.arc(x, y, 13, 0, Math.PI * 2);
  cctx.stroke();

  cctx.fillStyle = COLORS.v;
  cctx.beginPath();
  cctx.arc(x, y, 2.8, 0, Math.PI * 2);
  cctx.fill();
}

/* ---------------- rendu : graphiques ---------------- */

function drawChart(ctx, canvas, series, opts) {
  const w = canvas._w || 0;
  const h = canvas._h || 0;
  if (!w || !h) return;

  const s = dpr();
  ctx.setTransform(s, 0, 0, s, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;

  const now = performance.now();
  const t0 = now - WINDOW_MS;

  const padL = 46;
  const padR = 14;
  const padT = 26;
  const padB = 22;
  const pw = Math.max(1, w - padL - padR);
  const ph = Math.max(1, h - padT - padB);

  // échelle
  let dMax = 0;
  let dMin = 0;
  for (const smp of samples) {
    for (const sr of series) {
      const val = smp[sr.key];
      if (val > dMax) dMax = val;
      if (val < dMin) dMin = val;
    }
  }
  const target = opts.symmetric
    ? niceMax(Math.max(Math.abs(dMax), Math.abs(dMin)))
    : niceMax(dMax);
  opts.state.max += (target - opts.state.max) * 0.18;
  const max = Math.max(1e-6, opts.state.max);

  const yOf = opts.symmetric
    ? (v) => padT + ph / 2 - (v / max) * (ph / 2)
    : (v) => padT + ph - (v / max) * ph;
  const xOf = (t) => padL + ((t - t0) / WINDOW_MS) * pw;

  // grille horizontale
  const rows = [0, 0.25, 0.5, 0.75, 1];
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.grid;
  ctx.beginPath();
  for (const f of rows) {
    const y = Math.round(padT + ph * (1 - f)) + 0.5;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
  }
  ctx.stroke();

  // grille verticale (secondes)
  ctx.beginPath();
  for (let i = 0; i <= 3; i++) {
    const x = Math.round(xOf(t0 + (WINDOW_MS * i) / 3)) + 0.5;
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + ph);
  }
  ctx.stroke();

  // axe zéro (graphique symétrique)
  if (opts.symmetric) {
    ctx.strokeStyle = COLORS.axis;
    ctx.beginPath();
    const y = Math.round(yOf(0)) + 0.5;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
  }

  // étiquettes Y
  ctx.fillStyle = COLORS.text;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const f of rows) {
    const value = opts.symmetric ? (f * 2 - 1) * max : f * max;
    ctx.fillText(fmtAxis(value, max), padL - 8, yOf(value));
  }

  // étiquettes X
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 3; i++) {
    const label = i === 3 ? '0' : `${i - 3} s`;
    ctx.fillText(label, xOf(t0 + (WINDOW_MS * i) / 3), padT + ph + 6);
  }

  // légende (plusieurs séries uniquement)
  if (series.length > 1) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    let x = padL + pw;
    for (let i = series.length - 1; i >= 0; i--) {
      const sr = series[i];
      ctx.fillStyle = sr.color;
      const label = sr.label;
      ctx.fillText(label, x, 8);
      x -= ctx.measureText(label).width + 14;
    }
  }

  // courbes
  ctx.save();
  ctx.beginPath();
  ctx.rect(padL, padT, pw, ph);
  ctx.clip();

  if (samples.length > 1) {
    const trace = (key) => {
      ctx.beginPath();
      for (let i = 0; i < samples.length; i++) {
        const smp = samples[i];
        const x = xOf(smp.t);
        const y = yOf(smp[key]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    };

    for (const sr of series) {
      if (opts.fill) {
        trace(sr.key);
        const base = padT + ph;
        ctx.lineTo(xOf(samples[samples.length - 1].t), base);
        ctx.lineTo(xOf(samples[0].t), base);
        ctx.closePath();
        const g = ctx.createLinearGradient(0, padT, 0, padT + ph);
        g.addColorStop(0, withAlpha(sr.color, 0.26));
        g.addColorStop(1, withAlpha(sr.color, 0));
        ctx.fillStyle = g;
        ctx.fill();
      }

      trace(sr.key);
      ctx.strokeStyle = sr.color;
      ctx.lineWidth = 1.6;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // point courant
    const last = samples[samples.length - 1];
    for (const sr of series) {
      ctx.fillStyle = sr.color;
      ctx.beginPath();
      ctx.arc(xOf(last.t), yOf(last[sr.key]), 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    ctx.restore();
    ctx.fillStyle = COLORS.empty;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('en attente de mouvement', padL + pw / 2, padT + ph / 2);
    return;
  }

  ctx.restore();
}

/* ---------------- transformée de Fourier (FFT radix-2) ---------------- */

function fft(re, im) {
  const n = re.length;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + half];
        const bi = im[i + k + half];
        const vr = br * cr - bi * ci;
        const vi = br * ci + bi * cr;
        re[i + k] = ar + vr;
        im[i + k] = ai + vi;
        re[i + k + half] = ar - vr;
        im[i + k + half] = ai - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

function computeSpectrum(now) {
  const t0 = now - WINDOW_MS;
  let first = 0;
  while (first < samples.length && samples[first].t < t0) first++;
  const count = samples.length - first;

  if (count < 24) {
    for (let k = 0; k < BINS; k++) specMag[k] *= 0.7;
    specPeakHz = 0;
    return;
  }

  // rééchantillonnage linéaire sur NFFT points régulièrement espacés
  const last = samples.length - 1;
  let idx = first;
  for (let i = 0; i < NFFT; i++) {
    const t = t0 + (i / (NFFT - 1)) * WINDOW_MS;
    while (idx < last && samples[idx + 1].t <= t) idx++;
    const a = samples[idx];
    const b = samples[Math.min(idx + 1, last)];
    const span = b.t - a.t;
    const f = span > 0 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 0;
    specVals[i] = a.v + (b.v - a.v) * f;
  }

  // moyenne nulle + fenêtre de Hann (limite les fuites spectrales)
  let mean = 0;
  for (let i = 0; i < NFFT; i++) mean += specVals[i];
  mean /= NFFT;

  let wsum = 0;
  for (let i = 0; i < NFFT; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NFFT - 1));
    wsum += w;
    specRe[i] = (specVals[i] - mean) * w;
    specIm[i] = 0;
  }

  fft(specRe, specIm);

  let peakK = 0;
  let peakM = 0;
  for (let k = 0; k < BINS; k++) {
    const mag = (2 * Math.hypot(specRe[k], specIm[k])) / wsum;
    specMag[k] += (mag - specMag[k]) * 0.3;
    if (k > 0 && specMag[k] > peakM) {
      peakM = specMag[k];
      peakK = k;
    }
  }
  // un pic n'est retenu que s'il émerge du niveau de fond (vitesse ~constante = pas de pic)
  const threshold = Math.max(5, 0.05 * mean);
  specPeakHz = peakM > threshold ? peakK * DF : 0;
}

function drawSpectrum(ctx, canvas) {
  const w = canvas._w || 0;
  const h = canvas._h || 0;
  if (!w || !h) return;

  const s = dpr();
  ctx.setTransform(s, 0, 0, s, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;

  const padL = 46;
  const padR = 14;
  const padT = 26;
  const padB = 22;
  const pw = Math.max(1, w - padL - padR);
  const ph = Math.max(1, h - padT - padB);

  let dMax = 0;
  for (let k = 0; k < BINS; k++) if (specMag[k] > dMax) dMax = specMag[k];
  const target = niceMax(dMax);
  specState.max += (target - specState.max) * 0.18;
  const max = Math.max(1e-6, specState.max);

  const xOf = (hz) => padL + (hz / SPECTRUM_MAX_HZ) * pw;

  // grille horizontale
  const rows = [0, 0.25, 0.5, 0.75, 1];
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.grid;
  ctx.beginPath();
  for (const f of rows) {
    const y = Math.round(padT + ph * (1 - f)) + 0.5;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
  }
  ctx.stroke();

  // grille verticale (Hz)
  ctx.beginPath();
  for (const hz of [0, 5, 10, 15, 20]) {
    const x = Math.round(xOf(hz)) + 0.5;
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + ph);
  }
  ctx.stroke();

  // étiquettes Y / X
  ctx.fillStyle = COLORS.text;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const f of rows) {
    ctx.fillText(fmtAxis(f * max, max), padL - 8, padT + ph * (1 - f));
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const hz of [0, 5, 10, 15, 20]) {
    ctx.fillText(String(hz), xOf(hz), padT + ph + 6);
  }

  // barres spectrales
  let bestK = -1;
  let bestM = 0;
  for (let k = 0; k < BINS; k++) {
    if (specMag[k] > bestM) {
      bestM = specMag[k];
      bestK = k;
    }
  }

  if (bestM < 0.5) {
    ctx.fillStyle = COLORS.empty;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('en attente de mouvement', padL + pw / 2, padT + ph / 2);
    return;
  }

  const barW = pw / BINS;
  const base = padT + ph;
  for (let k = 0; k < BINS; k++) {
    const m = Math.min(max, specMag[k]);
    if (m <= 0) continue;
    const height = (m / max) * ph;
    const x = padL + k * barW + barW * 0.2;
    ctx.fillStyle = k === bestK ? COLORS.s : withAlpha(COLORS.s, 0.55);
    ctx.fillRect(x, base - height, Math.max(1, barW * 0.6), height);
  }
}

/* ---------------- statistiques ---------------- */

function updateStats(now) {
  if (now - lastStatT < 120) return;
  lastStatT = now;

  const t0 = now - WINDOW_MS;
  let peak = 0;
  let count = 0;
  let current = 0;
  let deriv = 0;

  for (const smp of samples) {
    if (smp.t < t0) continue;
    peak = Math.max(peak, smp.v);
    count++;
    current = smp.v;
    deriv = smp.a;
  }

  statNow.textContent = count ? `${fmtSpeed(current)} px/s` : '—';
  statPeak.textContent = count ? `${fmtSpeed(peak)} px/s` : '—';
  statDeriv.textContent = count ? `${deriv > 0 ? '+' : ''}${fmtSpeed(deriv)} px/s²` : '—';
  statFreq.textContent = specPeakHz > 0 ? `${specPeakHz.toFixed(1)} Hz` : '—';

  const sig = signatureOf(samples);
  if (sig) {
    if (mode === 'manuel') renderSignature('geste', sig);
    else if (mode === 'ia') renderSignature('modele', sig);
  }
}

/* ---------------- boucle d'animation ---------------- */

let lastFrameT = performance.now();

function frame(now) {
  const dt = Math.min(64, Math.max(0, now - lastFrameT));
  lastFrameT = now;

  stepSimulation(now, dt);
  update(now, dt);

  // collecte des couples (vx, vy) pour l'apprentissage : vitesse réelle
  // image par image (pas de lissage, sinon le tremblement du geste disparaît)
  if (ML.recording && mode === 'manuel' && pointer && dt > 0) {
    if (ML.lastPos) {
      ML.samples.push([
        ((pointer.x - ML.lastPos.x) / dt) * 1000,
        ((pointer.y - ML.lastPos.y) / dt) * 1000,
      ]);
      if (ML.samples.length > ML_MAX) ML.samples.splice(0, ML.samples.length - ML_MAX);
    }
    ML.lastPos = { x: pointer.x, y: pointer.y };
  } else if (!ML.recording) {
    ML.lastPos = null;
  }

  computeSpectrum(now);
  drawCapture(now);
  drawChart(
    sctx,
    speedEl,
    [{ key: 'v', color: COLORS.v, label: '|v|' }],
    { symmetric: false, fill: true, state: speedState },
  );
  drawChart(
    kctx,
    accelEl,
    [{ key: 'a', color: COLORS.a, label: 'dv/dt' }],
    { symmetric: true, fill: false, state: accelState },
  );
  drawSpectrum(pctx, spectrumEl);
  updateStats(now);

  if (now - lastStatusT > 400) {
    lastStatusT = now;
    updateStatus();
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
