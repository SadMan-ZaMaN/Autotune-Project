// =====================================================================
// Consonance — step-by-step graphs page (/steps/<job_id>).
//
// Shows every processing step one tuning job went through: the server
// keeps result.stages with each job (one entry per step that ran, each
// with its numbers, an explanation and 1-2 charts already reduced to plot
// data by src/autotune/stage_plots.py). This page fetches the job from
// /status/<job_id> and only draws. Opened from the main page's result;
// ?edited=<job_id> adds a Tuned/Edited switch, ?side=edited starts on it,
// ?style=<name> is the style label to show.
// =====================================================================

const $ = (id) => document.getElementById(id);

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtTime(seconds) {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

// Sizes a canvas for the screen's pixel density; returns {ctx, w, h} in CSS px.
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height, dpr };
}

// ---------- Step-by-step view ----------
// Chart types: wave, lines, bars, spectrogram.
const stagesEl = $('stages');
const stagesMeta = $('stages-meta');
const stageStepsEl = $('stage-steps');
const stageTitle = $('stage-title');
const stageSummary = $('stage-summary');
const stageCharts = $('stage-charts');
const stageStats = $('stage-stats');
const stageExplain = $('stage-explain');
const stagePrev = $('stage-prev');
const stageNext = $('stage-next');

let stageResult = null;   // the result whose steps are shown (Tuned or Edited)
let stageIndex = 0;

// stage_plots.py names colours by role: steel = the original voice, amber = the result
const CHART_COLORS = { steel: '--voice', amber: '--accent', edit: '--edit', muted: '--text-muted', text: '--text' };
const CHART_PAD = { left: 50, right: 12, top: 12, bottom: 28 };

function chartColor(name) {
  return cssVar(CHART_COLORS[name] || '--text');
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// "Nice" tick spacing: 1, 2 or 5 times a power of ten, about `count` ticks.
function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * power >= raw) return m * power;
  return 10 * power;
}

function fmtTick(v) {
  if (Math.abs(v) >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) >= 10 || v === 0) return `${Math.round(v)}`;
  return `${+v.toFixed(2)}`;
}

function fmtSeconds(t, duration) {
  return duration < 10 ? `${+t.toFixed(2)} s` : fmtTime(t);
}

// Plot area + value -> pixel maps. x can be logarithmic (frequency axes).
function makeFrame(w, h, xMin, xMax, yMin, yMax, xLog) {
  const plotW = w - CHART_PAD.left - CHART_PAD.right;
  const plotH = h - CHART_PAD.top - CHART_PAD.bottom;
  const xOf = xLog
    ? (x) => CHART_PAD.left + (Math.log(x) - Math.log(xMin)) / (Math.log(xMax) - Math.log(xMin)) * plotW
    : (x) => CHART_PAD.left + (x - xMin) / (xMax - xMin) * plotW;
  const yOf = (y) => CHART_PAD.top + (1 - (y - yMin) / (yMax - yMin)) * plotH;
  return { plotW, plotH, xOf, yOf };
}

function drawYGrid(ctx, w, f, ticks) {
  const muted = cssVar('--text-muted');
  ctx.font = `11px ${cssVar('--font-mono') || 'monospace'}`;
  ctx.textBaseline = 'middle';
  let lastY = Infinity;
  for (const [value, label] of ticks) {
    const y = Math.round(f.yOf(value)) + 0.5;
    if (y < CHART_PAD.top - 1 || y > CHART_PAD.top + f.plotH + 1) continue;
    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CHART_PAD.left, y);
    ctx.lineTo(w - CHART_PAD.right, y);
    ctx.stroke();
    if (Math.abs(lastY - y) >= 12) {   // skip labels that would overlap
      ctx.fillStyle = muted;
      ctx.fillText(label, 4, y);
      lastY = y;
    }
  }
}

function drawXAxis(ctx, h, f, ticks) {
  ctx.fillStyle = cssVar('--text-muted');
  ctx.font = `11px ${cssVar('--font-mono') || 'monospace'}`;
  ctx.textBaseline = 'alphabetic';
  let lastX = -Infinity;
  for (const [value, label] of ticks) {
    const x = f.xOf(value);
    if (x < CHART_PAD.left - 1 || x > CHART_PAD.left + f.plotW + 1) continue;
    const width = ctx.measureText(label).width;
    const left = Math.max(CHART_PAD.left, Math.min(x - width / 2, CHART_PAD.left + f.plotW - width));
    if (left < lastX + 8) continue;
    ctx.fillText(label, left, h - 8);
    lastX = left + width;
  }
}

function linearTicks(min, max, count, format) {
  const step = niceStep(max - min, count);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) {
    const value = Math.abs(v) < step * 1e-6 ? 0 : v;
    ticks.push([value, format ? format(value) : fmtTick(value)]);
  }
  return ticks;
}

function logTicks(min, max) {
  const ticks = [];
  for (let p = Math.floor(Math.log10(min)); p <= Math.ceil(Math.log10(max)); p++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, p);
      if (v >= min && v <= max) ticks.push([v, fmtTick(v)]);
    }
  }
  return ticks;
}

function timeTicks(duration, plotW) {
  return linearTicks(0, duration, Math.max(2, Math.floor(plotW / 80)), (t) => fmtSeconds(t, duration));
}

function drawWaveChart(canvas, chart) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  let peak = 1e-6;
  for (const s of chart.series) {
    for (let i = 0; i < s.min.length; i++) peak = Math.max(peak, Math.abs(s.min[i]), Math.abs(s.max[i]));
  }
  const f = makeFrame(w, h, 0, chart.duration, -peak * 1.05, peak * 1.05, false);
  drawYGrid(ctx, w, f, [[-peak, fmtTick(-peak)], [0, '0'], [peak, fmtTick(peak)]]);
  drawXAxis(ctx, h, f, timeTicks(chart.duration, f.plotW));
  chart.series.forEach((s, index) => {
    ctx.strokeStyle = chartColor(s.color);
    ctx.globalAlpha = index === 0 ? 0.9 : 0.75;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const n = s.min.length;
    for (let i = 0; i < n; i++) {
      const x = CHART_PAD.left + (i + 0.5) / n * f.plotW;
      ctx.moveTo(x, f.yOf(s.max[i]));
      ctx.lineTo(x, f.yOf(s.min[i]) + 0.5);
    }
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function drawLinesChart(canvas, chart) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const f = makeFrame(w, h, chart.x_min, chart.x_max, chart.y_min, chart.y_max, chart.x_log);
  const yTicks = chart.y_ticks || linearTicks(chart.y_min, chart.y_max, Math.max(3, Math.floor(f.plotH / 40)));
  drawYGrid(ctx, w, f, yTicks);
  let xTicks;
  if (chart.x_log) xTicks = logTicks(chart.x_min, chart.x_max);
  else if (chart.x_label.startsWith('time (s)')) xTicks = timeTicks(chart.x_max, f.plotW);
  else xTicks = linearTicks(chart.x_min, chart.x_max, Math.max(2, Math.floor(f.plotW / 80)));
  drawXAxis(ctx, h, f, xTicks);

  ctx.save();
  ctx.beginPath();
  ctx.rect(CHART_PAD.left, CHART_PAD.top, f.plotW, f.plotH);
  ctx.clip();
  for (const s of chart.series) {
    ctx.strokeStyle = chartColor(s.color);
    ctx.lineWidth = s.width || 1.5;
    ctx.lineJoin = 'round';
    ctx.setLineDash(s.dash ? [5, 4] : []);
    ctx.beginPath();
    let penDown = false;
    let prevY = null;
    let prevV = null;
    for (let i = 0; i < s.x.length; i++) {
      const v = s.y[i];
      if (v === null || v === undefined) { penDown = false; prevV = null; continue; }
      const x = f.xOf(s.x[i]);
      const y = f.yOf(v);
      // an octave glitch in the pitch track: lift the pen instead of drawing a spike
      const jumped = penDown && prevV !== null && (
        (s.max_jump && Math.abs(v - prevV) > s.max_jump) ||
        (s.jump_ratio && Math.max(v / prevV, prevV / v) > s.jump_ratio));
      prevV = v;
      if (!penDown || jumped) ctx.moveTo(x, y);
      else if (s.step) { ctx.lineTo(x, prevY); ctx.lineTo(x, y); }   // hold the value, then jump
      else ctx.lineTo(x, y);
      penDown = true;
      prevY = y;
    }
    ctx.stroke();
  }
  ctx.restore();
  ctx.setLineDash([]);
}

function drawBarsChart(canvas, chart) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const top = Math.max(1, ...chart.values) * 1.18;
  const f = makeFrame(w, h, 0, chart.values.length, 0, top, false);
  drawYGrid(ctx, w, f, linearTicks(0, top, 4, (v) => `${fmtTick(v)}%`));
  const slot = f.plotW / chart.values.length;
  ctx.font = `11px ${cssVar('--font-mono') || 'monospace'}`;
  chart.values.forEach((v, i) => {
    const x = CHART_PAD.left + i * slot + slot * 0.18;
    const bw = slot * 0.64;
    const y = f.yOf(v);
    ctx.fillStyle = chart.highlight[i] ? cssVar('--accent') : cssVar('--note-soft');
    ctx.fillRect(x, y, bw, f.yOf(0) - y);
    ctx.fillStyle = cssVar('--text-muted');
    ctx.textBaseline = 'alphabetic';
    const label = chart.labels[i];
    ctx.fillText(label, x + bw / 2 - ctx.measureText(label).width / 2, h - 8);
    if (v > 0 && bw > 18) {
      const text = `${Math.round(v)}`;
      ctx.fillText(text, x + bw / 2 - ctx.measureText(text).width / 2, y - 4);
    }
  });
}

// 0..255 -> colour: dark -> steel -> amber -> near white (the page's palette)
const SPEC_LUT = (() => {
  const stops = [[0, [16, 18, 21]], [0.4, [31, 58, 82]], [0.65, [92, 137, 172]], [0.85, [232, 163, 61]], [1, [255, 243, 214]]];
  const lut = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k];
    const [t1, c1] = stops[k + 1];
    const u = (t - t0) / (t1 - t0);
    lut.push([0, 1, 2].map((j) => Math.round(c0[j] + (c1[j] - c0[j]) * u)));
  }
  return lut;
})();

function drawSpectrogramChart(canvas, chart) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const gap = 16;
  const count = chart.panels.length;
  const plotW = w - CHART_PAD.left - CHART_PAD.right;
  const panelH = (h - CHART_PAD.top - CHART_PAD.bottom - gap * (count - 1)) / count;
  const muted = cssVar('--text-muted');
  ctx.font = `11px ${cssVar('--font-mono') || 'monospace'}`;

  chart.panels.forEach((panel, p) => {
    const top = CHART_PAD.top + p * (panelH + gap);
    // decode the 8-bit image (row 0 = lowest frequency) into pixels
    const bytes = Uint8Array.from(atob(panel.data), (c) => c.charCodeAt(0));
    const img = new ImageData(panel.cols, panel.rows);
    for (let r = 0; r < panel.rows; r++) {
      const outRow = panel.rows - 1 - r;   // high frequencies at the top
      for (let c = 0; c < panel.cols; c++) {
        const rgb = SPEC_LUT[bytes[r * panel.cols + c]];
        const o = (outRow * panel.cols + c) * 4;
        img.data[o] = rgb[0];
        img.data[o + 1] = rgb[1];
        img.data[o + 2] = rgb[2];
        img.data[o + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = panel.cols;
    off.height = panel.rows;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, CHART_PAD.left, top, plotW, panelH);

    // log-frequency axis
    const yOf = (freq) => top + (1 - (Math.log(freq) - Math.log(chart.fmin)) / (Math.log(chart.fmax) - Math.log(chart.fmin))) * panelH;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = muted;
    let lastY = Infinity;
    for (const [freq, label] of logTicks(chart.fmin, chart.fmax)) {
      const y = yOf(freq);
      if (Math.abs(lastY - y) < 11) continue;
      ctx.fillText(`${label}`, 4, y);
      lastY = y;
    }
    // panel name, on a dark chip so it reads over the image
    ctx.textBaseline = 'top';
    const tw = ctx.measureText(panel.label).width;
    ctx.fillStyle = 'rgba(16, 18, 21, 0.8)';
    ctx.fillRect(CHART_PAD.left + 4, top + 4, tw + 10, 16);
    ctx.fillStyle = cssVar('--text');
    ctx.fillText(panel.label, CHART_PAD.left + 9, top + 7);
  });
  const f = makeFrame(w, h, 0, chart.duration, 0, 1, false);
  drawXAxis(ctx, h, f, timeTicks(chart.duration, plotW));
  ctx.textBaseline = 'alphabetic';
}

const CHART_DRAWERS = { wave: drawWaveChart, lines: drawLinesChart, bars: drawBarsChart, spectrogram: drawSpectrogramChart };

function chartLegend(chart) {
  let items = [];
  if (chart.series) {
    items = chart.series.map((s) => {
      const color = chartColor(s.color);
      const swatch = s.dash ? `<i class="dash" style="border-color:${color}"></i>` : `<i style="background:${color}"></i>`;
      return `<span class="lg">${swatch}${escapeHtml(s.label)}</span>`;
    });
  } else if (chart.type === 'bars') {
    items = [`<span class="lg"><i style="background:${cssVar('--accent')}"></i>in the key</span>`,
      `<span class="lg"><i style="background:${cssVar('--note-soft')}"></i>not in the key</span>`];
  } else if (chart.type === 'spectrogram') {
    items = [`<span class="lg">brighter = louder · ${chart.db_range} dB range · log frequency</span>`];
  }
  return `<span class="stage-legend">${items.join('')}</span>`;
}

function drawStageCharts() {
  const stage = stageResult && stageResult.stages ? stageResult.stages[stageIndex] : null;
  if (!stage) return;
  stageCharts.querySelectorAll('canvas').forEach((canvas, i) => {
    const chart = stage.charts[i];
    CHART_DRAWERS[chart.type](canvas, chart);
  });
}

// Step explanations come from stage_plots.py with a tiny markup:
// `formula` spans, blank lines between paragraphs, and a last paragraph
// starting with "Graph: " (what to look for in the chart).
function prettyMath(text) {
  return escapeHtml(text)
    .replace(/\^(-?\w+)/g, '<sup>$1</sup>')        // x^2, z^-1, r^strength
    .replace(/\bsum_m\b/g, 'Σ<sub>m</sub>')
    .replace(/\bsum\b/g, 'Σ')
    .replace(/\bpi\b/g, 'π')
    .replace(/\balpha\b/g, 'α')
    .replace(/ ~ /g, ' ≈ ')
    .replace(/(\s)- /g, '$1− ');
}

function prettyProse(text) {
  return escapeHtml(text)
    .replace(/ - /g, ' — ')
    .replace(/~(?=\d)/g, '≈');
}

function renderInline(text) {
  return text.split('`').map((part, i) => {
    if (i % 2 === 0) return prettyProse(part);
    const cls = part.length <= 28 ? 'math short' : 'math';
    return `<code class="${cls}">${prettyMath(part)}</code>`;
  }).join('');
}

function renderExplain(text) {
  const paragraphs = String(text).split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const html = paragraphs.map((p) => {
    if (p.startsWith('Graph:')) {
      let rest = p.slice(6).trim();
      rest = rest.charAt(0).toUpperCase() + rest.slice(1);
      return `<p class="look"><span>${renderInline(rest)}</span></p>`;
    }
    const wholeFormula = /^`[^`]+`$/.test(p);
    return `<p${wholeFormula ? ' class="formula-line"' : ''}>${renderInline(p)}</p>`;
  }).join('');
  return `<h5 class="explain-heading">How it works</h5>${html}`;
}

function showStage(index) {
  const stages = stageResult.stages;
  stageIndex = Math.max(0, Math.min(stages.length - 1, index));
  const stage = stages[stageIndex];
  stageStepsEl.querySelectorAll('.stage-step').forEach((el, i) => {
    const on = i === stageIndex;
    el.classList.toggle('active', on);
    el.setAttribute('aria-selected', on ? 'true' : 'false');
    el.tabIndex = on ? 0 : -1;
  });
  stageTitle.textContent = `${String(stageIndex + 1).padStart(2, '0')} · ${stage.title}`;
  stageSummary.textContent = stage.summary;
  stageCharts.innerHTML = stage.charts.map((chart) => {
    const tall = chart.type === 'spectrogram' ? ' tall' : '';
    return `<div class="stage-chart"><div class="stage-chart-label"><span>${escapeHtml(chart.title)}</span>${chartLegend(chart)}</div>` +
      `<canvas class="stage-canvas${tall}" role="img" aria-label="${escapeHtml(chart.title)}"></canvas></div>`;
  }).join('');
  stageStats.innerHTML = stage.stats.map(([label, value]) =>
    `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('');
  stageExplain.innerHTML = renderExplain(stage.explain);
  stagePrev.disabled = stageIndex === 0;
  stageNext.disabled = stageIndex === stages.length - 1;
  drawStageCharts();
}

// Shows the steps of one version (Tuned or Edited), staying on the same
// step when switching between them.
function renderStages(source, label) {
  const previousId = stageResult && stageResult.stages[stageIndex] ? stageResult.stages[stageIndex].id : null;
  const changed = source !== stageResult;
  stageResult = source;
  stagesMeta.textContent = `${label} · ${source.stages.length} steps`;
  if (changed) {
    stageStepsEl.innerHTML = source.stages.map((stage, i) =>
      `<button type="button" class="stage-step" role="tab" data-index="${i}">` +
      `<span class="num">${String(i + 1).padStart(2, '0')}</span>${escapeHtml(stage.title)}</button>`).join('');
  }
  let index = source.stages.findIndex((s) => s.id === previousId);
  if (index < 0) index = changed && previousId ? 0 : stageIndex;
  showStage(index);
}

stageStepsEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.stage-step');
  if (btn) showStage(Number(btn.dataset.index));
});
stageStepsEl.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
  e.preventDefault();
  showStage(stageIndex + (e.key === 'ArrowRight' ? 1 : -1));
  stageStepsEl.querySelectorAll('.stage-step')[stageIndex].focus();
});
stagePrev.addEventListener('click', () => showStage(stageIndex - 1));
stageNext.addEventListener('click', () => showStage(stageIndex + 1));


// ---------- Loading the job ----------
const pageMessage = $('page-message');
const versionSwitch = $('version-switch');
const params = new URLSearchParams(location.search);
const styleLabel = params.get('style') || 'Custom';
const versions = {};   // 'tuned' / 'edited' -> result

async function fetchResult(jobId) {
  const res = await fetch(`/status/${encodeURIComponent(jobId)}`);
  const job = await res.json();
  if (!res.ok || job.state !== 'done' || !job.result || !job.result.stages) return null;
  return job.result;
}

function showVersion(side) {
  const result = versions[side];
  versionSwitch.querySelectorAll('.ab-option').forEach((el) => {
    const on = el.dataset.side === side;
    el.classList.toggle('active', on);
    el.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  renderStages(result, side === 'edited' ? `${styleLabel} + your edits` : styleLabel);
}

function describe(result) {
  const key = result.key_type === 'chromatic' ? 'all 12 notes'
    : `${result.key_root} ${result.key_type === 'major' ? 'major' : 'minor'}`;
  return `${fmtTime(result.duration)} of audio · key ${key} · processed in ${result.processing_time} s`;
}

async function loadPage() {
  const jobId = decodeURIComponent(location.pathname.split('/').filter(Boolean).pop() || '');
  try {
    versions.tuned = await fetchResult(jobId);
    if (params.get('edited')) versions.edited = await fetchResult(params.get('edited'));
  } catch (err) {
    console.error(err);
  }
  if (!versions.tuned) {
    pageMessage.textContent = "This result isn't on the server any more (it only keeps results until it restarts). Tune your voice again, then open the steps from the result.";
    pageMessage.hidden = false;
    return;
  }
  $('job-meta').textContent = describe(versions.tuned);
  stagesEl.hidden = false;
  if (versions.edited) {
    versionSwitch.hidden = false;
    versionSwitch.querySelectorAll('.ab-option').forEach((el) =>
      el.addEventListener('click', () => showVersion(el.dataset.side)));
  }
  showVersion(params.get('side') === 'edited' && versions.edited ? 'edited' : 'tuned');
}

// "Back to your recording": the main page restores the recording and the
// result for this tab by itself (script.js, "Keeping your work across
// pages"), so a plain link to / would do. When we came from the main page,
// going BACK in history is better still - the browser may show the page
// exactly as it was, without re-fetching anything.
document.querySelectorAll('.back-link').forEach((link) => {
  link.addEventListener('click', (e) => {
    let fromMainPage = false;
    try {
      const ref = new URL(document.referrer);
      fromMainPage = ref.origin === location.origin && ref.pathname === '/';
    } catch (err) { /* no referrer */ }
    if (fromMainPage && history.length > 1) {
      e.preventDefault();
      history.back();
    }
  });
});

let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(drawStageCharts, 120);
});

loadPage();
