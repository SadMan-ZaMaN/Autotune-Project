// =====================================================================
// Console — frontend for the pitch-correction pipeline.
//
// Flow: (1) get audio (file or microphone) -> decode it IN THE BROWSER and
// re-encode as a mono 44.1 kHz WAV, so the Python side only ever sees WAV
// (this is what makes MP3/M4A/WebM recordings work with no ffmpeg);
// (2) pick a style preset; (3) POST to /process, poll /status for progress,
// then play original vs tuned in a sample-synchronised A/B player and draw
// the pitch graph returned by the server.
// =====================================================================

const TARGET_SR = 44100;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const SCALE_STEPS = {
  major: [0, 2, 4, 5, 7, 9, 11],
  natural_minor: [0, 2, 3, 5, 7, 8, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

// ---------- State ----------
let audioCtx = null;
let inputWav = null;        // Blob (audio/wav) that gets uploaded
let inputSamples = null;    // Float32Array, for redrawing the input waveform
let inputUrl = null;
let presets = {};
let activePreset = null;
let lastResult = null;

// ---------- Elements ----------
const $ = (id) => document.getElementById(id);
const dropzone = $('dropzone');
const dropzoneText = $('dropzone-text');
const fileInput = $('file-input');
const recordBtn = $('record-btn');
const recordLabel = $('record-label');
const recMeter = $('rec-meter');
const recTime = $('rec-time');
const meterFill = $('meter-fill');
const sourceError = $('source-error');
const inputBlock = $('input-block');
const inputMeta = $('input-meta');
const traceInput = $('trace-input');
const audioInputEl = $('audio-input');

const presetsEl = $('presets');
const scaleRoot = $('scale-root');
const scaleType = $('scale-type');
const strengthSlider = $('strength');
const strengthReadout = $('strength-readout');
const retuneSlider = $('retune');
const retuneReadout = $('retune-readout');
const reverbSlider = $('reverb');
const reverbReadout = $('reverb-readout');
const usePolish = $('use-polish');
const useFollow = $('use-follow');
const useFormant = $('use-formant');
const usePhaseVocoder = $('use-phase-vocoder');
const usePreemphasis = $('use-preemphasis');
const runBtn = $('run-btn');
const progressEl = $('progress');
const progressFill = $('progress-fill');
const progressStage = $('progress-stage');
const statusLine = $('status-line');

const emptyState = $('empty-state');
const resultEl = $('result');
const playBtn = $('play-btn');
const iconPlay = $('icon-play');
const iconPause = $('icon-pause');
const playTime = $('play-time');
const abOptions = document.querySelectorAll('.ab-option');
const waveLabel = $('wave-label');
const traceOutput = $('trace-output');
const pitchGraph = $('pitch-graph');
const chipsEl = $('chips');
const downloadBtn = $('download-btn');
const scopeCanvas = $('scope');
const statusbar = $('statusbar');

// ---------- Small helpers ----------
function getCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtTime(seconds) {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

// ---------- Decoding: any format -> mono 44.1 kHz -> WAV ----------
// The browser's own decoder (decodeAudioData) handles MP3, M4A/AAC, OGG,
// FLAC, WAV and the WebM/Opus that MediaRecorder produces. An
// OfflineAudioContext with ONE output channel at 44.1 kHz then mixes
// stereo down to mono and resamples in one go.
async function decodeToMono(arrayBuffer) {
  const decoded = await getCtx().decodeAudioData(arrayBuffer);
  const length = Math.max(1, Math.ceil(decoded.duration * TARGET_SR));
  const offline = new OfflineAudioContext(1, length, TARGET_SR);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  return offline.startRendering();
}

// 16-bit PCM WAV: 44-byte header + samples (little-endian).
function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + n * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);            // fmt chunk size
  view.setUint16(20, 1, true);             // PCM
  view.setUint16(22, 1, true);             // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);             // block align
  view.setUint16(34, 16, true);            // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, n * 2, true);
  let offset = 44;
  for (let i = 0; i < n; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

// ---------- Waveform drawing ----------
function drawWaveformInto(ctx, w, h, data, color) {
  ctx.clearRect(0, 0, w, h);
  if (!data || !data.length) return;
  const step = Math.max(1, Math.ceil(data.length / w));
  const mid = h / 2;
  // scale so the loudest part fills ~90% of the height
  let peak = 1e-6;
  for (let i = 0; i < data.length; i += Math.max(1, Math.floor(step / 4))) {
    const v = Math.abs(data[i]);
    if (v > peak) peak = v;
  }
  const scale = (mid * 0.9) / peak;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x++) {
    let min = 0;
    let max = 0;
    const start = Math.floor(x * data.length / w);
    for (let i = 0; i < step && start + i < data.length; i++) {
      const v = data[start + i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, mid + min * scale);
    ctx.lineTo(x + 0.5, mid + max * scale + 0.5);
  }
  ctx.stroke();
}

function drawInputWaveform() {
  if (!inputSamples || inputBlock.hidden) return;
  const { ctx, w, h } = setupCanvas(traceInput);
  drawWaveformInto(ctx, w, h, inputSamples, cssVar('--steel'));
}

// ---------- Hero scope ----------
function drawIdleScope() {
  const { ctx, w, h } = setupCanvas(scopeCanvas);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(232, 163, 61, 0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const mid = h / 2;
  for (let x = 0; x < w; x++) {
    const y = mid + Math.sin(x * 0.02) * (h * 0.12) * Math.sin(x * 0.002);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawLiveScope(timeData) {
  const { ctx, w, h } = setupCanvas(scopeCanvas);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(232, 163, 61, 0.55)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const mid = h / 2;
  for (let x = 0; x < w; x++) {
    const v = timeData[Math.floor(x * timeData.length / w)];
    const y = mid + v * h * 0.8;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ---------- Loading a source ----------
async function loadSource(blob, name) {
  sourceError.textContent = '';
  dropzoneText.textContent = `Reading ${name}…`;
  runBtn.disabled = true;
  try {
    const mono = await decodeToMono(await blob.arrayBuffer());
    if (mono.duration < 0.5) throw new Error('too short');
    inputSamples = mono.getChannelData(0);
    inputWav = encodeWav(inputSamples, TARGET_SR);
    if (inputUrl) URL.revokeObjectURL(inputUrl);
    inputUrl = URL.createObjectURL(inputWav);
    audioInputEl.src = inputUrl;
    dropzoneText.textContent = name;
    inputBlock.hidden = false;
    inputMeta.textContent = `${fmtTime(mono.duration)} · mono · 44.1 kHz`;
    drawInputWaveform();
    runBtn.disabled = false;
    statusLine.textContent = 'Ready — pick a style, then press "Tune my voice".';
  } catch (err) {
    console.error(err);
    dropzoneText.textContent = 'Drop a recording here, or click to choose';
    sourceError.textContent = "Couldn't read that audio. Try a WAV or MP3 file, or record again (at least half a second).";
    if (inputWav) runBtn.disabled = false;
  }
}

// ---------- File input / drag-drop ----------
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) loadSource(e.dataTransfer.files[0], e.dataTransfer.files[0].name);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) loadSource(fileInput.files[0], fileInput.files[0].name);
  fileInput.value = '';   // so choosing the same file again still triggers
});

// ---------- Microphone recording (no time limit) ----------
let recorder = null;
let recStream = null;
let recSourceNode = null;
let recAnalyser = null;
let recChunks = [];
let recStartedAt = 0;
let recRaf = null;

function meterLoop() {
  const data = new Float32Array(recAnalyser.fftSize);
  recAnalyser.getFloatTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
  const rms = Math.sqrt(sum / data.length);
  const db = 20 * Math.log10(rms + 1e-9);
  const pct = Math.max(0, Math.min(100, (db + 60) / 60 * 100));   // -60 dB .. 0 dB
  meterFill.style.width = `${pct}%`;
  recTime.textContent = fmtTime((performance.now() - recStartedAt) / 1000);
  drawLiveScope(data);
  recRaf = requestAnimationFrame(meterLoop);
}

recordBtn.addEventListener('click', async () => {
  if (recorder && recorder.state === 'recording') {
    recorder.stop();
    return;
  }
  sourceError.textContent = '';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    sourceError.textContent = "This browser can't record audio. Use Chrome or Edge, or upload a file.";
    return;
  }
  try {
    // Echo cancellation / noise suppression / auto gain are tuned for
    // speech calls and damage sustained sung notes, so they're switched off.
    recStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    sourceError.textContent = 'Microphone access was blocked. Allow it (icon in the address bar) and try again.';
    return;
  }

  const ctx = getCtx();
  await ctx.resume();
  recSourceNode = ctx.createMediaStreamSource(recStream);
  recAnalyser = ctx.createAnalyser();
  recAnalyser.fftSize = 2048;
  recSourceNode.connect(recAnalyser);

  recChunks = [];
  recorder = new MediaRecorder(recStream);
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
  recorder.onstop = async () => {
    cancelAnimationFrame(recRaf);
    recStream.getTracks().forEach((t) => t.stop());
    recSourceNode.disconnect();
    const seconds = (performance.now() - recStartedAt) / 1000;
    recordBtn.classList.remove('recording');
    recordLabel.textContent = 'Record again';
    recMeter.hidden = true;
    drawIdleScope();
    const blob = new Blob(recChunks, { type: recorder.mimeType || 'audio/webm' });
    await loadSource(blob, `Microphone recording (${fmtTime(seconds)})`);
  };
  // a chunk every second keeps memory use flat for long recordings
  recorder.start(1000);
  recStartedAt = performance.now();
  recordBtn.classList.add('recording');
  recordLabel.textContent = 'Stop recording';
  recMeter.hidden = false;
  meterLoop();
});

// ---------- Presets ----------
const PRESET_ORDER = ['natural', 'studio', 'hard', 'raw'];

function renderPresets(defaultKey) {
  presetsEl.innerHTML = '';
  for (const key of PRESET_ORDER) {
    const p = presets[key];
    if (!p) continue;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset';
    btn.dataset.key = key;
    btn.setAttribute('role', 'radio');
    const badge = key === defaultKey ? '<span class="preset-badge">Recommended</span>' : '';
    btn.innerHTML =
      `<span class="preset-name">${p.label}${badge}</span>` +
      `<span class="preset-tagline">${p.tagline}</span>` +
      `<span class="preset-best">Best for: ${p.best_for}</span>`;
    btn.addEventListener('click', () => applyPreset(key));
    presetsEl.appendChild(btn);
  }
}

function markActivePreset() {
  presetsEl.querySelectorAll('.preset').forEach((el) => {
    const on = el.dataset.key === activePreset;
    el.classList.toggle('active', on);
    el.setAttribute('aria-checked', on ? 'true' : 'false');
  });
}

function applyPreset(key) {
  const p = presets[key];
  if (!p) return;
  activePreset = key;
  strengthSlider.value = p.correction_strength;
  retuneSlider.value = p.retune_ms;
  if (p.reverb === null) {
    usePolish.checked = false;
  } else {
    usePolish.checked = true;
    reverbSlider.value = p.reverb;
  }
  updateReadouts();
  markActivePreset();
  updateStatusbar();
}

function markCustom() {
  activePreset = null;
  markActivePreset();
  updateStatusbar();
}

async function loadPresets() {
  try {
    const res = await fetch('/presets');
    const data = await res.json();
    presets = data.presets;
    renderPresets(data.default);
    applyPreset(data.default);
  } catch (err) {
    presetsEl.innerHTML = '<p class="hint">Could not load presets — is <code>python app.py</code> running?</p>';
  }
}

// ---------- Sliders / toggles ----------
function updateReadouts() {
  strengthReadout.textContent = `${Math.round(parseFloat(strengthSlider.value) * 100)}%`;
  retuneReadout.textContent = `${retuneSlider.value} ms`;
  reverbReadout.textContent = `${Math.round(parseFloat(reverbSlider.value) * 100)}%`;
  reverbSlider.disabled = !usePolish.checked;
}

[strengthSlider, retuneSlider, reverbSlider].forEach((el) => {
  el.addEventListener('input', () => { updateReadouts(); markCustom(); });
});
usePolish.addEventListener('change', () => { updateReadouts(); markCustom(); });
[useFollow, useFormant, usePhaseVocoder, usePreemphasis].forEach((el) => el.addEventListener('change', updateStatusbar));

scaleRoot.addEventListener('change', () => {
  scaleType.disabled = scaleRoot.value === 'auto';
  updateStatusbar();
});
scaleType.addEventListener('change', updateStatusbar);

function updateStatusbar() {
  const key = scaleRoot.value === 'auto'
    ? 'key auto'
    : `key ${scaleRoot.value} ${scaleType.options[scaleType.selectedIndex].text.toLowerCase()}`;
  const style = activePreset && presets[activePreset] ? presets[activePreset].label : 'Custom';
  const polish = usePolish.checked ? `reverb ${Math.round(parseFloat(reverbSlider.value) * 100)}%` : 'no polish';
  const method = usePhaseVocoder.checked ? 'phase vocoder' : 'naive resampling';
  statusbar.textContent =
    `44.1 kHz · frame 2048 · hop 512 · ${method} · ${style} · ${key} · ` +
    `strength ${Math.round(parseFloat(strengthSlider.value) * 100)}% · retune ${retuneSlider.value} ms · ${polish}`;
}

// ---------- Running the correction ----------
function setProgress(fraction, stage) {
  progressFill.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  progressStage.textContent = `${stage}… ${Math.round(fraction * 100)}%`;
}

async function pollJob(jobId) {
  for (;;) {
    await sleep(350);
    const res = await fetch(`/status/${jobId}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.error || 'The server lost this job');
    setProgress(job.progress || 0, job.stage || 'Working');
    if (job.state === 'done') return job.result;
    if (job.state === 'error') throw new Error(job.error || 'Processing failed');
  }
}

async function fetchAndDecode(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Could not download the result');
  return getCtx().decodeAudioData(await res.arrayBuffer());
}

runBtn.addEventListener('click', async () => {
  if (!inputWav) return;
  getCtx().resume();
  player.stop();
  runBtn.disabled = true;
  progressEl.hidden = false;
  statusLine.textContent = '';
  setProgress(0, 'Uploading');

  const form = new FormData();
  form.append('audio', inputWav, 'input.wav');
  form.append('scale_root', scaleRoot.value);
  form.append('scale_type', scaleType.value);
  form.append('correction_strength', strengthSlider.value);
  form.append('retune_ms', retuneSlider.value);
  form.append('reverb_amount', reverbSlider.value);
  form.append('studio_polish', usePolish.checked);
  form.append('follow_tuning', useFollow.checked);
  form.append('use_formant_preservation', useFormant.checked);
  form.append('use_phase_vocoder', usePhaseVocoder.checked);
  form.append('use_preemphasis', usePreemphasis.checked);

  try {
    let res;
    try {
      res = await fetch('/process', { method: 'POST', body: form });
    } catch (err) {
      throw new Error('could not reach the server — is "python app.py" running?');
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'server error');
    const result = await pollJob(data.job_id);
    setProgress(1, 'Loading the result');
    const [originalBuf, tunedBuf] = await Promise.all([
      fetchAndDecode(result.original_url),
      fetchAndDecode(result.output_url),
    ]);
    showResult(result, originalBuf, tunedBuf);
    statusLine.textContent = `Done — ${fmtTime(result.duration)} of audio processed in ${result.processing_time} s.`;
  } catch (err) {
    console.error(err);
    statusLine.textContent = `Error: ${err.message}`;
  } finally {
    runBtn.disabled = false;
    progressEl.hidden = true;
  }
});

// ---------- A/B player ----------
// Both versions play at the same time through their own gain node; the
// Original/Tuned switch just cross-fades the two gains (20 ms), so you can
// flip back and forth mid-note with no gap and no loss of sync.
const player = {
  buffers: { original: null, tuned: null },
  gains: {},
  sources: [],
  side: 'tuned',
  playing: false,
  startedAt: 0,     // audioCtx time corresponding to position 0
  offset: 0,        // position (s) when paused
  duration: 0,

  load(originalBuf, tunedBuf) {
    this.stop();
    this.buffers.original = originalBuf;
    this.buffers.tuned = tunedBuf;
    this.duration = Math.max(originalBuf.duration, tunedBuf.duration);
    this.offset = 0;
    const ctx = getCtx();
    for (const side of ['original', 'tuned']) {
      if (!this.gains[side]) {
        this.gains[side] = ctx.createGain();
        this.gains[side].connect(ctx.destination);
      }
      this.gains[side].gain.value = side === this.side ? 1 : 0;
    }
  },

  position() {
    if (!this.playing) return this.offset;
    return Math.max(0, Math.min(getCtx().currentTime - this.startedAt, this.duration));
  },

  play() {
    if (!this.buffers.tuned) return;
    const ctx = getCtx();
    ctx.resume();
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    const when = ctx.currentTime + 0.03;
    for (const side of ['original', 'tuned']) {
      const src = ctx.createBufferSource();
      src.buffer = this.buffers[side];
      src.connect(this.gains[side]);
      src.start(when, Math.min(this.offset, src.buffer.duration));
      this.sources.push(src);
    }
    this.sources[this.sources.length - 1].onended = () => {
      if (this.playing) {
        this.killSources();
        this.playing = false;
        this.offset = 0;
        updatePlayerUI();
      }
    };
    this.startedAt = when - this.offset;
    this.playing = true;
    updatePlayerUI();
    animate();
  },

  pause() {
    this.offset = this.position();
    this.killSources();
    this.playing = false;
    updatePlayerUI();
  },

  stop() {
    this.killSources();
    this.playing = false;
    this.offset = 0;
  },

  killSources() {
    for (const src of this.sources) {
      src.onended = null;
      try { src.stop(); } catch (e) { /* already stopped */ }
      src.disconnect();
    }
    this.sources = [];
  },

  seek(seconds) {
    const wasPlaying = this.playing;
    if (wasPlaying) this.pause();
    this.offset = Math.max(0, Math.min(seconds, this.duration));
    if (wasPlaying) this.play(); else updatePlayerUI();
  },

  setSide(side) {
    this.side = side;
    const ctx = getCtx();
    const now = ctx.currentTime;
    for (const s of ['original', 'tuned']) {
      const g = this.gains[s];
      if (!g) continue;
      g.gain.cancelScheduledValues(now);
      g.gain.setValueAtTime(g.gain.value, now);
      g.gain.linearRampToValueAtTime(s === side ? 1 : 0, now + 0.02);
    }
  },
};

function setSide(side) {
  player.setSide(side);
  abOptions.forEach((el) => {
    const on = el.dataset.side === side;
    el.classList.toggle('active', on);
    el.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  waveLabel.textContent = side === 'tuned' ? 'Tuned signal' : 'Original signal';
  drawOutputBase();
  drawOverlays();
}

abOptions.forEach((el) => el.addEventListener('click', () => setSide(el.dataset.side)));

playBtn.addEventListener('click', () => {
  if (player.playing) player.pause(); else player.play();
});

function updatePlayerUI() {
  iconPlay.hidden = player.playing;
  iconPause.hidden = !player.playing;
  playBtn.setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
  playTime.textContent = `${fmtTime(player.position())} / ${fmtTime(player.duration)}`;
  drawOverlays();
}

function animate() {
  if (!player.playing) return;
  playTime.textContent = `${fmtTime(player.position())} / ${fmtTime(player.duration)}`;
  drawOverlays();
  requestAnimationFrame(animate);
}

document.addEventListener('keydown', (e) => {
  if (resultEl.hidden) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (['input', 'select', 'textarea'].includes(tag)) return;
  if (e.code === 'Space') {
    // a focused button/link/summary already reacts to Space by itself
    if (['button', 'summary', 'a'].includes(tag)) return;
    e.preventDefault();
    if (player.playing) player.pause(); else player.play();
  } else if (e.key === 't' || e.key === 'T') {
    setSide(player.side === 'tuned' ? 'original' : 'tuned');
  }
});

// ---------- Result drawing (base layer cached, playhead drawn on top) ----------
const baseLayers = new Map();   // canvas -> offscreen canvas holding the static drawing

function cacheBase(canvas) {
  const off = document.createElement('canvas');
  off.width = canvas.width;
  off.height = canvas.height;
  off.getContext('2d').drawImage(canvas, 0, 0);
  baseLayers.set(canvas, off);
}

const PITCH_PAD = { left: 44, right: 10, top: 10, bottom: 22 };

function drawOutputBase() {
  if (resultEl.hidden || !player.buffers.tuned) return;
  const { ctx, w, h } = setupCanvas(traceOutput);
  const buf = player.buffers[player.side];
  drawWaveformInto(ctx, w, h, buf.getChannelData(0), cssVar(player.side === 'tuned' ? '--amber' : '--steel'));
  cacheBase(traceOutput);
}

function niceTimeStep(duration, width) {
  const target = duration / Math.max(2, Math.floor(width / 90));
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  for (const s of steps) if (s >= target) return s;
  return 600;
}

function pitchRange(result) {
  const vals = [];
  for (const arr of [result.pitch.detected, result.pitch.corrected]) {
    for (const v of arr) if (v !== null) vals.push(v);
  }
  if (!vals.length) return null;
  vals.sort((a, b) => a - b);
  let lo = vals[Math.floor(vals.length * 0.02)] - 1.5;
  let hi = vals[Math.floor(vals.length * 0.98)] + 1.5;
  if (hi - lo < 8) {
    const mid = (hi + lo) / 2;
    lo = mid - 4;
    hi = mid + 4;
  }
  return { lo, hi };
}

function drawPitchBase() {
  if (!lastResult) return;
  const { ctx, w, h } = setupCanvas(pitchGraph);
  ctx.clearRect(0, 0, w, h);
  const result = lastResult;
  const range = pitchRange(result);
  const plotW = w - PITCH_PAD.left - PITCH_PAD.right;
  const plotH = h - PITCH_PAD.top - PITCH_PAD.bottom;
  const muted = cssVar('--text-muted');
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;

  if (!range) {
    ctx.fillStyle = muted;
    ctx.fillText('No sung notes were detected in this recording.', PITCH_PAD.left, h / 2);
    cacheBase(pitchGraph);
    return;
  }

  const duration = player.duration || result.duration;
  const xOf = (t) => PITCH_PAD.left + (t / duration) * plotW;
  const offset = (result.tuning_offset_cents || 0) / 100;
  const yOf = (m) => PITCH_PAD.top + (1 - (m - range.lo) / (range.hi - range.lo)) * plotH;

  // note lines: brighter for notes of the key, labelled on the left
  const steps = SCALE_STEPS[result.key_type] || SCALE_STEPS.chromatic;
  const rootPc = NOTE_NAMES.indexOf(result.key_root);
  const pxPerSemitone = plotH / (range.hi - range.lo);
  ctx.textBaseline = 'middle';
  for (let m = Math.ceil(range.lo - offset); m <= Math.floor(range.hi - offset); m++) {
    const pc = ((m % 12) + 12) % 12;
    const inKey = steps.includes((pc - rootPc + 12) % 12);
    const y = Math.round(yOf(m + offset)) + 0.5;
    ctx.strokeStyle = inKey ? '#353B42' : '#1D2126';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PITCH_PAD.left, y);
    ctx.lineTo(w - PITCH_PAD.right, y);
    ctx.stroke();
    const showLabel = inKey && (pxPerSemitone >= 7 || pc === rootPc);
    if (showLabel) {
      ctx.fillStyle = muted;
      ctx.fillText(`${NOTE_NAMES[pc]}${Math.floor(m / 12) - 1}`, 6, y);
    }
  }

  // time axis
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = muted;
  const tStep = niceTimeStep(duration, plotW);
  for (let t = 0; t <= duration + 1e-6; t += tStep) {
    const x = xOf(t);
    ctx.fillText(fmtTime(t), Math.max(PITCH_PAD.left + 3, Math.min(x - 8, w - 34)), h - 6);
  }

  // a pitch track: break the line at unvoiced frames and at big jumps
  const drawTrack = (values, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let penDown = false;
    let prev = null;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || v < range.lo - 3 || v > range.hi + 3) { penDown = false; prev = null; continue; }
      const x = xOf(result.pitch.times[i]);
      const y = yOf(v);
      if (!penDown || (prev !== null && Math.abs(v - prev) > 3)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      penDown = true;
      prev = v;
    }
    ctx.stroke();
  };
  drawTrack(result.pitch.detected, cssVar('--steel'), 1.5);
  drawTrack(result.pitch.corrected, cssVar('--amber'), 2);
  cacheBase(pitchGraph);
}

function drawPlayhead(canvas, xCss, dpr) {
  const base = baseLayers.get(canvas);
  if (!base) return;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(base, 0, 0);
  const x = Math.round(xCss * dpr) + 0.5;
  ctx.strokeStyle = 'rgba(236, 234, 228, 0.85)';
  ctx.lineWidth = Math.max(1, dpr);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, canvas.height);
  ctx.stroke();
}

function drawOverlays() {
  if (resultEl.hidden || !player.duration) return;
  const dpr = window.devicePixelRatio || 1;
  const frac = player.position() / player.duration;
  const wWave = traceOutput.getBoundingClientRect().width;
  drawPlayhead(traceOutput, frac * wWave, dpr);
  const wPitch = pitchGraph.getBoundingClientRect().width;
  const plotW = wPitch - PITCH_PAD.left - PITCH_PAD.right;
  drawPlayhead(pitchGraph, PITCH_PAD.left + frac * plotW, dpr);
}

traceOutput.addEventListener('click', (e) => {
  const rect = traceOutput.getBoundingClientRect();
  player.seek(((e.clientX - rect.left) / rect.width) * player.duration);
});
pitchGraph.addEventListener('click', (e) => {
  const rect = pitchGraph.getBoundingClientRect();
  const plotW = rect.width - PITCH_PAD.left - PITCH_PAD.right;
  const frac = (e.clientX - rect.left - PITCH_PAD.left) / plotW;
  player.seek(Math.max(0, Math.min(1, frac)) * player.duration);
});

function keyLabel(result) {
  if (result.key_type === 'chromatic') {
    return result.key_auto ? 'all 12 notes (no single key fitted)' : 'all 12 notes';
  }
  const kind = result.key_type === 'major' ? 'major' : 'minor';
  return `${result.key_root} ${kind}`;
}

function renderChips(result) {
  const chips = [];
  chips.push(`Key: <b>${keyLabel(result)}</b>${result.key_auto ? ' · auto-detected' : ''}`);
  if (useFollow.checked) {
    const c = result.tuning_offset_cents;
    const txt = Math.abs(c) < 3 ? 'right on standard pitch' : `${c > 0 ? '+' : ''}${Math.round(c)} cents ${c > 0 ? 'sharp' : 'flat'} — followed`;
    chips.push(`Your tuning: <b>${txt}</b>`);
  }
  chips.push(`Style: <b>${activePreset && presets[activePreset] ? presets[activePreset].label : 'Custom'}</b>`);
  chips.push(`Length <b>${fmtTime(result.duration)}</b> · processed in <b>${result.processing_time} s</b>`);
  chipsEl.innerHTML = chips.map((c) => `<span class="chip">${c}</span>`).join('');
}

function showResult(result, originalBuf, tunedBuf) {
  lastResult = result;
  player.load(originalBuf, tunedBuf);
  emptyState.hidden = true;
  resultEl.hidden = false;
  downloadBtn.href = result.output_url;
  setSide('tuned');
  drawPitchBase();
  renderChips(result);
  updatePlayerUI();
  const top = resultEl.getBoundingClientRect().top;
  if (top > window.innerHeight * 0.6) resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- Resize ----------
let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!recorder || recorder.state !== 'recording') drawIdleScope();
    drawInputWaveform();
    if (!resultEl.hidden) {
      drawOutputBase();
      drawPitchBase();
      drawOverlays();
    }
  }, 120);
});

// ---------- Start ----------
drawIdleScope();
updateReadouts();
loadPresets();
