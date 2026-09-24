// =====================================================================
// Console — frontend for the pitch-correction pipeline.
//
// Flow: (1) get audio (file or microphone) -> decode it IN THE BROWSER and
// re-encode as a mono 44.1 kHz WAV, so the Python side only ever sees WAV
// (this is what makes MP3/M4A/WebM recordings work with no ffmpeg);
// (2) pick a style preset; (3) POST to /process, poll /status for progress,
// then play original vs tuned in a sample-synchronised A/B player and draw
// the pitch graph returned by the server.
// (4) optional note editing: the graph shows one bar per automatic note;
// dragging bars and pressing "Apply edits" asks the server to re-render the
// same job with those notes moved (/rerender). The automatic result is kept
// as "Tuned"; the re-render plays as a third A/B side, "Edited".
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
let lastResult = null;      // the automatic result (never replaced by edits)
let editedResult = null;    // the latest re-render with note edits, or null

// Note editing. Notes are identified by their frame range
// "start_frame:end_frame" - the server's segmentation is deterministic, so
// the same note keeps the same key in every re-render.
const MAX_EDIT = 12;          // semitones, same limit as the server
const noteEdits = new Map();  // note key -> semitones (only non-zero edits stored)
let appliedEditsKey = '';     // editsKey() of the edits editedResult was rendered with
let selectedNote = null;      // note key
let drag = null;              // {note, startY, startEdit, pxPerSemitone, moved}
let pitchView = null;         // {lo, hi} MIDI range, frozen while dragging
let editRendering = false;
let editProgress = 0;
let editError = '';

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
const noiseSlider = $('noise');
const noiseReadout = $('noise-readout');
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
const abEdited = $('ab-edited');
const editSummary = $('edit-summary');
const applyEditsBtn = $('apply-edits-btn');
const resetEditsBtn = $('reset-edits-btn');
const dlOriginal = $('dl-original');
const dlTuned = $('dl-tuned');
const dlEdited = $('dl-edited');
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
  const noise = parseFloat(noiseSlider.value);
  noiseReadout.textContent = noise > 0 ? `${Math.round(noise * 100)}%` : 'Off';
  reverbSlider.disabled = !usePolish.checked;
}

[strengthSlider, retuneSlider, reverbSlider].forEach((el) => {
  el.addEventListener('input', () => { updateReadouts(); markCustom(); });
});
usePolish.addEventListener('change', () => { updateReadouts(); markCustom(); });
// noise reduction isn't part of the style presets - it's about the room, not the sound
noiseSlider.addEventListener('input', () => { updateReadouts(); updateStatusbar(); });
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
  const noise = parseFloat(noiseSlider.value);
  const denoise = noise > 0 ? `noise reduction ${Math.round(noise * 100)}%` : 'no noise reduction';
  statusbar.textContent =
    `44.1 kHz · frame 2048 · hop 512 · ${method} · ${style} · ${key} · ` +
    `strength ${Math.round(parseFloat(strengthSlider.value) * 100)}% · retune ${retuneSlider.value} ms · ${polish} · ${denoise}`;
}

// ---------- Running the correction ----------
function setProgress(fraction, stage) {
  progressFill.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  progressStage.textContent = `${stage}… ${Math.round(fraction * 100)}%`;
}

async function pollJob(jobId, onProgress = setProgress) {
  for (;;) {
    await sleep(350);
    const res = await fetch(`/status/${jobId}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.error || 'The server lost this job');
    onProgress(job.progress || 0, job.stage || 'Working');
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
  form.append('noise_reduction', noiseSlider.value);
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
    result.style_label = activePreset && presets[activePreset] ? presets[activePreset].label : 'Custom';
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
// Every loaded version plays at the same time through its own gain node;
// the Original/Tuned/Edited switch just cross-fades the gains (20 ms), so
// you can flip back and forth mid-note with no gap and no loss of sync.
const SIDES = ['original', 'tuned', 'edited'];
const SIDE_LABELS = { original: 'Original signal', tuned: 'Tuned signal', edited: 'Edited signal' };
const SIDE_COLORS = { original: '--steel', tuned: '--amber', edited: '--edit' };

const player = {
  buffers: { original: null, tuned: null, edited: null },
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
    this.buffers.edited = null;
    if (this.side === 'edited') this.side = 'tuned';
    this.offset = 0;
    this.updateDuration();
    const ctx = getCtx();
    for (const side of SIDES) {
      if (!this.gains[side]) {
        this.gains[side] = ctx.createGain();
        this.gains[side].connect(ctx.destination);
      }
      this.gains[side].gain.value = side === this.side ? 1 : 0;
    }
  },

  // Swap in (or remove, with null) the re-rendered version without losing
  // the play position.
  setEdited(buf) {
    const wasPlaying = this.playing;
    if (wasPlaying) this.pause();
    this.buffers.edited = buf;
    this.updateDuration();
    if (wasPlaying) this.play();
  },

  updateDuration() {
    let d = 0;
    for (const side of SIDES) if (this.buffers[side]) d = Math.max(d, this.buffers[side].duration);
    this.duration = d;
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
    for (const side of SIDES) {
      if (!this.buffers[side]) continue;
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
    for (const s of SIDES) {
      const g = this.gains[s];
      if (!g) continue;
      g.gain.cancelScheduledValues(now);
      g.gain.setValueAtTime(g.gain.value, now);
      g.gain.linearRampToValueAtTime(s === side ? 1 : 0, now + 0.02);
    }
  },
};

function availableSides() {
  return SIDES.filter((side) => player.buffers[side]);
}

function setSide(side) {
  if (!player.buffers[side]) side = 'tuned';
  player.setSide(side);
  abOptions.forEach((el) => {
    const on = el.dataset.side === side;
    el.classList.toggle('active', on);
    el.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  waveLabel.textContent = SIDE_LABELS[side];
  updateDownloads();
  renderStages();
  drawOutputBase();
  drawOverlays();
}

// One link per version: the recording as you gave it, the automatic
// tuning, and (once edits are applied) the edited tuning.
function updateDownloads() {
  if (!lastResult) return;
  dlOriginal.href = lastResult.input_url;
  dlTuned.href = lastResult.output_url;
  dlEdited.hidden = !editedResult;
  if (editedResult) dlEdited.href = editedResult.output_url;
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
    const sides = availableSides();
    setSide(sides[(sides.indexOf(player.side) + 1) % sides.length]);
  } else if (selectedNote && !editRendering) {
    const note = findNote(selectedNote);
    if (!note) return;
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      setNoteEdit(note, editOf(note) + (e.key === 'ArrowUp' ? 1 : -1));
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      setNoteEdit(note, 0);
    } else if (e.key === 'Escape') {
      selectedNote = null;
      redrawPitch();
    }
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
  drawWaveformInto(ctx, w, h, buf.getChannelData(0), cssVar(SIDE_COLORS[player.side]));
  cacheBase(traceOutput);
}

function niceTimeStep(duration, width) {
  const target = duration / Math.max(2, Math.floor(width / 90));
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  for (const s of steps) if (s >= target) return s;
  return 600;
}

function noteKey(note) {
  return `${note.start_frame}:${note.end_frame}`;
}

function findNote(key) {
  if (!lastResult || !lastResult.notes) return null;
  return lastResult.notes.find((n) => noteKey(n) === key) || null;
}

function editOf(note) {
  return noteEdits.get(noteKey(note)) || 0;
}

// Visible MIDI range: covers what was sung, the tuned result, and every
// note bar where it currently sits (so a dragged note never leaves the graph).
function pitchRange() {
  const vals = [];
  const tracks = [lastResult.pitch.detected, lastResult.pitch.corrected];
  if (editedResult) tracks.push(editedResult.pitch.corrected);
  for (const arr of tracks) {
    for (const v of arr) if (v !== null) vals.push(v);
  }
  if (!vals.length) return null;
  vals.sort((a, b) => a - b);
  let lo = vals[Math.floor(vals.length * 0.02)] - 1.5;
  let hi = vals[Math.floor(vals.length * 0.98)] + 1.5;
  for (const note of lastResult.notes || []) {
    const m = note.midi + editOf(note);
    if (editOf(note) !== 0) {
      lo = Math.min(lo, m - 1.5);
      hi = Math.max(hi, m + 1.5);
    }
  }
  if (hi - lo < 8) {
    const mid = (hi + lo) / 2;
    lo = mid - 4;
    hi = mid + 4;
  }
  return { lo, hi };
}

// Everything needed to map time/MIDI <-> canvas pixels, shared by the
// drawing code and the mouse hit-testing so the two always agree.
function pitchGeometry() {
  if (!lastResult) return null;
  if (!drag || !pitchView) pitchView = pitchRange();   // frozen mid-drag so the grid doesn't jump
  const range = pitchView;
  if (!range) return null;
  const rect = pitchGraph.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  const plotW = w - PITCH_PAD.left - PITCH_PAD.right;
  const plotH = h - PITCH_PAD.top - PITCH_PAD.bottom;
  const duration = player.duration || lastResult.duration;
  const pxPerSemitone = plotH / (range.hi - range.lo);
  return {
    w, h, plotW, plotH, range, duration, pxPerSemitone,
    barH: Math.max(6, pxPerSemitone * 0.8),
    xOf: (t) => PITCH_PAD.left + (t / duration) * plotW,
    yOf: (m) => PITCH_PAD.top + (1 - (m - range.lo) / (range.hi - range.lo)) * plotH,
  };
}

function drawNoteBars(ctx, g) {
  const steel = cssVar('--steel');
  const edit = cssVar('--edit');
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;
  ctx.textBaseline = 'middle';
  for (const note of lastResult.notes || []) {
    const x0 = g.xOf(note.start);
    const width = Math.max(2, g.xOf(note.end) - x0);
    const shift = editOf(note);
    const yAuto = g.yOf(note.midi) - g.barH / 2;
    const selected = noteKey(note) === selectedNote;

    if (shift !== 0) {
      // where the auto-tune put it: dashed outline
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(139, 143, 148, 0.6)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, yAuto + 0.5, width - 1, g.barH - 1);
      ctx.setLineDash([]);
      // where you dragged it
      const y = g.yOf(note.midi + shift) - g.barH / 2;
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = edit;
      ctx.fillRect(x0, y, width, g.barH);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = selected ? cssVar('--text') : edit;
      ctx.lineWidth = selected ? 2 : 1;
      ctx.strokeRect(x0 + 0.5, y + 0.5, width - 1, g.barH - 1);
      const label = `${shift > 0 ? '+' : ''}${shift}`;
      ctx.fillStyle = edit;
      ctx.fillText(label, x0 + width + 3, y + g.barH / 2);
    } else {
      ctx.globalAlpha = 0.22;
      ctx.fillStyle = steel;
      ctx.fillRect(x0, yAuto, width, g.barH);
      ctx.globalAlpha = selected ? 1 : 0.7;
      ctx.strokeStyle = selected ? cssVar('--text') : steel;
      ctx.lineWidth = selected ? 2 : 1;
      ctx.strokeRect(x0 + 0.5, yAuto + 0.5, width - 1, g.barH - 1);
      ctx.globalAlpha = 1;
    }
  }
}

function drawPitchBase() {
  if (!lastResult) return;
  const { ctx, w, h } = setupCanvas(pitchGraph);
  ctx.clearRect(0, 0, w, h);
  const result = lastResult;
  const g = pitchGeometry();
  const muted = cssVar('--text-muted');
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;

  if (!g) {
    ctx.fillStyle = muted;
    ctx.fillText('No sung notes were detected in this recording.', PITCH_PAD.left, h / 2);
    cacheBase(pitchGraph);
    return;
  }

  const { range, xOf, yOf, plotW, duration, pxPerSemitone } = g;
  const offset = (result.tuning_offset_cents || 0) / 100;

  // note lines: brighter for notes of the key, labelled on the left
  const steps = SCALE_STEPS[result.key_type] || SCALE_STEPS.chromatic;
  const rootPc = NOTE_NAMES.indexOf(result.key_root);
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

  // the draggable note bars, under the pitch lines so those stay readable
  drawNoteBars(ctx, g);

  // a pitch track: break the line at unvoiced frames and at big jumps
  const drawTrack = (times, values, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let penDown = false;
    let prev = null;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || v < range.lo - 3 || v > range.hi + 3) { penDown = false; prev = null; continue; }
      const x = xOf(times[i]);
      const y = yOf(v);
      if (!penDown || (prev !== null && Math.abs(v - prev) > 3)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      penDown = true;
      prev = v;
    }
    ctx.stroke();
  };
  drawTrack(result.pitch.times, result.pitch.detected, cssVar('--steel'), 1.5);
  drawTrack(result.pitch.times, result.pitch.corrected, cssVar('--amber'), 2);
  if (editedResult) {
    drawTrack(editedResult.pitch.times, editedResult.pitch.corrected, cssVar('--edit'), 2);
  }
  cacheBase(pitchGraph);
}

function redrawPitch() {
  drawPitchBase();
  drawOverlays();
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

// ---------- Note editing on the pitch graph ----------
// Press on a bar and drag vertically: every pxPerSemitone pixels moves the
// note one semitone. A press that doesn't move just selects the note (and
// jumps playback to it); a press on empty graph seeks, as before.
function seekFromPitchX(clientX) {
  const rect = pitchGraph.getBoundingClientRect();
  const plotW = rect.width - PITCH_PAD.left - PITCH_PAD.right;
  const frac = (clientX - rect.left - PITCH_PAD.left) / plotW;
  player.seek(Math.max(0, Math.min(1, frac)) * player.duration);
}

function noteAt(clientX, clientY) {
  const g = pitchGeometry();
  if (!g || !lastResult.notes) return null;
  const rect = pitchGraph.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const tolerance = Math.max(g.barH / 2, 7);   // easy to grab even when bars are thin
  let best = null;
  let bestDist = Infinity;
  for (const note of lastResult.notes) {
    const x0 = g.xOf(note.start) - 3;
    const x1 = Math.max(g.xOf(note.end), g.xOf(note.start) + 2) + 3;
    if (x < x0 || x > x1) continue;
    const dist = Math.abs(y - g.yOf(note.midi + editOf(note)));
    if (dist <= tolerance && dist < bestDist) {
      best = note;
      bestDist = dist;
    }
  }
  return best;
}

function setNoteEdit(note, semitones) {
  const value = Math.max(-MAX_EDIT, Math.min(MAX_EDIT, semitones));
  if (value === 0) noteEdits.delete(noteKey(note)); else noteEdits.set(noteKey(note), value);
  editError = '';
  updateEditUI();
  redrawPitch();
}

pitchGraph.addEventListener('pointerdown', (e) => {
  if (!lastResult || e.button !== 0) return;
  const note = editRendering ? null : noteAt(e.clientX, e.clientY);
  if (!note) {
    if (selectedNote) { selectedNote = null; redrawPitch(); }
    seekFromPitchX(e.clientX);
    return;
  }
  e.preventDefault();
  pitchGraph.focus({ preventScroll: true });
  selectedNote = noteKey(note);
  drag = {
    note,
    startY: e.clientY,
    startEdit: editOf(note),
    pxPerSemitone: pitchGeometry().pxPerSemitone,
    moved: false,
  };
  pitchGraph.setPointerCapture(e.pointerId);
  pitchGraph.style.cursor = 'grabbing';
  redrawPitch();
});

pitchGraph.addEventListener('pointermove', (e) => {
  if (!drag) {
    if (lastResult && !editRendering) pitchGraph.style.cursor = noteAt(e.clientX, e.clientY) ? 'grab' : '';
    return;
  }
  const dy = drag.startY - e.clientY;
  if (Math.abs(dy) > 3) drag.moved = true;
  const target = drag.startEdit + Math.round(dy / drag.pxPerSemitone);
  if (target !== editOf(drag.note) && Math.abs(target) <= MAX_EDIT) setNoteEdit(drag.note, target);
});

function endDrag() {
  if (!drag) return;
  const { note, moved } = drag;
  drag = null;
  pitchGraph.style.cursor = '';
  if (!moved) player.seek(note.start);
  redrawPitch();   // unfreezes the view so a far-dragged note fits
}
pitchGraph.addEventListener('pointerup', endDrag);
pitchGraph.addEventListener('pointercancel', endDrag);

// On touch screens, stop the page scrolling only when the finger lands on a
// note bar - anywhere else on the graph still scrolls the page normally.
pitchGraph.addEventListener('touchstart', (e) => {
  const t = e.touches[0];
  if (t && !editRendering && noteAt(t.clientX, t.clientY)) e.preventDefault();
}, { passive: false });

function editsKey() {
  return Array.from(noteEdits.entries()).sort().map(([k, v]) => `${k}=${v}`).join(',');
}

function updateEditUI() {
  const count = noteEdits.size;
  const pending = editsKey() !== appliedEditsKey;
  const plural = count === 1 ? '' : 's';
  applyEditsBtn.disabled = editRendering || !pending;
  resetEditsBtn.disabled = editRendering || (count === 0 && !editedResult);
  if (editRendering) {
    editSummary.textContent = `Re-tuning with your edits… ${Math.round(editProgress * 100)}%`;
  } else if (editError) {
    editSummary.textContent = `Error: ${editError}`;
  } else if (count === 0 && !editedResult) {
    editSummary.textContent = 'Drag a note bar up or down to force it onto a different note.';
  } else if (count === 0) {
    editSummary.innerHTML = 'All edits undone — press <b>Apply edits</b> to remove the Edited version.';
  } else if (pending) {
    editSummary.innerHTML = `<b>${count} note${plural} edited</b> — press Apply edits to hear the change.`;
  } else {
    editSummary.innerHTML = `<b>${count} note${plural} edited</b> — playing as <b>Edited</b>.`;
  }
}

function clearEditedVersion() {
  editedResult = null;
  appliedEditsKey = '';
  player.setEdited(null);
  abEdited.hidden = true;
  if (player.side === 'edited') setSide('tuned');
  updateDownloads();
}

function resetEdits() {
  noteEdits.clear();
  selectedNote = null;
  editError = '';
  clearEditedVersion();
  updateEditUI();
  redrawPitch();
}

async function applyEdits() {
  if (!lastResult || editRendering) return;
  if (noteEdits.size === 0) { resetEdits(); return; }
  const forResult = lastResult;
  const key = editsKey();
  const overrides = Array.from(noteEdits.entries()).map(([k, semitones]) => {
    const [startFrame, endFrame] = k.split(':').map(Number);
    return { start_frame: startFrame, end_frame: endFrame, semitones };
  });
  editRendering = true;
  editProgress = 0;
  editError = '';
  updateEditUI();
  try {
    const res = await fetch(`/rerender/${forResult.job_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ overrides }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'server error');
    const result = await pollJob(data.job_id, (fraction) => { editProgress = fraction; updateEditUI(); });
    const buf = await fetchAndDecode(result.output_url);
    if (lastResult !== forResult) return;   // a new recording was tuned meanwhile
    result.style_label = forResult.style_label;
    editedResult = result;
    appliedEditsKey = key;
    player.setEdited(buf);
    abEdited.hidden = false;
    setSide('edited');
    redrawPitch();
  } catch (err) {
    console.error(err);
    editError = err.message;
  } finally {
    editRendering = false;
    updateEditUI();
  }
}

applyEditsBtn.addEventListener('click', applyEdits);
resetEditsBtn.addEventListener('click', resetEdits);

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
  const nr = result.noise_reduction;
  if (nr) {
    let txt;
    if (nr.applied) txt = `background noise (${nr.noise_db} dBFS) turned down up to ${Math.round(nr.max_reduction_db)} dB`;
    else if (nr.reason === 'off') txt = 'off';
    else txt = `nothing removed — ${nr.reason}`;
    chips.push(`Noise reduction: <b>${txt}</b>`);
  }
  chips.push(`Style: <b>${activePreset && presets[activePreset] ? presets[activePreset].label : 'Custom'}</b>`);
  chips.push(`Length <b>${fmtTime(result.duration)}</b> · processed in <b>${result.processing_time} s</b>`);
  chipsEl.innerHTML = chips.map((c) => `<span class="chip">${c}</span>`).join('');
}

function showResult(result, originalBuf, tunedBuf) {
  lastResult = result;
  stageResult = null;
  stageIndex = 0;
  editedResult = null;
  noteEdits.clear();
  appliedEditsKey = '';
  selectedNote = null;
  editError = '';
  abEdited.hidden = true;
  player.load(originalBuf, tunedBuf);
  emptyState.hidden = true;
  resultEl.hidden = false;
  setSide('tuned');
  updateEditUI();
  drawPitchBase();
  renderChips(result);
  updatePlayerUI();
  const top = resultEl.getBoundingClientRect().top;
  if (top > window.innerHeight * 0.6) resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- Step-by-step processing view ----------
// The server sends result.stages: one entry per processing step that ran
// (the list depends on the style), each with its numbers, an explanation
// and 1-2 charts already reduced to plot data (src/autotune/stage_plots.py).
// This code only draws them. Chart types: wave, lines, bars, spectrogram.
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

const CHART_COLORS = { steel: '--steel', amber: '--amber', edit: '--edit', muted: '--text-muted', text: '--text' };
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
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;
  ctx.textBaseline = 'middle';
  let lastY = Infinity;
  for (const [value, label] of ticks) {
    const y = Math.round(f.yOf(value)) + 0.5;
    if (y < CHART_PAD.top - 1 || y > CHART_PAD.top + f.plotH + 1) continue;
    ctx.strokeStyle = '#23282E';
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
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;
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
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;
  chart.values.forEach((v, i) => {
    const x = CHART_PAD.left + i * slot + slot * 0.18;
    const bw = slot * 0.64;
    const y = f.yOf(v);
    ctx.fillStyle = chart.highlight[i] ? cssVar('--amber') : 'rgba(92, 137, 172, 0.45)';
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
  ctx.font = `10px ${cssVar('--font-mono') || 'monospace'}`;

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
    items = [`<span class="lg"><i style="background:${cssVar('--amber')}"></i>in the key</span>`,
      `<span class="lg"><i style="background:rgba(92,137,172,0.45)"></i>not in the key</span>`];
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
  stageExplain.textContent = stage.explain;
  stagePrev.disabled = stageIndex === 0;
  stageNext.disabled = stageIndex === stages.length - 1;
  drawStageCharts();
}

// Shows the steps of the version you're listening to (Tuned, or Edited
// once there is one), staying on the same step when switching.
function renderStages() {
  const source = player.side === 'edited' && editedResult ? editedResult : lastResult;
  if (!source || !source.stages) { stagesEl.hidden = true; return; }
  stagesEl.hidden = false;
  const previousId = stageResult && stageResult.stages[stageIndex] ? stageResult.stages[stageIndex].id : null;
  const changed = source !== stageResult;
  stageResult = source;
  const label = source === editedResult ? `${source.style_label || 'Custom'} + your edits` : (source.style_label || 'Custom');
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
      drawStageCharts();
    }
  }, 120);
});

// ---------- Start ----------
drawIdleScope();
updateReadouts();
loadPresets();
