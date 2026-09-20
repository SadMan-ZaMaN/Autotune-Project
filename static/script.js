// ---------- State ----------
let inputBlob = null;
let audioCtx = new (window.AudioContext || window.webkitAudioContext)();
let mediaRecorder = null;
let recordedChunks = [];

// ---------- Elements ----------
const dropzone = document.getElementById('dropzone');
const dropzoneText = document.getElementById('dropzone-text');
const fileInput = document.getElementById('file-input');
const recordBtn = document.getElementById('record-btn');
const recordStatus = document.getElementById('record-status');
const runBtn = document.getElementById('run-btn');
const statusLine = document.getElementById('status-line');
const audioInputEl = document.getElementById('audio-input');
const audioOutputEl = document.getElementById('audio-output');
const downloadBtn = document.getElementById('download-btn');
const traceInput = document.getElementById('trace-input');
const traceOutput = document.getElementById('trace-output');
const scopeCanvas = document.getElementById('scope');
const statusbar = document.getElementById('statusbar');

const strengthSlider = document.getElementById('strength');
const strengthReadout = document.getElementById('strength-readout');
const retuneSlider = document.getElementById('retune');
const retuneReadout = document.getElementById('retune-readout');
const scaleRoot = document.getElementById('scale-root');
const scaleType = document.getElementById('scale-type');

// ---------- Waveform drawing ----------
// Draws real decoded audio samples onto a canvas - this is actual signal
// content, not a decorative shape, matching what the rest of the project
// treats as the hero visual (a waveform).
function drawWaveform(canvas, audioBuffer, color) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / rect.width);
  const mid = rect.height / 2;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;

  for (let x = 0; x < rect.width; x++) {
    let min = 1.0, max = -1.0;
    const start = x * step;
    for (let i = 0; i < step && (start + i) < data.length; i++) {
      const v = data[start + i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x, mid + min * mid * 0.9);
    ctx.lineTo(x, mid + max * mid * 0.9);
  }
  ctx.stroke();
}

async function drawFromBlob(blob, canvas, color) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  drawWaveform(canvas, audioBuffer, color);
}

// Idle scope trace in the hero, before any audio is loaded - a gentle
// generated sine so the instrument doesn't look "dead" on first load.
function drawIdleScope() {
  const dpr = window.devicePixelRatio || 1;
  const rect = scopeCanvas.getBoundingClientRect();
  scopeCanvas.width = rect.width * dpr;
  scopeCanvas.height = rect.height * dpr;
  const ctx = scopeCanvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.strokeStyle = 'rgba(232, 163, 61, 0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const mid = rect.height / 2;
  for (let x = 0; x < rect.width; x++) {
    const y = mid + Math.sin(x * 0.02) * (rect.height * 0.12) * Math.sin(x * 0.002);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
drawIdleScope();
window.addEventListener('resize', drawIdleScope);

// ---------- File input / drag-drop ----------
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  inputBlob = file;
  dropzoneText.textContent = file.name;
  audioInputEl.src = URL.createObjectURL(file);
  drawFromBlob(file, traceInput, getComputedStyle(document.documentElement).getPropertyValue('--steel'));
  runBtn.disabled = false;
  statusLine.textContent = '';
}

// ---------- Microphone recording ----------
recordBtn.addEventListener('click', async () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      handleFile(new File([blob], 'recording.webm', { type: 'audio/webm' }));
      recordStatus.textContent = '';
      recordBtn.textContent = 'Record from microphone';
      stream.getTracks().forEach(t => t.stop());
    };
    mediaRecorder.start();
    recordBtn.textContent = 'Stop recording';
    recordStatus.textContent = 'Recording...';
  } catch (err) {
    recordStatus.textContent = 'Microphone access denied';
  }
});

// ---------- Slider readouts ----------
strengthSlider.addEventListener('input', () => {
  strengthReadout.textContent = parseFloat(strengthSlider.value).toFixed(2);
  updateStatusbar();
});
retuneSlider.addEventListener('input', () => {
  retuneReadout.textContent = `${retuneSlider.value} ms`;
  updateStatusbar();
});
scaleRoot.addEventListener('change', updateStatusbar);
scaleType.addEventListener('change', updateStatusbar);

function updateStatusbar() {
  const strength = parseFloat(strengthSlider.value).toFixed(2);
  const retune = retuneSlider.value;
  const root = scaleRoot.value;
  const type = scaleType.options[scaleType.selectedIndex].text;
  statusbar.textContent = `44.1 kHz · frame 2048 · hop 512 · ${root} ${type} · strength ${strength} · retune ${retune} ms`;
}
updateStatusbar();

// ---------- Run correction ----------
runBtn.addEventListener('click', async () => {
  if (!inputBlob) return;

  runBtn.disabled = true;
  statusLine.textContent = 'Processing...';

  const formData = new FormData();
  formData.append('audio', inputBlob, 'input.wav');
  formData.append('scale_root', scaleRoot.value);
  formData.append('scale_type', scaleType.value);
  formData.append('correction_strength', strengthSlider.value);
  formData.append('retune_ms', retuneSlider.value);
  formData.append('use_phase_vocoder', document.getElementById('use-phase-vocoder').checked);
  formData.append('use_formant_preservation', document.getElementById('use-formant').checked);
  formData.append('use_preemphasis', document.getElementById('use-preemphasis').checked);

  try {
    const res = await fetch('/process', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      statusLine.textContent = `Error: ${data.error}`;
      runBtn.disabled = false;
      return;
    }

    audioOutputEl.src = data.output_url;
    downloadBtn.href = data.output_url;
    downloadBtn.style.display = 'inline-block';

    const outBlob = await (await fetch(data.output_url)).blob();
    await drawFromBlob(outBlob, traceOutput, getComputedStyle(document.documentElement).getPropertyValue('--amber'));

    statusLine.textContent = 'Done.';
  } catch (err) {
    statusLine.textContent = 'Error: could not reach server';
  }
  runBtn.disabled = false;
});
