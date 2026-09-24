# Autotune / Pitch Correction System

CSE 220 (Signals and Linear Systems) Sessional Project — Team "Cells Interlinked" (2305068, 2305075)

A DSP-based pitch correction tool built from scratch: detects the pitch of a
recorded voice using autocorrelation, quantizes it to the nearest note in a
chosen musical scale, and (upcoming) shifts pitch using an STFT-based phase
vocoder — the same core technique behind commercial autotune tools.

## Project layout
- `src/autotune/config.py` — central config (sample rate, frame/hop size, window)
- `src/autotune/io_utils.py` — WAV load/save, mono conversion, resampling
- `src/autotune/framing.py` — framing, Hann windowing, weighted overlap-add reconstruction
- `src/autotune/pitch_detection.py` — FFT-based autocorrelation pitch detection
- `src/autotune/scales.py` — MIDI/frequency conversion, scale-note quantization
- `tests/` — unit tests
- `notebooks/` — exploration and tuning
- `data/raw` / `data/processed` — input/output audio
- `results/` — plots and audio samples for the report
- `docs/` — final report

## Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app (web UI)
```bash
python app.py        # then open http://localhost:5000
```
1. Upload a recording (WAV, MP3, M4A, FLAC, OGG — any length) or record
   with the microphone (no time limit).
2. Pick a style: **Natural**, **Studio Pop** (recommended), **Hard Tune**
   (robotic T-Pain effect) or **Pitch only**. Leave the key on
   *Auto-detect*.
3. Press **Tune my voice**, then flip between *Original* and *Tuned*
   while it plays (or press `T`).

## Other entry points
```bash
python run_demo.py                     # CLI: edit the variables at the top
python main.py                         # framing + overlap-add reconstruction test
python -m unittest discover tests -v   # regression tests
```

## Status
- [x] Week 1: signal I/O, framing, windowing, overlap-add reconstruction
- [x] Week 2: autocorrelation pitch detection
- [x] Note/scale quantization (MIDI mapping, nearest-note snapping)
- [x] Week 3: naive resampling pitch shift (baseline for comparison)
- [x] Week 4-5: phase vocoder pitch shifting
- [x] Week 5-6: full pipeline + customization features (scale selector, correction strength)
- [ ] Week 6: formant preservation, Z-transform preprocessing filter
- [ ] Week 7: report + demo polish





## Shared Workflow
- [x] Week 1: signal I/O, framing, windowing, overlap-add reconstruction
- [x] Week 2: autocorrelation pitch detection
- [x] Note/scale quantization (MIDI mapping, nearest-note snapping)
- [s|r] Week 3: naive resampling pitch shift (Sadman) + Z-transform filter module (Rajin)
- [s|r] Week 4-5: phase vocoder pitch shifting (Sadman) + visualization functions (Rajin)
- [ ] Week 5-6: pipeline integration (joint) + customization config (Rajin)
- [ ] Week 6: formant preservation (Sadman) + filter wired into pipeline (Rajin)
- [ ] Week 7: report + demo polish + UI (shared)

## will have to make presentation for this