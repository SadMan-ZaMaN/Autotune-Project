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

## Run (current: framing + overlap-add reconstruction test)
```bash
python main.py
```
Reads `data/raw/test_voice.wav`, frames it, reconstructs it via overlap-add,
and writes the result to `data/processed/reconstructed_test.wav`.

## Status
- [x] Week 1: signal I/O, framing, windowing, overlap-add reconstruction
- [x] Week 2: autocorrelation pitch detection
- [x] Note/scale quantization (MIDI mapping, nearest-note snapping)
- [ ] Week 3: naive resampling pitch shift (baseline for comparison)
- [ ] Week 4-5: phase vocoder pitch shifting
- [ ] Week 5-6: full pipeline + customization features (scale selector, correction strength)
- [ ] Week 6: formant preservation, Z-transform preprocessing filter
- [ ] Week 7: report + demo polish
