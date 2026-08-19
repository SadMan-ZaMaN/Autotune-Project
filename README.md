# Autotune / Pitch Correction System

CSE 220 (Signals and Linear Systems) Sessional Project — Team "Cells Interlinked" (2305068, 2305075)

A DSP-based pitch correction tool built from scratch: detects the pitch of a
recorded voice using autocorrelation, quantizes it to the nearest note in a
chosen musical scale, and shifts pitch using an STFT-based phase vocoder —
the same core technique behind commercial autotune tools.

## Features
- Selectable musical scale/key (C major, A minor, chromatic, custom)
- Adjustable correction strength/speed (natural <-> classic robotic hard-tune)
- Naive resampling vs. phase-vocoder pitch-shift comparison (demonstrates aliasing)
- Formant preservation (stretch goal)
- Z-transform-designed preprocessing filter with pole-zero / stability analysis

## Project layout
- `src/autotune/` — core package, one module per pipeline stage
- `tests/` — unit tests (e.g. pitch detection validated against known sine waves)
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

## Run
```bash
python run_demo.py --input data/raw/voice.wav --scale Cmajor --strength 0.7
```

## Status
- [x] Week 1: signal I/O, framing, windowing
- [ ] Week 2: autocorrelation pitch detection
- [ ] Week 3: note quantization + naive pitch shift
- [ ] Week 4-5: phase vocoder
- [ ] Week 5-6: full pipeline + customization features
- [ ] Week 6: formant preservation, Z-transform filter
- [ ] Week 7: report + demo polish
