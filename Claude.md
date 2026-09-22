# Autotune / Pitch Correction System — Project Context

CSE 220 (Signals and Linear Systems) Sessional Project.
Team "Cells Interlinked" — Sadman (2305068), Rajin (2305075).

Read this whole file before making changes. It contains hard-won context —
especially the "Critical Bug History" section — that took a long debugging
session to discover. Do not silently "simplify" or "clean up" the phase
vocoder back toward an approach already proven broken.

## What this project is

A DSP-based pitch correction tool (like Auto-Tune/Melodyne) built from
scratch using classical signal processing — no ML/DL. Detects pitch via
autocorrelation, quantizes to the nearest note in a musical scale, and
shifts pitch using a phase vocoder (STFT-based, phase-coherent). Built for
a university Signals and Systems course, so code favors clarity and
explicit steps over clever/compressed one-liners — see "Coding
conventions" below.

## Tech stack

- Python 3, numpy/scipy for all DSP (deliberately avoid high-level
  shortcut libraries where a manual implementation teaches the concept)
- soundfile for WAV I/O, sounddevice for mic recording
- matplotlib for report plots
- Flask backend + vanilla HTML/CSS/JS frontend (local web UI, not deployed)

## How to run

```
.venv\Scripts\activate
pip install -r requirements.txt
python main.py              # basic framing/OLA test
python run_demo.py          # CLI pipeline demo, edit variables at top of file
python app.py                # web UI, open http://localhost:5000
python -m unittest tests.test_phase_vocoder -v   # regression tests
```

## Project structure

```
autotune-project/
├── src/autotune/
│   ├── config.py           # AutoTuneConfig: frame_size, hop_size, sample_rate,
│   │                        #   scale_root, scale_type, correction_strength, retune_ms
│   ├── io_utils.py         # load_audio, save_audio (WAV only currently — see Known Issues)
│   ├── framing.py          # frame_signal, overlap_add (Hann window, weighted OLA)
│   ├── pitch_detection.py  # detect_pitch_autocorrelation (FFT-based, Wiener-Khinchin)
│   ├── scales.py           # MIDI conversion, nearest_scale_note, build_scale_midi_set
│   ├── pitch_shift.py      # naive_pitch_shift, compute_shift_ratios, smooth_shift_ratios
│   ├── phase_vocoder.py    # THE CORE FILE — see below, read Bug History before touching
│   ├── filters.py          # Rajin's pre-emphasis filter (Z-transform), pole-zero/freq plots
│   ├── visualization.py    # Rajin's waveform/spectrogram/pitch-contour plots
│   └── pipeline.py         # run_pipeline() — chains everything together
├── app.py                  # Flask backend, wraps run_pipeline for the web UI
├── templates/index.html    # frontend
├── static/style.css        # "console/instrument" design — amber+steel on graphite
├── static/script.js        # waveform canvas drawing, recording, fetch to /process
├── record_voice.py         # standalone mic-recording script (Rajin)
├── run_demo.py             # simple CLI entry point, edit variables at top
├── tests/test_phase_vocoder.py   # regression suite — RUN THIS after any phase_vocoder.py change
├── CONTRACTS.md            # team interface agreement (may be stale re: phase vocoder internals)
└── data/raw, data/processed  # audio in/out (gitignored)
```

## Pipeline flow (run_pipeline in pipeline.py)

```
load audio (raw_audio kept untouched for honest before/after comparison)
  -> pre-emphasis filter (optional, filters.py)
  -> frame_signal (framing.py)
  -> detect_pitch_for_all_frames (pitch_detection.py)
  -> nearest_scale_note per frame (scales.py)
  -> compute_shift_ratios (pitch_shift.py) — raw_ratio ** correction_strength
  -> smooth_shift_ratios (pitch_shift.py) — EMA in log-space, avoids robotic snap
  -> phase_vocoder_shift (phase_vocoder.py) OR naive_pitch_shift (baseline)
  -> formant_preserve per frame (phase_vocoder.py) if enabled
  -> overlap_add (framing.py)
```

## CRITICAL: Bug history in phase_vocoder.py — read before editing

The phase vocoder went through 3 real, verified-broken implementations
before landing on the current correct one. **Do not revert to approaches
1 or 2** even if they look simpler — they were tested and proven wrong.

**Attempt 1 (broken): per-bin independent phase modulation, same hop.**
Kept each FFT bin's magnitude in its own native bin, only modified its
phase to evolve at a shifted rate. Worked for a single pure sine tone, but
completely broke on real multi-harmonic signals (voice). A bin acts like a
narrow bandpass filter — you can't push its phase to represent a
frequency far outside that bin's native range and expect correct output.
Verified broken at realistic ratios: `ratio=1.05` produced 78Hz output
when 157.5Hz was expected.

**Attempt 2 (broken): peak-locked phase, still no bin remapping.**
Added peak detection so neighboring bins locked their phase to their
nearest harmonic peak (fixed a threshold bug where quiet-but-real
harmonics were rejected — use MEAN-based threshold, not max-based, since
harmonics can legitimately vary 50x in loudness). This fixed detection
but NOT shifting — still left magnitude in the original bin, so it had
the same fundamental flaw as attempt 1. Still broken at ratio=1.05.

**Attempt 3 (CURRENT, correct): spectrum remapping + peak locking.**
Located in `remap_and_lock_spectrum()`. Actually MOVES each harmonic
peak's magnitude to a new bin near its shifted frequency (not just
adjusting phase in place), and drags neighboring/sidelobe bins along with
it via `assign_regions()`, phase-locked relative to their peak. This is
the frequency-domain equivalent of resampling, done per-bin so it still
fits the fixed-hop frame-in/frame-out contract.

Verified against `tests/test_phase_vocoder.py` (9/9 passing), covering:
realistic ratios (1.02–1.5), the exact 1.05 case that broke attempts 1&2,
time-varying per-frame ratios (needed since different notes need
different correction amounts), and the original single-tone sanity check.

**If you touch `remap_and_lock_spectrum`, `find_peaks`, `assign_regions`,
or `shift_and_resynthesize_phase`: rerun the test suite before trusting
any change.** If tests fail, do not "fix" by reverting toward attempt 1/2.

## formant_preserve (also in phase_vocoder.py)

Cepstral smoothing technique: log(magnitude) -> inverse FFT -> keep first
~30 coefficients (formant shape) -> zero the rest (pitch harmonic detail)
-> FFT back -> exp. Used to strip the shifted frame's own (wrong) formant
envelope and replace it with the original frame's envelope, so pitch
shifts don't drag timbre with them (avoids "chipmunk" effect). Verified:
without it, formant peak drifts with shift ratio (1249/1163/1357 Hz
across ratios 1.3/1.5/2.0); with it, stays locked near ~1184 Hz regardless
of shift amount.

## Known issues / open TODO

1. **MP3/M4A not supported for upload.** `io_utils.load_audio` uses
   `soundfile`, which does NOT decode MP3/M4A/AAC reliably across
   platforms (libsndfile support is inconsistent). WAV/FLAC/OGG work.
   Fix options: (a) add a librosa fallback (uses audioread/ffmpeg,
   handles MP3 better) when soundfile fails, or (b) convert client-side —
   the browser's Web Audio API can decode almost anything
   (mp3/m4a/webm/ogg) via `decodeAudioData`, then re-encode to WAV in JS
   before uploading to Flask. Option (b) also fixes issue #2 below in one
   pass, since it means the backend only ever receives WAV regardless of
   what the user picked or recorded.

2. **Mic recording likely broken end-to-end — UNVERIFIED, check first.**
   `static/script.js`'s recorder uses `MediaRecorder` which produces a
   `webm` blob, sent directly to the Flask `/process` endpoint. But
   `io_utils.load_audio` (via `soundfile`) cannot decode webm. This was
   never actually tested with a live recording — verify with a real
   browser recording before assuming it works. Same fix as #1 (encode to
   WAV client-side before upload) solves this too.

3. **"Not sounding great" — needs listening-based debugging.** User
   reports current output isn't melodious/natural yet. Things to check,
   roughly in order of likely impact:
   - `correction_strength` default is 1.0 (full hard-snap correction) —
     try 0.5–0.7 for natural sound, confirm the person testing knows this
     is adjustable and isn't just using the hard-snap default.
   - `retune_ms` default is 40 — try tuning 20–80 and A/B by ear.
   - `find_peaks`' `min_rel_height=0.05` threshold in phase_vocoder.py was
     tuned against synthetic test tones, not real recorded voice with
     background noise/breath sounds — may need adjustment for real mic
     input (noisy frames could produce spurious peaks).
   - `compute_spectral_envelope`'s `num_coeffs=30` in formant_preserve
     was not extensively tuned — too few coefficients over-smooths (loses
     real formant detail), too many leaks pitch harmonics back in.
   - Check whether `detect_pitch_autocorrelation`'s `fmin=80, fmax=1000`
     defaults suit the actual voice being tested (adjust per singer's
     range).
   - Get an actual recorded voice sample (not synthetic tone) into
     `tests/` as a fixture for repeatable listening tests.

4. **`phase_vocoder.py` has ~300 lines of dead code** from earlier fix
   attempts sitting inside a `'''...'''` comment block at the bottom of
   the file (harmless, doesn't execute, but bloats the file). Safe to
   delete once the current implementation is trusted.

5. **`CONTRACTS.md` is stale** — still describes an earlier, simpler
   phase-shift approach. Should be updated to reflect the remap+lock
   architecture if anyone (professor, Rajin) reads it as documentation.

6. **Frontend visual appearance unverified.** The Flask backend and API
   were tested end-to-end (real file upload -> real pipeline -> valid WAV
   response, confirmed via curl). The actual rendered look (spacing,
   colors, canvas rendering) has NOT been visually verified — no
   screenshot/browser tool was available when it was built. Check it live
   in a browser and treat layout/spacing bugs as expected-possible, not
   surprising.

## Coding conventions (please follow)

- Written for a course project — prefer explicit, readable code over
  clever one-liners or heavy use of high-level library shortcuts. Plain
  `for` loops over stacked list comprehensions/broadcasting tricks, even
  where numpy could do it in fewer lines.
- Every non-trivial function has a docstring explaining the WHY, often
  with a worked numeric example (see `pitch_detection.py`,
  `phase_vocoder.py` for the established style) — continue this pattern,
  the student needs to explain this code in a viva.
- New functions should be testable in isolation with a quick synthetic
  signal before being trusted in the full pipeline (this is how every bug
  above was actually found — single-frame or single-tone tests can hide
  real bugs that only appear on multi-frame, multi-harmonic signals;
  always test the full frame -> shift -> overlap-add path, not an
  isolated function call).
- `CONTRACTS.md` documents agreed function signatures between the two
  team members. If you change a signature, note it there too.

## Testing philosophy (learned the hard way this session)

A phase-vocoder pitch shift bug will NOT show up in a single-frame test —
frequency shift is an emergent property of how multiple overlapping
frames combine via overlap-add, not something visible in one frame's
FFT alone. Also test at REALISTIC ratios (1.02–1.1 for autotune use), not
just extreme ones (1.5–2.0) — attempt 1 above passed extreme-ratio checks
by coincidence-adjacent behavior while failing badly at ratio=1.05, which
is a completely normal correction amount. `tests/test_phase_vocoder.py`
encodes these lessons — extend it rather than replacing it.