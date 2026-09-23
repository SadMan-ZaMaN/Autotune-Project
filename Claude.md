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
python -m unittest discover tests -v   # ALL regression tests (27) - run after any DSP change
```

## Project structure

```
autotune-project/
├── src/autotune/
│   ├── config.py           # AutoTuneConfig: frame_size, hop_size, sample_rate,
│   │                        #   scale_root, scale_type, correction_strength, retune_ms,
│   │                        #   auto_key, follow_singer_tuning, note_hysteresis,
│   │                        #   studio_polish, reverb_amount
│   ├── io_utils.py         # load_audio, save_audio (WAV/FLAC/OGG/MP3 via soundfile)
│   ├── framing.py          # frame_signal, overlap_add (Hann window, weighted OLA)
│   ├── pitch_detection.py  # autocorrelation + Boersma window correction, parabolic
│   │                        #   interpolation, octave guard, clean_pitch_track
│   ├── scales.py           # MIDI conversion, build_scale_midi_set, choose_target_notes (hysteresis)
│   ├── key_detection.py    # estimate_tuning_offset (circular mean), detect_key (coverage + K-S)
│   ├── pitch_shift.py      # naive_pitch_shift, compute_shift_ratios, smooth_shift_ratios
│   ├── phase_vocoder.py    # THE CORE FILE — see below, read Bug History before touching
│   ├── effects.py          # studio polish: biquad EQ, compressor, Schroeder reverb (convolution)
│   ├── presets.py          # Natural / Studio Pop / Hard Tune / Pitch only settings bundles
│   ├── filters.py          # Rajin's pre-emphasis filter (Z-transform), pole-zero/freq plots
│   ├── visualization.py    # Rajin's waveform/spectrogram/pitch-contour plots
│   └── pipeline.py         # run_pipeline() — chains everything together
├── app.py                  # Flask backend: background job + /status polling, /presets, /audio
├── templates/index.html    # frontend (3 steps: Voice -> Style -> Result)
├── static/style.css        # "console/instrument" design — amber+steel on graphite
├── static/script.js        # in-browser decode->WAV, recorder, presets, A/B player, pitch graph
├── record_voice.py         # standalone mic-recording script (Rajin)
├── run_demo.py             # simple CLI entry point, edit variables at top
├── tests/test_phase_vocoder.py   # vocoder regression suite — RUN after any phase_vocoder.py change
├── tests/test_pipeline.py        # detection, key/tuning, hysteresis, effects, full pipeline
├── CONTRACTS.md            # team interface agreement (updated for attempt 4)
└── data/raw, data/processed  # audio in/out - ALL audio gitignored (only .gitkeep tracked);
                              #   regenerate data/raw/test_voice.wav with gen_test_tone.py
```

## Pipeline flow (run_pipeline in pipeline.py)

```
load audio (raw_audio kept untouched for honest before/after comparison)
  -> frame_signal (framing.py) on the RAW audio  <- this is what gets shifted
  -> pre-emphasis filter on a COPY, framed separately, used ONLY for detection
  -> detect_pitch_for_all_frames (pitch_detection.py, incl. clean_pitch_track)
  -> estimate_tuning_offset + detect_key (key_detection.py) if auto
  -> choose_target_notes (scales.py) — nearest scale note WITH hysteresis
  -> compute_shift_ratios (pitch_shift.py) — raw_ratio ** correction_strength
  -> smooth_shift_ratios (pitch_shift.py) — EMA in log-space, avoids robotic snap
  -> phase_vocoder_shift (formant preservation is INSIDE it now) OR naive_pitch_shift
  -> overlap_add (framing.py)
  -> detect pitch of the result (for the UI graph / report)
  -> studio_polish (effects.py) if enabled: HPF+EQ -> compressor -> reverb -> -1 dBFS
```

`progress_callback(stage, fraction)` is threaded through for the web UI's progress bar.

## CRITICAL: Bug history in phase_vocoder.py — read before editing

The phase vocoder went through 2 verified-broken implementations, a
correct-but-rough third, and the current fourth. **Do not revert to
approaches 1 or 2** even if they look simpler — they were tested and
proven wrong. Attempt 4 is a refinement of 3, not a different idea.

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

**Attempt 3 (correct, but rough): spectrum remapping + peak locking.**
`remap_and_lock_spectrum()` actually MOVES each harmonic peak's magnitude
to a new bin near its shifted frequency (not just adjusting phase in
place), and drags neighboring/sidelobe bins along with it via
`assign_regions()`, phase-locked relative to their peak. This is the
frequency-domain equivalent of resampling, done per-bin so it still fits
the fixed-hop frame-in/frame-out contract. Pitch was right (9/9 tests),
but it kept ONE RUNNING PHASE PER OUTPUT BIN: whenever vibrato/drift moved
a harmonic into a neighbouring bin, that bin's stored phase was stale ->
phase glitch -> roughness/"phasiness". It also re-synthesized EVERY
frame, so even unshifted consonants/breaths came out altered.

**Attempt 4 (CURRENT): attempt 3 + peak tracking with rotations
(Laroche & Dolson 1999).** Same remap + lock, but:
- each peak is TRACKED to the previous frame's peak whose region contains
  its bin, and continues THAT peak's phase;
- phase is stored as a ROTATION on top of the original phase:
  `out_phase = in_phase + rotation - pi*shift_bins`, and each frame
  `rotation += 2*pi*(r-1)*f*hop/fs`. The `-pi*shift_bins` term is the
  Hann-window linear-phase correction (FFT phase is measured from the frame
  start, the window is centred) — removing it costs ~3 dB clarity;
- frames with ratio == 1 pass through bit-exact and reset rotations to 0;
- `find_peaks` threshold 0.001 (-60 dB) instead of 0.05 (quiet upper
  harmonics were lumped into the last loud peak's region);
- formant preservation moved INSIDE the remap (see below).
Measured (synthetic singer with vibrato/glides/consonants, vs an ideal
re-synthesis): clarity (HNR) 25.7 -> 31.1 dB (input 32.4), spectral error
3.9 -> 1.8 dB, consonant distortion 6.0 -> 0.24 dB, ~3x faster. On the real
recording in data/raw: old lost 3.2-4.1 dB clarity at ratios 0.97-1.06,
new loses ~0. `TestAttempt4PeakTracking` in tests/test_phase_vocoder.py
FAILS on attempt 3 (9.9 dB clarity loss on a vibrato voice) — keep it.

**Pipeline bug fixed at the same time (not in phase_vocoder.py):** the
pre-emphasis filter (1 - 0.95z^-1, ~-25 dB at 150 Hz) was applied to the
audio that got shifted and saved, and never undone -> output thin, tinny,
~7x quieter. It now only feeds the pitch detector. This was the single
biggest cause of "doesn't sound good".

**If you touch `remap_and_lock_spectrum`, `find_peaks`, `assign_regions`,
or `shift_and_resynthesize_phase`: rerun the test suite before trusting
any change.** If tests fail, do not "fix" by reverting toward attempt 1/2.

## Formant preservation (inside phase_vocoder.py)

`compute_spectral_envelope` = cepstral smoothing: log(magnitude) -> inverse
FFT -> keep first ~30 coefficients (formant shape) -> FFT back (returns the
LOG envelope). In `remap_and_lock_spectrum`, a region moved by
`shift_bins` is multiplied by `formant_gain` = env(target)/env(source) of
the ORIGINAL frame's envelope (clipped to +/-12 dB), so each harmonic
slides along the original vocal-tract shape instead of carrying it.

The old `formant_preserve(shifted_frame, original_frame)` after-step was
removed: it divided by the SHIFTED frame's envelope, and the exact zeros
remapping leaves between regions (log(1e-8) = -18) dragged that envelope
down -> +1.6 dB loudness bias and slightly worse spectral error than no
formant step. In-remap version: neutral at autotune-size shifts, much
better at big ones (spectral error vs ideal at ratio 1.5: 8.4 -> 4.1 dB
without/with). num_coeffs 20/30/40 measured ~identical.

## Pitch detection / key / targets (the "music theory" layer)

- `detect_pitch_autocorrelation`: Boersma window-ACF normalization,
  local-peak list + octave guard (first peak >= 0.9x best — REQUIRED once
  Boersma normalization is on, otherwise ~25% octave-down errors),
  parabolic interpolation (median error 2-4 -> 0.3 cents), confidence 0.5.
- `clean_pitch_track`: -40 dB loudness gate, drop voiced runs < 4 frames,
  5-frame median in semitones. Removed the ~1000 Hz / 80 Hz spikes on
  breaths/consonants of the real recording (398 -> 25 frames > 600 Hz).
- `key_detection.estimate_tuning_offset`: circular mean of cents deviation.
- `key_detection.detect_key`: coverage-first (keys containing all sung
  notes), Krumhansl-Kessler correlation as tie-break, chromatic fallback
  below 85% coverage. Plain K-S alone picked wrong keys on short melodies.
  Tested on 20 synthetic melodies: auto+follow-tuning 3.4% wrong-note
  frames vs 6.7% for the CORRECT key typed in by hand (because of singers
  who are consistently sharp/flat).
- `scales.choose_target_notes`: hysteresis 0.3 st stops target flip-flop.
- The real recording in data/raw has notes C#..G heavily chromatic ->
  auto key correctly falls back to chromatic (71% best coverage).

## Known issues / open TODO

Resolved (kept here so nobody re-investigates):
- MP3/M4A upload and mic recording: the browser now decodes EVERYTHING
  (decodeAudioData) and uploads a mono 44.1 kHz WAV (static/script.js
  `decodeToMono` + `encodeWav`). Verified in headless Edge: stereo MP3,
  AAC-in-MP4 (m4a), fake-microphone WebM recording. Recorder has no time
  limit (chunks every 1 s), shows a timer + level meter, and disables
  echo cancellation / noise suppression / AGC (they damage sung notes).
- Dead code at the bottom of phase_vocoder.py: deleted (it is in git history).
- CONTRACTS.md: updated for attempt 4 + new functions.
- Frontend visually verified with Playwright + Edge screenshots at 1440 px
  and 390 px (no horizontal scroll).

Still open:
1. **No human listening test yet.** All quality claims above are
   objective proxies (HNR, spectral distance vs an ideal re-synthesis,
   pyin-measured pitch). Someone should A/B by ear on real singing.
2. Only one real recording exists (data/raw/upload_*.wav, 32 s, male,
   chromatic-ish). A second real sung recording as a test fixture would
   help, especially a female voice and a clearly diatonic song.
3. Pre-existing hygiene: `__pycache__/*.pyc` files are
   tracked in git (a .gitignore now exists but does not untrack them;
   `git rm --cached` them if the team agrees).
4. Preset numbers (strength/retune/reverb) were chosen from the metrics
   and common autotune practice, not tuned by ear.

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

Synthetic test signals must be REALISTIC: a formant model with Gaussian
resonance tails made a female voice almost all even harmonics and sent the
pitch detector an octave up — a test-signal bug, not a detector bug. Use
the cascade 2-pole resonance model in `make_vibrato_voice` (tests) and
include vibrato: attempt 3's flaw was invisible on steady tones.
A new test should be checked to FAIL on the code it guards against (the
attempt-4 tests were run against attempt 3 from git and do fail).