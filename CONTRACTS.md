# CONTRACTS.md — Autotune Project Interface Agreements

This file defines the exact function signatures both teammates will build against
for the remaining work. Treat this as an agreement: if a signature needs to
change, update this file first and tell the other person, don't just change
your code silently.

Existing modules (already built, Weeks 1-2) — do not need contracts, already stable:
- `config.py` — AutoTuneConfig
- `io_utils.py` — load_audio, save_audio
- `framing.py` — frame_signal, overlap_add
- `pitch_detection.py` — detect_pitch_for_all_frames
- `scales.py` — build_scale_midi_set, nearest_scale_note

---

## Owner: Sadman — Core pitch-shifting chain

### naive_pitch_shift
**File:** `src/autotune/pitch_shift.py`
**Status:** [ ] not started

```python
def naive_pitch_shift(frame: np.ndarray, shift_ratio: float) -> np.ndarray
```
- `frame`: shape (frame_size,), float32 — one windowed frame
- `shift_ratio`: float, e.g. 1.05 = shift up 5%, 0.95 = shift down 5%
- Returns: shape (frame_size,), float32 — pitch-shifted frame (same length,
  achieved via resample + pad/truncate back to frame_size)
- Purpose: baseline for comparison, demonstrates aliasing/quality issues

### compute_shift_ratios
**File:** `src/autotune/pitch_shift.py`
**Status:** [ ] not started

```python
def compute_shift_ratios(detected_pitches: np.ndarray, target_pitches: np.ndarray,
                          strength: float = 1.0) -> np.ndarray
```
- `detected_pitches`: shape (num_frames,) — from pitch_detection.py, 0.0 = unvoiced
- `target_pitches`: shape (num_frames,) — nearest scale note per frame, from scales.py
- `strength`: 0.0 = no correction, 1.0 = full correction to target
- Returns: shape (num_frames,) — ratio per frame to feed into shift functions.
  Unvoiced frames (detected_pitch == 0) must return ratio 1.0 (no shift).

### phase_vocoder_shift
**File:** `src/autotune/phase_vocoder.py`
**Status:** [x] done — attempt 4 (see CLAUDE.md bug history)

```python
def phase_vocoder_shift(frames: np.ndarray, shift_ratios: np.ndarray,
                         config: AutoTuneConfig, preserve_formants: bool = True,
                         progress_callback=None) -> np.ndarray
```
- `frames`: shape (num_frames, frame_size) — output of frame_signal()
- `shift_ratios`: shape (num_frames,) — output of compute_shift_ratios()
- `config`: existing AutoTuneConfig object
- `preserve_formants`: keep the voice's timbre (formant envelope) fixed
- `progress_callback`: optional `f(fraction)` for the web UI
- Returns: shape (num_frames, frame_size) — shifted frames, same shape as
  input, ready to pass into overlap_add() unchanged
- Internals: spectrum remapping + peak locking, with each peak's phase
  TRACKED across frames as a rotation on top of the original phase. Frames
  with ratio == 1 come out bit-exact (unshifted consonants are untouched).

### formant preservation — CHANGED
The separate `formant_preserve(shifted_frame, original_frame, config)`
after-step was removed (it had a loudness bias, see CLAUDE.md). Formant
preservation is now the `preserve_formants` flag above; the pieces are
`compute_spectral_envelope(magnitude, config, num_coeffs=30)` (returns the
LOG envelope of a magnitude spectrum) and `formant_gain(log_envelope,
source_bins, target_bins)`.

### New: key / tuning / target notes
```python
# src/autotune/key_detection.py
def estimate_tuning_offset(detected_pitches) -> float            # cents, [-50, 50)
def detect_key(detected_pitches, tuning_offset_cents=0.0) -> (root: str, scale_type: str, confidence: float)
# src/autotune/scales.py
def choose_target_notes(detected_pitches, scale_midi_set,
                        tuning_offset_cents=0.0, hysteresis=0.3) -> np.ndarray   # Hz, 0 = unvoiced
```

### New: studio effects
```python
# src/autotune/effects.py
def studio_polish(audio, sample_rate, reverb_amount=0.2) -> np.ndarray   # EQ, compressor, reverb, -1 dBFS
```
Also exposes the biquad designers (`design_highpass`, `design_peaking_eq`,
`design_high_shelf`) as (b, a) pairs that work with Rajin's
`plot_pole_zero` / `plot_frequency_response`, and
`reverb_impulse_response(sample_rate)` (plot it for the report).

### detect_pitch_for_all_frames — extended (backward compatible)
```python
def detect_pitch_for_all_frames(frames, sample_rate, fmin=70, fmax=1000,
                                window=None, clean=True) -> np.ndarray
```
`clean=False` returns the raw per-frame detections (for "before clean-up"
plots).

---

## Owner: Rajin — Filter module, visualization, customization

### design_preemphasis_filter
**File:** `src/autotune/filters.py`
**Status:** [ ] not started

```python
def design_preemphasis_filter(coeff: float = 0.95) -> tuple[np.ndarray, np.ndarray]
```
- Returns: (b, a) — numerator/denominator coefficients (standard scipy filter format)

### apply_filter
**File:** `src/autotune/filters.py`
**Status:** [ ] not started

```python
def apply_filter(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray
```
- `audio`: shape (num_samples,) — full audio signal (not frames)
- Returns: shape (num_samples,) — filtered audio, same length

### plot_pole_zero / plot_frequency_response
**File:** `src/autotune/filters.py`
**Status:** [ ] not started

```python
def plot_pole_zero(b: np.ndarray, a: np.ndarray, save_path: str) -> None
def plot_frequency_response(b: np.ndarray, a: np.ndarray, sample_rate: int, save_path: str) -> None
```
- Both save a .png to save_path, no return value

### plot_waveform_comparison / plot_spectrogram_comparison / plot_pitch_contour
**File:** `src/autotune/visualization.py`
**Status:** [ ] not started

```python
def plot_waveform_comparison(original: np.ndarray, processed: np.ndarray,
                              sample_rate: int, save_path: str) -> None

def plot_spectrogram_comparison(original: np.ndarray, processed: np.ndarray,
                                 sample_rate: int, save_path: str) -> None

def plot_pitch_contour(detected_pitches: np.ndarray, target_pitches: np.ndarray,
                        corrected_pitches: np.ndarray, hop_size: int,
                        sample_rate: int, save_path: str) -> None
```
- Can be built and tested against fake/dummy numpy arrays before real
  pipeline output exists — don't wait on Sadman's code to start these

### Config additions for customization
**File:** `src/autotune/config.py`
**Status:** [ ] not started

Add to `AutoTuneConfig.__init__`:
```python
scale_root: str = "C"
scale_type: str = "major"       # "major" | "natural_minor" | "chromatic"
correction_strength: float = 1.0  # 0.0-1.0
```

---

## Joint — built after both sides above are ready

### run_pipeline
**File:** `src/autotune/pipeline.py`
**Status:** [x] done

```python
def run_pipeline(input_path: str, config: AutoTuneConfig,
                  use_phase_vocoder: bool = True, use_preemphasis: bool = True,
                  use_formant_preservation: bool = True,
                  progress_callback=None) -> dict
```
- Returns dict with keys:
  `raw_audio`, `original_audio` (both = the untouched input),
  `corrected_audio`, `sample_rate`, `detected_pitches`, `target_pitches`,
  `shift_ratios`, `corrected_pitches` (pitch of the output — feed to
  `plot_pitch_contour`), `key_root`, `key_type`, `key_confidence`,
  `tuning_offset_cents`
- Pre-emphasis is applied ONLY to the copy used for pitch detection.
- New config fields: `auto_key`, `follow_singer_tuning`,
  `note_hysteresis`, `studio_polish`, `reverb_amount`.
- Called by `run_demo.py` and the Flask app (`app.py`).

---

## Ground rules
1. If you need to change a signature above, edit this file first, commit it,
   and message the other person before changing your code.
2. Build against these signatures even before the other person's code exists —
   use dummy/fake numpy arrays of the right shape to test your own piece in isolation.
3. One branch per function/feature, PR into main, other person reviews before merge.