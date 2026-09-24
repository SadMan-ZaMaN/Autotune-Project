import base64
import numpy as np
from scipy.signal import freqz

from .framing import frame_signal
from .noise_reduction import frame_spectra
from .scales import freq_to_midi, NOTE_NAMES, SCALE_INTERVALS
from .key_detection import pitch_class_histogram
from .filters import design_preemphasis_filter
from .effects import (design_highpass, design_peaking_eq, design_high_shelf,
                      reverb_impulse_response)

"""
Step-by-step graphs of the whole processing chain (for the web UI)
=====================================================================
run_pipeline(..., collect_stages=True) keeps the signal after every step in
a "trace" dict; build_stages(trace) turns that into one entry per step:

    {"id", "title", "summary", "explain", "stats": [[label, value], ...],
     "charts": [chart, ...]}

Only steps that actually RAN are included, so the list depends on the
style: "Pitch only" has no studio-polish steps, "Hard Tune" (retune 0 ms)
shows no smoothing in the correction step, "Natural" shows the 80% strength.

The charts are small, plain-JSON descriptions (the page just draws them):
    wave         min/max envelope of one or more signals over time
    lines        x/y line series (frequency responses, pitch tracks, ...)
    bars         a bar chart (the key histogram)
    spectrogram  time x log-frequency images, 0..255 = -80..0 dB, base64
Big signals are reduced first (a waveform becomes ~900 min/max pairs), so a
whole song's worth of steps stays a few hundred kB of JSON.
"""

WAVE_COLUMNS = 900         # min/max pairs per waveform
MAX_TRACK_POINTS = 1500    # points per per-frame track (pitch, ratios)
SPEC_COLUMNS = 420
SPEC_ROWS = 140
SPEC_DB_RANGE = 80.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def db_power(x):
    return 10 * np.log10(max(float(x), 1e-20))


def db_amplitude(x):
    return 20 * np.log10(max(float(x), 1e-10))


def note_name(midi):
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def fmt_db(value):
    return f"{value:+.1f} dB"


def fmt_cents(value):
    return f"{value:.0f} cents"


def waveform_envelope(audio, columns=WAVE_COLUMNS):
    """
    Reduces a signal to `columns` (min, max) pairs - what a waveform view
    draws anyway: one vertical line per pixel column from the lowest to the
    highest sample in it. 30 s at 44.1 kHz = 1,323,000 samples -> 900 pairs.
    """
    audio = np.asarray(audio)
    columns = max(1, min(columns, len(audio)))
    mins = []
    maxs = []
    for c in range(columns):
        start = c * len(audio) // columns
        end = max(start + 1, (c + 1) * len(audio) // columns)
        block = audio[start:end]
        mins.append(round(float(np.min(block)), 4))
        maxs.append(round(float(np.max(block)), 4))
    return mins, maxs


def wave_series(label, audio, color):
    mins, maxs = waveform_envelope(audio)
    return {"label": label, "color": color, "min": mins, "max": maxs}


def frame_time(i, config):
    """Centre of frame i in the original recording (frame_signal pads frame_size zeros first)."""
    return (i * config.hop_size - config.frame_size / 2) / config.sample_rate


def thinned_indices(n, max_points=MAX_TRACK_POINTS):
    step = max(1, int(np.ceil(n / max_points)))
    return list(range(0, n, step))


def track_series(label, values, config, color, indices, width=1.5, dash=False, step=False,
                 max_jump=None, jump_ratio=None):
    """
    A per-frame track (None = gap) as a time series. max_jump (same units
    as the values) or jump_ratio (for Hz: 1.19 = 3 semitones) makes the
    line break instead of drawing a vertical spike when the detector
    briefly jumps an octave - same rule as the main pitch graph.
    """
    xs = []
    ys = []
    for i in indices:
        t = frame_time(i, config)
        if t < 0:
            continue
        xs.append(round(t, 4))
        v = values[i]
        ys.append(None if v is None or not np.isfinite(v) else round(float(v), 3))
    series = {"label": label, "color": color, "x": xs, "y": ys, "width": width,
              "dash": dash, "step": step}
    if max_jump is not None:
        series["max_jump"] = max_jump
    if jump_ratio is not None:
        series["jump_ratio"] = jump_ratio
    return series


def midi_track(pitches):
    """Hz per frame -> MIDI per frame (None where unvoiced)."""
    out = []
    for f in pitches:
        out.append(float(freq_to_midi(f)) if f > 0 else None)
    return out


def note_ticks(lo, hi):
    """[midi, "C4"] for every semitone in view - the page skips labels that would overlap."""
    ticks = []
    for m in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
        ticks.append([m, note_name(m)])
    return ticks


def midi_range(*tracks):
    values = []
    for track in tracks:
        for v in track:
            if v is not None:
                values.append(v)
    if not values:
        return 48.0, 72.0
    values.sort()
    lo = values[int(len(values) * 0.02)] - 1.5
    hi = values[int(len(values) * 0.98)] + 1.5
    if hi - lo < 8:
        mid = (hi + lo) / 2
        lo, hi = mid - 4, mid + 4
    return lo, hi


def response_db(b, a, freqs, sample_rate):
    """|H(e^jw)| in dB at the given frequencies (Hz)."""
    w = 2 * np.pi * np.asarray(freqs) / sample_rate
    _, h = freqz(b, a, worN=w)
    return 20 * np.log10(np.abs(h) + 1e-12)


def log_freqs(fmin, fmax, count):
    return np.exp(np.linspace(np.log(fmin), np.log(fmax), count))


def spectrogram_panel(audio, config, label, fmin, fmax, ref_db=None,
                      columns=SPEC_COLUMNS, rows=SPEC_ROWS):
    """
    Log-frequency spectrogram as a small 8-bit image.

    1. STFT with the pipeline's own frames (Hann, 2048, hop 512).
    2. Neighbouring frames are averaged (in power) into `columns` columns.
    3. Each column is read at `rows` log-spaced frequencies between fmin
       and fmax (linear interpolation between FFT bins) - log frequency
       because pitch is logarithmic: every octave gets the same height.
    4. dB relative to ref_db (the loudest point; pass the same ref_db for a
       before/after pair so the two images are directly comparable), kept
       to the top 80 dB and scaled to 0..255.
    Returns (panel dict, the ref_db used).
    """
    spectra, _ = frame_spectra(np.asarray(audio, dtype=np.float32), config)
    power = np.abs(spectra) ** 2
    num_frames = len(power)
    columns = max(1, min(columns, num_frames))
    bin_freqs = np.arange(power.shape[1]) * config.sample_rate / config.frame_size
    row_freqs = log_freqs(fmin, fmax, rows)

    image_db = np.zeros((rows, columns))
    for c in range(columns):
        start = c * num_frames // columns
        end = max(start + 1, (c + 1) * num_frames // columns)
        column_power = np.mean(power[start:end], axis=0)
        image_db[:, c] = 10 * np.log10(np.interp(row_freqs, bin_freqs, column_power) + 1e-20)

    if ref_db is None:
        ref_db = float(np.max(image_db))
    scaled = (image_db - (ref_db - SPEC_DB_RANGE)) / SPEC_DB_RANGE * 255.0
    image = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    panel = {"label": label, "rows": rows, "cols": columns,
             "data": base64.b64encode(image.tobytes()).decode("ascii")}
    return panel, ref_db


def spectrogram_chart(title, signals, config, fmin, fmax):
    """signals: list of (label, audio). Panels share one dB scale."""
    panels = []
    ref_db = None
    for label, audio in signals:
        panel, ref_db = spectrogram_panel(audio, config, label, fmin, fmax, ref_db)
        panels.append(panel)
    duration = len(signals[0][1]) / config.sample_rate
    return {"type": "spectrogram", "title": title, "duration": round(duration, 3),
            "fmin": fmin, "fmax": fmax, "db_range": SPEC_DB_RANGE, "panels": panels}


def frame_powers(audio, config):
    frames, _ = frame_signal(audio, config)
    powers = np.zeros(len(frames))
    for i in range(len(frames)):
        powers[i] = np.mean(frames[i] ** 2)
    return powers


def pick_example_frame(detected_pitches, shift_ratios):
    """
    The frame used to illustrate single-frame steps (windowing, the phase
    vocoder): the voiced frame with the BIGGEST correction, preferring one
    in the middle of a note (both neighbours voiced) so it isn't an onset.
    Falls back to the middle of the recording.
    """
    best = None
    best_amount = -1.0
    for i in range(1, len(detected_pitches) - 1):
        if detected_pitches[i] <= 0 or detected_pitches[i - 1] <= 0 or detected_pitches[i + 1] <= 0:
            continue
        amount = abs(np.log(shift_ratios[i]))
        if amount > best_amount:
            best_amount = amount
            best = i
    if best is None:
        best = len(detected_pitches) // 2
    return best


# ---------------------------------------------------------------------------
# One builder per step
# ---------------------------------------------------------------------------

def stage_input(t):
    audio = t["raw_audio"]
    sr = t["config"].sample_rate
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    rms = np.sqrt(np.mean(audio ** 2)) if len(audio) else 0.0
    return {
        "id": "input", "title": "Original recording",
        "summary": "Your voice as a sampled signal x[n] - nothing changed yet.",
        "explain": (
            f"The recording is a list of numbers x[n], one every 1/{sr} s "
            f"({sr} samples per second, so frequencies up to fs/2 = {sr // 2} Hz can be "
            "represented - the Nyquist limit). The graph shows its waveform: for every "
            "pixel column, the lowest and highest sample in that stretch of time. "
            "Loud parts are sung notes, the thin parts are breaths and pauses."),
        "stats": [["Length", f"{len(audio) / sr:.1f} s"],
                  ["Samples", f"{len(audio):,}"],
                  ["Sample rate", f"{sr} Hz"],
                  ["Peak", f"{db_amplitude(peak):.1f} dBFS"],
                  ["RMS level", f"{db_amplitude(rms):.1f} dBFS"]],
        "charts": [{"type": "wave", "title": "Waveform x[n]",
                    "duration": round(len(audio) / sr, 3),
                    "series": [wave_series("original", audio, "steel")]}],
    }


def stage_noise(t):
    config = t["config"]
    info = t["noise_info"]
    raw = t["raw_audio"]
    if not info["applied"]:
        return {
            "id": "denoise", "title": "Noise reduction",
            "summary": f"Nothing removed - {info['reason']}.",
            "explain": (
                "Noise reduction learns the background noise from moments where nobody is "
                "singing and the level stays steady (a fan, traffic, hiss). This recording "
                f"had none ({info['reason']}), so it was left exactly as it was - a clean "
                "take can never be made worse by this step."),
            "stats": [["Result", "unchanged"], ["Reason", info["reason"]]],
            "charts": [{"type": "wave", "title": "Waveform (unchanged)",
                        "duration": round(len(raw) / config.sample_rate, 3),
                        "series": [wave_series("original", raw, "steel")]}],
        }

    cleaned = t["denoised_audio"]
    # measured effect: quietest 15% of frames (the background) vs loudest 50% (the singing)
    p_raw = frame_powers(raw, config)
    p_clean = frame_powers(cleaned, config)
    order = np.argsort(p_raw)
    quiet = order[:max(1, int(len(order) * 0.15))]
    loud = order[len(order) // 2:]
    quiet_change = db_power(np.sum(p_clean[quiet]) / max(np.sum(p_raw[quiet]), 1e-20))
    loud_change = db_power(np.sum(p_clean[loud]) / max(np.sum(p_raw[loud]), 1e-20))
    return {
        "id": "denoise", "title": "Noise reduction",
        "summary": "Steady background noise turned down with a Wiener filter, the voice kept.",
        "explain": (
            "In each STFT frame X[k] = S[k] + D[k] (voice + noise). The noise spectrum |D[k]|^2 "
            "is measured in the pauses, then every bin is multiplied by the Wiener gain "
            "G = SNR / (1 + SNR): a voice harmonic (SNR ~ 100) keeps G ~ 1, a bin that is only "
            f"noise is turned down - never by more than {info['max_reduction_db']:.0f} dB, and "
            "bins at the harmonics of a sung note are never turned down at all. Compare the "
            "two spectrograms: the haze between the harmonic lines and in the pauses fades, "
            "the harmonic lines themselves stay."),
        "stats": [["Noise level", f"{info['noise_db']} dBFS"],
                  ["Max reduction", f"{info['max_reduction_db']:.0f} dB"],
                  ["Pauses (background)", fmt_db(quiet_change)],
                  ["Singing (loud parts)", fmt_db(loud_change)]],
        "charts": [spectrogram_chart("Spectrogram before / after", [("before", raw), ("after", cleaned)],
                                     config, 50.0, 8000.0)],
    }


def stage_framing(t):
    config = t["config"]
    audio = t["audio"]
    i = t["example_index"]
    N = config.frame_size
    H = config.hop_size
    sr = config.sample_rate
    padded = np.concatenate([np.zeros(N), audio, np.zeros(N)])
    segment = padded[i * H: i * H + N]
    if len(segment) < N:
        segment = np.concatenate([segment, np.zeros(N - len(segment))])
    windowed = segment * config.window
    peak = max(np.max(np.abs(segment)), 1e-9)

    xs = []
    raw_y = []
    win_y = []
    out_y = []
    for n in range(0, N, 2):   # every other sample is plenty for 2048
        xs.append(round(n / sr * 1000, 3))
        raw_y.append(round(float(segment[n] / peak), 4))
        win_y.append(round(float(config.window[n]), 4))
        out_y.append(round(float(windowed[n] / peak), 4))
    num_frames = (len(audio) + 2 * N - N) // H + 1
    return {
        "id": "framing", "title": "Framing + Hann window",
        "summary": f"Cut into overlapping {N}-sample frames, each tapered by a Hann window.",
        "explain": (
            "Pitch changes over time, so the signal is analysed in short frames that are "
            f"roughly stationary: N = {N} samples = {N / sr * 1000:.1f} ms, a new one every "
            f"H = {H} samples ({(N - H) / N * 100:.0f}% overlap). Each frame is multiplied by the "
            "Hann window w[n] = 0.5 - 0.5 cos(2 pi n / (N-1)), which goes smoothly to 0 at both "
            "ends: cutting the signal abruptly would add fake high frequencies (spectral "
            "leakage) to its FFT. The graph shows one frame (the one with the biggest "
            f"correction, at {frame_time(i, config):.2f} s) before and after windowing."),
        "stats": [["Frame size N", f"{N} ({N / sr * 1000:.1f} ms)"],
                  ["Hop H", f"{H} ({H / sr * 1000:.1f} ms)"],
                  ["Overlap", f"{(N - H) / N * 100:.0f}%"],
                  ["Frames", f"{num_frames:,}"],
                  ["FFT bin spacing", f"{sr / N:.1f} Hz"]],
        "charts": [{"type": "lines", "title": f"Frame at {frame_time(i, config):.2f} s",
                    "x_label": "time in frame (ms)", "y_label": "amplitude (normalised)",
                    "x_min": 0, "x_max": round(N / sr * 1000, 2), "y_min": -1.05, "y_max": 1.05,
                    "series": [
                        {"label": "frame x[n]", "color": "muted", "x": xs, "y": raw_y, "width": 1},
                        {"label": "Hann window w[n]", "color": "steel", "x": xs, "y": win_y, "width": 1.5, "dash": True},
                        {"label": "windowed x[n]w[n]", "color": "amber", "x": xs, "y": out_y, "width": 1.5},
                    ]}],
    }


def stage_preemphasis(t):
    config = t["config"]
    sr = config.sample_rate
    b, a = design_preemphasis_filter(coeff=0.95)
    freqs = log_freqs(50.0, sr / 2 * 0.95, 200)
    h_db = response_db(b, a, freqs, sr)

    # average spectrum of the voiced frames, before and after the filter
    detected = t["detected_pitches"]
    before, _ = frame_spectra(t["audio"], config)
    after, _ = frame_spectra(t["detection_audio"], config)
    voiced = [i for i in range(min(len(detected), len(before))) if detected[i] > 0]
    if not voiced:
        voiced = list(range(len(before)))
    bin_freqs = np.arange(before.shape[1]) * sr / config.frame_size
    avg_before = np.mean(np.abs(before[voiced]) ** 2, axis=0)
    avg_after = np.mean(np.abs(after[voiced]) ** 2, axis=0)
    ref = 10 * np.log10(np.max(avg_before) + 1e-20)
    spec_before = 10 * np.log10(np.interp(freqs, bin_freqs, avg_before) + 1e-20) - ref
    spec_after = 10 * np.log10(np.interp(freqs, bin_freqs, avg_after) + 1e-20) - ref

    def h_at(f):
        return float(response_db(b, a, [f], sr)[0])

    xs = [round(float(f), 1) for f in freqs]
    return {
        "id": "preemphasis", "title": "Pre-emphasis filter",
        "summary": "H(z) = 1 - 0.95 z^-1 tilts the spectrum towards the highs - for the pitch detector only.",
        "explain": (
            "A first-order FIR filter, y[n] = x[n] - 0.95 x[n-1]. Its zero at z = 0.95 (close to "
            "z = 1, i.e. DC) cuts low frequencies far more than high ones: -25 dB at 150 Hz but "
            "only -3 dB at 5 kHz (it only goes above 0 dB near the top of the band). That tilt "
            "flattens the voice's naturally falling spectrum, so the upper harmonics count as "
            "much as the fundamental in the autocorrelation. The filtered copy is used ONLY for detecting pitch - the "
            "audio that gets tuned and saved is never filtered (that used to be a bug: the output "
            "came out thin and ~7x quieter)."),
        "stats": [["H(z)", "1 - 0.95 z^-1"],
                  ["|H| at 150 Hz", fmt_db(h_at(150))],
                  ["|H| at 1 kHz", fmt_db(h_at(1000))],
                  ["|H| at 5 kHz", fmt_db(h_at(5000))]],
        "charts": [{"type": "lines", "title": "Frequency response and average voice spectrum",
                    "x_label": "frequency (Hz)", "y_label": "dB", "x_log": True,
                    "x_min": 50, "x_max": round(sr / 2 * 0.95), "y_min": -90, "y_max": 12,
                    "series": [
                        {"label": "voice spectrum before", "color": "steel", "x": xs,
                         "y": [round(float(v), 2) for v in spec_before], "width": 1.2},
                        {"label": "after pre-emphasis", "color": "amber", "x": xs,
                         "y": [round(float(v), 2) for v in spec_after], "width": 1.2},
                        {"label": "filter |H(e^jw)|", "color": "text", "x": xs,
                         "y": [round(float(v), 2) for v in h_db], "width": 2, "dash": True},
                    ]}],
    }


def stage_pitch(t):
    config = t["config"]
    detected = t["detected_pitches"]
    voiced = detected[detected > 0]
    indices = thinned_indices(len(detected))
    hz = [float(f) if f > 0 else None for f in detected]
    stats = [["Voiced frames", f"{len(voiced) / max(len(detected), 1) * 100:.0f}%"]]
    y_max = 500.0
    if len(voiced):
        median = float(np.median(voiced))
        lo, hi = np.percentile(voiced, [5, 95])
        stats.append(["Median pitch", f"{median:.0f} Hz ({note_name(freq_to_midi(median))})"])
        stats.append(["Range (5-95%)", f"{lo:.0f}-{hi:.0f} Hz"])
        y_max = float(min(1000.0, hi * 1.4))
    stats.append(["Search range", "70-1000 Hz"])
    return {
        "id": "pitch", "title": "Pitch detection",
        "summary": "Autocorrelation finds how often each frame repeats: f0 = fs / lag.",
        "explain": (
            "For every frame the autocorrelation r[l] = sum x[n] x[n+l] is computed (via the FFT). "
            "A periodic voice matches itself best when shifted by one period, so the first "
            "strong peak at lag l gives f0 = fs / l. Refinements: dividing by the window's own "
            "autocorrelation (Boersma), an octave guard (take the first peak at least 0.9x the "
            "best, otherwise it jumps an octave down), parabolic interpolation between lags "
            "(~0.3 cent accuracy), and a clean-up pass (drop quiet frames and runs shorter than "
            "4 frames, 5-frame median). Gaps in the line are unvoiced frames: breaths, "
            "consonants, pauses."),
        "stats": stats,
        "charts": [{"type": "lines", "title": "Detected pitch f0 per frame",
                    "x_label": "time (s)", "y_label": "Hz", "x_min": 0,
                    "x_max": round(len(t["raw_audio"]) / config.sample_rate, 3),
                    "y_min": 50, "y_max": round(y_max),
                    "series": [track_series("detected f0", hz, config, "steel", indices, 1.5, jump_ratio=1.19)]}],
    }


def stage_key(t):
    detected = t["detected_pitches"]
    offset = t["tuning_offset"]
    hist = pitch_class_histogram(detected, offset)
    root = t["key_root"]
    key_type = t["key_type"]
    root_pc = NOTE_NAMES.index(root)
    steps = SCALE_INTERVALS[key_type]
    highlight = []
    for pc in range(12):
        highlight.append(((pc - root_pc) % 12) in steps)
    total = max(float(np.sum(hist)), 1.0)
    if key_type == "chromatic":
        key_label = "all 12 notes"
    else:
        key_label = f"{root} {'major' if key_type == 'major' else 'minor'}"
    if t["auto_key"]:
        how = ("Auto-detect: first the keys that contain (almost) all sung notes are kept "
               "(coverage), then the Krumhansl-Kessler key profile breaks ties by how well "
               "the note counts correlate with it; below 85% coverage it falls back to all "
               "12 notes.")
    else:
        how = "The key was chosen by hand."
    if t["follow_tuning"]:
        tuning = (f"Follow-my-tuning is on: the singer's average offset is {offset:+.0f} cents "
                  "(circular mean of every frame's deviation from the nearest note), and the "
                  "whole scale is slid by it.")
    else:
        tuning = "Follow-my-tuning is off: notes are tuned to standard pitch (A4 = 440 Hz)."
    stats = [["Key", key_label + (" (auto)" if t["auto_key"] else "")],
             ["Tuning offset", f"{offset:+.0f} cents" if t["follow_tuning"] else "off (A4 = 440 Hz)"]]
    if t["key_confidence"] is not None:
        stats.append(["Confidence", f"{t['key_confidence']:.2f}"])
    return {
        "id": "key", "title": "Key + tuning",
        "summary": f"Which notes are allowed: {key_label}.",
        "explain": (
            "Each voiced frame is turned into a note number, MIDI = 69 + 12 log2(f / 440), and "
            "counted by pitch class (C, C#, ... B), ignoring slides between notes. "
            + how + " " + tuning + " Highlighted bars are the notes of the key."),
        "stats": stats,
        "charts": [{"type": "bars", "title": "How often each note was sung",
                    "labels": NOTE_NAMES,
                    "values": [round(float(v / total * 100), 2) for v in hist],
                    "highlight": highlight, "y_label": "% of sung frames"}],
    }


def stage_targets(t):
    config = t["config"]
    detected = midi_track(t["detected_pitches"])
    targets = midi_track(t["target_pitches"])
    lo, hi = midi_range(detected, targets)
    indices = thinned_indices(len(detected))
    edits = t["num_overrides"]
    stats = [["Notes found", str(t["num_notes"])],
             ["Hysteresis", f"{config.note_hysteresis:.1f} semitones"]]
    if edits:
        stats.append(["Notes you edited", str(edits)])
    return {
        "id": "targets", "title": "Target notes",
        "summary": "Each frame gets the nearest allowed note - with hysteresis so it doesn't flicker.",
        "explain": (
            "The target of a frame is the nearest note of the key. On its own that flips back and "
            "forth when the voice sits between two notes (a warble), so there is hysteresis, like "
            f"a thermostat: the target only changes when the voice is closer to another note by "
            f"more than {config.note_hysteresis} semitones. The flat amber steps are the targets; "
            "the blue line is what was sung."
            + (" Notes you dragged in the editor use your note instead." if edits else "")),
        "stats": stats,
        "charts": [{"type": "lines", "title": "Sung pitch and target note (MIDI)",
                    "x_label": "time (s)", "y_label": "note", "x_min": 0,
                    "x_max": round(len(t["raw_audio"]) / config.sample_rate, 3),
                    "y_min": round(lo, 2), "y_max": round(hi, 2), "y_ticks": note_ticks(lo, hi),
                    "series": [track_series("sung", detected, config, "steel", indices, 1.5, max_jump=3),
                               track_series("target", targets, config, "amber", indices, 2.5, step=True, max_jump=3)]}],
    }


def stage_ratios(t):
    config = t["config"]
    detected = t["detected_pitches"]
    targets = t["target_pitches"]
    before = t["ratios_before_smoothing"]
    after = t["shift_ratios"]
    needed = []
    with_strength = []
    applied = []
    for i in range(len(detected)):
        if detected[i] > 0 and targets[i] > 0:
            needed.append(1200 * np.log2(targets[i] / detected[i]))
            with_strength.append(1200 * np.log2(before[i]))
            applied.append(1200 * np.log2(after[i]))
        else:
            needed.append(None)
            with_strength.append(None)
            applied.append(None)
    indices = thinned_indices(len(detected))
    real_applied = [abs(v) for v in applied if v is not None]
    real_needed = [abs(v) for v in needed if v is not None]
    limit = 100.0
    if real_needed:
        limit = float(min(300.0, max(60.0, np.percentile(real_needed, 99) * 1.2)))

    strength = config.correction_strength
    retune = config.retune_ms
    if retune <= 0:
        smoothing = ("Retune speed is 0 ms, so there is NO smoothing: every frame jumps straight "
                     "to its target - the amber line sits exactly on the dashed one. That instant "
                     "snap is the robotic Hard Tune sound.")
    else:
        alpha = 1 - np.exp(-(config.hop_size / config.sample_rate * 1000) / retune)
        smoothing = (f"Then the retune speed ({retune:.0f} ms) smooths it with an exponential moving "
                     f"average in log-ratio: each frame moves {alpha * 100:.0f}% of the way to its new "
                     "value (alpha = 1 - exp(-hop_time / retune_ms)), so notes glide onto pitch "
                     "instead of snapping.")
    return {
        "id": "ratios", "title": "Correction amount",
        "summary": f"How far each frame is moved: strength {strength * 100:.0f}%, retune {retune:.0f} ms.",
        "explain": (
            "The shift for a frame is the ratio r = target / detected (r = 1.059 is one semitone "
            "up). It is scaled by the correction strength ON A LOG SCALE, r^strength, because "
            f"pitch is perceived logarithmically - at {strength * 100:.0f}% strength a 50-cent error "
            f"is corrected by {50 * strength:.0f} cents. " + smoothing +
            " Shown in cents (100 cents = 1 semitone): grey = the full correction needed, "
            "dashed = after strength, amber = what was actually applied."),
        "stats": [["Strength", f"{strength * 100:.0f}%"],
                  ["Retune speed", f"{retune:.0f} ms"],
                  ["Typical error", fmt_cents(np.median(real_needed)) if real_needed else "-"],
                  ["Typical correction", fmt_cents(np.median(real_applied)) if real_applied else "-"]],
        "charts": [{"type": "lines", "title": "Correction per frame (cents)",
                    "x_label": "time (s)", "y_label": "cents", "x_min": 0,
                    "x_max": round(len(t["raw_audio"]) / config.sample_rate, 3),
                    "y_min": round(-limit), "y_max": round(limit),
                    "series": [track_series("full correction", needed, config, "muted", indices, 1),
                               track_series("after strength", with_strength, config, "steel", indices, 1.2, dash=True),
                               track_series("applied (after retune)", applied, config, "amber", indices, 2)]}],
    }


def stage_vocoder(t):
    config = t["config"]
    i = t["example_index"]
    sr = config.sample_rate
    N = config.frame_size
    ratio = float(t["shift_ratios"][i])
    f0 = float(t["detected_pitches"][i])
    x_in = np.fft.rfft(t["example_in_frame"])
    x_out = np.fft.rfft(t["example_out_frame"])
    ref = 20 * np.log10(np.max(np.abs(x_in)) + 1e-12)
    fmax = 3000.0
    if f0 > 0:
        fmax = float(min(sr / 2, max(1500.0, f0 * 12)))
    xs = []
    y_in = []
    y_out = []
    for k in range(int(fmax * N / sr) + 1):
        xs.append(round(k * sr / N, 1))
        y_in.append(round(float(20 * np.log10(np.abs(x_in[k]) + 1e-12) - ref), 2))
        y_out.append(round(float(20 * np.log10(np.abs(x_out[k]) + 1e-12) - ref), 2))

    method = "phase vocoder" if t["use_phase_vocoder"] else "naive resampling"
    before_audio = t["audio"]
    after_audio = t["corrected_pre_polish"]
    spec_top = 2000.0
    if f0 > 0:
        spec_top = float(min(4000.0, max(1000.0, f0 * 8)))
    stats = [["Method", method],
             ["Example frame", f"{frame_time(i, config):.2f} s"],
             ["Shift there", f"{1200 * np.log2(ratio):+.0f} cents (r = {ratio:.4f})"]]
    if f0 > 0:
        stats.append(["f0 there", f"{f0:.1f} -> {f0 * ratio:.1f} Hz"])
    if t["use_phase_vocoder"]:
        stats.append(["Formants", "preserved" if t["use_formants"] else "not preserved"])
    if t["use_phase_vocoder"]:
        explain = (
            "Each frame's FFT is searched for harmonic peaks; each peak's magnitude is MOVED to the "
            "bin nearest its new frequency r x f (with the bins around it, phase-locked to it), and "
            "its phase keeps rotating at the new frequency from frame to frame (peak tracking, "
            "Laroche & Dolson 1999) so overlapping frames still add up coherently. "
            + ("With formant preservation each moved harmonic takes the loudness of the ORIGINAL "
               "spectral envelope at its new place (cepstral smoothing), so the voice's timbre "
               "doesn't shift with the pitch. " if t["use_formants"] else "")
            + "Top: one frame's spectrum - the harmonic peaks slide to r times their frequency. "
            "Bottom: spectrogram of the voice before and after - watch the harmonic lines "
            "straighten onto the notes.")
    else:
        explain = (
            "Naive resampling: each frame is read faster or slower (np.interp at positions n x r) "
            "with no anti-aliasing filter - kept for comparison. It shifts the pitch but also "
            "stretches the formants (\"chipmunk\" voice) and can alias.")
    return {
        "id": "vocoder", "title": "Pitch shifting",
        "summary": f"Every frame is moved by its ratio using the {method}.",
        "explain": explain,
        "stats": stats,
        "charts": [
            {"type": "lines", "title": f"One frame's spectrum before / after ({frame_time(i, config):.2f} s)",
             "x_label": "frequency (Hz)", "y_label": "dB", "x_min": 0, "x_max": round(fmax),
             "y_min": -80, "y_max": 5,
             "series": [{"label": "before", "color": "steel", "x": xs, "y": y_in, "width": 1.2},
                        {"label": "after", "color": "amber", "x": xs, "y": y_out, "width": 1.5}]},
            spectrogram_chart("Voice spectrogram before / after shifting",
                              [("before", before_audio), ("after", after_audio)], config, 60.0, spec_top),
        ],
    }


def stage_result(t):
    config = t["config"]
    detected = midi_track(t["detected_pitches"])
    targets = midi_track(t["target_pitches"])
    corrected = midi_track(t["corrected_pitches"])
    lo, hi = midi_range(detected, corrected)
    indices = thinned_indices(len(detected))

    err_before = []
    err_after = []
    for i in range(min(len(detected), len(corrected))):
        if detected[i] is not None and targets[i] is not None and corrected[i] is not None:
            err_before.append(abs(detected[i] - targets[i]) * 100)
            err_after.append(abs(corrected[i] - targets[i]) * 100)
    stats = []
    if err_before:
        stats.append(["Off-target before", fmt_cents(np.median(err_before))])
        stats.append(["Off-target after", fmt_cents(np.median(err_after))])
    stats.append(["Strength", f"{config.correction_strength * 100:.0f}%"])
    return {
        "id": "result", "title": "Overlap-add = tuned voice",
        "summary": "The shifted frames are added back together; the pitch is measured again.",
        "explain": (
            "Overlap-add: every shifted frame is windowed again and added at its original "
            f"position (hop {config.hop_size}), then divided by the sum of the squared windows so "
            "the overlaps don't change the loudness: y[n] = sum_m y_m[n - mH] w[n - mH] / "
            "sum_m w^2[n - mH]. A frame whose ratio was exactly 1 comes out bit-identical. The "
            "result is run through the same pitch detector: the amber line should now sit on "
            "the targets (fully at 100% strength, part-way at lower strengths)."),
        "stats": stats,
        "charts": [{"type": "lines", "title": "Pitch before and after tuning (MIDI)",
                    "x_label": "time (s)", "y_label": "note", "x_min": 0,
                    "x_max": round(len(t["raw_audio"]) / config.sample_rate, 3),
                    "y_min": round(lo, 2), "y_max": round(hi, 2), "y_ticks": note_ticks(lo, hi),
                    "series": [track_series("target", targets, config, "muted", indices, 3, step=True, max_jump=3),
                               track_series("sung", detected, config, "steel", indices, 1.2, max_jump=3),
                               track_series("after tuning", corrected, config, "amber", indices, 1.8, max_jump=3)]}],
    }


def stage_eq(t):
    sr = t["config"].sample_rate
    freqs = log_freqs(20.0, sr / 2 * 0.95, 240)
    parts = [("high-pass 80 Hz", design_highpass(80.0, sr)),
             ("-2 dB at 300 Hz", design_peaking_eq(300.0, -2.0, sr, q=1.0)),
             ("+2.5 dB at 3 kHz", design_peaking_eq(3000.0, 2.5, sr, q=0.9)),
             ("+3 dB shelf above 10 kHz", design_high_shelf(10000.0, 3.0, sr))]
    total = np.zeros(len(freqs))
    series = []
    xs = [round(float(f), 1) for f in freqs]
    for label, (b, a) in parts:
        h = response_db(b, a, freqs, sr)
        total += h     # cascaded filters multiply -> their dB responses add
        series.append({"label": label, "color": "muted", "x": xs,
                       "y": [round(float(v), 2) for v in h], "width": 1, "dash": True})
    series.append({"label": "whole EQ", "color": "amber", "x": xs,
                   "y": [round(float(v), 2) for v in total], "width": 2.2})
    polish = t["polish"]
    return {
        "id": "eq", "title": "EQ (biquad filters)",
        "summary": "Four 2nd-order IIR filters in a row shape the tone.",
        "explain": (
            "Each filter is a biquad, H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2), "
            "designed with the Audio EQ Cookbook formulas: a high-pass at 80 Hz removes rumble "
            "below the lowest sung note, a small cut at 300 Hz removes 'boxiness', a boost at "
            "3 kHz adds presence (clearer words) and a shelf above 10 kHz adds 'air'. Filters in "
            "series multiply their responses, so in dB they ADD - the amber curve is the sum of "
            "the dashed ones."),
        "stats": [["Filters", "4 biquads (2 poles + 2 zeros each)"],
                  ["At 50 Hz", fmt_db(float(np.interp(50, freqs, total)))],
                  ["At 300 Hz", fmt_db(float(np.interp(300, freqs, total)))],
                  ["At 3 kHz", fmt_db(float(np.interp(3000, freqs, total)))],
                  ["At 12 kHz", fmt_db(float(np.interp(12000, freqs, total)))]],
        "charts": [{"type": "lines", "title": "Frequency response |H(e^jw)|",
                    "x_label": "frequency (Hz)", "y_label": "dB", "x_log": True,
                    "x_min": 20, "x_max": round(sr / 2 * 0.95), "y_min": -24, "y_max": 6,
                    "series": series},
                   {"type": "wave", "title": "Signal before / after EQ",
                    "duration": round(len(polish["eq"]) / sr, 3),
                    "series": [wave_series("before", t["corrected_pre_polish"], "muted"),
                               wave_series("after EQ", polish["eq"], "amber")]}],
    }


def stage_compressor(t):
    sr = t["config"].sample_rate
    polish = t["polish"]
    gain = polish["gain_db"]
    columns = min(WAVE_COLUMNS, len(gain))
    xs = []
    ys = []
    for c in range(columns):
        start = c * len(gain) // columns
        end = max(start + 1, (c + 1) * len(gain) // columns)
        xs.append(round(start / sr, 3))
        ys.append(round(float(np.min(gain[start:end])), 2))   # the strongest reduction in that stretch
    # normalise both waveforms to the same peak so the SHAPE change is visible
    before = polish["eq"] / max(np.max(np.abs(polish["eq"])), 1e-9)
    after = polish["compressed"] / max(np.max(np.abs(polish["compressed"])), 1e-9)
    return {
        "id": "compressor", "title": "Compressor",
        "summary": "Loud parts are turned down (3:1 above -24 dB), so every word is heard.",
        "explain": (
            "A level meter (square the signal, 10 ms one-pole low-pass: y[n] = (1-c) x^2[n] + "
            "c y[n-1]) measures the short-term loudness. Above the -24 dB threshold every extra "
            "3 dB in becomes only 1 dB out (ratio 3:1); far below the singing (under -50 dB: "
            "hiss, room noise) the level is turned DOWN instead so later boosts don't raise the "
            "noise. The gain is smoothed (80 ms) so it moves gently. The graph shows that gain; "
            "the waveforms (scaled to the same peak) show loud and soft parts coming closer "
            "together."),
        "stats": [["Threshold", "-24 dB"], ["Ratio", "3:1"],
                  ["Most reduction", fmt_db(float(np.min(gain)))],
                  ["Typical gain (singing)", fmt_db(float(np.median(gain[gain < -0.5]))) if np.any(gain < -0.5) else "0 dB"]],
        "charts": [{"type": "lines", "title": "Compressor gain over time",
                    "x_label": "time (s)", "y_label": "gain (dB)", "x_min": 0,
                    "x_max": round(len(gain) / sr, 3), "y_min": round(min(-3.0, float(np.min(gain)) - 1)),
                    "y_max": 1,
                    "series": [{"label": "gain", "color": "amber", "x": xs, "y": ys, "width": 1.5}]},
                   {"type": "wave", "title": "Before / after compression (same peak)",
                    "duration": round(len(before) / sr, 3),
                    "series": [wave_series("before", before, "muted"),
                               wave_series("after", after, "amber")]}],
    }


def reverb_decay_time(h, sample_rate):
    """
    RT60 estimate by Schroeder backward integration: the energy still to come
    after time t, E(t) = sum_{n >= t} h^2[n], falls in a straight line in dB.
    Fit the -5..-25 dB part and extrapolate to -60 dB (the "T20" method).
    """
    energy = np.cumsum((h ** 2)[::-1])[::-1]
    edc = 10 * np.log10(energy / energy[0] + 1e-20)
    start = np.argmax(edc <= -5)
    end = np.argmax(edc <= -25)
    if end <= start:
        return None
    slope = (edc[end] - edc[start]) / ((end - start) / sample_rate)   # dB per second
    return -60.0 / slope


def stage_reverb(t):
    sr = t["config"].sample_rate
    polish = t["polish"]
    amount = t["config"].reverb_amount
    h = reverb_impulse_response(sr)
    shown = h[:int(1.6 * sr)]
    rt60 = reverb_decay_time(h, sr)
    stats = [["Amount (wet)", f"{amount * 100:.0f}%"],
             ["Impulse response", f"{len(h) / sr:.1f} s"],
             ["Pre-delay", "20 ms"]]
    if rt60:
        stats.append(["Decay time RT60", f"{rt60:.2f} s"])
    return {
        "id": "reverb", "title": "Reverb (convolution)",
        "summary": "y = x + amount x (x * h): the voice convolved with a room's impulse response.",
        "explain": (
            "A room is an LTI system, so it is fully described by its impulse response h[n] "
            "(what you'd record after one hand clap). Reverb is then just convolution, y = x * h, "
            "computed with FFTs (convolution in time = multiplication in frequency) because h is "
            f"{len(h) / sr:.0f} s = {len(h):,} samples long. h is designed the classic Schroeder way: "
            "8 feedback comb filters (echoes between walls, each repeat a bit duller) in parallel, "
            "then 4 all-pass filters that smear every echo into many. Top: h[n]; bottom: the dry "
            "and the reverberant signal."),
        "stats": stats,
        "charts": [{"type": "wave", "title": "Impulse response h[n] (first 1.6 s)",
                    "duration": round(len(shown) / sr, 3),
                    "series": [wave_series("h[n]", shown / max(np.max(np.abs(shown)), 1e-9), "steel")]},
                   {"type": "wave", "title": "Dry / with reverb",
                    "duration": round(len(polish["reverb"]) / sr, 3),
                    "series": [wave_series("with reverb", polish["reverb"], "amber"),
                               wave_series("dry", polish["compressed"], "muted")]}],
    }


def stage_final(t):
    sr = t["config"].sample_rate
    polish = t["polish"]
    final = polish["final"]
    before_peak = np.max(np.abs(polish["reverb"]))
    gain = db_amplitude(np.max(np.abs(final))) - db_amplitude(before_peak)
    return {
        "id": "final", "title": "Normalise = final track",
        "summary": "Scaled so the loudest sample is at -1 dBFS.",
        "explain": (
            "The whole signal is multiplied by one constant so its highest peak sits at -1 dBFS "
            "(0.891 of full scale): as loud as possible without clipping, with a little headroom "
            "for MP3 encoding. The graph compares the finished track with your original "
            "recording."),
        "stats": [["Peak", "-1.0 dBFS"], ["Gain applied", fmt_db(gain)],
                  ["Length", f"{len(final) / sr:.1f} s"]],
        "charts": [{"type": "wave", "title": "Original vs final",
                    "duration": round(len(final) / sr, 3),
                    "series": [wave_series("final", final, "amber"),
                               wave_series("original", t["raw_audio"], "steel")]}],
    }


def build_stages(trace):
    """
    trace: the dict collected by run_pipeline(collect_stages=True).
    Returns the list of steps, in processing order, containing only the
    steps that actually ran with these settings.
    """
    config = trace["config"]
    stages = [stage_input(trace)]
    if config.noise_reduction > 0:
        stages.append(stage_noise(trace))
    stages.append(stage_framing(trace))
    if trace["use_preemphasis"]:
        stages.append(stage_preemphasis(trace))
    stages.append(stage_pitch(trace))
    stages.append(stage_key(trace))
    stages.append(stage_targets(trace))
    stages.append(stage_ratios(trace))
    stages.append(stage_vocoder(trace))
    stages.append(stage_result(trace))
    if trace["polish"] is not None:
        stages.append(stage_eq(trace))
        stages.append(stage_compressor(trace))
        if config.reverb_amount > 0:
            stages.append(stage_reverb(trace))
        stages.append(stage_final(trace))
    return stages
