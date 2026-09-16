import numpy as np

"""
Naive Pitch Shifting via Resampling
=====================================

The idea: to change pitch, we change how "densely" we read the original
samples. If we read faster (skip ahead), the waveform compresses in time,
which raises pitch when played back at the normal rate. If we read slower
(re-visit nearby points via interpolation), the waveform stretches, which
lowers pitch.

We do this using simple linear interpolation (np.interp) - and deliberately
so, because linear interpolation does NOT apply a low-pass filter before
resampling. This is exactly what makes it "naive" and lets us demonstrate
a real course concept: ALIASING.

Reminder from the sampling theorem: before you reduce your effective sample
rate (skip over samples), you must remove frequency content above the new
Nyquist limit, or higher frequencies "fold back" and appear as false lower
frequencies - this is aliasing. Proper resamplers (like scipy's resample_poly,
which you already used in io_utils.py) apply this filter. Here, we skip that
filter on purpose so you can hear/see the difference against the phase
vocoder later.

How the shift actually works:

original indices:   0, 1, 2, 3, ..., frame_size-1
read positions:      0, shift_ratio, 2*shift_ratio, 3*shift_ratio, ...

Example: frame_size=8, shift_ratio=2.0 (shift up an octave)
read positions: 0, 2, 4, 6, 8, 10, 12, 14
Notice positions 8-14 don't exist in our original frame (max index is 7)!
np.interp will just clamp these to the last sample - this clamping is itself
one of the naive method's artifacts (not a bug, just a real limitation).

Example: frame_size=8, shift_ratio=0.5 (shift down an octave)
read positions: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
We only ever use the FIRST HALF of the frame, stretched to fill the whole
output - the back half of the original frame (indices 4-7) never gets used.
This is why naive pitch shifting also distorts information, not just pitch.
"""

def naive_pitch_shift(frame, shift_ratio):
    """
    frame: np.ndarray, shape (frame_size,) - one windowed audio frame
    shift_ratio: float - e.g. 1.05 = shift up ~1 semitone, 0.95 = shift down

    Returns: np.ndarray, shape (frame_size,) - shifted frame, same length,
    so it can drop straight into overlap_add() unchanged.
    """
    n = len(frame)
    original_indices = np.arange(n)
    read_positions = np.arange(n) * shift_ratio

    # np.interp reads frame[] at these possibly-fractional positions,
    # linearly blending between the two nearest real samples.
    # Positions beyond n-1 are clamped to the last sample automatically.
    shifted = np.interp(read_positions, original_indices, frame)

    return shifted.astype(np.float32)


"""
Computing Shift Ratios from Detected + Target Pitch
=====================================================

Once pitch_detection.py tells us the detected pitch per frame, and scales.py
tells us the nearest correct note (target pitch), we need a single number:
how much to shift each frame by.

The obvious approach: ratio = target_pitch / detected_pitch
But we also want a "correction strength" knob (0.0 = no correction at all,
1.0 = full correction) so we can demo both natural, subtle correction and
the classic hard robotic snap.

Why we use exponentiation instead of linear blending for partial strength:
Pitch is perceived LOGARITHMICALLY - this is the exact same reasoning
scales.py already uses (MIDI numbers are a log2-based scale, see freq_to_midi).
So "50% correction" should mean "halfway in semitone-distance", not "halfway
in raw Hz-ratio". This is done with:

    ratio = (target/detected) ** strength

When strength = 1.0: ratio = target/detected (full correction)
When strength = 0.0: ratio = 1.0 (no change at all, since anything^0 = 1)
When strength = 0.5: ratio = halfway between them ON THE LOG SCALE
"""

def compute_shift_ratios(detected_pitches, target_pitches, strength=1.0):
    """
    detected_pitches: np.ndarray, shape (num_frames,) - from pitch_detection.py
                       0.0 means unvoiced/silent frame
    target_pitches:   np.ndarray, shape (num_frames,) - nearest scale note,
                       from scales.py's nearest_scale_note(), per frame
    strength:         float, 0.0 to 1.0 - correction amount

    Returns: np.ndarray, shape (num_frames,) - one shift ratio per frame,
    ready to feed into naive_pitch_shift() or later phase_vocoder_shift()
    """
    ratios = np.ones_like(detected_pitches, dtype=np.float32)

    # Only compute a real ratio where the frame is actually voiced -
    # unvoiced frames (silence, noise) get ratio=1.0 (no shift, leave as is)
    voiced = detected_pitches > 0

    raw_ratio = np.ones_like(detected_pitches, dtype=np.float32)
    raw_ratio[voiced] = target_pitches[voiced] / detected_pitches[voiced]

    ratios[voiced] = raw_ratio[voiced] ** strength

    return ratios



def smooth_shift_ratios(shift_ratios, detected_pitches, hop_size, sample_rate, retune_ms=40.0):
    """
    Smooths the per-frame correction ratio over time so pitch glides toward
    the target note instead of snapping fully in a single ~11ms frame. This
    is what separates a natural-sounding correction from the hard, robotic
    "T-Pain" snap - compute_shift_ratios() alone recomputes a fresh target
    every frame with no memory of the previous frame, so any jitter in the
    detected pitch (very normal with autocorrelation + natural vibrato)
    shows up directly as flutter in the corrected audio.

    Uses an exponential moving average in the LOG of the ratio, not the raw
    ratio - shift ratios are multiplicative (a ratio of 2.0 up and 0.5 down
    are equally "one octave"), so averaging in log space is what keeps the
    glide symmetric between upward and downward corrections.

    The average resets at the start of every voiced run (i.e. after a
    silence/unvoiced gap) so a new note starts clean instead of gliding in
    from whatever the previous note's ratio happened to be.

    retune_ms: how many milliseconds it takes to glide most of the way to
    the target pitch.
      0        -> no smoothing at all (identical to the old instant-snap behavior)
      ~30-80   -> natural-sounding correction
      100+     -> audible pitch bends/glides rather than a "correction"
    """
    if retune_ms <= 0:
        return shift_ratios

    log_ratios = np.log(shift_ratios.astype(np.float64))
    smoothed_log = np.zeros_like(log_ratios)

    hop_time_ms = (hop_size / sample_rate) * 1000.0
    alpha = 1.0 - np.exp(-hop_time_ms / retune_ms)

    prev = None
    for i in range(len(log_ratios)):
        if detected_pitches[i] <= 0:
            # unvoiced: nothing to glide, and reset so the next note doesn't
            # inherit a stale glide-in-progress from before the gap
            smoothed_log[i] = log_ratios[i]
            prev = None
            continue
        if prev is None:
            smoothed_log[i] = log_ratios[i]  # first voiced frame of a run: start clean, no glide-in
        else:
            smoothed_log[i] = prev + alpha * (log_ratios[i] - prev)
        prev = smoothed_log[i]

    return np.exp(smoothed_log).astype(np.float32)