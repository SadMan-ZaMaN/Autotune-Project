import sys
sys.path.insert(0, '.')
import numpy as np

"""
Phase Vocoder Pitch Shifting
==============================
Summary of the pieces (each has its own docstring with a worked example):
  1. compute_stft - magnitude + phase per frame
  2. compute_instantaneous_frequency - precise per-bin frequency via phase tracking
  3. find_peaks / assign_regions - locate spectral peaks (harmonics) and figure
     out which bins "belong" to which peak
  4. compute_spectral_envelope - the smooth formant shape of a frame (used
     to keep the voice's timbre while its pitch moves)
  5. remap_and_lock_spectrum - moves each peak's region of bins to its
     shifted frequency, phase-locked to the peak, with the peak's phase
     TRACKED from the previous frame
  6. phase_vocoder_shift - ties it all together (the CONTRACTS.md function)

--------------------------------------------------------------------------
History (see CLAUDE.md "Critical bug history" before changing this file):

Attempts 1 and 2 left each harmonic's magnitude in its ORIGINAL bin and only
changed its phase. Each FFT bin behaves like a narrow bandpass filter
(about +/- sample_rate/frame_size wide), so it can't output energy far from
its own frequency - ratio=1.05 produced 78 Hz instead of 157.5 Hz. Never go
back to that.

Attempt 3 fixed it by actually MOVING each harmonic peak (plus the sidelobe
bins around it) to a bin near its shifted frequency, phase-locked to the peak.

Attempt 4 (this file) keeps attempt 3's remap + lock, and fixes how the
PHASE is carried from one frame to the next:
  - Attempt 3 kept one running phase per OUTPUT BIN. A sung note wobbles
    (vibrato, drift), so a harmonic often lands in bin 93 in one frame and
    bin 94 in the next - and bin 94's running phase was stale (it last held
    some other harmonic, or nothing). Every such hop was a small phase
    glitch, heard as roughness/"phasiness", worse with vibrato.
  - Now each PEAK is tracked: we find which peak it was in the previous
    frame and continue THAT peak's phase, wherever its bin moved to.
  - The phase is stored as a ROTATION added on top of the original phase
    (Laroche & Dolson, 1999). When shift_ratio == 1 the rotation stays 0,
    so frames that need no correction (silence, breaths, "s"/"t"
    consonants) come out bit-for-bit identical to the input instead of
    being re-synthesized.
  - Formant preservation now happens inside the remap (each moved
    harmonic takes the loudness the ORIGINAL envelope has at its NEW
    frequency) instead of as a separate after-step - see
    compute_spectral_envelope for why the old after-step was biased.

Measured on a synthetic singer (vibrato, glides, consonants) against an
ideal re-synthesis, attempt 3 -> attempt 4:
    harmonic-to-noise ratio (clarity)   25.7 dB -> 31.1 dB (input: 32.4 dB)
    spectral error vs ideal             3.9 dB  -> 1.8 dB
    consonant (unvoiced) distortion     6.0 dB  -> 0.24 dB
    speed                               ~3x faster
tests/test_phase_vocoder.py still passes (all ratios incl. 1.05).
--------------------------------------------------------------------------
"""


"""
Piece 1: Converting frames to frequency-domain (magnitude + phase)
====================================================================
np.fft.rfft(frame) gives a COMPLEX number per frequency bin, which encodes:
    magnitude = np.abs(X[k])    -> "how much" of this frequency is present
    phase     = np.angle(X[k])  -> "where in its cycle" that frequency is
To SHIFT pitch cleanly we need both: magnitude says what to move, phase is
what lets consecutive frames line up smoothly when we overlap-add them.
"""

def compute_stft(frames, config):
    """
    frames: shape (num_frames, frame_size) - already windowed, output of frame_signal()
    Returns magnitudes, phases - both shape (num_frames, frame_size//2 + 1)
    (frame_size//2 + 1 = number of bins rfft gives for a real input)
    """
    num_frames = frames.shape[0]
    num_bins = config.frame_size // 2 + 1
    magnitudes = np.zeros((num_frames, num_bins))
    phases = np.zeros((num_frames, num_bins))
    for i in range(num_frames):
        spectrum = np.fft.rfft(frames[i])
        magnitudes[i] = np.abs(spectrum)
        phases[i] = np.angle(spectrum)
    return magnitudes, phases


"""
Piece 2: Instantaneous Frequency via Phase Difference
========================================================
Bin frequencies are coarse (44100/2048 = 21.5 Hz apart). By comparing how much
a bin's phase ACTUALLY advanced between two frames with how much it would
advance if the signal sat exactly at the bin's centre, we get the true
frequency much more precisely.

Worked example (frame_size=2048, hop_size=512, sample_rate=44100):
    bin k=20 centre = 20 * 44100/2048 = 430.7 Hz
    expected advance per hop = 2*pi * 20 * 512/2048 = 31.416 rad
    a real 440 Hz tone advances 2*pi * 440 * 512/44100 = 32.076 rad
    deviation = 0.660 rad per hop
    extra freq = 0.660/(2*pi) * (44100/512) = 9.04 Hz
    estimate = 430.7 + 9.04 = 439.7 Hz  (vs. coarse 430.7 Hz)

np.angle() only returns values in [-pi, pi], so the deviation must be
wrapped back into that range first (wrap_phase), or the estimate is wildly off.
"""

def wrap_phase(phase_diff):
    """
    Wraps a phase (radians) into [-pi, pi].
    Example: wrap_phase(7.5) = 7.5 - 2*pi = 1.22
    Example: wrap_phase(-4.0) = -4.0 + 2*pi = 2.28
    """
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


def compute_instantaneous_frequency(phases, config):
    """
    phases: shape (num_frames, num_bins) - from compute_stft()
    Returns: shape (num_frames, num_bins) - precise frequency (Hz) per bin per frame.
    The first frame has no previous frame, so it uses the bin centre frequencies.
    """
    num_frames, num_bins = phases.shape
    freqs = np.zeros((num_frames, num_bins))
    bin_center_freqs = np.arange(num_bins) * config.sample_rate / config.frame_size
    expected_advance = 2 * np.pi * np.arange(num_bins) * config.hop_size / config.frame_size
    freqs[0] = bin_center_freqs
    for i in range(1, num_frames):
        actual_advance = phases[i] - phases[i - 1]
        deviation = wrap_phase(actual_advance - expected_advance)
        extra_freq = deviation / (2 * np.pi) * (config.sample_rate / config.hop_size)
        freqs[i] = bin_center_freqs + extra_freq
    return freqs


"""
Piece 3: Peaks and their regions
==================================
A sung note is a comb of harmonics: in the magnitude spectrum each harmonic
is a peak a few bins wide (the Hann window's main lobe is 4 bins wide). To
move a harmonic we must move its WHOLE shape - the peak bin and the bins
around it - together, keeping their phases locked to the peak. The bins a
peak "owns" are its region: everything closer to it than to the neighbouring
peak.
"""

def find_peaks(magnitude, min_rel_height=0.001):
    """
    Local maxima in the magnitude spectrum, i.e. roughly one bin per harmonic.

    min_rel_height: a bin only counts as a peak if it is at least this
    fraction of the frame's loudest bin. 0.001 = -60 dB.
    Why so low: a voice's harmonics fall off steeply with frequency - the
    20th harmonic can easily be 40-50 dB below the fundamental, yet it is
    still part of the voice's "brightness". With the old 0.05 (-26 dB) limit
    every quiet upper harmonic was lumped into the region of the last loud
    one and shifted by the wrong amount (measured: 1.8 dB vs 3.3 dB spectral
    error against an ideal shift). Tiny noise peaks that sneak in are
    harmless - they just become small regions of their own.
    """
    num_bins = len(magnitude)
    if num_bins < 3:
        return np.array([], dtype=int)
    peak_max = magnitude.max()
    if peak_max <= 1e-12:
        return np.array([], dtype=int)  # digital silence: nothing to shift
    peak_mask = np.zeros(num_bins, dtype=bool)
    peak_mask[1:-1] = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:])
    peak_mask &= magnitude > (min_rel_height * peak_max)
    peaks = np.flatnonzero(peak_mask)
    if len(peaks) == 0:
        # no clear local maxima - fall back to the single loudest bin so the
        # frame still has one region covering everything
        peaks = np.array([int(np.argmax(magnitude))])
    return peaks


def region_bounds(peaks, num_bins):
    """
    For each peak, the first and one-past-last bin of its region.
    Regions split at the midpoint between neighbouring peaks.

    Example: peaks at bins [10, 20, 36], num_bins = 50
        peak 10 owns bins  0..15   (midpoint of 10 and 20 is 15)
        peak 20 owns bins 16..28   (midpoint of 20 and 36 is 28)
        peak 36 owns bins 29..49
    -> starts = [0, 16, 29], ends = [16, 29, 50]
    """
    starts = []
    ends = []
    for i in range(len(peaks)):
        if i == 0:
            starts.append(0)
        else:
            starts.append((peaks[i - 1] + peaks[i]) // 2 + 1)
        if i == len(peaks) - 1:
            ends.append(num_bins)
        else:
            ends.append((peaks[i] + peaks[i + 1]) // 2 + 1)
    return starts, ends


def assign_regions(peaks, num_bins):
    """
    For every bin, which peak owns it (region[bin] = that peak's bin index).
    Used for peak TRACKING: "which peak of the previous frame owned the bin
    where this frame's peak now sits?" - that is the same harmonic, one hop
    earlier.
    """
    region = np.zeros(num_bins, dtype=int)
    starts, ends = region_bounds(peaks, num_bins)
    for i in range(len(peaks)):
        region[starts[i]:ends[i]] = peaks[i]
    return region


"""
Piece 4: Formant envelope (keeps the voice sounding like the same person)
==========================================================================
Pitch = which note (spacing of the harmonic comb). Formants = the smooth
"shape" the comb sits under, set by the singer's throat/mouth - it's what
makes a voice recognisable. If we shift the comb AND the shape together, a
big shift sounds like a chipmunk. We want to move the comb only.

Cepstral smoothing finds the shape:
    log(magnitude) -> inverse FFT ("cepstrum") -> keep first ~30 values,
    zero the rest -> FFT back
The slowly-varying formant shape lives in the first cepstral values, the
fast ripple of the individual harmonics lives further out, so keeping only
the first ~30 is a low-pass filter on the spectrum's shape.
"""

def compute_spectral_envelope(magnitude, config, num_coeffs=30):
    """
    magnitude: shape (frame_size//2 + 1,) - one frame's magnitude spectrum
    num_coeffs: cepstral values kept. Smaller = smoother envelope (loses
                formant detail); larger = starts following individual
                harmonics. ~20-40 is reasonable at our settings (measured:
                20, 30 and 40 give nearly identical results).

    Returns: shape (frame_size//2 + 1,) - the LOG of the smooth envelope.
    (log, because we only ever need ratios of the envelope: a ratio of two
    envelopes is a difference of two log-envelopes.)
    """
    log_magnitude = np.log(magnitude + 1e-8)  # +1e-8 avoids log(0)
    cepstrum = np.fft.irfft(log_magnitude, n=config.frame_size)

    # keep the first and last few coefficients (both ends are needed for the
    # result to stay real-valued, same symmetry reasoning as rfft)
    liftered = np.zeros_like(cepstrum)
    liftered[:num_coeffs] = cepstrum[:num_coeffs]
    liftered[-(num_coeffs - 1):] = cepstrum[-(num_coeffs - 1):]

    smooth_log_magnitude = np.fft.rfft(liftered, n=config.frame_size).real
    return smooth_log_magnitude


def formant_gain(log_envelope, source_bins, target_bins, max_gain=4.0):
    """
    How much louder/quieter a harmonic must become when it moves from
    source_bins to target_bins, so that it follows the ORIGINAL formant
    envelope at its new position:
        gain = envelope(new frequency) / envelope(old frequency)
             = exp(log_env[target] - log_env[source])

    Example: a harmonic at 650 Hz sits just below a formant peak at 700 Hz.
    Shifted up 12% to 728 Hz it lands right ON the formant peak, where the
    original envelope is, say, 1.4x higher -> gain 1.4. Without this, the
    harmonic would keep its old loudness and the formant peak would move up
    by 12% along with it ("chipmunk").

    max_gain limits the correction to +/-12 dB so a noisy envelope estimate
    can never blow a harmonic up or wipe it out.

    Why this replaced the old formant_preserve() after-step: that version
    divided the SHIFTED frame by its own envelope. Remapping leaves exact
    zeros between moved regions (log(1e-8) = -18), which dragged the
    shifted envelope far down, so the division boosted everything: measured
    +1.6 dB loudness error and slightly WORSE spectral error than no formant
    step at all. Here only the ORIGINAL frame's envelope is used (it has no
    artificial gaps), so there is no bias.
    """
    log_ratio = log_envelope[target_bins] - log_envelope[source_bins]
    limit = np.log(max_gain)
    log_ratio = np.clip(log_ratio, -limit, limit)
    return np.exp(log_ratio)


"""
Piece 5: Moving each harmonic, with its phase tracked across frames
=====================================================================
For a peak p with true frequency f (from Piece 2) and ratio r:

(a) WHERE it goes: move its whole region by
        shift_bins = round((r - 1) * f / bin_spacing)
    Example: f = 1000 Hz, r = 1.05 -> +50 Hz -> 50/21.5 = 2.3 -> 2 bins.
    Example: f = 150 Hz,  r = 1.05 -> +7.5 Hz -> 0.35 -> 0 bins (!)
    The second case is fine: the exact new frequency is set by the PHASE
    (next step), and a bin can represent any frequency within about half a
    bin of its centre. Moving magnitude is what makes the big jumps
    possible (the bug in attempts 1 and 2); phase does the fine tuning.

(b) WHAT phase it gets: a sinusoid at frequency f advances its phase by
    2*pi*f*hop/fs every hop. The input already does that on its own. We
    want 2*pi*(r*f)*hop/fs, i.e. an EXTRA
        delta_rotation = 2*pi*(r - 1)*f*hop/fs   per hop.
    We keep a running total of that extra ("rotation") for each harmonic
    and add it to the harmonic's original phase:
        output_phase = input_phase + rotation
    Example: f = 200 Hz, r = 1.05, hop = 512, fs = 44100
        delta_rotation = 2*pi*0.05*200*512/44100 = 0.729 rad per hop
    If r = 1, delta_rotation = 0 and the output IS the input.

(c) WHICH rotation to continue: the harmonic that is at bin p now was, one
    hop ago, the peak whose region contained bin p (it can only move a
    little in 11.6 ms). We continue THAT peak's rotation. This is the fix
    for attempt 3, which continued the phase stored at the output bin
    instead, and so broke continuity whenever vibrato moved a harmonic to
    a neighbouring bin.

(d) Linear-phase correction -pi*shift_bins: the FFT measures phase from the
    START of the frame but the Hann window is centred in the MIDDLE, so a
    windowed sinusoid's phase at bin k contains an extra -pi*k term
    (alternating sign between neighbouring bins). A region moved by
    shift_bins bins must lose pi*shift_bins to stay consistent.
    (Measured: without it clarity drops 3 dB.)
"""

def remap_and_lock_spectrum(magnitude, phase, true_freq, shift_ratio,
                            prev_region, prev_rotation, config, log_envelope=None):
    """
    Builds the pitch-shifted spectrum for ONE frame.

    magnitude, phase, true_freq: shape (num_bins,) - this frame's analysis
    shift_ratio: float - e.g. 1.05 = up ~0.85 semitone
    prev_region: from assign_regions() for the previous frame (or None)
    prev_rotation: dict {previous peak bin -> its rotation} (or None)
    log_envelope: from compute_spectral_envelope() to preserve formants,
                  or None to let formants move with the pitch.

    Returns:
        new_spectrum: complex, shape (num_bins,)
        region:       assign_regions() for this frame (for the next frame)
        rotation:     dict {this frame's peak bin -> rotation} (for the next frame)
    """
    num_bins = len(magnitude)
    bin_spacing = config.sample_rate / config.frame_size
    hop_over_sr = config.hop_size / config.sample_rate

    peaks = find_peaks(magnitude)
    starts, ends = region_bounds(peaks, num_bins)
    region = assign_regions(peaks, num_bins)

    # --- step (c)+(b): continue each tracked harmonic's rotation ---------
    rotation = {}
    for p in peaks:
        previous = 0.0
        if prev_region is not None and prev_rotation is not None:
            previous = prev_rotation.get(prev_region[p], 0.0)
        rotation[p] = previous + 2 * np.pi * (shift_ratio - 1.0) * true_freq[p] * hop_over_sr

    # --- step (a)+(d): move every region, phase-locked to its peak -------
    new_spectrum = np.zeros(num_bins, dtype=complex)
    new_magnitude = np.zeros(num_bins)   # to resolve overlaps (see below)

    for i in range(len(peaks)):
        p = peaks[i]
        shift_bins = int(round((shift_ratio - 1.0) * true_freq[p] / bin_spacing))

        # source bins [src_start, src_end) move to [src_start+shift_bins, ...),
        # clipped so we never write outside 0..num_bins-1
        dst_start = max(starts[i] + shift_bins, 0)
        dst_end = min(ends[i] + shift_bins, num_bins)
        if dst_start >= dst_end:
            continue  # the whole region was pushed past Nyquist (or below 0 Hz)
        source = np.arange(dst_start - shift_bins, dst_end - shift_bins)
        target = np.arange(dst_start, dst_end)

        moved_magnitude = magnitude[source]
        if log_envelope is not None and shift_bins != 0:
            moved_magnitude = moved_magnitude * formant_gain(log_envelope, source, target)

        moved_phase = phase[source] + rotation[p] - np.pi * shift_bins

        # When shifting DOWN, neighbouring regions can overlap by a bin; keep
        # whichever is louder there (the loud one is the real harmonic, the
        # quiet one is only a sidelobe tail). "louder" is a True/False mask
        # over this region's target bins - same as an if-statement per bin,
        # but done for the whole region at once (this line runs ~1000 times
        # per frame, so a plain Python loop here made long songs slow).
        louder = moved_magnitude > new_magnitude[target]
        new_magnitude[target[louder]] = moved_magnitude[louder]
        new_spectrum[target[louder]] = moved_magnitude[louder] * np.exp(1j * moved_phase[louder])

    return new_spectrum, region, rotation


def shift_and_resynthesize_phase(magnitudes, phases, true_freqs, shift_ratios, config,
                                 preserve_formants=True, progress_callback=None):
    """
    Runs remap_and_lock_spectrum frame by frame, carrying the peak-tracking
    state (previous regions + rotations) from each frame to the next.

    Frames with shift_ratio == 1 (unvoiced/silent frames, see
    compute_shift_ratios) are passed through UNCHANGED and reset the
    rotations to 0. Because the rotation formulation gives input == output
    at ratio 1, the next voiced frame starts from exactly where the input
    is - no jump. (Measured: resetting here halves the error around note
    starts/ends compared to carrying rotations through the gap.)

    Returns: complex spectra, shape (num_frames, num_bins)
    """
    num_frames, num_bins = phases.shape
    spectra = np.zeros((num_frames, num_bins), dtype=complex)

    prev_region = None
    prev_rotation = None

    for i in range(num_frames):
        magnitude = magnitudes[i]
        phase = phases[i]
        ratio = float(shift_ratios[i])

        if abs(ratio - 1.0) < 1e-6 or magnitude.max() <= 1e-12:
            spectra[i] = magnitude * np.exp(1j * phase)
            prev_region = None
            prev_rotation = None
        else:
            log_envelope = None
            if preserve_formants:
                log_envelope = compute_spectral_envelope(magnitude, config)
            spectra[i], prev_region, prev_rotation = remap_and_lock_spectrum(
                magnitude, phase, true_freqs[i], ratio,
                prev_region, prev_rotation, config, log_envelope)

        if progress_callback is not None and i % 100 == 0:
            progress_callback(i / num_frames)

    return spectra


def reconstruct_frame(spectrum, config):
    """
    Complex spectrum -> time-domain frame (the reverse of rfft in compute_stft).
    """
    frame = np.fft.irfft(spectrum, n=config.frame_size)
    return frame.astype(np.float32)


def phase_vocoder_shift(frames, shift_ratios, config, preserve_formants=True, progress_callback=None):
    """
    frames: shape (num_frames, frame_size) - output of frame_signal()
    shift_ratios: shape (num_frames,) - one ratio per frame
    config: AutoTuneConfig
    preserve_formants: keep the voice's timbre while the pitch moves (Piece 4)
    progress_callback: optional function(fraction 0..1), for the web UI

    Returns: shape (num_frames, frame_size) - shifted frames, ready for overlap_add()
    """
    magnitudes, phases = compute_stft(frames, config)
    true_freqs = compute_instantaneous_frequency(phases, config)
    spectra = shift_and_resynthesize_phase(magnitudes, phases, true_freqs, shift_ratios,
                                           config, preserve_formants, progress_callback)

    num_frames = frames.shape[0]
    shifted_frames = np.zeros((num_frames, config.frame_size), dtype=np.float32)
    for i in range(num_frames):
        shifted_frames[i] = reconstruct_frame(spectra[i], config)
    return shifted_frames
