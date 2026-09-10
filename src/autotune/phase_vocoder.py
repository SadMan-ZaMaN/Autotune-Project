import sys
sys.path.insert(0, '.')
import numpy as np


"""
# Piece 1: Converting frames to frequency-domain (magnitude + phase)

Phase Vocoder Pitch Shifting — Overview
=========================================

Recall from pitch_detection.py: np.fft.rfft(frame) gives us a COMPLEX array,
one complex number per frequency bin. Each complex number encodes TWO things:

    magnitude = np.abs(X[k])        -> "how much" of this frequency is present
    phase     = np.angle(X[k])      -> "what point in its cycle" this frequency
                                        was at, at the start of this frame

For pitch detection, we only used magnitude (via the power spectrum). But to
properly SHIFT pitch without destroying the sound, we need both. Here's why:

If we simply move energy from bin k to bin k*ratio (to shift pitch), but
reconstruct each frame's phase independently (e.g., starting from 0 every
time), the frames won't line up smoothly when we overlap-add them back
together. Consecutive frames need their phases to evolve smoothly and
consistently, or you get a robotic/buzzy "phasiness" artifact.

So step 1 (this piece): extract magnitude AND phase from every frame,
so we have the full picture to work with before we touch anything.
"""

def compute_stft(frames, config):
    """
    frames: np.ndarray, shape (num_frames, frame_size) - already windowed,
            this is exactly the output of frame_signal() from framing.py

    Returns two arrays, both shape (num_frames, frame_size//2 + 1):
        magnitudes: how much energy is in each frequency bin, per frame
        phases:     the phase angle (in radians) of each frequency bin, per frame

    Note: frame_size//2 + 1 is the number of bins rfft gives us for a
    real-valued input of length frame_size (same reasoning you already
    saw in pitch_detection.py's use of np.fft.rfft).
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

Why we need this: bin frequencies are coarse (locked to frame_size/sample_rate
spacing, ~21.5 Hz apart at our settings). Real voice pitch can be anywhere,
not just at these exact bin centers. By comparing how much the phase actually
advanced between two consecutive frames vs. how much we'd EXPECT it to
advance if the signal were sitting exactly at the bin's center frequency,
we can figure out the TRUE frequency much more precisely.

Worked numeric example (frame_size=2048, hop_size=512, sample_rate=44100):
    bin k=20 center frequency = 20 * 44100/2048 = 430.7 Hz
    expected phase advance per hop for this bin:
        = 2*pi * k * hop_size / frame_size
        = 2*pi * 20 * 512/2048
        = 2*pi * 5.0
        = 31.416 radians

    Suppose the ACTUAL signal is exactly 440 Hz (slightly higher than the
    bin's 430.7 Hz center). Its true phase advance per hop would be:
        = 2*pi * 440 * (512/44100)
        = 2*pi * 5.1043
        = 32.076 radians

    deviation = actual - expected = 32.076 - 31.416 = 0.660 radians per hop
    extra frequency = deviation / (2*pi) * (sample_rate/hop_size)
                     = 0.660 / (2*pi) * (44100/512)
                     = 0.105 * 86.13
                     = 9.04 Hz

    true frequency estimate = bin center + extra = 430.7 + 9.04 = 439.7 Hz
    -> much closer to the real 440 Hz than the coarse 430.7 Hz bin estimate!

The phase-wrapping problem: np.angle() always returns values in [-pi, pi].
Our raw phase difference (actual - expected) might genuinely be, say, 7.5
radians, but np.angle()-based subtraction will show something like
7.5 - 2*pi = 1.2 instead. We MUST correct for this by wrapping the deviation
back into [-pi, pi] ourselves before converting it to a frequency - otherwise
our "extra frequency" calculation above would be wildly wrong.
"""

def wrap_phase(phase_diff):
    """
    Wraps a phase difference (in radians) into the range [-pi, pi].

    Example: wrap_phase(7.5) 
        7.5 is more than pi (3.14) so it's "too big" - it actually represents
        the same angle as 7.5 - 2*pi = 1.22 radians
    Example: wrap_phase(-4.0)
        -4.0 is less than -pi, so it wraps to -4.0 + 2*pi = 2.28 radians
    """
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


def compute_instantaneous_frequency(phases, config):
    """
    phases: np.ndarray, shape (num_frames, num_bins) - output of compute_stft()
    config: AutoTuneConfig - needs hop_size, frame_size, sample_rate

    Returns: np.ndarray, shape (num_frames, num_bins) - precise frequency (Hz)
    estimate for every bin, in every frame.

    Note: the very first frame has no "previous frame" to compare against,
    so we just use the bin's center frequency as a reasonable starting guess.
    """
    num_frames, num_bins = phases.shape
    freqs = np.zeros((num_frames, num_bins))

    # bin_center_freqs[k] = the "coarse" frequency that bin k represents
    bin_center_freqs = np.arange(num_bins) * config.sample_rate / config.frame_size

    # expected phase advance per hop, if a signal sat exactly at each bin's center
    expected_advance = 2 * np.pi * np.arange(num_bins) * config.hop_size / config.frame_size

    freqs[0] = bin_center_freqs  # first frame: no previous phase to compare, use coarse estimate

    for i in range(1, num_frames):
        actual_advance = phases[i] - phases[i - 1]
        deviation = wrap_phase(actual_advance - expected_advance)

        # convert the deviation (radians per hop) into an extra frequency (Hz)
        extra_freq = deviation / (2 * np.pi) * (config.sample_rate / config.hop_size)

        freqs[i] = bin_center_freqs + extra_freq

    return freqs





"""
Piece 3: Shifting Frequency and Re-Synthesizing Phase
=========================================================

Now that we know the TRUE frequency in each bin (from Piece 2), shifting
pitch is conceptually simple: multiply every bin's true frequency by our
shift_ratio. E.g., if shift_ratio=1.05, a bin whose true content was 440 Hz
now represents content at 462 Hz.

But we can't just "put" this shifted frequency into a fresh phase from
scratch each frame - remember, that causes robotic artifacts. Instead we
must ACCUMULATE phase smoothly across frames, using the shifted frequency.

Think of a phase accumulator as a runner going around a track:
    new_phase = previous_synthesized_phase + (shifted_frequency's expected
                phase advance for this hop)

This is exactly like: position = previous_position + speed * time
Except here "speed" is angular frequency and "time" is hop_size/sample_rate.

Worked example:
    true frequency = 440 Hz, shift_ratio = 1.05
    shifted frequency = 440 * 1.05 = 462 Hz

    phase advance per hop for 462 Hz:
        = 2*pi * 462 * (hop_size/sample_rate)
        = 2*pi * 462 * (512/44100)
        = 2*pi * 5.363
        = 33.70 radians

    if previous synthesized phase (for this bin) was, say, 1.0 radian:
        new synthesized phase = 1.0 + 33.70 = 34.70 radians
    (we do NOT wrap this - phase accumulates continuously across the
    whole signal, we only wrap when actually reading with np.sin/cos
    which handles huge angles fine automatically)
"""

def shift_and_resynthesize_phase(true_freqs, shift_ratios, config):
    """
    true_freqs: np.ndarray, shape (num_frames, num_bins) - output of
                compute_instantaneous_frequency()
    shift_ratios: np.ndarray, shape (num_frames,) - one ratio per frame,
                  output of compute_shift_ratios() from pitch_shift.py
    config: AutoTuneConfig - needs hop_size, sample_rate

    Returns: np.ndarray, shape (num_frames, num_bins) - new synthesized
    phase for every bin, in every frame, accumulated smoothly across time.
    """
    num_frames, num_bins = true_freqs.shape
    synthesized_phase = np.zeros((num_frames, num_bins))

    for i in range(num_frames):
        # shift this frame's true frequencies by this frame's shift ratio
        shifted_freqs = true_freqs[i] * shift_ratios[i]

        # how much phase this shifted frequency should advance in one hop
        phase_advance = 2 * np.pi * shifted_freqs * (config.hop_size / config.sample_rate)

        if i == 0:
            synthesized_phase[i] = phase_advance  # start accumulating from 0
        else:
            synthesized_phase[i] = synthesized_phase[i - 1] + phase_advance

    return synthesized_phase






"""
Piece 4: Reconstructing a Time-Domain Frame
==============================================

We now have, for each bin, in each frame:
    magnitude       - unchanged from the original (we keep the same "tone color")
    synthesized phase - our new smoothly-accumulated, shifted-pitch phase

A complex number can be built from magnitude and phase using Euler's formula:
    complex_value = magnitude * (cos(phase) + j*sin(phase))
    (equivalently: magnitude * exp(j*phase), same thing)

Once we rebuild the full complex spectrum this way, np.fft.irfft() converts
it back into a real time-domain frame - exactly the reverse of np.fft.rfft()
we used in Piece 1.
"""

def reconstruct_frame(magnitude, synthesized_phase, config):
    """
    magnitude: np.ndarray, shape (num_bins,) - one frame's magnitude spectrum
    synthesized_phase: np.ndarray, shape (num_bins,) - one frame's new phase
    config: AutoTuneConfig - needs frame_size

    Returns: np.ndarray, shape (frame_size,) - reconstructed time-domain frame
    """
    complex_spectrum = magnitude * (np.cos(synthesized_phase) + 1j * np.sin(synthesized_phase))
    frame = np.fft.irfft(complex_spectrum, n=config.frame_size)
    return frame.astype(np.float32)






"""
Piece 5: The Full Phase Vocoder Pipeline
===========================================

This ties Pieces 1-4 together into one function matching the signature
agreed in CONTRACTS.md:

    phase_vocoder_shift(frames, shift_ratios, config) -> shifted_frames

Workflow per frame:
    1. Convert all frames to magnitude + phase        (Piece 1)
    2. Get precise true frequency per bin, per frame   (Piece 2)
    3. Shift frequencies + accumulate new phase        (Piece 3)
    4. Rebuild each frame from magnitude + new phase   (Piece 4)
"""

def phase_vocoder_shift(frames, shift_ratios, config):
    """
    frames: np.ndarray, shape (num_frames, frame_size) - output of frame_signal()
    shift_ratios: np.ndarray, shape (num_frames,) - output of compute_shift_ratios()
    config: AutoTuneConfig

    Returns: np.ndarray, shape (num_frames, frame_size) - shifted frames,
    same shape as input, ready to pass straight into overlap_add()
    """
    magnitudes, phases = compute_stft(frames, config)
    true_freqs = compute_instantaneous_frequency(phases, config)
    new_phases = shift_and_resynthesize_phase(true_freqs, shift_ratios, config)

    num_frames = frames.shape[0]
    shifted_frames = np.zeros((num_frames, config.frame_size), dtype=np.float32)

    for i in range(num_frames):
        shifted_frames[i] = reconstruct_frame(magnitudes[i], new_phases[i], config)

    return shifted_frames




                                        #Template

if __name__ == "__main__":
    from src.autotune.config import AutoTuneConfig
    from src.autotune.io_utils import load_audio, save_audio
    from src.autotune.framing import frame_signal, overlap_add
    from src.autotune.pitch_detection import detect_pitch_for_all_frames
    from src.autotune.scales import build_scale_midi_set, nearest_scale_note
    from src.autotune.pitch_shift import compute_shift_ratios

    config = AutoTuneConfig()
    audio, sr = load_audio("data/raw/test_voice.wav", target_sr=config.sample_rate)
    frames, pad_len = frame_signal(audio, config)

    detected = detect_pitch_for_all_frames(frames, sr)
    scale = build_scale_midi_set("C", "major")
    target = np.array([nearest_scale_note(f, scale) if f > 0 else 0.0 for f in detected])
    ratios = compute_shift_ratios(detected, target, strength=1.0)

    shifted_frames = phase_vocoder_shift(frames, ratios, config)

    output = overlap_add(shifted_frames, config, pad_len)
    save_audio("data/processed/phase_vocoder_shifted.wav", output, sr)
    print("Saved data/processed/phase_vocoder_shifted.wav")
    print("Compare this against naive_shifted.wav - this should sound noticeably cleaner!")