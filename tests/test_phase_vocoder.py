import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autotune.config import AutoTuneConfig
from autotune.framing import frame_signal, overlap_add
from autotune.phase_vocoder import phase_vocoder_shift
from autotune.pitch_detection import detect_pitch_for_all_frames


def make_harmonic_signal(f0, sample_rate, duration, num_harmonics=12):
    """A synthetic vowel-like signal: fundamental + harmonics, like real
    singing (which is exactly the case that exposed the original bug -
    a single pure tone was not enough to catch it)."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = np.zeros_like(t)
    for h in range(1, num_harmonics + 1):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    signal /= np.max(np.abs(signal))
    return signal.astype(np.float32)


def shift_and_measure(audio, ratio, config):
    """Runs audio through phase_vocoder_shift at a constant ratio and
    measures the resulting pitch via the project's own pitch detector."""
    frames, pad_len = frame_signal(audio, config)
    shift_ratios = np.full(frames.shape[0], ratio, dtype=np.float32)
    shifted = phase_vocoder_shift(frames, shift_ratios, config)
    output = overlap_add(shifted, config, pad_len)

    out_frames, _ = frame_signal(output, config)
    pitches = detect_pitch_for_all_frames(out_frames, config.sample_rate)
    voiced = pitches[pitches > 0]
    if len(voiced) == 0:
        return output, 0.0
    return output, float(np.median(voiced))


class TestPhaseVocoderShift(unittest.TestCase):

    def setUp(self):
        self.config = AutoTuneConfig()
        self.sr = self.config.sample_rate

    # --- Basic sanity / contract checks -----------------------------------

    def test_output_shape_matches_input(self):
        audio = make_harmonic_signal(200.0, self.sr, 0.5)
        frames, _ = frame_signal(audio, self.config)
        ratios = np.ones(frames.shape[0], dtype=np.float32)
        shifted = phase_vocoder_shift(frames, ratios, self.config)
        self.assertEqual(shifted.shape, frames.shape)

    def test_no_nan_or_inf(self):
        audio = make_harmonic_signal(180.0, self.sr, 0.5)
        frames, _ = frame_signal(audio, self.config)
        ratios = np.full(frames.shape[0], 1.4, dtype=np.float32)
        shifted = phase_vocoder_shift(frames, ratios, self.config)
        self.assertFalse(np.isnan(shifted).any())
        self.assertFalse(np.isinf(shifted).any())

    def test_ratio_one_leaves_pitch_unchanged(self):
        audio = make_harmonic_signal(220.0, self.sr, 0.75)
        _, measured = shift_and_measure(audio, 1.0, self.config)
        self.assertAlmostEqual(measured, 220.0, delta=3.0)

    def test_silence_does_not_crash(self):
        audio = np.zeros(self.sr, dtype=np.float32)
        frames, _ = frame_signal(audio, self.config)
        ratios = np.full(frames.shape[0], 1.2, dtype=np.float32)
        shifted = phase_vocoder_shift(frames, ratios, self.config)
        self.assertFalse(np.isnan(shifted).any())

    # --- The actual bug: multi-harmonic signals at realistic ratios --------
    # A single pure tone is not enough to catch this bug (it happened to
    # work fine even in the old buggy code). These use a harmonic-rich
    # signal, like real singing, which is where it broke.

    def test_small_correction_stays_accurate(self):
        # ~1 semitone-ish correction, the realistic autotune case
        audio = make_harmonic_signal(150.0, self.sr, 1.0)
        _, measured = shift_and_measure(audio, 1.05, self.config)
        self.assertAlmostEqual(measured, 157.5, delta=3.0)

    def test_moderate_shift_up(self):
        audio = make_harmonic_signal(150.0, self.sr, 1.0)
        _, measured = shift_and_measure(audio, 1.10, self.config)
        self.assertAlmostEqual(measured, 165.0, delta=3.0)

    def test_large_shift_up_does_not_collapse(self):
        # This is the case that most dramatically failed before the fix:
        # a 2x shift ended up measuring LOWER than the original pitch.
        audio = make_harmonic_signal(150.0, self.sr, 1.0)
        _, measured = shift_and_measure(audio, 2.0, self.config)
        self.assertGreater(measured, 250.0)  # must have moved up, not collapsed
        self.assertAlmostEqual(measured, 300.0, delta=5.0)

    def test_shift_down(self):
        audio = make_harmonic_signal(220.0, self.sr, 1.0)
        _, measured = shift_and_measure(audio, 0.75, self.config)
        self.assertAlmostEqual(measured, 165.0, delta=3.0)

    def test_time_varying_ratio_is_stable(self):
        # Real autotune uses a different ratio per frame, not a constant one.
        audio = make_harmonic_signal(180.0, self.sr, 1.0)
        frames, pad_len = frame_signal(audio, self.config)
        ratios = np.linspace(1.0, 0.9, frames.shape[0]).astype(np.float32)
        shifted = phase_vocoder_shift(frames, ratios, self.config)
        output = overlap_add(shifted, self.config, pad_len)
        self.assertFalse(np.isnan(output).any())
        self.assertGreater(np.max(np.abs(output)), 0.01)  # not silent/dead


def make_vibrato_voice(midi_center, sample_rate, duration, vibrato_semitones=0.25,
                       formants=((700, 110), (1220, 120), (2600, 170))):
    """Harmonic 'sung vowel' with 5.5 Hz vibrato and fixed formant
    resonances - the case that exposed attempt 3's per-bin phase bug
    (vibrato keeps moving harmonics into neighbouring bins)."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    midi = midi_center + vibrato_semitones * np.sin(2 * np.pi * 5.5 * t)
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    phase = np.cumsum(2 * np.pi * f0 / sample_rate)
    signal = np.zeros_like(t)
    for h in range(1, 40):
        fh = h * f0
        gain = np.ones_like(fh)
        for F, B in formants:
            gain = gain * F ** 2 / np.sqrt((F ** 2 - fh ** 2) ** 2 + (B * fh) ** 2)
        signal += (fh < 6000) * gain / h * np.sin(h * phase)
    return (signal / np.max(np.abs(signal)) * 0.5).astype(np.float32)


def periodicity_db(audio, sample_rate, skip=4096):
    """Mean harmonic-to-noise ratio (dB) from the window-corrected
    autocorrelation peak. Phase glitches/'phasiness' lower it."""
    from autotune.pitch_detection import autocorrelate, window_autocorrelation
    N = 2048
    wc = window_autocorrelation(np.hanning(N))
    values = []
    for s in range(skip, len(audio) - N - skip, 1024):
        frame = audio[s:s + N] * np.hanning(N)
        r = autocorrelate(frame)
        r = r / r[0] / np.maximum(wc, 1e-3)
        peak = min(np.max(r[int(sample_rate / 900):int(sample_rate / 70)]), 0.9999)
        values.append(10 * np.log10(peak / (1 - peak)))
    return float(np.mean(values))


class TestAttempt4PeakTracking(unittest.TestCase):
    """Attempt 4: phase continued per tracked PEAK (rotation formulation),
    identity at ratio 1, formant gain inside the remap."""

    def setUp(self):
        self.config = AutoTuneConfig()
        self.sr = self.config.sample_rate

    def test_ratio_one_is_exact_identity(self):
        # frames with nothing to correct (silence, consonants, in-tune notes)
        # must come out untouched, not re-synthesized
        audio = make_vibrato_voice(57, self.sr, 1.0)
        frames, _ = frame_signal(audio, self.config)
        shifted = phase_vocoder_shift(frames, np.ones(frames.shape[0]), self.config)
        self.assertLess(np.max(np.abs(shifted - frames)), 1e-5)

    def test_vibrato_voice_keeps_clarity(self):
        # attempt 3 lost ~6 dB of periodicity on vibrato voices (phase
        # glitches whenever a harmonic changed bin); attempt 4 loses ~1 dB
        audio = make_vibrato_voice(52, self.sr, 2.0)
        frames, pad_len = frame_signal(audio, self.config)
        shifted = phase_vocoder_shift(frames, np.full(frames.shape[0], 1.04), self.config)
        output = overlap_add(shifted, self.config, pad_len)
        drop = periodicity_db(audio, self.sr) - periodicity_db(output, self.sr)
        self.assertLess(drop, 3.0)

    def test_vibrato_voice_pitch_accuracy(self):
        audio = make_vibrato_voice(60, self.sr, 2.0, vibrato_semitones=0.0)
        _, measured = shift_and_measure(audio, 1.06, self.config)
        expected = 440.0 * 2 ** ((60 - 69) / 12) * 1.06
        self.assertAlmostEqual(measured, expected, delta=expected * 0.005)  # < ~9 cents

    def test_formants_stay_put(self):
        # shift up 30%: with formant preservation the strongest envelope
        # region (around F1 = 700 Hz) must not move up by 30% too
        from autotune.phase_vocoder import compute_spectral_envelope
        audio = make_vibrato_voice(48, self.sr, 1.5, vibrato_semitones=0.0)
        frames, pad_len = frame_signal(audio, self.config)
        ratios = np.full(frames.shape[0], 1.3)

        def envelope_peak_hz(sig):
            fr, _ = frame_signal(sig, self.config)
            mag = np.abs(np.fft.rfft(fr[len(fr) // 2]))
            env = compute_spectral_envelope(mag, self.config)
            freqs = np.arange(len(env)) * self.sr / self.config.frame_size
            band = (freqs > 300) & (freqs < 1000)
            return freqs[band][np.argmax(env[band])]

        original_peak = envelope_peak_hz(audio)
        kept = overlap_add(phase_vocoder_shift(frames, ratios, self.config, preserve_formants=True), self.config, pad_len)
        moved = overlap_add(phase_vocoder_shift(frames, ratios, self.config, preserve_formants=False), self.config, pad_len)
        self.assertLess(abs(envelope_peak_hz(kept) - original_peak), abs(envelope_peak_hz(moved) - original_peak))
        self.assertLess(abs(envelope_peak_hz(kept) - original_peak), 120.0)


if __name__ == '__main__':
    unittest.main()