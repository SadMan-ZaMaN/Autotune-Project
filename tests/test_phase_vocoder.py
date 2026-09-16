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


if __name__ == '__main__':
    unittest.main()