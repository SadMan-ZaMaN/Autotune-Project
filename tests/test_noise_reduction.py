import sys
import os
import tempfile
import unittest
import numpy as np
from scipy.signal import butter, lfilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))   # for the shared test signal helpers

from autotune.config import AutoTuneConfig
from autotune.framing import frame_signal, overlap_add
from autotune.pitch_detection import detect_pitch_for_all_frames
from autotune.scales import midi_to_freq
from autotune.noise_reduction import (frame_spectra, estimate_noise_profile,
                                      compute_suppression_gains, protect_harmonics, reduce_noise)

from test_phase_vocoder import make_vibrato_voice

"""
Tests for noise_reduction.py. The point of the module is to remove
background noise WITHOUT hurting the voice, so most tests measure the voice
directly: the exact gains computed from the noisy recording are applied to
the CLEAN voice alone, and we check how much that voice changed.

Several tests guard against versions that were tried and were wrong:
- learning the noise from "the quietest 10% of frames" learned the fading
  tails of notes as noise on a clean melody (test_clean_melody_is_untouched)
- without harmonic protection the voice lost ~0.6 dB at 5 dB SNR
  (test_voice_level_kept_in_heavy_fan_noise)
- with noise reduction off, noisy notes are tuned up to 36 cents wrong
  (test_noisy_recording_tuned_to_right_notes)
Run with:  python -m unittest discover tests -v
"""

SR = 44100
CONFIG = AutoTuneConfig()
# slightly off-pitch notes, all in C major (A3 B3 C4 D4 E4 D4 C4 A3)
NOTES = [57.3, 59.2, 60.3, 62.1, 64.2, 62.0, 60.4, 57.1]
GAP_S = 0.35
CONSONANT_S = 0.08
NOTE_S = 0.9


def consonant(seed):
    """An 's'-like sound: 4-9 kHz noise burst, 80 ms, faded in and out."""
    rng = np.random.default_rng(seed)
    b, a = butter(4, [4000 / (SR / 2), 9000 / (SR / 2)], 'band')
    burst = lfilter(b, a, rng.normal(size=int(CONSONANT_S * SR)))
    return burst / np.sqrt(np.mean(burst ** 2)) * 0.05 * np.hanning(len(burst))


def melody():
    """Silent gap (digital zero) -> consonant -> sung note, for every note.
    The consonants matter: their quiet tails are exactly what an unsafe
    noise estimate mistakes for background noise."""
    parts = []
    for i in range(len(NOTES)):
        parts.append(np.zeros(int(GAP_S * SR)))
        parts.append(consonant(i))
        parts.append(make_vibrato_voice(NOTES[i], SR, NOTE_S) * 0.3)
    parts.append(np.zeros(int(GAP_S * SR)))
    return np.concatenate(parts).astype(np.float32)


def fan_noise(n, seed=0):
    """A fan: 100 Hz hum with harmonics + broadband rush (low-passed at 2 kHz)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    hum = np.zeros(n)
    for k in range(6):
        hum += 0.5 ** k * np.sin(2 * np.pi * 100 * (k + 1) * t + rng.uniform(0, 6))
    b, a = butter(2, 2000 / (SR / 2))
    rush = lfilter(b, a, rng.normal(size=n))
    x = hum / np.std(hum) * 0.5 + rush / np.std(rush)
    return x / np.std(x)


def traffic_noise(n, seed=1):
    """Traffic: rumble below ~300 Hz whose level swells as cars pass."""
    rng = np.random.default_rng(seed)
    b, a = butter(2, 300 / (SR / 2))
    x = lfilter(b, a, rng.normal(size=n))
    t = np.arange(n) / SR
    x = x / np.std(x) * (1 + 0.6 * np.sin(2 * np.pi * 0.15 * t) ** 2)
    return x / np.std(x)


def voiced_mask(clean):
    envelope = np.convolve(clean ** 2, np.ones(1024) / 1024, 'same')
    return envelope > np.max(envelope) * 1e-3


def add_noise(clean, noise, snr_db):
    """Scales the noise so the SINGING is snr_db louder than it."""
    vm = voiced_mask(clean)
    noise = noise * np.sqrt(np.mean(clean[vm] ** 2) / np.mean(noise ** 2)) * 10 ** (-snr_db / 20)
    return (clean + noise).astype(np.float32), noise.astype(np.float32)


def plain_pitches(audio):
    frames, _ = frame_signal(audio, CONFIG)
    return detect_pitch_for_all_frames(frames, SR, window=CONFIG.window)


def apply_gains(signal, gains):
    spectra, pad_len = frame_spectra(signal, CONFIG)
    frames = np.zeros((len(spectra), CONFIG.frame_size), dtype=np.float32)
    for i in range(len(spectra)):
        frames[i] = np.fft.irfft(spectra[i] * gains[i], n=CONFIG.frame_size)
    out = overlap_add(frames, CONFIG, pad_len)[:len(signal)]
    return np.concatenate([out, np.zeros(len(signal) - len(out))])


def db(power_ratio):
    return 10 * np.log10(power_ratio)


class TestNoiseReductionSafety(unittest.TestCase):

    def test_unit_gains_reconstruct_exactly(self):
        # the STFT -> gains -> overlap-add path itself must be transparent
        clean = melody()
        spectra, _ = frame_spectra(clean, CONFIG)
        out = apply_gains(clean, np.ones(spectra.shape))
        self.assertLess(np.max(np.abs(out - clean)), 1e-5)

    def test_clean_melody_is_untouched(self):
        # regression: "quietest 10% of frames" learned the fading tails of
        # the notes as noise and changed a CLEAN voice. There is no steady
        # noise floor here, so nothing may be changed at all.
        clean = melody()
        out, info = reduce_noise(clean, CONFIG, 0.5, plain_pitches(clean))
        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(out, clean)

    def test_voice_level_kept_in_heavy_fan_noise(self):
        # regression: without harmonic protection the voice itself lost
        # ~0.6 dB at 5 dB SNR (weak upper harmonics shaved off)
        clean = melody()
        noisy, _ = add_noise(clean, fan_noise(len(clean)), snr_db=5)
        pitches = plain_pitches(noisy)
        spectra, _ = frame_spectra(noisy, CONFIG)
        noise_power, reason = estimate_noise_profile(spectra, CONFIG, len(noisy), pitches)
        self.assertEqual(reason, "ok")
        gains = protect_harmonics(compute_suppression_gains(spectra, noise_power, 12.0), pitches, CONFIG)
        voice = apply_gains(clean, gains)
        vm = voiced_mask(clean)
        level_change = db(np.mean(voice[vm] ** 2) / np.mean(clean[vm] ** 2))
        self.assertGreater(level_change, -0.25)

    def test_noise_removed_and_voice_unchanged(self):
        clean = melody()
        for noise_fn in (fan_noise, traffic_noise):
            noisy, noise = add_noise(clean, noise_fn(len(clean)), snr_db=10)
            pitches = plain_pitches(noisy)
            spectra, _ = frame_spectra(noisy, CONFIG)
            noise_power, _ = estimate_noise_profile(spectra, CONFIG, len(noisy), pitches)
            gains = protect_harmonics(compute_suppression_gains(spectra, noise_power, 12.0), pitches, CONFIG)
            vm = voiced_mask(clean)
            # the noise itself, in the gaps, gets ~12 dB quieter
            noise_after = apply_gains(noise, gains)
            self.assertLess(db(np.mean(noise_after[~vm] ** 2) / np.mean(noise[~vm] ** 2)), -9.0)
            # the voice: same level, and almost exactly the same signal
            voice = apply_gains(clean, gains)
            self.assertGreater(db(np.mean(voice[vm] ** 2) / np.mean(clean[vm] ** 2)), -0.2)
            self.assertLess(db(np.mean((voice - clean)[vm] ** 2) / np.mean(clean[vm] ** 2)), -20.0)

    def test_gain_floor_is_respected(self):
        clean = melody()
        noisy, _ = add_noise(clean, fan_noise(len(clean)), snr_db=10)
        spectra, _ = frame_spectra(noisy, CONFIG)
        noise_power, _ = estimate_noise_profile(spectra, CONFIG, len(noisy), plain_pitches(noisy))
        gains = compute_suppression_gains(spectra, noise_power, 12.0)
        self.assertGreaterEqual(np.min(gains), 10 ** (-12 / 20) - 1e-9)
        self.assertLessEqual(np.max(gains), 1.0)


class TestNoiseReductionInPipeline(unittest.TestCase):

    def run_on(self, audio, **kwargs):
        import soundfile as sf
        from autotune.pipeline import run_pipeline
        settings = dict(scale_root="C", scale_type="major", retune_ms=0.0)
        settings.update(kwargs)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "in.wav")
            sf.write(path, audio, SR)
            return run_pipeline(path, AutoTuneConfig(**settings))

    def note_errors_cents(self, result):
        """Median corrected pitch in the steady middle of each note vs the
        note it should have been tuned to (the nearest semitone)."""
        corrected = result["corrected_pitches"]
        errors = []
        position = 0.0
        for m in NOTES:
            position += GAP_S + CONSONANT_S
            lo = position + 0.25
            hi = position + NOTE_S - 0.25
            position += NOTE_S
            frame_pitches = []
            for i in range(len(corrected)):
                t = (i * CONFIG.hop_size - CONFIG.frame_size / 2) / SR
                if lo <= t <= hi and corrected[i] > 0:
                    frame_pitches.append(corrected[i])
            errors.append(abs(1200 * np.log2(np.median(frame_pitches) / midi_to_freq(round(m)))))
        return np.array(errors)

    def test_clean_input_bit_identical(self):
        clean = melody()
        off = self.run_on(clean)
        on = self.run_on(clean, noise_reduction=0.5)
        np.testing.assert_array_equal(on["corrected_audio"], off["corrected_audio"])

    def test_noisy_recording_tuned_to_right_notes(self):
        # fan noise at 5 dB SNR: without noise reduction the pitch detector
        # is thrown off and notes came out up to 36 cents off-pitch; with it,
        # every note lands within ~4 cents
        clean = melody()
        noisy, _ = add_noise(clean, fan_noise(len(clean)), snr_db=5)
        result = self.run_on(noisy, noise_reduction=0.5)
        self.assertTrue(result["noise_reduction"]["applied"])
        self.assertLess(np.max(self.note_errors_cents(result)), 10.0)


if __name__ == '__main__':
    unittest.main()
