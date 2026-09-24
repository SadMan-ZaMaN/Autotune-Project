import sys
import os
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))   # for the shared test signal helpers

from autotune.config import AutoTuneConfig
from autotune.framing import frame_signal
from autotune.pitch_detection import detect_pitch_for_all_frames
from autotune.scales import (build_scale_midi_set, choose_target_notes, midi_to_freq, freq_to_midi,
                             segment_notes, apply_note_overrides)
from autotune.key_detection import estimate_tuning_offset, detect_key
from autotune import effects

from test_phase_vocoder import make_vibrato_voice

"""
Tests for everything around the phase vocoder: pitch detection, key and
tuning detection, target-note choice, studio effects, and the full
pipeline. Run with:  python -m unittest discover tests -v
"""

SR = 44100


def cents(f_measured, f_true):
    return 1200 * np.log2(f_measured / f_true)


def midi_melody_pitches(midi_notes, frames_per_note=40, cents_offset=0.0, jitter_cents=0.0, seed=0):
    """Fake per-frame pitch track (Hz) for a melody - for testing the
    note/key logic without synthesizing audio."""
    rng = np.random.default_rng(seed)
    pitches = []
    for m in midi_notes:
        for _ in range(frames_per_note):
            dev = cents_offset + rng.normal(0, jitter_cents) if jitter_cents else cents_offset
            pitches.append(midi_to_freq(m + dev / 100.0))
    return np.array(pitches)


class TestPitchDetection(unittest.TestCase):

    def setUp(self):
        self.config = AutoTuneConfig()

    def detect_median(self, audio):
        frames, _ = frame_signal(audio, self.config)
        p = detect_pitch_for_all_frames(frames, SR)
        return float(np.median(p[p > 0])), p

    def test_low_male_voice_is_accurate(self):
        audio = make_vibrato_voice(45, SR, 1.0, vibrato_semitones=0.0)   # A2 = 110 Hz
        measured, _ = self.detect_median(audio)
        self.assertLess(abs(cents(measured, 110.0)), 3.0)

    def test_high_voice_is_accurate(self):
        # period ~50 samples: without parabolic interpolation the error
        # would be up to ~17 cents here
        audio = make_vibrato_voice(79, SR, 1.0, vibrato_semitones=0.0)   # G5 = 784 Hz
        measured, _ = self.detect_median(audio)
        self.assertLess(abs(cents(measured, midi_to_freq(79))), 3.0)

    def test_no_octave_errors(self):
        # a voice whose 2nd harmonic sits on a formant (strong even harmonics)
        # tempts autocorrelation into octave mistakes
        audio = make_vibrato_voice(62, SR, 1.5)
        _, p = self.detect_median(audio)
        voiced = p[p > 0]
        off = np.abs(cents(voiced, midi_to_freq(62)))
        self.assertLess(np.mean(off > 600), 0.02)

    def test_silence_and_noise_are_unvoiced(self):
        rng = np.random.default_rng(1)
        audio = np.concatenate([np.zeros(SR // 2),
                                0.1 * rng.standard_normal(SR // 2)]).astype(np.float32)
        frames, _ = frame_signal(audio, self.config)
        p = detect_pitch_for_all_frames(frames, SR)
        self.assertLess(np.mean(p > 0), 0.05)


class TestKeyAndTargets(unittest.TestCase):

    def test_tuning_offset_is_found(self):
        # singer consistently 40 cents sharp (some frames wrap past +50)
        p = midi_melody_pitches([55, 57, 59, 60, 62], cents_offset=40, jitter_cents=8)
        self.assertAlmostEqual(estimate_tuning_offset(p), 40.0, delta=5.0)

    def test_key_of_g_major_melody(self):
        # G A B C D E F# with lots of G/D, like a real tune in G
        notes = [55, 59, 62, 67, 62, 59, 57, 55, 60, 64, 66, 67, 62, 55]
        root, scale_type, confidence = detect_key(midi_melody_pitches(notes))
        g_major_notes = set(build_scale_midi_set("G", "major") % 12)
        detected_notes = set(build_scale_midi_set(root, scale_type) % 12)
        self.assertEqual(detected_notes, g_major_notes)   # G major or E minor: same notes
        self.assertGreater(confidence, 0.95)

    def test_chromatic_fallback_when_nothing_fits(self):
        notes = list(range(60, 72)) * 2   # all 12 notes equally
        _, scale_type, _ = detect_key(midi_melody_pitches(notes))
        self.assertEqual(scale_type, "chromatic")

    def test_hysteresis_stops_note_flipping(self):
        # hovering right between C (60) and C# (61) with small wobble
        rng = np.random.default_rng(3)
        midi = 60.5 + rng.uniform(-0.12, 0.12, 200)
        pitches = np.array([midi_to_freq(m) for m in midi])
        scale = build_scale_midi_set("C", "chromatic")
        targets = choose_target_notes(pitches, scale, hysteresis=0.3)
        switches = np.sum(np.abs(np.diff(targets)) > 1e-6)
        no_hyst = choose_target_notes(pitches, scale, hysteresis=0.0)
        self.assertEqual(switches, 0)
        self.assertGreater(np.sum(np.abs(np.diff(no_hyst)) > 1e-6), 10)

    def test_real_note_change_still_switches(self):
        pitches = midi_melody_pitches([60, 62, 64])
        targets = choose_target_notes(pitches, build_scale_midi_set("C", "major"))
        target_midi = [round(freq_to_midi(t)) for t in targets[::40]]
        self.assertEqual(target_midi, [60, 62, 64])


class TestEffects(unittest.TestCase):

    def test_filters_are_stable(self):
        designs = [effects.design_highpass(80, SR),
                   effects.design_peaking_eq(300, -2, SR),
                   effects.design_peaking_eq(3000, 2.5, SR, 0.9),
                   effects.design_high_shelf(10000, 3, SR)]
        for b, a in designs:
            self.assertTrue(np.all(np.abs(np.roots(a)) < 1.0))   # all poles inside unit circle

    def test_polish_output_is_safe(self):
        audio = make_vibrato_voice(57, SR, 2.0) * 0.1
        out = effects.studio_polish(audio, SR, 0.2)
        self.assertEqual(len(out), len(audio))
        self.assertFalse(np.isnan(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 0.9)          # -1 dBFS peak, no clipping


class TestFullPipeline(unittest.TestCase):

    def run_on(self, audio, **config_kwargs):
        import soundfile as sf
        from autotune.pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "in.wav")
            sf.write(path, audio, SR)
            return run_pipeline(path, AutoTuneConfig(**config_kwargs))

    def test_preemphasis_does_not_leak_into_output(self):
        # regression: the pre-emphasis filter used to be applied to the
        # audio that got shifted and saved (output ~7x quieter and tinny).
        # An in-tune note should come out essentially unchanged.
        audio = make_vibrato_voice(57, SR, 1.5, vibrato_semitones=0.0)   # exactly A3
        result = self.run_on(audio, scale_root="A", scale_type="major")
        out = result["corrected_audio"][:len(audio)]
        rms_ratio = np.sqrt(np.mean(out ** 2) / np.mean(audio ** 2))
        self.assertAlmostEqual(rms_ratio, 1.0, delta=0.1)

    def test_off_key_note_gets_corrected(self):
        audio = make_vibrato_voice(57.4, SR, 1.5, vibrato_semitones=0.0)   # 40 cents sharp of A3
        result = self.run_on(audio, scale_root="A", scale_type="major", retune_ms=0.0)
        out = result["corrected_pitches"]
        voiced = out[out > 0]
        self.assertLess(abs(cents(np.median(voiced), midi_to_freq(57))), 5.0)

    def test_auto_key_and_polish_run(self):
        audio = make_vibrato_voice(60, SR, 1.5)
        result = self.run_on(audio, auto_key=True, follow_singer_tuning=True, studio_polish=True)
        self.assertIn(result["key_type"], ("major", "natural_minor", "chromatic"))
        self.assertLessEqual(np.max(np.abs(result["corrected_audio"])), 0.9)


class TestNoteEditing(unittest.TestCase):
    """The web UI's draggable note bars: segment_notes + note_overrides."""

    def test_segment_notes_splits_on_gaps_and_note_changes(self):
        targets = np.array([0, 220, 220, 220, 247, 247, 0, 0, 220], dtype=float)
        self.assertEqual(segment_notes(targets),
                         [(1, 4, 220.0), (4, 6, 247.0), (8, 9, 220.0)])

    def test_override_moves_only_that_note_by_semitones(self):
        targets = np.array([0, 220, 220, 247, 247, 0], dtype=float)
        moved, mask = apply_note_overrides(targets, [{"start_frame": 1, "end_frame": 3, "semitones": 2}])
        self.assertAlmostEqual(moved[1], 220 * 2 ** (2 / 12))
        self.assertAlmostEqual(moved[2], 220 * 2 ** (2 / 12))
        self.assertEqual(list(moved[3:]), [247, 247, 0])       # other note untouched
        self.assertEqual(list(mask), [False, True, True, False, False, False])

    def test_dragged_note_lands_on_new_pitch_at_low_strength(self):
        # A3 sung in tune, dragged +2 semitones in the UI -> must come out
        # as B3 even though the global correction strength is only 50%
        # (edits are corrected at full strength), through the full
        # frame -> phase vocoder -> overlap-add path.
        from autotune.pipeline import run_pipeline
        import soundfile as sf
        audio = make_vibrato_voice(57, SR, 1.5, vibrato_semitones=0.0)
        config = AutoTuneConfig(scale_root="A", scale_type="major",
                                correction_strength=0.5, retune_ms=0.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "in.wav")
            sf.write(path, audio, SR)
            auto = run_pipeline(path, config)
            self.assertEqual(len(auto["notes"]), 1)
            start, end, _ = auto["notes"][0]
            edited = run_pipeline(path, config, note_overrides=[
                {"start_frame": start, "end_frame": end, "semitones": 2}])
        self.assertEqual(edited["notes"], auto["notes"])        # bars keep their identity
        out = edited["corrected_pitches"]
        middle = out[len(out) // 4: 3 * len(out) // 4]
        voiced = middle[middle > 0]
        self.assertLess(abs(cents(np.median(voiced), midi_to_freq(59))), 10.0)


if __name__ == '__main__':
    unittest.main()
