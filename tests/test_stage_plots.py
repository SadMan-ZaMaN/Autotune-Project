import sys
import os
import json
import base64
import tempfile
import unittest
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))   # for the shared test signal helpers

from autotune.config import AutoTuneConfig
from autotune.pipeline import run_pipeline
from autotune.presets import PRESETS

from test_noise_reduction import melody, fan_noise, add_noise, SR

"""
Tests for the step-by-step graphs (stage_plots.py + run_pipeline's
collect_stages). The graphs must show what REALLY happened: collecting them
may not change the audio, each style must list exactly the steps it runs,
and the numbers in the charts must follow the style's settings.
Run with:  python -m unittest discover tests -v
"""

CHART_TYPES = {"wave", "lines", "bars", "spectrogram"}


def preset_config(key, **extra):
    p = PRESETS[key]
    settings = dict(scale_root="C", scale_type="major",
                    correction_strength=p["correction_strength"], retune_ms=p["retune_ms"],
                    studio_polish=p["reverb"] is not None, reverb_amount=p["reverb"] or 0.0,
                    noise_reduction=0.5)
    settings.update(extra)
    return AutoTuneConfig(**settings)


class TestStagePlots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        clean = melody()
        noisy, _ = add_noise(clean, fan_noise(len(clean)), snr_db=15)   # so noise reduction runs
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "in.wav")
        sf.write(cls.path, noisy, SR)
        cls.results = {}
        for key in PRESETS:
            cls.results[key] = run_pipeline(cls.path, preset_config(key), collect_stages=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stage(self, key, stage_id):
        for s in self.results[key]["stages"]:
            if s["id"] == stage_id:
                return s
        return None

    def applied_series(self, key, label):
        chart = self.stage(key, "ratios")["charts"][0]
        for s in chart["series"]:
            if s["label"] == label:
                return s["y"]
        raise KeyError(label)

    def test_collecting_stages_does_not_change_the_audio(self):
        for key in ("studio", "raw"):
            plain = run_pipeline(self.path, preset_config(key))
            np.testing.assert_array_equal(plain["corrected_audio"], self.results[key]["corrected_audio"])

    def test_each_style_lists_the_steps_it_runs(self):
        core = ["input", "denoise", "framing", "preemphasis", "pitch", "key",
                "targets", "ratios", "vocoder", "result"]
        polish = ["eq", "compressor", "reverb", "final"]
        for key in ("natural", "studio", "hard"):
            self.assertEqual([s["id"] for s in self.results[key]["stages"]], core + polish)
        # "Pitch only": no studio polish at all
        self.assertEqual([s["id"] for s in self.results["raw"]["stages"]], core)

    def test_optional_steps_disappear_when_switched_off(self):
        result = run_pipeline(self.path, preset_config("raw", noise_reduction=0.0),
                              use_preemphasis=False, collect_stages=True)
        ids = [s["id"] for s in result["stages"]]
        self.assertNotIn("denoise", ids)
        self.assertNotIn("preemphasis", ids)

    def test_hard_tune_shows_no_smoothing(self):
        # retune 0 ms: what's applied IS the (full-strength) correction
        after_strength = self.applied_series("hard", "after strength")
        applied = self.applied_series("hard", "applied (after retune)")
        for a, b in zip(after_strength, applied):
            if a is not None:
                self.assertAlmostEqual(a, b, places=1)

    def test_natural_shows_its_strength_and_smoothing(self):
        # strength 0.8: "after strength" = 0.8 x the full correction, in cents
        full = self.applied_series("natural", "full correction")
        after_strength = self.applied_series("natural", "after strength")
        applied = self.applied_series("natural", "applied (after retune)")
        pairs = [(f, s) for f, s in zip(full, after_strength) if f is not None and abs(f) > 5]
        self.assertTrue(pairs)
        for f, s in pairs:
            self.assertAlmostEqual(s, 0.8 * f, delta=0.5)
        # retune 90 ms: the applied curve differs from the unsmoothed one somewhere
        diffs = [abs(a - s) for a, s in zip(applied, after_strength) if a is not None and s is not None]
        self.assertGreater(max(diffs), 1.0)

    def test_charts_are_well_formed_json(self):
        for key, result in self.results.items():
            text = json.dumps(result["stages"])
            self.assertLess(len(text), 1_500_000)      # stays light enough to send to the page
            for stage in result["stages"]:
                for field in ("id", "title", "summary", "explain", "stats", "charts"):
                    self.assertIn(field, stage)
                self.assertTrue(stage["charts"])
                for chart in stage["charts"]:
                    self.assertIn(chart["type"], CHART_TYPES)
                    if chart["type"] == "lines":
                        for s in chart["series"]:
                            self.assertEqual(len(s["x"]), len(s["y"]))
                    if chart["type"] == "wave":
                        for s in chart["series"]:
                            self.assertEqual(len(s["min"]), len(s["max"]))
                    if chart["type"] == "spectrogram":
                        for panel in chart["panels"]:
                            data = base64.b64decode(panel["data"])
                            self.assertEqual(len(data), panel["rows"] * panel["cols"])

    def test_noise_step_reports_voice_kept(self):
        stats = dict(self.stage("studio", "denoise")["stats"])
        singing = float(stats["Singing (loud parts)"].split()[0])
        pauses = float(stats["Pauses (background)"].split()[0])
        self.assertGreater(singing, -0.5)     # the voice is kept
        self.assertLess(pauses, -6.0)         # the background is turned down


if __name__ == '__main__':
    unittest.main()
