# this was created Week 5-6: full pipeline + customization features (scale selector, correction strength)
# to run the full autotune pipeline from start to finish, using the modules i built in previous weeks.

import sys
sys.path.insert(0, '.')
from src.autotune.config import AutoTuneConfig
from src.autotune.pipeline import run_pipeline
from src.autotune.io_utils import save_audio
from src.autotune.presets import PRESETS

"""
run_demo.py: Simple Entry Point (edit variables below, then run)
====================================================================

No command-line flags - just change these variables directly each time
you want to try a different setting, then run:
    python run_demo.py

The web UI (python app.py) does the same thing with buttons.
"""

# ---- EDIT THESE SETTINGS ----
INPUT_FILE = "data/raw/test_voice.wav"     # WAV / FLAC / OGG / MP3
OUTPUT_FILE = "data/processed/output.wav"
STYLE = "studio"              # "natural", "studio", "hard" (robotic T-Pain) or "raw" (no effects)
SCALE_ROOT = "auto"           # "auto" = detect the key from the recording, or "C", "A", "G", ...
SCALE_TYPE = "major"          # only used when SCALE_ROOT isn't "auto": "major", "natural_minor", "chromatic"
USE_PHASE_VOCODER = True      # False = use naive resampling shift instead (for comparison)
# ------------------------------

preset = PRESETS[STYLE]
config = AutoTuneConfig(scale_root=SCALE_ROOT if SCALE_ROOT != "auto" else "C",
                        scale_type=SCALE_TYPE,
                        auto_key=(SCALE_ROOT == "auto"),
                        follow_singer_tuning=True,
                        correction_strength=preset["correction_strength"],
                        retune_ms=preset["retune_ms"],
                        studio_polish=preset["reverb"] is not None,
                        reverb_amount=preset["reverb"] or 0.0)

result = run_pipeline(INPUT_FILE, config, use_phase_vocoder=USE_PHASE_VOCODER)
save_audio(OUTPUT_FILE, result["corrected_audio"], result["sample_rate"])

print(f"Saved corrected audio to {OUTPUT_FILE}")
print(f"Style: {preset['label']} (strength={preset['correction_strength']}, retune={preset['retune_ms']} ms)")
print(f"Key: {result['key_root']} {result['key_type']}"
      f"{' (auto-detected)' if config.auto_key else ''}, "
      f"singer's tuning offset: {result['tuning_offset_cents']:+.0f} cents")
print(f"Method: {'phase vocoder' if USE_PHASE_VOCODER else 'naive'}")




'''
Koyekta Custom Test             python run_demo.py

   OUTPUT_FILE = "data/processed/demo_hard.wav"
   STYLE = "hard"



    OUTPUT_FILE = "data/processed/demo_natural.wav"
    STYLE = "natural"



    OUTPUT_FILE = "data/processed/demo_naive.wav"
    STYLE = "raw"
    USE_PHASE_VOCODER = False



'''
