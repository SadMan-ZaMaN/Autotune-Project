# this was created Week 5-6: full pipeline + customization features (scale selector, correction strength)
# to run the full autotune pipeline from start to finish, using the modules i built in previous weeks.

import sys
sys.path.insert(0, '.')
from src.autotune.config import AutoTuneConfig
from src.autotune.pipeline import run_pipeline
from src.autotune.io_utils import save_audio

"""
run_demo.py: Simple Entry Point (edit variables below, then run)
====================================================================

No command-line flags - just change these variables directly each time
you want to try a different setting, then run:
    python run_demo.py
"""

# ---- EDIT THESE SETTINGS ----
INPUT_FILE = "data/raw/test_voice.wav"
OUTPUT_FILE = "data/processed/output.wav"
SCALE_ROOT = "C"              # e.g. "C", "A", "G"
SCALE_TYPE = "major"          # "major", "natural_minor", or "chromatic"
CORRECTION_STRENGTH = 1.0     # 0.0 = no correction, 1.0 = full correction
USE_PHASE_VOCODER = True      # False = use naive resampling shift instead
# ------------------------------

config = AutoTuneConfig(scale_root=SCALE_ROOT,
                         scale_type=SCALE_TYPE,
                         correction_strength=CORRECTION_STRENGTH)

result = run_pipeline(INPUT_FILE, config, use_phase_vocoder=USE_PHASE_VOCODER)
save_audio(OUTPUT_FILE, result["corrected_audio"], result["sample_rate"])

print(f"Saved corrected audio to {OUTPUT_FILE}")
print(f"Settings: scale={SCALE_ROOT} {SCALE_TYPE}, strength={CORRECTION_STRENGTH}, "
      f"method={'phase vocoder' if USE_PHASE_VOCODER else 'naive'}")




'''
Koyekta Custom Test             python run_demo.py

   OUTPUT_FILE = "data/processed/demo_full_strength.wav"
   CORRECTION_STRENGTH = 1.0
   USE_PHASE_VOCODER = True



    OUTPUT_FILE = "data/processed/demo_half_strength.wav"
    CORRECTION_STRENGTH = 0.5



    OUTPUT_FILE = "data/processed/demo_naive.wav"
    USE_PHASE_VOCODER = False



'''