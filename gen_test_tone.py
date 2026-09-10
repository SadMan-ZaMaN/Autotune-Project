import sys
sys.path.insert(0, '.')
import numpy as np
from src.autotune.io_utils import save_audio

sr = 44100
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 220 Hz tone with slight vibrato/wobble, so pitch detection has something
# realistic to track (not perfectly flat like a pure tone)
audio = 0.4 * np.sin(2 * np.pi * 220 * t + 3 * np.sin(2 * np.pi * 5 * t))

save_audio("data/raw/test_voice.wav", audio.astype(np.float32), sr)
print("Saved data/raw/test_voice.wav")