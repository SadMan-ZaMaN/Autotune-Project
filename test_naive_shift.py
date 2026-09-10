import sys
sys.path.insert(0, '.')
import numpy as np
from src.autotune.config import AutoTuneConfig
from src.autotune.io_utils import load_audio, save_audio
from src.autotune.framing import frame_signal, overlap_add
from src.autotune.pitch_detection import detect_pitch_for_all_frames
from src.autotune.scales import build_scale_midi_set, nearest_scale_note
from src.autotune.pitch_shift import naive_pitch_shift, compute_shift_ratios

config = AutoTuneConfig()
audio, sr = load_audio("data/raw/test_voice.wav", target_sr=config.sample_rate)
frames, pad_len = frame_signal(audio, config)

detected = detect_pitch_for_all_frames(frames, sr)

scale = build_scale_midi_set("C", "major")
target = np.array([nearest_scale_note(f, scale) if f > 0 else 0.0 for f in detected])

ratios = compute_shift_ratios(detected, target, strength=1.0)

shifted_frames = np.array([naive_pitch_shift(frames[i], ratios[i]) for i in range(len(frames))])

output = overlap_add(shifted_frames, config, pad_len)
save_audio("data/processed/naive_shifted.wav", output, sr)
print("Saved data/processed/naive_shifted.wav - listen and compare to original")

# python test_naive_shift.py