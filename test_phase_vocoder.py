import sys
sys.path.insert(0, '.')
import numpy as np

from src.autotune.config import AutoTuneConfig
from src.autotune.framing import frame_signal, overlap_add
from src.autotune.phase_vocoder import phase_vocoder_shift
from src.autotune.pitch_detection import autocorrelate

def generate_vowelsynth_signal(f0=150.0, duration=1.0, sample_rate=44100, num_harmonics=12):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, num_harmonics + 1):
        amplitude = 1.0 / h
        signal += amplitude * np.sin(2 * np.pi * (f0 * h) * t)
    return (signal / np.max(np.abs(signal))).astype(np.float32)

def detect_pitch_precise(frame, sample_rate, fmin=80, fmax=1000):
    """Autocorrelation with parabolic sub-sample peak refinement."""
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(int(sample_rate / fmin), len(frame) - 1)
    
    corr = autocorrelate(frame)
    if corr[0] <= 1e-8:
        return 0.0
    corr_norm = corr / corr[0]
    
    search_region = corr_norm[min_lag:max_lag]
    if len(search_region) == 0:
        return 0.0
    
    peak_idx = np.argmax(search_region) + min_lag
    if corr_norm[peak_idx] < 0.3:
        return 0.0
        
    # Parabolic sub-sample peak refinement
    if 0 < peak_idx < len(corr_norm) - 1:
        alpha = corr_norm[peak_idx - 1]
        beta = corr_norm[peak_idx]
        gamma = corr_norm[peak_idx + 1]
        delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-12)
        true_lag = peak_idx + delta
    else:
        true_lag = float(peak_idx)
        
    return sample_rate / true_lag


def main():
    config = AutoTuneConfig()
    sr = config.sample_rate
    
    print("Generating 150 Hz multi-harmonic signal...")
    f0_target_input = 150.0
    shift_ratio = 1.05  # 1 semitone correction
    expected_shifted_f0 = f0_target_input * shift_ratio  # 157.50 Hz
    
    raw_signal = generate_vowelsynth_signal(f0=f0_target_input, duration=1.0, sample_rate=sr)
    frames, pad_len = frame_signal(raw_signal, config)
    num_frames = frames.shape[0]
    
    # Measure original signal directly in time domain
    avg_orig_pitch = detect_pitch_precise(raw_signal[sr//4 : 3*sr//4], sr)
    print(f"Original Detected Pitch : {avg_orig_pitch:.2f} Hz (Expected: {f0_target_input} Hz)")
    
    # Apply phase vocoder shift
    ratios = np.full(num_frames, shift_ratio, dtype=np.float32)
    shifted_frames = phase_vocoder_shift(frames, ratios, config)
    reconstructed = overlap_add(shifted_frames, config, pad_len)
    
    # Measure shifted signal directly in time domain (avoids double-windowing distortion)
    avg_shifted_pitch = detect_pitch_precise(reconstructed[sr//4 : 3*sr//4], sr)
    print(f"Shifted Detected Pitch  : {avg_shifted_pitch:.2f} Hz (Expected: {expected_shifted_f0:.2f} Hz)")
    
    error = abs(avg_shifted_pitch - expected_shifted_f0)
    if error < 3.0:
        print("\n[SUCCESS] Phase Vocoder successfully shifted the multi-harmonic pitch!")
    else:
        print(f"\n[FAILURE] Shift failed. Error: {error:.2f} Hz")

if __name__ == "__main__":
    main()