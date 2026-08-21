"""
Week 1 — Signal Basics for the Autotune Project
=================================================
Goal this week: get comfortable with the raw building blocks before touching
pitch detection or the phase vocoder.

Concepts covered (and how they map to your course):
  1. Discrete-time signals        -> the audio array itself
  2. Basic operations on signals  -> loading, indexing, scaling
  3. Framing (windowing in time)  -> chopping a long signal into short chunks
  4. Window functions (Hann)      -> tapering each frame to avoid spectral leakage

Run this on YOUR OWN MACHINE (not here) so you can use your microphone.
Install requirements first:
    pip install numpy scipy librosa soundfile sounddevice matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# STEP 1: Get a signal — either record from mic or load a WAV file
# ----------------------------------------------------------------------

def record_from_mic(duration_sec=3, sample_rate=44100):
    """Records audio from your microphone. Run this on your own laptop."""
    import sounddevice as sd
    print(f"Recording for {duration_sec} seconds... sing/talk now!")
    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate,
                    channels=1, dtype='float64')
    sd.wait()
    print("Done recording.")
    return audio.flatten(), sample_rate


def load_wav(path, sample_rate=None):
    """Loads an existing WAV file. librosa auto-converts to mono + resamples."""
    import librosa
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio, sr


def generate_test_tone(freq=220.0, duration_sec=2.0, sample_rate=44100):
    """
    No mic/file? Use this to test your pipeline on a KNOWN signal.
    A pure sine wave at 220 Hz (the note A3) — useful because you know
    exactly what the "correct" pitch detection answer should be.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio, sample_rate


# ----------------------------------------------------------------------
# STEP 2: Basic signal operations (course concept: signal operations)
# ----------------------------------------------------------------------

def normalize(x):
    """Scale signal so max absolute amplitude is 1. Prevents clipping later."""
    return x / (np.max(np.abs(x)) + 1e-12)


def plot_waveform(x, sr, title="Waveform"):
    t = np.arange(len(x)) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(t, x, linewidth=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("waveform.png", dpi=120)
    plt.close()
    print("Saved waveform.png")


# ----------------------------------------------------------------------
# STEP 3: Framing — split the signal into short overlapping chunks
# ----------------------------------------------------------------------

def frame_signal(x, frame_size=2048, hop_size=512):
    """
    Splits x into overlapping frames.
    frame_size: samples per frame (~46ms at 44.1kHz) — short enough that
                pitch is roughly constant within one frame.
    hop_size:   how far we slide forward each time (frame_size/4 is typical,
                giving 75% overlap — needed later for smooth phase vocoder output).

    Returns a 2D array: shape (num_frames, frame_size)
    """
    num_frames = 1 + (len(x) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = x[start:start + frame_size]
    return frames


# ----------------------------------------------------------------------
# STEP 4: Windowing — taper each frame with a Hann window
# ----------------------------------------------------------------------

def apply_window(frames):
    """
    Multiplies each frame by a Hann window (bell-shaped curve, zero at edges).

    WHY this matters (this is real course material): chopping a signal into
    a frame is like multiplying by a rectangular window in the time domain.
    By the convolution theorem, multiplication in time = convolution in
    frequency. A rectangular window has a spectrum with big side-lobes
    (it "smears" energy into wrong frequencies — this is called spectral
    leakage). A Hann window has much smaller side-lobes, giving a cleaner,
    more accurate frequency-domain picture — which pitch detection depends on.
    """
    window = np.hanning(frames.shape[1])
    return frames * window  # broadcasts across all frames


def plot_window_effect(frame_size=2048):
    """Visualize a single frame before/after windowing, and the window itself."""
    window = np.hanning(frame_size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(window)
    axes[0].set_title("Hann window shape")
    axes[1].plot(np.ones(frame_size), label="rectangular (no window)", alpha=0.6)
    axes[1].plot(window, label="Hann window")
    axes[1].legend()
    axes[1].set_title("Rectangular vs Hann")
    plt.tight_layout()
    plt.savefig("window_shapes.png", dpi=120)
    plt.close()
    print("Saved window_shapes.png")


# ----------------------------------------------------------------------
# DEMO — run this file directly to see it in action on a test tone
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Using a generated tone here since this environment has no mic.
    # On your own machine, swap this for: record_from_mic() or load_wav("your_file.wav")
    audio, sr = generate_test_tone(freq=220.0, duration_sec=1.0)
    #audio, sr = record_from_mic(duration_sec=3)
    audio = normalize(audio)

    print(f"Signal length: {len(audio)} samples, sample rate: {sr} Hz")
    plot_waveform(audio[:2000], sr, title="First ~45ms of 220Hz test tone")

    frames = frame_signal(audio, frame_size=2048, hop_size=512)
    print(f"Number of frames: {frames.shape[0]}, samples per frame: {frames.shape[1]}")

    windowed_frames = apply_window(frames)
    plot_window_effect(frame_size=2048)

    print("\nWeek 1 pipeline check: OK")
    print("Next: implement autocorrelation-based pitch detection on each frame.")
