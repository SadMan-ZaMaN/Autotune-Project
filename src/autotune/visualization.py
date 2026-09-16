import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_waveform_comparison(original: np.ndarray, processed: np.ndarray,
                              sample_rate: int, save_path: str) -> None:
    t_orig = np.arange(len(original)) / sample_rate
    t_proc = np.arange(len(processed)) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    ax1.plot(t_orig, original, color="#7f8c8d", linewidth=0.7)
    ax1.set_title("Original")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_proc, processed, color="#27ae60", linewidth=0.7)
    ax2.set_title("Processed")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_spectrogram_comparison(original: np.ndarray, processed: np.ndarray,
                                 sample_rate: int, save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, signal, title in [(ax1, original, "Original"), (ax2, processed, "Processed")]:
        f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=1024, noverlap=768)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        im = ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 4000)  # voice content mostly lives below ~4kHz

    ax1.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=[ax1, ax2], label="Power (dB)")
    fig.suptitle("Spectrogram Comparison")
    fig.savefig(save_path)
    plt.close(fig)

def plot_pitch_contour(detected_pitches: np.ndarray, target_pitches: np.ndarray,
                        corrected_pitches: np.ndarray, hop_size: int,
                        sample_rate: int, save_path: str) -> None:
    num_frames = len(detected_pitches)
    t = np.arange(num_frames) * hop_size / sample_rate

    def mask_unvoiced(pitches):
        return np.where(pitches > 0, pitches, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, mask_unvoiced(detected_pitches), label="Detected", 
            color="#e67e22", linewidth=1.5)
    ax.plot(t, mask_unvoiced(target_pitches), label="Target (scale)", 
            color="#2980b9", linestyle="--", linewidth=1.5)
    ax.plot(t, mask_unvoiced(corrected_pitches), label="Corrected (output)", 
            color="#27ae60", linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Pitch Correction Contour")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)