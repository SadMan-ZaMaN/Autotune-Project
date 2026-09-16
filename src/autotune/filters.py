import numpy as np
from scipy.signal import lfilter
from scipy.signal import freqz

import matplotlib.pyplot as plt
from scipy.signal import tf2zpk

"""
Why do we need pre-emphasis?
A speech signal generally has more energy at lower 
frequencies than at higher frequencies.So high-frequency components of speech can be relatively weak.
For pitch detection, it can be useful to make the higher frequencies stronger.

pre emphasis equation: y[n] = x[n] - ax[n-1]
x[n] = current input sample
x[n-1] = previous input sample
y[n] = output sample
a = pre-emphasis coefficient

"""
def design_preemphasis_filter(coeff = 0.95):
    """
        A(z)=1
        H(z) = (1-0.95z^-1)/1
        H(z) = 1-0,95z^-1

        this is an FIR: FIR stands for Finite Impulse Response.
        It is a type of digital filter where the output depends on the current 
        and previous input samples, but not on previous output samples.

        FIR:

        y[n]=x[n]-0.95x[n-1] 

        Only previous inputs are used.

        IIR:

        y[n]=x[n]-0.95x[n-1]+0.5y[n-1] 
    """
    b = np.array([1.0, -coeff]) #b=[1, -0.95] coeffs of x[n-1] and x[n]
    a = np.array([1.0]) #denominator coefficient later needed for filtration and Z transform
    return b,a

def apply_filter(audio, b, a):
    filtered = lfilter(b,a,audio)
    return filtered.astype(audio.dtype)

def plot_pole_zero(b, a, save_path):
    zeros, poles, _ = tf2zpk(b,a)

    fig, ax = plt.subplots(figsize=(5,5))

    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linestyle = "--", color="gray")

    ax.scatter(zeros.real, zeros.imag, marker="o", facecolors="none", edgecolors="blue", s=80, label="Zeros")
    ax.scatter(poles.real, poles.imag, marker="x", color="red", s=80, label="Poles")

    ax.axhline(0, color="black", linewidth=.5)
    ax.axvline(0, color="black", linewidth=.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Pole-Zero Plot")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_frequency_response(b, a, sample_rate, save_path):
    w, h = freqz(b,a, worN=2048, fs=sample_rate)

    magnitude_db = 20*np.log10(np.abs(h)+1e-12)
    phase_deg = np.unwrap(np.angle(h))*(180/np.pi)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True)

    ax1.plot(w, magnitude_db, color="#2980b9")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Frequency Response")
    ax1.grid(True, alpha=0.3)

    ax2.plot(w, phase_deg, color="#e67e22")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)



        



