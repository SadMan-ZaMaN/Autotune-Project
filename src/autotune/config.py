import numpy as np

"""
Overall idea layout:

1. A voice signal is basically something like x[n] which might contains several
seconds of audio. We cannot treat the entire recording as one signal because the pitch
changes with time. So we divide the audio into short overlapping pieces.

Then each frame goes through:
Audio -> Framing -> Windowing -> Pitch detection -> Find nearest musical note
-> Calculate pitch correction -> Phase vocoder pitch shifting -> Overlap add -> Corrected audio


2. Basic Starting calculations:

Frame size = 2048: meaning each individual pice of audio we analyze contains 2048 samples
Sampling rate = 44100 Hz: meaning there are 44100 samples/second 
so 2048 samples correspond to: 2048/44100 = 0.0464 secs (46.4 ms)


Hop Size = 512: tells us how many samples we move forward before taking the next frame
our frame has 2048 samples but we jump only 512 samples, so the next frame starts 512 samples later.

As a result the frame "Overlap". We have N = 2048 and H = 512
amount of overlap = 2048 - 512 = 1536
% of overlap = (1536/2048)*100 = 75% or 25% hop

3. Why overlapping is important?

without overlapping:
======== | ======== | ========
frame 1     frame 2.   frame 3

When processing each frame independently the boundaries become audible as clicks, discontinuities,
amplitude fluctuations, choppy sounds.

With overlap we can smoothly combine them. That's the "Overlap-add" process


Using Nyquist THeorem we can determine the max representable frequency: fmax = fs/2 = 44100/2 = 22050 Hz
Human speech occupies much less than this so 44.1 kHz is okay

The Nyquist sampling theorem tells us how fast we need to sample a continuous signal so that we can accurately reconstruct it from its samples.

4. Windowing:
self.window = np.hanning(frame_size) creates a Hann window containing 2048 values

Why do we need a window in the first place?
If we cut a piece of continuous voice signal, at the boudaries we are suddenly cutting the waveforms. From 
FFT'a perspective, it looks like the frame abruptly starts and stops. This creates "Spectral Leakage".

To solve this problem we do xw[n] = x[n]*w[n] /w[n] is the Hann Window
as w[0] ~ 0 and w[N-1]~0 the signal smoothly approaches zero at boundaries.

What happens mathematically?
original frame: x[n]
windowed frame : xw[n] = x[n]w[n]
taking FFT:
X[k] = FFT{xw[n]} this gives a cleaner spectrum than FFT{x[n]}


"""
class AutoTuneConfig:
    """
    Central place for all the parameters that the rest of the pipeline
    depends on. Passing this object around instead of loose variables means every module
    agress on frame size, hop size, sample rate
    """

    def __init__(self, frame_size = 2048, hop_size = 512, sample_rate = 44100):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window = np.hanning(frame_size)

