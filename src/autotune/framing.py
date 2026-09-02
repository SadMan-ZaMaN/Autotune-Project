import numpy as np
from .config import AutoTuneConfig

"""
1. parameters: audio is a numpy array of size= time*sampling rate, config is the AutotuneConfig object

2. padding: if original audio->[A B C D E...]
padded audio->[0 0 0 ... A B C D E ... 0 0 0]
We need to pad it for the Hann windowing. We know w[0] and w[N-1]~0
so if audio:A B C D E F
and window: 0 0.5 1.0 0.5 0.0
for x*w we get: 0 .5B C .5D 0; A and E complete disappears

3. Calculating number of frames:

suppose padded length = 10000 and N = 2048 so the first frame occupies: 0->2047
second frame occupies: 2048->4095 and so on. so the latest starting position = 10000-2048=7952

max starting position = len(padded)-N
no of hops we can make = (len(padded)-N)/H
as the first frame starts at 0 we add 1.

4. frames.shape = (frame number, N)

so          ---------N samples-------
Frame 0     x x x x ....
Frame 1     x x x x ....
Frame 2
....
Frame framenum-1

work flow: original audio -> padding -> split into overlapping frames
-> Hann window -> frames

"""
def frame_signal(audio, config: AutoTuneConfig):
    N = config.frame_size
    H = config.hop_size
    pad_len = N
    padded = np.concatenate([np.zeros(pad_len), audio, np.zeros(pad_len)])

    frame_number = ((len(padded)-N)//H)+1
    frames = np.zeros((frame_number, N), dtype=np.float32)

    for i in range(frame_number):
        start = i*H
        frames[i] = padded[start: start+N]*config.window

    return frames, pad_len

"""
this is basically the opposite/reverse stage of frame_signal
work flow: frames -> overlap+add -> normalize -> remove padding
-> reconstructed audio

suppose N=4, H=2, frame number = 4
frame 0: 0-3
frame 1: 2-5
frame 2: 4-7
frame 3:6-9

so output length = (last frame number)*H+N
=(number of frames-1)*H+N

CRUCIAL part:
frames[i] was already Hann windowed now we are doing it again during reconstruction
frames[i]*config.window is basically: x[n]*(w[n])^2

to remove the scaling factor we again divide the output by w[n]^2

"""
def overlap_add(frames, config: AutoTuneConfig, pad_len):
    N = config.frame_size
    H = config.hop_size
    frame_number = frames.shape[0]

    output_len = (frame_number - 1)*H+N
    output = np.zeros(output_len, dtype=np.float32)
    window_sum = np.zeros(output_len, dtype=np.float32)

    for i in range(frame_number):
        start = i*H
        output[start:start+N] += frames[i]*config.window
        window_sum[start: start+N] += config.window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    #we remove the padding by starting from pad_len and ending at padlen samples before the end
    return output[pad_len: -pad_len]
