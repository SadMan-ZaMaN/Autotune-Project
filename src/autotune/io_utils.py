import numpy as np
import soundfile as sf

from scipy.signal import resample_poly
from math import gcd

"""
Some things to know before hand:
what are "mono" and "stereo" audios

A channel is one independent stream of audio samples.
for example , our audio contains: x[n] = [0.1, 0.3, 0.2, -0.1, ...]
There's only one sequence of samples so one channel.

A mono audio has only one channel. When we record outselves singing with a microphone. The 
audio file contains only one sequence.

For stereo audio there are two channels, left and right channels. The two channels contain
separate sequences of samples. For example;
Left: [0.1, 0.3, 0.5, 0.2, ... ]
Right: [0.2, 0.4, 0.45, 0.1, ...] They dont have to be identitcal
This allows us to create a sense of spatial position.

If the audio is of stereo type:  we convert it to mono using axis=1
Suppose:
             Left       Right
Sample 1     0.20       0.40
Sample 2     0.60       0.20
Sample 3    -0.40      -0.20

Sample 1 = (0.2+0.4)/2 = 0.3 and so on

"""

#two args: path or location of the audio file ex, path = "my_voice.wav"
#and target sampling rate
def load_audio(path, target_sr = 44100):
    """ we load a wav file as mono float32. Raises error if sample rate doesnot match

    soundfile reads the audio file and returns two things:
    the actual audio samples and the sampling rate
    for ex. voice.wav has sample rate = 44100Hz and duration = 3s and mono

    So 441000*3 = 132300 samples are loaded. so audio might be a NumPy array of
    length = 132300 and sr = 44100

    "always_2d = False" tells the soundfile to not force the mono audio to have an
    unnecessary second dimension.
    
    
    """

    # or we could just use librosa..librosa auto converts to mono + resamples
    
    audio, sr = sf.read(path, always_2d = False)

    #check whether the audio is stereo
    if audio.ndim > 1:
        audio = audio.mean(axis = 1)

    #if the sample rate of the input file is different than the target sr
    if sr != target_sr:
       audio = _resample(audio, sr, target_sr)
       sr = target_sr

    return audio.astype(np.float32), sr

"""
    Resampling Guideline:
    We need to match the audio file's sample rate to the target sample rate. 
    
    Suppose given sample rate = 48000, target sr = 44100
    gcd(48000, 44100) = 300

    up = 44100/300 = 147 -> upsample by 147
    down = 48000/300 = 160 -> downsample by 160

    upsampling:
    upsampling factor of 147 means introducing (147-1)=146 zero valued samples between every two
    original sample.

    downsampling:
    this reduces the number of samples. like, take only every mth sample but we cannot just throw 
    away samples. This can cause aliasing. So before downsampling we need a low pass filter.

    Aliasing:
    suppose we have fs= 48000 Hz; Nyquist frequency = 48000/2 = 24000 Hz
    our target sample rate = 44100 Hz; new Nyquist frequency = 44100/2 = 22050 Hz
    so frequencies. between 22050Hz and 24000Hz cannot safely exist.

    suppose fs=1000Hz, fN=500Hz if our original signal contains:700Hz then it aliases to |700-1000|=300Hz
    this is problematic as our digital signal now contains a freq that was not actually there in that form.

    so after up sampling resample_poly uses a "Low Pass Filter" then down samples so that only low frequencies can pass.
    Then we get : (48000*147)/160 = 44100

"""
def _resample(audio, original_sr, target_sr):

    g = gcd(original_sr, target_sr)
    up = target_sr // g
    down = original_sr // g

    return resample_poly(audio, up, down)


def save_audio(path, audio, sample_rate):
    sf.write(path, audio, sample_rate)
    