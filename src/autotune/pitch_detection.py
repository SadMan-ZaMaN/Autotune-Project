import numpy as np

"""
Discrete Fourier Transform:
X[k] = ∑n=0 to N-1 x[n]e^(-j2pikn/N

Continuous Fourier Transform:
X(f) = ∫x(t)e^(-j2pift)dt

In python we find the integration using trapezoidal rule which basically converts CFT
calculation to DFT calculation. FFT(Fast Fourier Transform) is an algorithm that calculates
the DFT much faster. it takes O(NlogN) time instead O(N^2)

"""

def autocorrelate(frame):
    """
        1. Autocorrelation measure how much a signal resembles a shifted/delayed version of itself.
        For pitch detection we are basically asking if we move the audio waveform to the right then
        at what distance does it line up with itself again.

        suppose for x[n] the period ,T=5 samples. if the audio sampling rate is 1000 samples/sec:
        then f=sampling rate/time period in samples= 1000/5 = 200Hz

        x = [1 0 -1 0 1 0 -1 0] there is a pattern
        if we shift the array by 4 then the pattern lines up again perfectly. so autocorrelation tests
        the lag where we get a strong match. [lag = number of samples we shift]

        The Autocorrelation Formula:

        R(k) = ∑n=0 to N-k-1 x[n]x[n+k]
        let k = 4
        then we are comparing
        x[n]:   1 0 -1 0
        x[n+4]: 1 0 -1 0
        We multiply corresponding values: 1*1+(-1)(-1) = 2
        This is a large number beacuse the two portions are very similar
        If the lag does not match, suppose k =1
        x[n]:   1 0 -1 0 1 0 -1
        x[n+4]: 0 -1 0 1 0 -1 0 ; the product is = 0

        Suppose a sung note has T=100 samples then autocorrelation will have a string 
        peak around k =100 so kpeak~100 and f0=fs/100=44100/100 = 441Hz, we estimate the singer's fundamental pitch as approx 441 Hz

        We cannot directly use FFT because if we just say take fpeak then other frequencies will be distorted.

        2. we exclude k=0 because at that point a signal will obviously be maximally similar to itself so we search for the next meaningful peak.


        3. We can narrow down the lag range for our search:
        fs=44100 and human pitch is approx: 80 Hz<= f<= 1000Hz
        we know k = fs/f so 44<= k <= 551

        4. Normalization Problem:
        R(k) =  ∑n=0 to N-k-1 x[n]x[n+k] for k=0: there are N terms. for k=100 there are N-100 terms
        so as k gets larger we are comparing fewer terms.
        Now suppose two lags produce exactly the same average similarity
        Suppose avg contributio is 5
        At lag 10, we compare 990 samples, R(10) = 990.5 = 4950
        at lag 500, we compare 500 samples R(500) = 500*5 = 2500
        so raw autocorrelation says R(10)>R(500) even tho the average similarity was identical. This is the
        Shrinking window problem. To fix this we divide by the no of overlapping samples:N-k

        Rnormalized(k) = R(k)/(N-k)
    """

    n = len(frame)
    #Zero pad to avoid circular correlation wraparound artifacts
    #For an N size sample we can shift from -(N-1) to N-1 so there are 2N-1 shifts possible
    #so we pad to the size of 2N
    padded_len = 2*n
    #rfft = real input fast fourier transform
    fft_frame = np.fft.rfft(frame, n = padded_len)
    power_spectrum = fft_frame*np.conj(fft_frame)
    result = np.fft.irfft(power_spectrum)

    #R[k] = IFFT(FFT(x).FFT(x)*) : Wiener–Khinchin theorem.

    return result[:n]

def detect_pitch_autocorrelation(frame, sample_rate, fmin=80, fmax = 1000):

    #T = fs/f example: fs = 44100 and fmax=1000 then Tmin = 44.1
    min_lag = int(sample_rate/fmax)
    max_lag = int(sample_rate/fmin)
    max_lag = min(max_lag, len(frame)-1)

    #we dont want to seach at lag 0 as R(0)=n∑​x[n]^2 we are comparing x[n] with itself
    if min_lag<1:
        min_lag = 1

    #corr[k]=R[k] where k=lag
    corr = autocorrelate(frame)

    #R(0) = ∑|x[n]|^2 s essentially the signal's energy, if the frame is [0, 0, 0, ..]
    #then there's isnt enough signal to determine a pitch
    if corr[0] <= 1e-8:
        return 0.0
    corr_normalized = corr/corr[0]

    search_region = corr_normalized[min_lag: max_lag]
    if len(search_region) == 0:
        return 0.0

    """
    search_region:

    position       value

    0              0.10
    1              0.20
    2              0.30
    3              0.80  ← largest
    4              0.40
    argmax return 3. 

    search_region[0] = corr_normalized[44]

    search_region[1] = corr_normalized[45]

    search_region[2] = corr_normalized[46]

    search_region[3] = corr_normalized[47]  

    so the actual peak lag is = 44+3 = 47=min lag+argmax val
    """
    peak_lag = np.argmax(search_region)+min_lag
    peak_value = corr_normalized[peak_lag]

    confidence_threshold = 0.3
    if peak_value < confidence_threshold:
        return 0.0
    frequency = sample_rate/peak_lag
    return frequency

def detect_pitch_for_all_frames(frames, sample_rate, fmin=80, fmax=1000):
    pitches = np.zeros(frames.shape[0])

    for i, frame in enumerate(frames):
        pitches[i] = detect_pitch_autocorrelation(frame, sample_rate, fmin, fmax)
    return pitches
