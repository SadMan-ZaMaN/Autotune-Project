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

def window_autocorrelation(window):
    """
    Autocorrelation of the analysis window itself, normalized so value at
    lag 0 is 1. Used to undo the "taper bias" in detect_pitch_autocorrelation.

    Why this is needed (Boersma's method, 1993):
    Our frames are Hann-windowed before we autocorrelate them, so what we
    actually compute is the autocorrelation of x[n]*w[n], not of x[n].
    Because the window fades to 0 at both edges, shifting the frame by k
    samples lines up LESS and LESS of the window with itself - so even a
    perfectly periodic signal gets a smaller R(k) at larger lags.

    Worked example (N = 2048, Hann window):
        lag k = 100  -> window-overlap factor r_w(k) ~ 0.93
        lag k = 400  -> r_w(k) ~ 0.64
    So a perfect 110 Hz tone (period 400 samples) would only reach ~0.64,
    while a random bump at lag 100 could reach 0.9 - the detector gets
    biased toward short lags (= too-high pitches, octave errors).

    Fix: divide the frame's autocorrelation by the window's autocorrelation:
        r_x(k) ~ r_xw(k) / r_w(k)
    Now a perfectly periodic signal scores ~1.0 at its period, no matter how
    long that period is.
    """
    corr = autocorrelate(window)
    return corr / corr[0]


def parabolic_peak_offset(left, center, right):
    """
    Sub-sample refinement of a peak location.

    Lags are whole numbers of samples, so on its own the detector can only
    say "the period is 100 samples" or "101 samples" - at 44100 Hz that is
    441 Hz vs 436.6 Hz, a jump of ~17 cents. For a higher voice (period ~50
    samples) one sample is ~35 cents - bigger than the errors we are trying
    to correct!

    Fix: fit a parabola through the peak and its two neighbours and use
    the parabola's vertex as the true peak position.
        offset = 0.5 * (left - right) / (left - 2*center + right)
    Example: left=0.80, center=0.95, right=0.90
        offset = 0.5 * (0.80-0.90) / (0.80-1.90+0.90) = 0.25
    -> true peak is 0.25 samples to the right of the integer peak.
    """
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    # a real peak's vertex can't be more than half a sample away
    return float(np.clip(offset, -0.5, 0.5))


def detect_pitch_autocorrelation(frame, sample_rate, fmin=70, fmax=1000,
                                 window_corr=None, confidence_threshold=0.5,
                                 octave_tolerance=0.9):
    """
    Estimates the fundamental frequency (pitch) of one windowed frame.
    Returns 0.0 if the frame doesn't look like a clear musical note
    (silence, breath, "s"/"sh" consonants...).

    window_corr: output of window_autocorrelation() for the window that was
                 applied to this frame (optional but strongly recommended).
    confidence_threshold: how periodic the frame must be (0..1) to count
                 as a sung note. 0.5 = "at least half the energy repeats
                 exactly one period later".
    octave_tolerance: see step 4 below.
    """

    #T = fs/f example: fs = 44100 and fmax=1000 then Tmin = 44.1
    min_lag = int(sample_rate/fmax)
    max_lag = int(sample_rate/fmin)
    max_lag = min(max_lag, len(frame)-2)

    #we dont want to seach at lag 0 as R(0)=n∑​x[n]^2 we are comparing x[n] with itself
    if min_lag<2:
        min_lag = 2

    #corr[k]=R[k] where k=lag
    corr = autocorrelate(frame)

    #R(0) = ∑|x[n]|^2 s essentially the signal's energy, if the frame is [0, 0, 0, ..]
    #then there's isnt enough signal to determine a pitch
    if corr[0] <= 1e-8:
        return 0.0
    corr_normalized = corr/corr[0]

    # Step 1: undo the window's taper bias (see window_autocorrelation)
    if window_corr is not None:
        corr_normalized = corr_normalized / np.maximum(window_corr, 1e-3)

    if max_lag <= min_lag:
        return 0.0

    # Step 2: collect every LOCAL peak in the allowed lag range.
    # (A local peak = bigger than both its neighbours.)
    peak_lags = []
    for k in range(min_lag, max_lag):
        if corr_normalized[k] > corr_normalized[k-1] and corr_normalized[k] >= corr_normalized[k+1]:
            peak_lags.append(k)
    if len(peak_lags) == 0:
        return 0.0

    # Step 3: the strongest peak tells us how periodic this frame is at all
    best_value = 0.0
    for k in peak_lags:
        if corr_normalized[k] > best_value:
            best_value = corr_normalized[k]
    if best_value < confidence_threshold:
        return 0.0

    # Step 4: octave-error guard.
    # A signal with period T also repeats at 2T, 3T, ... so R(2T) is often
    # almost exactly as big as R(T). Plain argmax sometimes picks 2T, which
    # reads the note an octave too LOW. The real period is the SHORTEST lag
    # that is (nearly) as good as the best one, so we take the first peak
    # within octave_tolerance of the best.
    # Example: R(200)=0.93, R(400)=0.95 -> 0.93 >= 0.9*0.95, so pick 200.
    peak_lag = peak_lags[0]
    for k in peak_lags:
        if corr_normalized[k] >= octave_tolerance * best_value:
            peak_lag = k
            break

    # Step 5: sub-sample accuracy (see parabolic_peak_offset)
    offset = parabolic_peak_offset(corr_normalized[peak_lag-1],
                                   corr_normalized[peak_lag],
                                   corr_normalized[peak_lag+1])
    frequency = sample_rate / (peak_lag + offset)
    return frequency


def clean_pitch_track(pitches, frame_rms, silence_db=-40.0, min_run_frames=4, median_size=5):
    """
    Post-processing on the whole pitch track (one value per frame). Single
    frames are judged in isolation by detect_pitch_autocorrelation, so a few
    get it wrong - and every wrong frame becomes a wrong pitch SHIFT, which is
    heard as a chirp or a garbled consonant. This removes those mistakes using
    the fact that a real sung note lasts much longer than one frame (~11 ms).

    1. Loudness gate: frames more than |silence_db| dB quieter than the loud
       part of the recording are background noise -> unvoiced.
       Example: loud frames have RMS 0.1; -40 dB = 0.1 * 10^(-40/20) = 0.001,
       so anything below RMS 0.001 is treated as silence.
    2. Remove tiny "voiced islands": a run of voiced frames shorter than
       min_run_frames (4 frames ~ 46 ms) is almost always a breath or a
       consonant that happened to look periodic, not a sung note.
    3. Median filter (in semitones) inside each voiced run: replaces each
       frame's pitch with the middle value of its neighbourhood, which
       deletes one-frame octave jumps but keeps real note changes (a median
       ignores outliers, unlike an average which would smear them).
       Example window [150, 151, 302, 150, 152] -> median 151, the 302 Hz
       octave glitch disappears.
    """
    num_frames = len(pitches)
    cleaned = np.array(pitches, dtype=np.float64)

    # 1. loudness gate, relative to the loud (95th percentile) frames
    loud_level = np.percentile(frame_rms, 95)
    gate = loud_level * 10 ** (silence_db / 20.0)
    for i in range(num_frames):
        if frame_rms[i] < gate:
            cleaned[i] = 0.0

    # find voiced runs as (start, end) pairs, end exclusive
    runs = []
    i = 0
    while i < num_frames:
        if cleaned[i] > 0:
            start = i
            while i < num_frames and cleaned[i] > 0:
                i += 1
            runs.append((start, i))
        else:
            i += 1

    # 2. drop voiced islands that are too short to be a real note
    kept_runs = []
    for start, end in runs:
        if end - start < min_run_frames:
            cleaned[start:end] = 0.0
        else:
            kept_runs.append((start, end))

    # 3. median filter in semitones inside each run
    half = median_size // 2
    result = cleaned.copy()
    for start, end in kept_runs:
        semitones = 12 * np.log2(cleaned[start:end] / 440.0)
        for j in range(end - start):
            lo = max(0, j - half)
            hi = min(end - start, j + half + 1)
            result[start + j] = 440.0 * 2 ** (np.median(semitones[lo:hi]) / 12)
    return result


def detect_pitch_for_all_frames(frames, sample_rate, fmin=70, fmax=1000, window=None, clean=True):
    """
    Runs detect_pitch_autocorrelation on every frame, then clean_pitch_track
    on the result.

    window: the window that frame_signal applied (config.window). If None we
            assume a Hann window of the frame length, which is what
            frame_signal uses.
    clean:  set False to see the raw per-frame detections (useful for plots
            in the report, to show what the clean-up step removes).
    """
    frame_size = frames.shape[1]
    if window is None:
        window = np.hanning(frame_size)
    window_corr = window_autocorrelation(window)

    pitches = np.zeros(frames.shape[0])
    frame_rms = np.zeros(frames.shape[0])

    for i, frame in enumerate(frames):
        pitches[i] = detect_pitch_autocorrelation(frame, sample_rate, fmin, fmax, window_corr)
        frame_rms[i] = np.sqrt(np.mean(frame ** 2))

    if clean and len(pitches) > 0:
        pitches = clean_pitch_track(pitches, frame_rms)
    return pitches
