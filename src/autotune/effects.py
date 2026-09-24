import numpy as np
from scipy.signal import lfilter, fftconvolve

"""
Studio polish: the effects every released vocal goes through
===============================================================
Pitch correction makes the notes right. What makes a recording sound like a
"singer on a record" rather than "someone into a laptop mic" is mostly this
chain, and every stage is a classic signals-and-systems building block:

    high-pass filter -> EQ -> compressor -> reverb -> loudness normalisation
    (biquad IIR)     (biquad IIR) (IIR envelope follower) (convolution with
                                                           an impulse response)

1. High-pass (80 Hz): removes rumble/handling noise below the lowest sung
   note. Voices have nothing useful down there.
2. EQ: a small cut around 300 Hz ("boxy/muddy" region of cheap mics), a
   small boost around 3 kHz ("presence" - makes words clear and forward),
   and a high-shelf boost above 10 kHz ("air"/sparkle).
3. Compressor: evens out loud and soft parts so every word is heard.
4. Reverb: puts the voice in a room/hall instead of a dry closet - the
   single biggest difference between a raw and a produced vocal.
5. Normalise: bring the peak to -1 dBFS so the result is as loud as a
   normal song without clipping.
"""


# ---------------------------------------------------------------------------
# 1 + 2: Biquad filters (2nd-order IIR), Robert Bristow-Johnson's "Audio EQ
# Cookbook" formulas. Each returns (b, a) for
#     H(z) = (b0 + b1 z^-1 + b2 z^-2) / (a0 + a1 z^-1 + a2 z^-2)
# i.e. the difference equation
#     a0 y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
# (2 poles + 2 zeros; plot_pole_zero / plot_frequency_response in
# filters.py work on these directly.)
# ---------------------------------------------------------------------------

def _biquad_common(freq_hz, sample_rate, q):
    w0 = 2 * np.pi * freq_hz / sample_rate      # centre frequency in rad/sample
    alpha = np.sin(w0) / (2 * q)                # sets the bandwidth
    return w0, alpha


def design_highpass(freq_hz, sample_rate, q=0.707):
    """
    2nd-order high-pass. q=0.707 = Butterworth (flattest passband).
    Two zeros at z=1 (DC) are what kill the low frequencies:
    at 20 Hz this one (fc = 80 Hz) attenuates by ~24 dB, at 200 Hz by ~0.
    """
    w0, alpha = _biquad_common(freq_hz, sample_rate, q)
    cos_w0 = np.cos(w0)
    b = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2])
    a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
    return b / a[0], a / a[0]


def design_peaking_eq(freq_hz, gain_db, sample_rate, q=1.0):
    """
    Boosts (gain_db > 0) or cuts (gain_db < 0) a band around freq_hz,
    leaving everything far away untouched (|H| = 1 at DC and Nyquist).
    Example: design_peaking_eq(3000, +2.5, 44100) -> |H(3 kHz)| = 10^(2.5/20) = 1.33
    """
    w0, alpha = _biquad_common(freq_hz, sample_rate, q)
    A = 10 ** (gain_db / 40.0)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cos_w0, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cos_w0, 1 - alpha / A])
    return b / a[0], a / a[0]


def design_high_shelf(freq_hz, gain_db, sample_rate):
    """
    Boosts/cuts everything ABOVE freq_hz by gain_db (like the treble knob).
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq_hz / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2 * np.sqrt(2)   # shelf slope S = 1
    sqrt_A = np.sqrt(A)
    b = np.array([A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha),
                  -2 * A * ((A - 1) + (A + 1) * cos_w0),
                  A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)])
    a = np.array([(A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha,
                  2 * ((A - 1) - (A + 1) * cos_w0),
                  (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha])
    return b / a[0], a / a[0]


def vocal_eq(audio, sample_rate):
    """High-pass + the three EQ moves described at the top of this file."""
    b, a = design_highpass(80.0, sample_rate)
    audio = lfilter(b, a, audio)
    b, a = design_peaking_eq(300.0, -2.0, sample_rate, q=1.0)
    audio = lfilter(b, a, audio)
    b, a = design_peaking_eq(3000.0, 2.5, sample_rate, q=0.9)
    audio = lfilter(b, a, audio)
    b, a = design_high_shelf(10000.0, 3.0, sample_rate)
    audio = lfilter(b, a, audio)
    return audio


# ---------------------------------------------------------------------------
# 3: Compressor
# ---------------------------------------------------------------------------

def one_pole_smoother(time_constant_ms, sample_rate):
    """
    (b, a) for y[n] = (1-c) x[n] + c y[n-1], a first-order IIR low-pass
    (single pole at z = c). It "follows" its input with a delay of about
    time_constant_ms.
        c = exp(-1 / (tau * fs))
    Example: tau = 10 ms, fs = 44100 -> c = exp(-1/441) = 0.9977
    """
    c = np.exp(-1.0 / (time_constant_ms / 1000.0 * sample_rate))
    return np.array([1.0 - c]), np.array([1.0, -c])


def compress(audio, sample_rate, threshold_db=-24.0, ratio=3.0, gate_db=-50.0, return_gain=False):
    """
    Evens out loudness. Steps:
      1. Level meter: square the signal and smooth it with a 10 ms one-pole
         filter -> short-term power (the square root of which is the RMS).
      2. Gain computer, in dB: above threshold, every extra 3 dB in only
         becomes 1 dB out (ratio 3:1).
             level -12 dB, threshold -24 dB: 12 dB over -> keep 12/3 = 4 dB
             -> output -20 dB, i.e. gain = -8 dB
         Far below the singing level (gate_db: background hiss, room noise
         between phrases) we turn the level DOWN instead (a gentle
         "expander"), so the later loudness boost doesn't raise the noise.
      3. Smooth the gain (80 ms) so it moves gently and doesn't distort.
    The input is first scaled so its loud parts sit near -18 dBFS; that
    makes the fixed threshold mean the same thing for quiet and loud
    recordings.
    return_gain=True also returns the smoothed gain curve in dB (one value
    per sample) - that's what the web UI's "Compressor" graph shows.
    """
    b, a = one_pole_smoother(10.0, sample_rate)
    power = lfilter(b, a, audio ** 2)
    level_db = 10 * np.log10(power + 1e-12)

    # put the loud (95th percentile) level at -18 dB. (Never trust a level
    # more than 20 dB under the peak as "loud" - a recording that is mostly
    # silence would otherwise get a huge boost.)
    loud_db = max(np.percentile(level_db, 95), level_db.max() - 20.0)
    audio = audio * 10 ** ((-18.0 - loud_db) / 20.0)
    level_db = level_db + (-18.0 - loud_db)

    gain_db = np.zeros_like(level_db)
    over = level_db > threshold_db
    gain_db[over] = (threshold_db + (level_db[over] - threshold_db) / ratio) - level_db[over]
    under = level_db < gate_db
    gain_db[under] = np.maximum(level_db[under] - gate_db, -20.0)  # 2:1 downward, max -20 dB

    b, a = one_pole_smoother(80.0, sample_rate)
    gain_db = lfilter(b, a, gain_db)
    out = audio * 10 ** (gain_db / 20.0)
    if return_gain:
        return out, gain_db
    return out


# ---------------------------------------------------------------------------
# 4: Reverb = convolution with an impulse response
# ---------------------------------------------------------------------------
# A room is (to a very good approximation) an LTI system, so it is fully
# described by its impulse response h[n] - what you'd record if you clapped
# once. The reverberated voice is then just y = x * h (convolution), which we
# compute with FFTs (convolution theorem: convolution in time = multiplication
# in frequency) because h is ~2 s = 88,200 samples long.
#
# We DESIGN h the classic way (Schroeder 1962 / "Freeverb"): a few feedback
# comb filters in parallel (each one = an echo that repeats every D samples
# and decays, like sound bouncing between two walls) followed by all-pass
# filters in series (smear each echo into many, without colouring the tone).
# ---------------------------------------------------------------------------

# Freeverb's tuning at 44.1 kHz - mutually "unrelated" delay lengths so the
# echoes of different combs never line up (which would sound metallic).
COMB_DELAYS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
ALLPASS_DELAYS = [556, 441, 341, 225]


def comb_filter_impulse(length, delay, feedback, damping):
    """
    Impulse response of one damped feedback comb filter, by running its
    difference equation on x = [1, 0, 0, ...]:
        y[n]  = x[n - D] + feedback * lp[n - D]
        lp[n] = (1 - damping) * y[n] + damping * lp[n - 1]   (one-pole low-pass)
    The low-pass inside the loop makes every repeat a little duller than
    the last, like real walls absorbing high frequencies more than low ones.
    """
    y = np.zeros(length)
    lp = np.zeros(length)
    lp_prev = 0.0
    for n in range(length):
        value = 0.0
        if n == delay:
            value = 1.0            # the (delayed) impulse itself
        if n >= delay:
            value += feedback * lp[n - delay]
        y[n] = value
        lp_prev = (1 - damping) * value + damping * lp_prev
        lp[n] = lp_prev
    return y


def allpass_filter(signal, delay, gain=0.5):
    """
    Schroeder all-pass: y[n] = -g x[n] + x[n-D] + g y[n-D]
    H(z) = (-g + z^-D) / (1 - g z^-D): |H| = 1 at every frequency (poles
    and zeros are mirror images), so it doesn't change the tone - it only
    scatters each echo in time, making the tail dense and smooth.
    """
    b = np.zeros(delay + 1)
    b[0] = -gain
    b[delay] = 1.0
    a = np.zeros(delay + 1)
    a[0] = 1.0
    a[delay] = -gain
    return lfilter(b, a, signal)


_ir_cache = {}


def reverb_impulse_response(sample_rate, room_size=0.84, damping=0.25, length_s=2.0, predelay_ms=20.0):
    """
    Builds (and caches) the reverb's impulse response h[n].
    room_size = comb feedback: 0.84 gives a ~1.5 s "medium hall" decay.
    predelay_ms: silence before the first echo - keeps the words clear
    (the dry voice is heard first, the room follows).
    """
    key = (sample_rate, room_size, damping, length_s, predelay_ms)
    if key in _ir_cache:
        return _ir_cache[key]

    length = int(length_s * sample_rate)
    scale = sample_rate / 44100.0   # delays above are tuned for 44.1 kHz
    h = np.zeros(length)
    for d in COMB_DELAYS:
        h += comb_filter_impulse(length, int(d * scale), room_size, damping)
    for d in ALLPASS_DELAYS:
        h = allpass_filter(h, int(d * scale))

    predelay = int(predelay_ms / 1000.0 * sample_rate)
    h = np.concatenate([np.zeros(predelay), h])[:length]
    h = h / np.sqrt(np.sum(h ** 2))   # unit energy, so "amount" means the same for any settings
    _ir_cache[key] = h
    return h


def add_reverb(audio, sample_rate, amount=0.2):
    """
    y = dry + amount * (x * h)
    amount 0.1 = subtle, 0.2 = clearly "produced", 0.35+ = big/washy.
    The output keeps the input's length; the last echoes after the end of
    the recording are cut.
    """
    if amount <= 0:
        return audio
    h = reverb_impulse_response(sample_rate)
    wet = fftconvolve(audio, h)[:len(audio)]
    return audio + amount * wet


# ---------------------------------------------------------------------------
# 5 + full chain
# ---------------------------------------------------------------------------

def normalize_peak(audio, peak_db=-1.0):
    """Scale so the loudest sample sits at peak_db dBFS (-1 dB = 0.891)."""
    peak = np.max(np.abs(audio))
    if peak <= 1e-9:
        return audio
    return audio * (10 ** (peak_db / 20.0) / peak)


def studio_polish(audio, sample_rate, reverb_amount=0.2):
    """The whole chain described at the top of this file."""
    return studio_polish_stages(audio, sample_rate, reverb_amount)["final"]


def studio_polish_stages(audio, sample_rate, reverb_amount=0.2):
    """
    The same chain as studio_polish, but keeping the signal after every
    stage (for the step-by-step graphs in the web UI). studio_polish itself
    calls this, so the graphs always show exactly what was applied.

    Returns a dict:
        "eq"          after high-pass + EQ
        "compressed"  after the compressor
        "gain_db"     the compressor's gain curve (dB, one value per sample)
        "reverb"      after the reverb
        "final"       after peak normalisation (float32) = studio_polish output
    """
    stages = {}
    audio = np.asarray(audio, dtype=np.float64)
    audio = vocal_eq(audio, sample_rate)
    stages["eq"] = audio
    audio, gain_db = compress(audio, sample_rate, return_gain=True)
    stages["compressed"] = audio
    stages["gain_db"] = gain_db
    audio = add_reverb(audio, sample_rate, reverb_amount)
    stages["reverb"] = audio
    stages["final"] = normalize_peak(audio, -1.0).astype(np.float32)
    return stages


def match_loudness(reference, audio, sample_rate):
    """
    Scales `audio` so its loud parts are as loud as `reference`'s loud parts
    (95th percentile of 50 ms RMS), limited so it never clips. Used to make
    the "original" in the before/after player as loud as the corrected
    version - otherwise people just pick the louder one as "better".
    """
    def loud_rms(x):
        win = int(0.05 * sample_rate)
        n = len(x) // win
        if n == 0:
            return np.sqrt(np.mean(x ** 2)) + 1e-12
        blocks = x[:n * win].reshape(n, win)
        return np.percentile(np.sqrt(np.mean(blocks ** 2, axis=1)), 95) + 1e-12

    gain = loud_rms(reference) / loud_rms(audio)
    peak = np.max(np.abs(audio)) + 1e-12
    gain = min(gain, 0.98 / peak)
    return (audio * gain).astype(np.float32)
