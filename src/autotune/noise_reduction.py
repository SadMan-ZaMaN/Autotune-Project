import numpy as np
from .framing import frame_signal, overlap_add

"""
Noise reduction: removing steady background noise (fans, AC, traffic hum, hiss)
==================================================================================

The idea (spectral subtraction / Wiener filtering):
A recording is  x[n] = s[n] + d[n]  - the voice s plus background noise d.
In the frequency domain (one STFT frame at a time) that is
    X[k] = S[k] + D[k]
Voice and noise are (roughly) uncorrelated, so their POWERS add:
    |X[k]|^2 ~ |S[k]|^2 + |D[k]|^2
If we know the noise power |D[k]|^2 in every frequency bin, we can turn each
bin down by how much of it is noise, and keep the ORIGINAL phase:
    Y[k] = G[k] * X[k],     0 < G[k] <= 1

Where do we learn |D[k]|^2? From the gaps between phrases - the quietest
moments of the recording are "just the room". A fan or a road sounds the
same during the gaps as under the singing (it is STATIONARY), so the gap
spectrum is a good estimate of the noise under the voice too.

The gain: the Wiener filter
    G = SNR / (1 + SNR)          (SNR = voice power / noise power in that bin)
    SNR = 100 (20 dB, a voice harmonic) -> G = 0.99  (-0.09 dB, untouched)
    SNR = 1   (0 dB, half noise)        -> G = 0.5   (-6 dB)
    SNR = 0.01 (only noise)             -> G = 0.01  -> floored (see below)
Voice harmonics stand far above the noise, so they pass almost unchanged;
bins between harmonics and in the gaps are mostly noise and get turned down.

Three things keep the VOICE undamaged (the point of doing this carefully):
1. Gain floor: no bin is ever turned down by more than max_reduction_db
   (12 dB at the default amount). Removing noise completely is what
   produces the underwater "musical noise" sound; leaving a little, quiet,
   natural-sounding noise is what every commercial denoiser does.
2. Decision-directed SNR (Ephraim & Malah 1984): the SNR of a bin is
   averaged with the previous frame's CLEANED estimate, so random noise
   peaks don't flicker on and off as little tones ("musical noise"), while
   a real voice harmonic - present frame after frame - keeps G ~ 1.
3. If the recording has no quiet part that could be "just noise" (or it is
   already clean), nothing is changed at all.

Uses the same Hann frames + weighted overlap-add as the rest of the project
(framing.py): with G = 1 everywhere the output is the input exactly.
"""

# Frames more than this far below the loudest frame are digital silence
# (e.g. zero padding) - there is no noise in them to learn from.
SILENCE_DB = -90.0


def frame_spectra(audio, config):
    """
    Hann-windowed STFT of the whole recording, one rfft per frame.
    Returns (spectra, pad_len): spectra has shape (num_frames, frame_size//2 + 1)
    (complex); pad_len is what overlap_add needs to undo frame_signal's padding.
    """
    frames, pad_len = frame_signal(audio, config)
    num_bins = config.frame_size // 2 + 1
    spectra = np.zeros((len(frames), num_bins), dtype=np.complex128)
    for i in range(len(frames)):
        spectra[i] = np.fft.rfft(frames[i])
    return spectra, pad_len


def estimate_noise_profile(spectra, config, num_samples, detected_pitches=None,
                           floor_spread_db=6.0, min_noise_seconds=0.25, min_voice_gap_db=3.0):
    """
    Estimates the noise power spectrum |D[k]|^2 from the frames that are
    "just the room" - and refuses (returns None) when it can't be sure.

    Steps:
    1. Usable frames: fully inside the recording (frame_signal pads
       frame_size zeros at both ends; a half-zero frame looks quieter than
       the real noise), not digital silence, and - when the pitch track is
       given - UNVOICED. A frame where the detector heard a sung note is
       never used as "noise", however quiet it is.
    2. The noise FLOOR: take the 10th-percentile frame power of those frames
       and keep every frame within floor_spread_db (6 dB) of it. Steady
       noise (a fan, a road, hiss) sits at an almost constant level, so its
       frames pile up in this band. The fading tail of a note does not: it
       falls through 60 dB in a few frames, so only 1-2 of its frames land
       in any 6 dB band. This is what stops soft singing being learned as
       noise (the first version of this function, using simply "the
       quietest 10% of frames", did exactly that on a clean melody).
    3. There must be at least min_noise_seconds (0.25 s = 22 frames at
       hop 512) of floor frames, and they must be clearly quieter
       (min_voice_gap_db, 3 dB) than the voiced frames. Otherwise -> None.
    4. Per frequency bin, the MEDIAN power over the floor frames, not the
       mean: a breath or click in one of them can't skew it. The power of
       a noise bin in one frame is exponentially distributed, whose median
       is ln(2) = 0.693 x its mean, so we divide by ln(2) to get the mean.

    Returns: (noise_power array of shape (num_bins,) or None, reason string)
    """
    N = config.frame_size
    H = config.hop_size
    pad_len = N

    frame_power = np.zeros(len(spectra))
    for i in range(len(spectra)):
        frame_power[i] = np.sum(np.abs(spectra[i]) ** 2)

    loudest = np.max(frame_power) if len(frame_power) else 0.0
    if loudest <= 0:
        return None, "silent recording"

    candidates = []
    voiced_power = []
    for i in range(len(spectra)):
        start = i * H
        inside = start >= pad_len and start + N <= pad_len + num_samples
        not_silence = frame_power[i] > loudest * 10 ** (SILENCE_DB / 10.0)
        if not (inside and not_silence):
            continue
        if detected_pitches is not None and i < len(detected_pitches) and detected_pitches[i] > 0:
            voiced_power.append(frame_power[i])
        else:
            candidates.append(i)

    min_frames = int(np.ceil(min_noise_seconds * config.sample_rate / H))
    if len(candidates) < min_frames:
        return None, "no quiet gaps to learn the noise from"

    # 2. the floor cluster
    candidates = np.array(candidates)
    floor_level = np.percentile(frame_power[candidates], 10)
    limit = floor_level * 10 ** (floor_spread_db / 10.0)
    floor_frames = []
    for i in candidates:
        if frame_power[i] <= limit:
            floor_frames.append(i)
    floor_frames = np.array(floor_frames)

    # 3. enough of it, and clearly below the singing
    if len(floor_frames) < min_frames:
        return None, "no steady background noise found"
    if len(voiced_power) > 0:
        reference = np.median(voiced_power)
    else:
        reference = np.median(frame_power[candidates])
    gap_db = 10 * np.log10(reference / max(np.median(frame_power[floor_frames]), 1e-30))
    if gap_db < min_voice_gap_db:
        return None, "background noise is as loud as the voice"

    # 4. median noise spectrum
    noise_power = np.zeros(spectra.shape[1])
    for k in range(spectra.shape[1]):
        noise_power[k] = np.median(np.abs(spectra[floor_frames, k]) ** 2) / np.log(2)
    return noise_power, "ok"


def compute_suppression_gains(spectra, noise_power, max_reduction_db=12.0, smoothing=0.96):
    """
    Wiener gain G[frame, bin] with the decision-directed SNR estimate.

    Per frame and bin:
        gamma = |X|^2 / noise                  "a posteriori" SNR (what we see,
                                                voice + noise over noise)
        xi = smoothing * (G_prev^2 * gamma_prev)       <- last frame's CLEAN SNR
             + (1 - smoothing) * max(gamma - 1, 0)      <- this frame's raw guess
        G  = xi / (1 + xi), then at least the floor

    Why the averaging (smoothing = 0.96): in a noise-only bin, gamma jumps
    around randomly (0.1, 3.2, 0.4 ...). Plain spectral subtraction would let
    each random peak through for one frame - a tiny 12 ms beep, heard as
    "musical noise". Averaged, xi stays small and steady in noise-only bins,
    while a voice harmonic (gamma = 100 frame after frame) builds xi up
    within 1-2 frames: frame 1 xi = 0.04*99 = 4 (G = 0.8), frame 2
    xi = 0.96*0.8^2*100 + 4 = 65 (G = 0.98).

    Floor example: max_reduction_db = 12 -> floor = 10^(-12/20) = 0.25, so a
    bin is never made more than 4x (12 dB) quieter.
    """
    floor = 10 ** (-max_reduction_db / 20.0)
    noise = np.maximum(noise_power, 1e-20)
    gains = np.ones(spectra.shape)

    prev_clean_snr = None     # G_prev^2 * gamma_prev, per bin
    for i in range(len(spectra)):
        gamma = np.abs(spectra[i]) ** 2 / noise
        instant = np.maximum(gamma - 1.0, 0.0)
        if prev_clean_snr is None:
            xi = instant
        else:
            xi = smoothing * prev_clean_snr + (1.0 - smoothing) * instant
        g = xi / (1.0 + xi)
        g = np.maximum(g, floor)
        gains[i] = g
        prev_clean_snr = g ** 2 * gamma
    return gains


def protect_harmonics(gains, detected_pitches, config, half_width_bins=1):
    """
    In frames where a note is being sung, forces the gain to 1 at the bins
    of every harmonic of the detected pitch - the voice itself is never
    turned down, only the noise BETWEEN the harmonics and in the gaps.

    Why this is safe to do: noise lying under a loud harmonic is masked by
    it (the ear can't hear a quiet sound right next to a loud one in
    frequency), so leaving it costs almost no audible noise reduction,
    while the weak upper harmonics - which the Wiener gain would otherwise
    shave off because they sit close to the noise level - keep the voice's
    brightness. Without this, voice level dropped by up to 0.6 dB at 5 dB
    SNR in our tests.

    Harmonic h of pitch f0 lands in bin  k = h * f0 * frame_size / sample_rate.
    Example: f0 = 220 Hz, frame 2048, fs 44100 -> bin spacing 21.5 Hz,
    harmonic 1 at bin 10.2, harmonic 2 at 20.4 ... we protect the nearest
    bin and half_width_bins on each side (the Hann main lobe).
    """
    protected = gains.copy()
    N = config.frame_size
    num_bins = gains.shape[1]
    for i in range(min(len(gains), len(detected_pitches))):
        f0 = detected_pitches[i]
        if f0 <= 0:
            continue
        h = 1
        while h * f0 < config.sample_rate / 2:
            centre = int(round(h * f0 * N / config.sample_rate))
            for k in range(centre - half_width_bins, centre + half_width_bins + 1):
                if 0 <= k < num_bins:
                    protected[i, k] = 1.0
            h += 1
    return protected


def reduce_noise(audio, config, amount=0.5, detected_pitches=None):
    """
    Removes steady background noise from a whole recording.

    audio: np.ndarray, mono
    config: AutoTuneConfig (frame_size, hop_size, window, sample_rate)
    detected_pitches: optional per-frame pitch track (same frames, 0 =
            unvoiced) - frames with a sung note are never learned as noise.
            The pipeline always passes it.
    amount: 0..1 -> maximum reduction = amount * 24 dB
            0.5 (default) = up to 12 dB quieter noise, natural sounding
            1.0 = up to 24 dB, strongest (more risk of a "watery" sound)

    Returns: (cleaned_audio, info) - info is a dict with
        "applied": bool, "reason": str, "noise_db": noise level in dBFS
        (None when not applied), "max_reduction_db": float
    If nothing could be learned (clean or gapless recording), the audio is
    returned unchanged, so turning this on can never make a clean take worse.
    """
    max_reduction_db = 24.0 * float(np.clip(amount, 0.0, 1.0))
    info = {"applied": False, "reason": "off", "noise_db": None,
            "max_reduction_db": max_reduction_db}
    if max_reduction_db <= 0:
        return audio.copy(), info

    spectra, pad_len = frame_spectra(audio, config)
    noise_power, reason = estimate_noise_profile(spectra, config, len(audio), detected_pitches)
    info["reason"] = reason
    if noise_power is None:
        return audio.copy(), info

    # noise level in dBFS, for the UI: Parseval for a Hann-windowed rfft,
    # mean-square = sum(|X|^2, counting both halves) / (N * sum(w^2))
    N = config.frame_size
    one_sided = noise_power.copy()
    one_sided[1:-1] *= 2
    noise_rms = np.sqrt(np.sum(one_sided) / (N * np.sum(config.window ** 2)))
    info["noise_db"] = round(float(20 * np.log10(max(noise_rms, 1e-12))), 1)

    gains = compute_suppression_gains(spectra, noise_power, max_reduction_db)
    if detected_pitches is not None:
        gains = protect_harmonics(gains, detected_pitches, config)

    frames = np.zeros((len(spectra), N), dtype=np.float32)
    for i in range(len(spectra)):
        frames[i] = np.fft.irfft(spectra[i] * gains[i], n=N)
    cleaned = overlap_add(frames, config, pad_len)[:len(audio)]
    if len(cleaned) < len(audio):
        cleaned = np.concatenate([cleaned, np.zeros(len(audio) - len(cleaned), dtype=np.float32)])

    info["applied"] = True
    return cleaned.astype(np.float32), info
