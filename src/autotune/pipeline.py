# this was created  Week 5-6: full pipeline + customization features (scale selector, correction strength)


import numpy as np
from .io_utils import load_audio, save_audio
from .framing import frame_signal, overlap_add
from .pitch_detection import detect_pitch_for_all_frames
from .scales import build_scale_midi_set, choose_target_notes
from .key_detection import estimate_tuning_offset, detect_key
from .pitch_shift import compute_shift_ratios, smooth_shift_ratios, naive_pitch_shift
from .phase_vocoder import phase_vocoder_shift
from .filters import design_preemphasis_filter, apply_filter
from .effects import studio_polish

"""
run_pipeline: Full Autotune Chain, Start to Finish
=====================================================

This function is the "spine" that connects every module, in order:

    load audio -> frame it -> detect pitch per frame
    -> (auto) find the singer's tuning offset and key
    -> choose a target note per frame (with hysteresis)
    -> compute + smooth how much to shift each frame
    -> shift (phase vocoder or naive) -> stitch back together (overlap-add)
    -> (optional) studio polish: EQ, compressor, reverb, loudness

Nothing new is invented here - this function just calls, in order, the
functions built and verified separately in the other modules.
"""

def run_pipeline(input_path, config, use_phase_vocoder=True, use_preemphasis=True,
                 use_formant_preservation=True, progress_callback=None):
    """
    input_path: str - path to an audio file (WAV/FLAC/OGG/MP3)
    config: AutoTuneConfig - frame/hop sizes, key settings, correction
            strength, retune speed, polish settings
    use_phase_vocoder: True = phase_vocoder_shift, False = naive_pitch_shift
                       (useful for comparison clips for the report)
    use_preemphasis: run the pitch DETECTOR on a pre-emphasized copy
    use_formant_preservation: keep the voice's timbre while shifting
    progress_callback: optional function(stage_name, fraction 0..1) - the
                       web UI uses it for its progress bar

    Returns a dict with everything useful for saving audio or making plots:
        raw_audio / original_audio, corrected_audio, sample_rate,
        detected_pitches, target_pitches, shift_ratios, corrected_pitches,
        key_root, key_type, key_confidence, tuning_offset_cents
    """
    def report(stage, fraction):
        if progress_callback is not None:
            progress_callback(stage, fraction)

    report("Reading audio", 0.0)
    raw_audio, sr = load_audio(input_path, target_sr=config.sample_rate)
    audio = raw_audio

        # Pre-emphasis filter: boosts high frequencies before pitch detection,
        # since voice naturally has weaker high-frequency energy (see filters.py
        # docstring). This is Rajin's Z-transform-designed filter (see
        # results/plots for pole-zero and frequency response analysis).
        #
        # BUG FIX: the filtered signal is used ONLY for pitch detection. It
        # used to also be the signal that got pitch-shifted and saved, and
        # nothing undid the filter afterwards. H(z) = 1 - 0.95z^-1 has
        # |H| ~ 0.05 near 150 Hz (a -25 dB cut right where a voice's
        # fundamental lives), so the output came out thin, tinny and ~7x
        # quieter than the input. Now the audio we SHIFT is the untouched
        # original, and the filter only helps the detector.

    detection_audio = audio
    if use_preemphasis:
        b, a = design_preemphasis_filter(coeff=0.95)
        detection_audio = apply_filter(audio, b, a)

    frames, pad_len = frame_signal(audio, config)
    detection_frames, _ = frame_signal(detection_audio, config)

    report("Detecting pitch", 0.05)
    detected_pitches = detect_pitch_for_all_frames(detection_frames, sr, window=config.window)

    # Key + tuning: either what the user chose, or worked out from the
    # recording itself (see key_detection.py)
    report("Finding the key", 0.2)
    tuning_offset = 0.0
    if config.follow_singer_tuning:
        tuning_offset = estimate_tuning_offset(detected_pitches)

    if config.auto_key:
        key_root, key_type, key_confidence = detect_key(detected_pitches, tuning_offset)
    else:
        key_root, key_type, key_confidence = config.scale_root, config.scale_type, None

    scale = build_scale_midi_set(key_root, key_type)

    # One target note per frame. Unvoiced frames get 0 (compute_shift_ratios
    # leaves them un-shifted). Hysteresis stops the target flickering
    # between two notes when the voice sits in between them.
    target_pitches = choose_target_notes(detected_pitches, scale, tuning_offset,
                                         config.note_hysteresis)

    shift_ratios = compute_shift_ratios(detected_pitches, target_pitches,
                                         config.correction_strength)

    shift_ratios = smooth_shift_ratios(shift_ratios, detected_pitches, config.hop_size, config.sample_rate, config.retune_ms)

    report("Shifting pitch", 0.25)
    if use_phase_vocoder:
        # formant preservation happens inside the phase vocoder (each moved
        # harmonic follows the original envelope) - see phase_vocoder.py
        shifted_frames = phase_vocoder_shift(
            frames, shift_ratios, config,
            preserve_formants=use_formant_preservation,
            progress_callback=lambda f: report("Shifting pitch", 0.25 + 0.6 * f))
    else:
        shifted_frames = np.zeros_like(frames)
        for i in range(len(frames)):
            shifted_frames[i] = naive_pitch_shift(frames[i], shift_ratios[i])

    corrected_audio = overlap_add(shifted_frames, config, pad_len)

    # overlap_add can come back up to hop_size-1 samples short (the last
    # partial hop); pad so input and output line up exactly for playback
    if len(corrected_audio) < len(raw_audio):
        corrected_audio = np.concatenate(
            [corrected_audio, np.zeros(len(raw_audio) - len(corrected_audio), dtype=np.float32)])

    # Measure the pitch of the result (before reverb, which would smear it)
    # so the report/UI can show detected vs. target vs. what we actually got
    report("Measuring the result", 0.87)
    out_frames, _ = frame_signal(corrected_audio, config)
    corrected_pitches = detect_pitch_for_all_frames(out_frames, sr, window=config.window)

    if config.studio_polish:
        report("Studio polish", 0.94)
        corrected_audio = studio_polish(corrected_audio, sr, config.reverb_amount)

    report("Done", 1.0)
    return {
        "raw_audio": raw_audio,
        "original_audio": raw_audio,
        "corrected_audio": corrected_audio,
        "sample_rate": sr,
        "detected_pitches": detected_pitches,
        "target_pitches": target_pitches,
        "shift_ratios": shift_ratios,
        "corrected_pitches": corrected_pitches,
        "key_root": key_root,
        "key_type": key_type,
        "key_confidence": key_confidence,
        "tuning_offset_cents": tuning_offset,
    }
