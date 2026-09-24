# this was created  Week 5-6: full pipeline + customization features (scale selector, correction strength)


import numpy as np
from .io_utils import load_audio, save_audio
from .framing import frame_signal, overlap_add
from .pitch_detection import detect_pitch_for_all_frames
from .scales import build_scale_midi_set, choose_target_notes, segment_notes, apply_note_overrides
from .key_detection import estimate_tuning_offset, detect_key
from .pitch_shift import compute_shift_ratios, smooth_shift_ratios, naive_pitch_shift
from .phase_vocoder import phase_vocoder_shift
from .filters import design_preemphasis_filter, apply_filter
from .effects import studio_polish, studio_polish_stages
from .noise_reduction import reduce_noise
from .stage_plots import build_stages, pick_example_frame

"""
run_pipeline: Full Autotune Chain, Start to Finish
=====================================================

This function is the "spine" that connects every module, in order:

    load audio -> (optional) remove background noise -> frame it -> detect pitch per frame
    -> (auto) find the singer's tuning offset and key
    -> choose a target note per frame (with hysteresis)
    -> compute + smooth how much to shift each frame
    -> shift (phase vocoder or naive) -> stitch back together (overlap-add)
    -> (optional) studio polish: EQ, compressor, reverb, loudness

Nothing new is invented here - this function just calls, in order, the
functions built and verified separately in the other modules.
"""

def run_pipeline(input_path, config, use_phase_vocoder=True, use_preemphasis=True,
                 use_formant_preservation=True, progress_callback=None, note_overrides=None,
                 collect_stages=False):
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
    note_overrides: optional list of {"start_frame", "end_frame", "semitones"}
                    - notes the user dragged to a different pitch in the web
                    UI (see scales.apply_note_overrides). Everything else
                    stays exactly as the automatic correction chose it.
    collect_stages: also return "stages" - one entry per processing step
                    with its graphs and numbers (stage_plots.py), for the
                    web UI's step-by-step view

    Returns a dict with everything useful for saving audio or making plots:
        raw_audio / original_audio, corrected_audio, sample_rate,
        detected_pitches, target_pitches, shift_ratios, corrected_pitches,
        key_root, key_type, key_confidence, tuning_offset_cents,
        notes (the AUTOMATIC notes from segment_notes, before any override),
        noise_reduction (info dict from noise_reduction.reduce_noise),
        stages (only with collect_stages=True)
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

    # Background noise removal (noise_reduction.py), BEFORE anything else:
    # the noise would otherwise be pitch-shifted along with the voice (a fan
    # hum that wobbles up and down with every correction), and it confuses
    # the pitch detector. Then the pitch we TUNE with is measured on the
    # CLEANED audio (below, the normal path, pre-emphasis included).
    # Measured on synthetic noisy singing (tests/test_noise_reduction.py):
    # with fan noise at 5 dB SNR the tuned notes were up to 36 cents off
    # without this step and within 4 cents with it; with hiss at 20 dB SNR
    # 23 -> 1.4 cents. Clean recordings come out bit-identical.
    #
    # The ROUGH pitch track given to reduce_noise (which frames are singing,
    # so they are never learned as noise and their harmonics are never
    # turned down) skips pre-emphasis: pre-emphasis boosts hiss, and on a
    # hissy recording the detector then found only ~35% of the sung frames.
    # For the final detection pre-emphasis stays as before - without it the
    # detector made ~7x more octave-type errors (> 600 Hz spikes) on the
    # real recordings in data/raw.
    noise_info = {"applied": False, "reason": "off", "noise_db": None, "max_reduction_db": 0.0}
    if config.noise_reduction > 0:
        report("Removing background noise", 0.03)
        plain_frames, _ = frame_signal(audio, config)
        rough_pitches = detect_pitch_for_all_frames(plain_frames, sr, window=config.window)
        audio, noise_info = reduce_noise(audio, config, config.noise_reduction, rough_pitches)

    frames, pad_len = frame_signal(audio, config)

    detection_audio = audio
    if use_preemphasis:
        b, a = design_preemphasis_filter(coeff=0.95)
        detection_audio = apply_filter(audio, b, a)
    detection_frames, _ = frame_signal(detection_audio, config)

    report("Detecting pitch", 0.1)
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

    # The note bars in the web UI are these automatic notes. Segment BEFORE
    # applying overrides so a note keeps the same frames (= the same bar)
    # however far it has been dragged.
    notes = segment_notes(target_pitches)

    overridden = np.zeros(len(target_pitches), dtype=bool)
    if note_overrides:
        target_pitches, overridden = apply_note_overrides(target_pitches, note_overrides)

    shift_ratios = compute_shift_ratios(detected_pitches, target_pitches,
                                         config.correction_strength)

    # Hand-edited notes are corrected at FULL strength: if the user dragged a
    # note to D, it should land on D even with a gentle 60% global setting.
    if overridden.any():
        hard_ratios = compute_shift_ratios(detected_pitches, target_pitches, 1.0)
        shift_ratios[overridden] = hard_ratios[overridden]

    ratios_before_smoothing = shift_ratios.copy()
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

    # one frame kept before/after shifting, for the step-by-step graphs
    example_index = pick_example_frame(detected_pitches, shift_ratios)
    example_in_frame = frames[example_index].copy()
    example_out_frame = shifted_frames[example_index].copy()

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

    corrected_pre_polish = corrected_audio
    polish = None
    if config.studio_polish:
        report("Studio polish", 0.94)
        if collect_stages:
            polish = studio_polish_stages(corrected_audio, sr, config.reverb_amount)
            corrected_audio = polish["final"]
        else:
            corrected_audio = studio_polish(corrected_audio, sr, config.reverb_amount)

    stages = None
    if collect_stages:
        report("Drawing the steps", 0.97)
        stages = build_stages({
            "config": config, "raw_audio": raw_audio, "audio": audio,
            "noise_info": noise_info, "denoised_audio": audio,
            "use_preemphasis": use_preemphasis, "detection_audio": detection_audio,
            "use_phase_vocoder": use_phase_vocoder, "use_formants": use_formant_preservation,
            "detected_pitches": detected_pitches, "tuning_offset": tuning_offset,
            "follow_tuning": config.follow_singer_tuning, "auto_key": config.auto_key,
            "key_root": key_root, "key_type": key_type, "key_confidence": key_confidence,
            "target_pitches": target_pitches, "num_notes": len(notes),
            "num_overrides": len(note_overrides) if note_overrides else 0,
            "ratios_before_smoothing": ratios_before_smoothing, "shift_ratios": shift_ratios,
            "example_index": example_index, "example_in_frame": example_in_frame,
            "example_out_frame": example_out_frame,
            "corrected_pre_polish": corrected_pre_polish, "corrected_pitches": corrected_pitches,
            "polish": polish,
        })

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
        "notes": notes,
        "noise_reduction": noise_info,
        "stages": stages,
    }
