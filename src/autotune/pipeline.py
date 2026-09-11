# this was created  Week 5-6: full pipeline + customization features (scale selector, correction strength)


import numpy as np
from .io_utils import load_audio, save_audio
from .framing import frame_signal, overlap_add
from .pitch_detection import detect_pitch_for_all_frames
from .scales import build_scale_midi_set, nearest_scale_note
from .pitch_shift import compute_shift_ratios, naive_pitch_shift
from .phase_vocoder import phase_vocoder_shift

"""
run_pipeline: Full Autotune Chain, Start to Finish
=====================================================

This function is the "spine" that connects every module you've built so
far, in the same order you've already tested them individually:

    load audio -> frame it -> detect pitch per frame -> find nearest
    scale note per frame -> compute how much to shift each frame ->
    shift (naive or phase vocoder) -> stitch back together (overlap-add)

Nothing new is invented here - this function just calls, in order, the
functions you already built and verified separately.
"""

def run_pipeline(input_path, config, use_phase_vocoder=True):
    """
    input_path: str - path to a WAV file to correct
    config: AutoTuneConfig - holds frame_size, hop_size, sample_rate,
            scale_root, scale_type, correction_strength
    use_phase_vocoder: bool - True uses phase_vocoder_shift, False uses
                       naive_pitch_shift (useful for generating comparison
                       clips for your report)

    Returns a dict with everything useful for saving audio or making plots:
        original_audio, corrected_audio, sample_rate,
        detected_pitches, target_pitches, shift_ratios
    """
    audio, sr = load_audio(input_path, target_sr=config.sample_rate)
    frames, pad_len = frame_signal(audio, config)

    detected_pitches = detect_pitch_for_all_frames(frames, sr)

    scale = build_scale_midi_set(config.scale_root, config.scale_type)

    # For each frame: if it's voiced (pitch > 0), find its nearest scale
    # note. If it's silent/unvoiced, target stays 0 (compute_shift_ratios
    # already knows to leave unvoiced frames un-shifted).
    target_pitches = np.zeros_like(detected_pitches)
    for i in range(len(detected_pitches)):
        if detected_pitches[i] > 0:
            target_pitches[i] = nearest_scale_note(detected_pitches[i], scale)

    shift_ratios = compute_shift_ratios(detected_pitches, target_pitches,
                                         config.correction_strength)

    if use_phase_vocoder:
        shifted_frames = phase_vocoder_shift(frames, shift_ratios, config)
    else:
        shifted_frames = np.zeros_like(frames)
        for i in range(len(frames)):
            shifted_frames[i] = naive_pitch_shift(frames[i], shift_ratios[i])

    corrected_audio = overlap_add(shifted_frames, config, pad_len)

    return {
        "original_audio": audio,
        "corrected_audio": corrected_audio,
        "sample_rate": sr,
        "detected_pitches": detected_pitches,
        "target_pitches": target_pitches,
        "shift_ratios": shift_ratios,
    }