import numpy as np

"""
A bit of music theory:
    frequency of A4 = 440Hz, A5 = 880Hz
    880 = 440.2; so one octave corresponds to doubling the frequency.
    But musical notes dont increase by a fixed number of Hz. So musical pitch is
    used in the logarithmic scale. MIDI(Musical Instrument Digital Interface( numbers give us a convenient representation. 

    ex: C4 -> 60
        C#$ -> 61 and so on
    one semi tone = 1 MIDI number
    one octave = 12 MIDI numbers
    MIDI has assigned A4 = 69 and at standard tuning A4=440Hz so we use this as a reference point

    f = 440*2^((n-69)/12); n = MIDI note number
    ex: A3 is one octave below A4 so n=69-12=57
    f = 440*2^(-12/12) = 220Hz
    A#4 is one semitone above A4

    f=440*2^((70-69)/12)=466.26Hz

    But our pitch detector gives f
    so n = 69+12log(f/440)
    MIDI number is not always an integer. If MIDI = 69.39 it means the detected pitch is about 39%
    of the way from MIDI note 69 to MIDI note 70
    rounding gives us the nearest chromatic note. Nearest integer to 69.39 is 69 so A4. If n=69,8 then
    rounding gives 70 which is A#4

    *This has a pretty big caveat*
    Suppose we choose C major scale, C major contains: C,D,E,F,G,A,B

"""

A4_FREQ = 440.0
A4_MIDI = 69

SCALE_INTERVALS = {
    "major": [0,2,4,5,7,9,11],
    "natural_minor": [0,2,3,5,7,8,10],
    "chromatic": list(range(12))
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_name_to_midi(note_name, octave=4):
    idx = NOTE_NAMES.index(note_name)
    return (octave+1)*12 +idx

def freq_to_midi(frequency):
    if frequency <= 0:
        return None
    return A4_MIDI + 12*np.log2(frequency/A4_FREQ)

def midi_to_freq(midi_note):
    return A4_FREQ*(2**((midi_note - A4_MIDI)/12))

def build_scale_midi_set(root_note_name, scale_type, midi_low=24, midi_high=108):
    root_pitch_class = NOTE_NAMES.index(root_note_name)
    intervals = SCALE_INTERVALS[scale_type]

    allowed = []
    for midi in range(midi_low, midi_high+1):
        pitch_class = midi%12
        offset = (pitch_class - root_pitch_class)%12
        if offset in intervals:
            allowed.append(midi)
    return np.array(allowed)

def nearest_scale_note(frequency, scale_midi_set):
    midi = freq_to_midi(frequency)
    if midi is None:
        return 0.0
    idx = np.argmin(np.abs(scale_midi_set - midi))
    nearest_midi = scale_midi_set[idx]

    return midi_to_freq(nearest_midi)


def choose_target_notes(detected_pitches, scale_midi_set, tuning_offset_cents=0.0, hysteresis=0.3):
    """
    Picks the target note for every frame (0.0 for unvoiced frames).

    Why not just nearest_scale_note() on every frame independently?
    Imagine someone singing right between C and D (MIDI 61.0), with a
    little natural wobble: 60.9, 61.1, 60.95, 61.05 ...
    nearest-note alone would answer C, D, C, D ... - the correction flips
    between two notes 40 times a second, heard as a nasty warble. Real
    singers don't change notes that fast, so we add HYSTERESIS (the same
    idea as a thermostat or a Schmitt trigger): once we've picked a note we
    stay on it until the voice is clearly closer to another scale note -
    closer by more than `hysteresis` semitones.

    Example (C major, current note C=60, hysteresis=0.3):
        sung 61.05: |61.05-60| - |61.05-62| = 1.05 - 0.95 = 0.10 < 0.3 -> stay on C
        sung 61.20: 1.20 - 0.80 = 0.40 > 0.3                          -> switch to D
    A new note (after a silence) always starts on the plain nearest note.

    tuning_offset_cents: the singer's overall offset from standard tuning
    (see key_detection.estimate_tuning_offset). The whole scale is slid by
    this amount, so a singer who is consistently 40 cents sharp is kept
    consistent with THEMSELVES instead of having some notes pulled down
    and others pushed up.
    """
    offset = tuning_offset_cents / 100.0
    targets = np.zeros(len(detected_pitches))
    current = None   # MIDI number of the note we're currently holding

    for i in range(len(detected_pitches)):
        f = detected_pitches[i]
        if f <= 0:
            current = None
            continue
        midi = freq_to_midi(f) - offset
        nearest = scale_midi_set[np.argmin(np.abs(scale_midi_set - midi))]

        if current is None:
            current = nearest
        elif nearest != current:
            if abs(midi - current) - abs(midi - nearest) > hysteresis:
                current = nearest

        targets[i] = midi_to_freq(current + offset)
    return targets

def segment_notes(target_pitches):
    """
    Groups the per-frame targets into NOTES: runs of consecutive frames that
    share the same target (0.0 = unvoiced frames, never part of a note).
    These runs are what the web UI draws as draggable bars.

    Why this is well defined: choose_target_notes() already has hysteresis,
    so the target only changes when the singer really moves to another
    note - a run of equal targets is one sung note, not vibrato flicker.

    Example (hop = 11.6 ms), targets in Hz:
        [0, 220, 220, 220, 247, 247, 0, 0, 220]
        -> [(1, 4, 220.0), (4, 6, 247.0), (8, 9, 220.0)]
    Frames 1-3 are one A3, frames 4-5 a B3 sung legato right after it, and
    frame 8 a new A3 after a gap (a separate note, even though it's the
    same pitch - it can be edited independently).

    Returns: list of (start_frame, end_frame, target_hz), end_frame EXCLUSIVE
    """
    notes = []
    start = None
    for i in range(len(target_pitches) + 1):
        # a note ends at an unvoiced frame, a target change, or the end
        if start is not None:
            ended = (i == len(target_pitches)
                     or target_pitches[i] <= 0
                     or target_pitches[i] != target_pitches[start])
            if ended:
                notes.append((start, i, float(target_pitches[start])))
                start = None
        if start is None and i < len(target_pitches) and target_pitches[i] > 0:
            start = i
    return notes


def apply_note_overrides(target_pitches, note_overrides):
    """
    Moves hand-edited notes to a different target: the user dragged a note
    bar up or down by a whole number of semitones.

    note_overrides: list of dicts {"start_frame", "end_frame", "semitones"}
    (end_frame exclusive, same convention as segment_notes).

    One semitone is a frequency ratio of 2^(1/12) = 1.0595, so moving a
    note by s semitones multiplies its target by 2^(s/12):
        A3 = 220 Hz, dragged +2 -> 220 * 2^(2/12) = 246.9 Hz (B3)
        A3 = 220 Hz, dragged -12 -> 220 * 0.5 = 110 Hz (A2, an octave down)
    Only voiced frames are moved (unvoiced frames stay 0 = "don't shift").

    Returns: (new_target_pitches, overridden) - `overridden` is a boolean
    mask of the frames that were edited, so the pipeline can correct them
    at full strength (an explicit edit should land exactly on its note,
    whatever the global correction strength is).
    """
    targets = np.array(target_pitches, dtype=np.float64)
    overridden = np.zeros(len(targets), dtype=bool)
    for edit in note_overrides:
        semitones = edit["semitones"]
        if semitones == 0:
            continue
        start = max(0, int(edit["start_frame"]))
        end = min(len(targets), int(edit["end_frame"]))
        factor = 2.0 ** (semitones / 12.0)
        for i in range(start, end):
            if targets[i] > 0:
                targets[i] = targets[i] * factor
                overridden[i] = True
    return targets, overridden
