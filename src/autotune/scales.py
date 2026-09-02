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