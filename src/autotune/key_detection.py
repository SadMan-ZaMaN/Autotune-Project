import numpy as np
from .scales import freq_to_midi, NOTE_NAMES

"""
Automatic key detection - so the user doesn't need to know music theory
==========================================================================
To correct a note we need to know which notes are "allowed" (the key). If
we snap to the WRONG key, correctly-sung notes get pushed to wrong notes,
which sounds worse than no correction at all. Most people using the app
don't know what key they sang in, so we work it out from the recording.

Step 1 - tuning offset (estimate_tuning_offset)
    Someone singing without a backing track is often a bit sharp or flat
    on EVERY note by the same amount (e.g. all notes ~40 cents high). That's
    fine musically - the melody is still right relative to itself - but if
    we snapped each note to standard A=440 Hz notes, some would be pulled
    down 40 cents and others (a bit sharper by chance) pushed UP 60 cents to
    the next note, breaking the melody. So we first measure this common
    offset and tune the scale to the singer.

Step 2 - pitch-class histogram (pitch_class_histogram)
    Count how many frames were sung on each of the 12 note names
    (C, C#, D ... B), ignoring which octave. A song in G major spends most
    of its time on G, A, B, C, D, E, F# and very little on the other 5.

Step 3 - choose the key (detect_key)
    a) Coverage: for each of the 12 possible 7-note major scales, what
       fraction of the sung frames fall on its notes? Keep the scales that
       contain (almost) everything that was sung - a key MISSING a sung note
       would push that note to a wrong one.
    b) Krumhansl-Schmuckler tie-break: music psychologists (Krumhansl &
       Kessler, 1982) measured how strongly each of the 12 notes "belongs"
       to a major or minor key. Among the scales kept in (a) we correlate
       our histogram with that profile, rotated to each candidate root, and
       pick the best match. This is the same idea as matched filtering /
       template matching from signals: compare against shifted templates,
       pick the highest correlation.
    If no scale covers at least 85% of the singing, we use all 12 notes
    (chromatic) - always safe.
"""

# Krumhansl-Kessler key profiles, index 0 = the key's root note
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_tuning_offset(detected_pitches):
    """
    Returns the singer's overall tuning offset in cents, in [-50, 50).

    The trick: a note's deviation is only meaningful modulo 100 cents
    (+60 cents sharp of C is the same as -40 cents flat of C#). Values that
    wrap around like this are angles, so we average them as angles - turn
    each deviation into a point on a circle, average the points, and read
    back the angle:
        angle_i = 2*pi * deviation_i / 100
        mean    = angle of  sum(cos(angle_i)) + j*sum(sin(angle_i))
    Example: a singer who is ~45 cents sharp on every note. Three frames
    measure +40, +45 and -48 cents. The -48 is really "+52 above the note
    below" - it was just a hair past the halfway point, so rounding assigned
    it to the next note up. A plain average gives (40+45-48)/3 = +12 cents
    (meaningless). The circular mean of 144, 162 and 187 degrees is ~164
    degrees = +46 cents - the true offset.
    (This is exactly the phase of the DFT of the deviations at 1 cycle per
    100 cents.)
    """
    cos_sum = 0.0
    sin_sum = 0.0
    count = 0
    for f in detected_pitches:
        if f <= 0:
            continue
        midi = freq_to_midi(f)
        deviation_cents = (midi - np.round(midi)) * 100.0
        angle = 2 * np.pi * deviation_cents / 100.0
        cos_sum += np.cos(angle)
        sin_sum += np.sin(angle)
        count += 1
    if count == 0:
        return 0.0
    mean_angle = np.arctan2(sin_sum, cos_sum)
    offset = mean_angle / (2 * np.pi) * 100.0
    if offset >= 50.0:
        offset -= 100.0
    return float(offset)


def pitch_class_histogram(detected_pitches, tuning_offset_cents=0.0, max_deviation_cents=35.0):
    """
    hist[pc] = number of voiced frames sung on pitch class pc (0=C ... 11=B).

    Frames that are more than max_deviation_cents away from any note (after
    removing the tuning offset) are skipped - those are slides between notes,
    not notes, and would only blur the histogram.
    Example: 196 Hz -> MIDI 55.0 -> pitch class 55 % 12 = 7 = G.
    """
    hist = np.zeros(12)
    for f in detected_pitches:
        if f <= 0:
            continue
        midi = freq_to_midi(f) - tuning_offset_cents / 100.0
        nearest = int(np.round(midi))
        if abs(midi - nearest) * 100.0 > max_deviation_cents:
            continue
        hist[nearest % 12] += 1
    return hist


MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]   # same as scales.SCALE_INTERVALS["major"]


def scale_coverage(hist, root):
    """
    Fraction of the sung frames whose note belongs to the major scale on
    `root` (the natural minor scale on root+9 has exactly the same notes).
    Example: hist has 90 frames on G A B C D E F# and 10 on F
        -> G major covers 90/100 = 0.90, C major covers (100 - F# frames)/100
    """
    inside = 0.0
    for step in MAJOR_STEPS:
        inside += hist[(root + step) % 12]
    return inside / hist.sum()


def detect_key(detected_pitches, tuning_offset_cents=0.0, min_voiced_frames=60,
               min_coverage=0.85, coverage_margin=0.02):
    """
    Returns (root_name, scale_type, confidence), e.g. ("G", "major", 0.97).
    confidence = fraction of the sung notes that are inside the chosen key.

    Why coverage first, correlation second:
    For AUTOTUNE, the dangerous mistake is choosing a key that is missing a
    note the singer really sang - that note then gets pushed to a wrong
    neighbour, which sounds worse than no correction. So we first keep only
    the keys that contain (almost) all the sung notes (coverage within
    coverage_margin of the best one). A short melody often fits several
    keys equally well (it may only use 5 or 6 different notes), so among
    those we pick the one whose Krumhansl-Kessler profile correlates best
    with the histogram (which also decides major vs. minor).
    (Tested on synthetic melodies: plain K-S correlation alone picked a key
    missing sung notes on a short 10-note melody; coverage-first did not.)

    Falls back to ("C", "chromatic", ...) - "snap to the nearest of all 12
    notes", which can never pick a wrong key - when there isn't enough sung
    material (< min_voiced_frames, ~0.7 s) or no key covers at least
    min_coverage of the sung notes (e.g. a very chromatic melody).

    Note: relative major/minor keys (C major and A minor, G major and E
    minor, ...) use exactly the same 7 notes, so they give an identical
    correction - the major/minor label is only for display.
    """
    hist = pitch_class_histogram(detected_pitches, tuning_offset_cents)
    if hist.sum() < min_voiced_frames:
        return "C", "chromatic", 0.0

    # Step 1: how well does each of the 12 possible note-sets cover the singing?
    coverage = np.zeros(12)
    for root in range(12):
        coverage[root] = scale_coverage(hist, root)
    best_coverage = coverage.max()
    if best_coverage < min_coverage:
        return "C", "chromatic", float(best_coverage)

    # Step 2: among the (near-)best note-sets, pick the most "key-like" one
    best_root = 0
    best_type = "major"
    best_corr = -2.0
    for root in range(12):
        if coverage[root] < best_coverage - coverage_margin:
            continue
        # the same 7 notes can be read as `root` major or (root+9) minor
        candidates = (("major", root, MAJOR_PROFILE), ("natural_minor", (root + 9) % 12, MINOR_PROFILE))
        for scale_type, tonic, profile in candidates:
            # rotate the profile so its index 0 (the tonic) lands on `tonic`
            template = np.roll(profile, tonic)
            corr = np.corrcoef(hist, template)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_root = tonic
                best_type = scale_type

    return NOTE_NAMES[best_root], best_type, float(best_coverage)
