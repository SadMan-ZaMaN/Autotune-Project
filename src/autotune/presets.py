"""
Style presets - settings bundles for people who don't know music theory
==========================================================================
The two numbers that decide how autotune SOUNDS are:

  correction_strength (0..1): how far each note is pulled to the right pitch.
      1.0 = all the way, 0.5 = halfway (in semitones, see compute_shift_ratios).
  retune_ms: how quickly the pull happens (see smooth_shift_ratios).
      0 ms  = instantly - every note is dead flat and note changes jump in
              steps. That is the famous robotic "T-Pain / Cher" effect.
      ~35   = fast but smooth - modern pop vocals.
      ~80+  = slow - the singer's own slides and vibrato survive, only the
              average pitch of each note is fixed. Sounds natural.

Everything else is decided automatically (key, tuning) or is a quality
setting that should just stay on (phase vocoder, formant preservation).

"reverb" is the amount for effects.studio_polish (None = no polish at all).
"""

PRESETS = {
    "natural": {
        "label": "Natural",
        "tagline": "In tune, still sounds like you",
        "best_for": "Slow or emotional songs, ballads, acoustic covers, "
                    "or when you just want to sound better without anyone noticing autotune.",
        "correction_strength": 0.8,
        "retune_ms": 90.0,
        "reverb": 0.16,
    },
    "studio": {
        "label": "Studio Pop",
        "tagline": "Clean, polished, radio-ready",
        "best_for": "Most songs: pop, R&B, Bollywood/Bangla songs. "
                    "Notes are clearly in tune but still smooth. Start here.",
        "correction_strength": 1.0,
        "retune_ms": 30.0,
        "reverb": 0.22,
    },
    "hard": {
        "label": "Hard Tune",
        "tagline": "The famous robotic T-Pain effect",
        "best_for": "Rap, hip-hop, trap, EDM hooks - or any time you want "
                    "the audience to clearly HEAR the autotune.",
        "correction_strength": 1.0,
        "retune_ms": 0.0,
        "reverb": 0.14,
    },
    "raw": {
        "label": "Pitch only",
        "tagline": "Correction with no studio effects",
        "best_for": "Comparing methods for the report: only the pitch changes, "
                    "nothing else (no EQ, compression or reverb).",
        "correction_strength": 1.0,
        "retune_ms": 40.0,
        "reverb": None,
    },
}

DEFAULT_PRESET = "studio"
