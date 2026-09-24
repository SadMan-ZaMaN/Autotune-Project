import os
import time
import uuid
import threading
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template, abort

import sys
sys.path.insert(0, '.')
from src.autotune.config import AutoTuneConfig
from src.autotune.pipeline import run_pipeline
from src.autotune.io_utils import save_audio
from src.autotune.effects import match_loudness
from src.autotune.presets import PRESETS, DEFAULT_PRESET
from src.autotune.scales import freq_to_midi

"""
app.py - Local web server for the autotune frontend
=======================================================
This is a thin layer: it does NOT contain any DSP logic itself. Its only
job is to take a file + settings from the browser, call the EXISTING,
already-tested run_pipeline(), and hand the result back.

Long recordings take a while (roughly 1 s of processing per 8-10 s of
audio), so processing runs in a background thread: /process returns a job
id immediately, and the page polls /status/<job_id> for a progress bar.

The browser converts every upload/recording (MP3, M4A, WebM, ...) to a WAV
before sending it (see static/script.js), so the backend only ever receives
WAV - no ffmpeg needed.

Note editing: every result lists the automatic NOTES (runs of frames with
the same target, see scales.segment_notes). The page draws them as bars you
can drag up/down; /rerender/<job_id> then re-runs that job - same file, same
settings - with those notes moved (run_pipeline's note_overrides).

Run with:  python app.py
Then open: http://localhost:5001
"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB - allows very long recordings

UPLOAD_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# job_id -> {"state", "stage", "progress", "result", "error", "context"}
# context = (input_path, config, options), kept so a job can be re-rendered
# with note edits using exactly the settings it was first run with
jobs = {}

MAX_EDIT_SEMITONES = 12
jobs_lock = threading.Lock()


def update_job(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def as_bool(value, default):
    if value is None:
        return default
    return value == "true"


def pitch_track_for_plot(pitches, config, max_points=6000):
    """
    Per-frame pitches -> (times, MIDI numbers or None) for the pitch graph.
    Frame i's centre sits at i*hop - frame_size/2 samples in the original
    recording (frame_signal pads frame_size zeros at the start).
    Long recordings are thinned to at most max_points so the JSON stays small.
    """
    step = max(1, int(np.ceil(len(pitches) / max_points)))
    times = []
    values = []
    for i in range(0, len(pitches), step):
        t = (i * config.hop_size - config.frame_size / 2) / config.sample_rate
        if t < 0:
            continue
        times.append(round(t, 4))
        if pitches[i] > 0:
            values.append(round(float(freq_to_midi(pitches[i])), 3))
        else:
            values.append(None)
    return times, values


def notes_for_plot(notes, config):
    """
    (start_frame, end_frame, target_hz) notes -> JSON for the note bars.
    Times use the same frame-centre convention as pitch_track_for_plot; a
    bar covers half a hop either side of its first/last frame centre.
    midi is the automatic target (fractional when follow-my-tuning shifted
    the scale), so the bar sits exactly on the note line the graph draws.
    """
    hop_s = config.hop_size / config.sample_rate
    out = []
    for start, end, target_hz in notes:
        first_centre = (start * config.hop_size - config.frame_size / 2) / config.sample_rate
        last_centre = ((end - 1) * config.hop_size - config.frame_size / 2) / config.sample_rate
        out.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "start": round(max(0.0, first_centre - hop_s / 2), 4),
            "end": round(max(0.0, last_centre + hop_s / 2), 4),
            "midi": round(float(freq_to_midi(target_hz)), 3),
        })
    return out


def run_job(job_id, input_path, config, options, note_overrides=None):
    started = time.time()
    try:
        def progress(stage, fraction):
            update_job(job_id, stage=stage, progress=round(float(fraction), 3))

        result = run_pipeline(input_path, config, progress_callback=progress,
                              note_overrides=note_overrides, collect_stages=True, **options)
        sr = result["sample_rate"]

        corrected = result["corrected_audio"]
        save_audio(os.path.join(OUTPUT_DIR, f"result_{job_id}.wav"), corrected, sr)

        # The "before" track for the A/B player: the untouched recording at
        # the same sample rate, same length and similar loudness as the
        # result, so switching between them compares the TUNING, not volume.
        original = match_loudness(corrected, result["raw_audio"], sr)
        save_audio(os.path.join(OUTPUT_DIR, f"original_{job_id}.wav"), original, sr)

        times, detected = pitch_track_for_plot(result["detected_pitches"], config)
        _, target = pitch_track_for_plot(result["target_pitches"], config)
        _, corrected_pitch = pitch_track_for_plot(result["corrected_pitches"], config)

        update_job(job_id, state="done", stage="Done", progress=1.0, result={
            "job_id": job_id,
            "output_url": f"/audio/{job_id}/result",
            "original_url": f"/audio/{job_id}/original",
            # a re-render reuses the first job's upload, so take the id from the file name
            "input_url": f"/audio/{os.path.basename(input_path)[len('upload_'):-len('.wav')]}/input",
            "duration": round(len(corrected) / sr, 2),
            "processing_time": round(time.time() - started, 1),
            "key_root": result["key_root"],
            "key_type": result["key_type"],
            "key_confidence": result["key_confidence"],
            "key_auto": config.auto_key,
            "tuning_offset_cents": round(result["tuning_offset_cents"], 1),
            "noise_reduction": result["noise_reduction"],
            "pitch": {
                "times": times,
                "detected": detected,
                "target": target,
                "corrected": corrected_pitch,
            },
            "notes": notes_for_plot(result["notes"], config),
            "overrides": note_overrides or [],
            # one entry per processing step, with its graphs (stage_plots.py)
            "stages": result["stages"],
        })
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, state="error", error=str(e))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/presets")
def presets():
    return jsonify({"presets": PRESETS, "default": DEFAULT_PRESET})


@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    job_id = uuid.uuid4().hex[:8]
    input_path = os.path.join(UPLOAD_DIR, f"upload_{job_id}.wav")
    audio_file.save(input_path)

    # Settings from the control panel. A preset only fills in these values
    # on the page - the backend always receives the actual numbers.
    scale_root = request.form.get("scale_root", "auto")
    scale_type = request.form.get("scale_type", "major")
    reverb = float(request.form.get("reverb_amount", 0.2))

    config = AutoTuneConfig(
        scale_root=scale_root if scale_root != "auto" else "C",
        scale_type=scale_type,
        correction_strength=float(request.form.get("correction_strength", 1.0)),
        retune_ms=float(request.form.get("retune_ms", 30.0)),
        auto_key=(scale_root == "auto"),
        follow_singer_tuning=as_bool(request.form.get("follow_tuning"), True),
        studio_polish=as_bool(request.form.get("studio_polish"), True),
        reverb_amount=reverb,
        noise_reduction=float(request.form.get("noise_reduction", 0.5)),
    )
    options = {
        "use_phase_vocoder": as_bool(request.form.get("use_phase_vocoder"), True),
        "use_formant_preservation": as_bool(request.form.get("use_formant_preservation"), True),
        "use_preemphasis": as_bool(request.form.get("use_preemphasis"), True),
    }

    start_job(job_id, input_path, config, options)
    return jsonify({"job_id": job_id})


def start_job(job_id, input_path, config, options, note_overrides=None):
    with jobs_lock:
        jobs[job_id] = {"state": "running", "stage": "Starting", "progress": 0.0,
                        "result": None, "error": None,
                        "context": (input_path, config, options)}
    worker = threading.Thread(target=run_job, daemon=True,
                              args=(job_id, input_path, config, options, note_overrides))
    worker.start()


@app.route("/rerender/<job_id>", methods=["POST"])
def rerender(job_id):
    """
    Re-runs a finished job with some notes moved. Body:
        {"overrides": [{"start_frame": 120, "end_frame": 164, "semitones": 2}, ...]}
    Starts a NEW job (so the automatic result stays available unchanged)
    and returns its id; poll /status/<new id> as usual.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        context = job["context"] if job is not None else None
    if context is None:
        return jsonify({"error": "Unknown job - run the tuning again first"}), 404

    body = request.get_json(silent=True) or {}
    overrides = []
    try:
        for edit in body.get("overrides", []):
            start = int(edit["start_frame"])
            end = int(edit["end_frame"])
            semitones = int(edit["semitones"])
            if end <= start or semitones == 0:
                continue
            semitones = max(-MAX_EDIT_SEMITONES, min(MAX_EDIT_SEMITONES, semitones))
            overrides.append({"start_frame": start, "end_frame": end, "semitones": semitones})
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Badly formed note edits"}), 400

    input_path, config, options = context
    new_id = uuid.uuid4().hex[:8]
    start_job(new_id, input_path, config, options, overrides)
    return jsonify({"job_id": new_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        public = {k: v for k, v in job.items() if k != "context"}
        return jsonify(public)


# kind -> (folder, file prefix, download name)
#   result:   the tuned output
#   original: the untouched input, loudness-matched to the result (for the A/B player)
#   input:    the recording EXACTLY as uploaded/recorded (for downloading)
AUDIO_KINDS = {
    "result": (OUTPUT_DIR, "result", "tuned.wav"),
    "original": (OUTPUT_DIR, "original", "original_matched.wav"),
    "input": (UPLOAD_DIR, "upload", "original.wav"),
}


@app.route("/audio/<job_id>/<kind>")
def get_audio(job_id, kind):
    if kind not in AUDIO_KINDS or not job_id.isalnum():
        abort(404)
    folder, prefix, download_name = AUDIO_KINDS[kind]
    path = os.path.join(folder, f"{prefix}_{job_id}.wav")
    if not os.path.exists(path):
        abort(404)
    # a re-render with note edits downloads under its own name
    with jobs_lock:
        job = jobs.get(job_id)
        edited = bool(job and job["result"] and job["result"].get("overrides"))
    if kind == "result" and edited:
        download_name = "tuned_edited.wav"
    return send_file(path, mimetype="audio/wav", download_name=download_name)


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
