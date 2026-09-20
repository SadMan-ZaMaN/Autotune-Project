import os
import io
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template

import sys
sys.path.insert(0, '.')
from src.autotune.config import AutoTuneConfig
from src.autotune.pipeline import run_pipeline
from src.autotune.io_utils import save_audio

"""
app.py - Local web server for the autotune frontend
=======================================================
This is a thin layer: it does NOT contain any DSP logic itself. Its only
job is to take a file + settings from the browser, call your EXISTING,
already-tested run_pipeline(), and hand the result back as an audio file.

Run with:  python app.py
Then open: http://localhost:5000
"""

app = Flask(__name__)

UPLOAD_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    job_id = uuid.uuid4().hex[:8]
    input_path = os.path.join(UPLOAD_DIR, f"upload_{job_id}.wav")
    audio_file.save(input_path)

    # Read settings sent from the frontend's control panel
    scale_root = request.form.get("scale_root", "C")
    scale_type = request.form.get("scale_type", "major")
    correction_strength = float(request.form.get("correction_strength", 1.0))
    retune_ms = float(request.form.get("retune_ms", 40.0))
    use_phase_vocoder = request.form.get("use_phase_vocoder", "true") == "true"
    use_formant_preservation = request.form.get("use_formant_preservation", "true") == "true"
    use_preemphasis = request.form.get("use_preemphasis", "true") == "true"

    config = AutoTuneConfig(
        scale_root=scale_root,
        scale_type=scale_type,
        correction_strength=correction_strength,
        retune_ms=retune_ms,
    )

    try:
        result = run_pipeline(
            input_path, config,
            use_phase_vocoder=use_phase_vocoder,
            use_preemphasis=use_preemphasis,
            use_formant_preservation=use_formant_preservation,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    output_path = os.path.join(OUTPUT_DIR, f"result_{job_id}.wav")
    save_audio(output_path, result["corrected_audio"], result["sample_rate"])

    return jsonify({
        "output_url": f"/audio/{job_id}",
        "sample_rate": result["sample_rate"],
    })


@app.route("/audio/<job_id>")
def get_audio(job_id):
    path = os.path.join(OUTPUT_DIR, f"result_{job_id}.wav")
    return send_file(path, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
