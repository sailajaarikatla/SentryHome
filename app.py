"""
app.py
-------
Tiny web server that exposes the SentryHome CV pipeline to the browser UI
(static/index.html). Run with:  python app.py   then open http://127.0.0.1:5000
"""
from __future__ import annotations
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory

import pipeline
from face_recognition_module import SentryHomeFaceRecognizer

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUT_DIR = pipeline.OUT_DIR

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
recognizer = SentryHomeFaceRecognizer()
enrolled: list[str] = []


def _save_upload(file_storage) -> str:
    suffix = Path(file_storage.filename or "img.jpg").suffix.lower() or ".jpg"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}{suffix}"
    file_storage.save(path)
    return str(path)


def _url(p) -> str | None:
    return f"/output/{Path(p).name}" if p else None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/output/<path:name>")
def output_file(name):
    return send_from_directory(OUT_DIR, name)


@app.get("/api/residents")
def residents():
    return jsonify({"residents": enrolled})


@app.post("/api/enroll")
def enroll():
    name = (request.form.get("name") or "").strip()
    files = request.files.getlist("photos")
    if not name or not files:
        return jsonify({"error": "Provide a name and at least one photo."}), 400
    paths = [_save_upload(f) for f in files]
    try:
        recognizer.enroll(name, paths)
    except Exception as e:  # e.g. no face found in the photo
        return jsonify({"error": str(e)}), 400
    if name not in enrolled:
        enrolled.append(name)
    return jsonify({"residents": enrolled})


@app.post("/api/detect")
def detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Upload an image."}), 400
    after_hours = request.form.get("after_hours") == "true"

    path = _save_upload(file)
    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "Could not read that image."}), 400
    h, w = img.shape[:2]

    alert = pipeline.run_detection_event(path, (0, 0, w, h), recognizer)
    notification = pipeline.format_notification(alert)

    # Context layer (not a CV model): an unmatched person after hours escalates
    # to an intruder alert -- same logic described on Screen 6 of the deck.
    unknown_person = any(
        e["category"] == "person" and not (e.get("face") and e["face"]["is_known"])
        for e in alert["events"]
    )
    level = "intruder" if (after_hours and unknown_person) else (
        "unknown" if unknown_person else ("known" if alert["events"] else "none")
    )

    events = []
    for e in alert["events"]:
        seg = e.get("segmentation") or {}
        events.append({
            "category": e["category"],
            "confidence": e["confidence"],
            "face": e.get("face"),
            "cutout": _url(seg.get("cutout_path")),
        })

    stem = Path(path).name
    faces_file = OUT_DIR / f"faces_{stem}"
    return jsonify({
        "level": level,
        "notification": notification,
        "annotated": _url(alert["annotated_image"]),
        "faces": _url(faces_file) if faces_file.exists() else None,
        "events": events,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
