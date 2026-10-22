"""
pipeline.py
------------
Orchestrates the three CV modules into one SentryHome detection event,
mirroring the real app's flow: motion in the zone (Screen 2) -> face
recognition if a person is involved (Screen 3) -> segmentation for a precise
mask (background Screen 2/5 detail) -> a structured alert payload that is
exactly what the phone UI on Screens 2-5 would render.

Run directly for a demo: `python3 pipeline.py`
"""

import json
from pathlib import Path

from ultralytics.utils import ASSETS

import object_detection
import segmentation
from face_recognition_module import SentryHomeFaceRecognizer

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def run_detection_event(image_path: str, zone: tuple, recognizer: SentryHomeFaceRecognizer) -> dict:
    """
    Runs one full SentryHome detection event on a single frame and returns
    the alert payload the app would show the user.
    """
    det_result = object_detection.detect(image_path, zone, str(OUT_DIR))

    alert = {
        "source_image": image_path,
        "annotated_image": det_result.annotated_path,
        "zone": det_result.zone,
        "events": [],
    }

    # Run face recognition once per image (not once per person) so the
    # annotated faces image isn't overwritten for every person detection.
    all_face_results = recognizer.recognize(image_path)
    if all_face_results:
        recognizer.draw_annotations(image_path, all_face_results, str(OUT_DIR / f"faces_{Path(image_path).name}"))

    for idx, d in enumerate(det_result.in_zone_detections):
        event = {
            "category": d.category,
            "confidence": round(d.confidence, 3),
            "box": [round(v, 1) for v in d.box],
        }

        if d.category == "person":
            # Screen 3: match this person's region against already-detected faces
            x1, y1, x2, y2 = d.box
            matched_faces = [
                f for f in all_face_results
                if x1 <= f.box[0] + f.box[2] / 2 <= x2 and y1 <= f.box[1] + f.box[3] / 2 <= y2
            ]
            if matched_faces:
                face = matched_faces[0]
                event["face"] = {
                    "label": face.label if face.is_known else "Unknown Visitor",
                    "is_known": face.is_known,
                    "distance": round(face.distance, 1),
                }
            else:
                event["face"] = None

            # Screen 2/5: segment this person out of the background
            mask_path, cutout_path, coverage = segmentation.segment_box(
                image_path, d.box, str(OUT_DIR), tag=f"person{idx}"
            )
            event["segmentation"] = {
                "mask_path": mask_path,
                "cutout_path": cutout_path,
                "foreground_coverage": round(coverage, 3),
            }

        alert["events"].append(event)

    return alert


def format_notification(alert: dict) -> str:
    """Turns a raw alert payload into the kind of banner text Screen 2 shows."""
    if not alert["events"]:
        return "No zone activity — no notification sent."

    lines = []
    for e in alert["events"]:
        if e["category"] == "person":
            face = e.get("face")
            if face and face["is_known"]:
                lines.append(f"Welcome home, {face['label']} — {100 - face['distance']:.0f}% confidence match")
            elif face:
                lines.append("Unknown Visitor Detected — tap to confirm or correct")
            else:
                lines.append(f"Person detected ({e['confidence'] * 100:.0f}% confidence), no clear face match")
        else:
            lines.append(f"{e['category'].capitalize()} detected ({e['confidence'] * 100:.0f}% confidence)")
    return "\n".join(lines)


def demo():
    """
    Runs the full pipeline end to end on the two sample photos that ship
    with the `ultralytics` package, so this script works out of the box
    with no external downloads beyond `pip install`.
    """
    zidane = str(ASSETS / "zidane.jpg")
    bus = str(ASSETS / "bus.jpg")

    # --- Enrollment: teach SentryHome one "known resident" ---------------
    recognizer = SentryHomeFaceRecognizer()
    recognizer.enroll("Zidane", [zidane])
    print("Enrolled 'Zidane' as a known resident from zidane.jpg\n")

    # --- Event 1: Zidane's own photo -> should be recognized as known ----
    print("=" * 60)
    print("EVENT 1: front-door frame containing an enrolled resident")
    print("=" * 60)
    zone1 = (0, 0, 1280, 720)  # whole frame counts as the zone for this demo
    alert1 = run_detection_event(zidane, zone1, recognizer)
    print(format_notification(alert1))
    print(json.dumps(alert1, indent=2, default=str))

    # --- Event 2: the bus photo -> different people, none enrolled -------
    print("\n" + "=" * 60)
    print("EVENT 2: front-door frame containing unrecognized people + a 'vehicle'")
    print("=" * 60)
    zone2 = (0, 0, 1280, 1024)
    alert2 = run_detection_event(bus, zone2, recognizer)
    print(format_notification(alert2))
    print(json.dumps(alert2, indent=2, default=str))

    print(f"\nAnnotated images and masks written to: {OUT_DIR}")


if __name__ == "__main__":
    demo()
