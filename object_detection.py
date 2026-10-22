"""
object_detection.py
--------------------
Real object detection for SentryHome's "Detection Zone" feature (Screen 1 and
Screen 2 of the storyboard).

Uses a pretrained YOLOv8n model (Ultralytics, trained on the COCO dataset) to
find people, package-like objects, and vehicles in a frame, then filters the
results down to only the detections that fall inside a user-defined
rectangular "detection zone" -- mirroring the zone the homeowner draws on
Screen 1 of the app.

COCO has no literal "package" class, so we treat backpack/handbag/suitcase as
package-like proxies, which is a reasonable stand-in for "something left at
the door" without training a custom model.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class names SentryHome cares about, grouped into the categories the
# app actually shows the user (see Screen 2's "Person 96%" / "Package 89%").
PERSON_CLASSES = {"person"}
PACKAGE_CLASSES = {"backpack", "handbag", "suitcase"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

CATEGORY_COLORS = {
    "person": (43, 183, 163),    # teal, BGR
    "package": (60, 150, 232),   # amber, BGR
    "vehicle": (74, 42, 27),     # navy, BGR
}


@dataclass
class Detection:
    label: str          # raw COCO class name, e.g. "person"
    category: str       # SentryHome category: person | package | vehicle
    confidence: float
    box: tuple          # (x1, y1, x2, y2) in pixels
    in_zone: bool


@dataclass
class DetectionResult:
    image_path: str
    annotated_path: str
    zone: tuple
    detections: list = field(default_factory=list)

    @property
    def in_zone_detections(self):
        return [d for d in self.detections if d.in_zone]


def _categorize(label: str):
    if label in PERSON_CLASSES:
        return "person"
    if label in PACKAGE_CLASSES:
        return "package"
    if label in VEHICLE_CLASSES:
        return "vehicle"
    return None


def _box_center_inside(box, zone):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def detect(image_path: str, zone: tuple, out_dir: str, conf_threshold: float = 0.35) -> DetectionResult:
    """
    Run object detection on an image and keep only the SentryHome-relevant
    classes (person / package-like / vehicle), flagging whether each
    detection's box-center falls inside the given detection zone.

    zone: (x1, y1, x2, y2) in pixel coordinates, matching the dashed
    "Detection Zone" rectangle the user drew on Screen 1.
    """
    model = _get_model()
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)[0]

    image = cv2.imread(image_path)
    detections = []

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        category = _categorize(label)
        if category is None:
            continue  # not something SentryHome alerts on (e.g. "tie", "cup")

        xyxy = tuple(float(v) for v in box.xyxy[0])
        confidence = float(box.conf[0])
        in_zone = _box_center_inside(xyxy, zone)

        detections.append(Detection(label=label, category=category, confidence=confidence, box=xyxy, in_zone=in_zone))

    annotated_path = str(Path(out_dir) / f"detected_{Path(image_path).name}")
    _draw_annotations(image, detections, zone, annotated_path)

    return DetectionResult(image_path=image_path, annotated_path=annotated_path, zone=zone, detections=detections)


def _draw_annotations(image, detections, zone, out_path):
    canvas = image.copy()

    # draw the detection zone (dashed-look rectangle, matching Screen 1's UI)
    zx1, zy1, zx2, zy2 = (int(v) for v in zone)
    _dashed_rect(canvas, (zx1, zy1), (zx2, zy2), (183, 115, 43), thickness=2)
    cv2.putText(canvas, "Detection Zone", (zx1, max(zy1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (183, 115, 43), 2)

    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = CATEGORY_COLORS.get(d.category, (200, 200, 200))
        if not d.in_zone:
            # outside the zone: SentryHome would not alert on this, draw muted/gray
            color = (190, 190, 190)
        thickness = 2 if d.in_zone else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        tag = f"{d.category.capitalize()} {d.confidence * 100:.0f}%"
        if not d.in_zone:
            tag += " (outside zone)"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        text_color = (255, 255, 255) if d.in_zone else (60, 60, 60)
        cv2.putText(canvas, tag, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    cv2.imwrite(out_path, canvas)


def _dashed_rect(canvas, pt1, pt2, color, thickness=2, dash_len=12):
    x1, y1 = pt1
    x2, y2 = pt2
    for (sx, sy, ex, ey) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
        _dashed_line(canvas, (sx, sy), (ex, ey), color, thickness, dash_len)


def _dashed_line(canvas, pt1, pt2, color, thickness, dash_len):
    import numpy as np
    pt1, pt2 = np.array(pt1, dtype=float), np.array(pt2, dtype=float)
    dist = np.linalg.norm(pt2 - pt1)
    if dist == 0:
        return
    direction = (pt2 - pt1) / dist
    n_dashes = int(dist // (dash_len * 2))
    for i in range(n_dashes + 1):
        start = pt1 + direction * (i * dash_len * 2)
        end = start + direction * dash_len
        cv2.line(canvas, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)
