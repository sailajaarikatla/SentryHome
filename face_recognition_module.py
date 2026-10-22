"""
face_recognition_module.py
---------------------------
Real face recognition for SentryHome's "known vs. unknown visitor" feature
(Screen 3), and the relabel/active-learning loop from Screen 4.

Uses OpenCV's built-in Haar cascade for face *detection* and the classic
LBPH (Local Binary Patterns Histograms) algorithm for face *recognition*
(cv2.face.LBPHFaceRecognizer). LBPH is a real, non-deep-learning face
recognition technique: it compares the texture pattern around each pixel of
a face against enrolled reference faces and returns a distance score, which
is exactly the kind of "similarity threshold" described in the deck's
Screen 3 integration notes. It needs no large pretrained weights and no GPU,
which makes it a practical stand-in for this prototype's CNN-embedding
description -- in a shipped product you'd likely swap this for a deep
embedding model (e.g. FaceNet/ArcFace), but the enrollment -> match ->
threshold -> known/unknown pipeline is identical.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Some opencv-contrib-python wheels ship cv2.data.haarcascades without the
# actual XML files bundled. Fall back to a local copy (fetched once from the
# official OpenCV GitHub repo) if the package's own copy isn't present.
_BUNDLED_CASCADE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
_LOCAL_CASCADE = Path(__file__).parent / "cascades" / "haarcascade_frontalface_default.xml"
FACE_CASCADE_PATH = str(_BUNDLED_CASCADE if _BUNDLED_CASCADE.exists() else _LOCAL_CASCADE)

# Lower LBPH distance = more similar. Below this, SentryHome calls it a match.
MATCH_THRESHOLD = 70.0


@dataclass
class FaceResult:
    box: tuple              # (x, y, w, h) in pixels
    label: str               # enrolled person's name, or "Unknown"
    distance: float          # LBPH distance (lower = more confident match)
    is_known: bool


class SentryHomeFaceRecognizer:
    """
    Wraps OpenCV's Haar cascade (detection) + LBPH recognizer (recognition)
    behind the same enroll / recognize / correct vocabulary the storyboard
    uses on Screens 3 and 4.
    """

    def __init__(self):
        self._detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._label_names = {}   # int label id -> person name
        self._next_label_id = 0
        self._trained = False

    def detect_faces(self, gray_image):
        faces = self._detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [tuple(int(v) for v in f) for f in faces]

    def enroll(self, name: str, face_image_paths: list):
        """
        Register a person as "known" from one or more reference photos,
        exactly like a homeowner enrolling a family member in SentryHome's
        settings. Multiple photos improve robustness (this is also what
        Screen 4's "This is a family member" correction effectively does --
        it adds another reference photo to the person's set).
        """
        if name not in self._label_names.values():
            self._label_names[self._next_label_id] = name
            label_id = self._next_label_id
            self._next_label_id += 1
        else:
            label_id = [k for k, v in self._label_names.items() if v == name][0]

        faces, labels = [], []
        for path in face_image_paths:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            boxes = self.detect_faces(gray)
            if not boxes:
                continue
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])  # largest face in the photo
            face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            faces.append(face_crop)
            labels.append(label_id)

        if not faces:
            raise ValueError(f"No face found in any reference photo for '{name}'")

        self._train_or_update(faces, labels)
        return label_id

    def add_correction(self, name: str, face_image_path: str, box: tuple):
        """
        Screen 4's feedback loop: the user says "this is actually <name>",
        so we crop the face that was just misclassified and add it to that
        person's reference set, then retrain -- an active-learning update
        instead of a one-time static enrollment.
        """
        gray = cv2.imread(face_image_path, cv2.IMREAD_GRAYSCALE)
        x, y, w, h = box
        face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

        if name not in self._label_names.values():
            label_id = self._next_label_id
            self._label_names[label_id] = name
            self._next_label_id += 1
        else:
            label_id = [k for k, v in self._label_names.items() if v == name][0]

        self._train_or_update([face_crop], [label_id])

    def _train_or_update(self, faces, labels):
        if self._trained:
            self._recognizer.update(faces, np.array(labels))
        else:
            self._recognizer.train(faces, np.array(labels))
            self._trained = True

    def recognize(self, image_path: str) -> list:
        """
        Detect every face in an image and classify each as a known resident
        or "Unknown Visitor", the same decision shown on Screen 3.
        """
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        boxes = self.detect_faces(gray)
        results = []

        for (x, y, w, h) in boxes:
            face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

            if self._trained:
                label_id, distance = self._recognizer.predict(face_crop)
                is_known = distance <= MATCH_THRESHOLD
                name = self._label_names.get(label_id, "Unknown") if is_known else "Unknown"
            else:
                label_id, distance, is_known, name = -1, float("inf"), False, "Unknown"

            results.append(FaceResult(box=(x, y, w, h), label=name, distance=float(distance), is_known=is_known))

        return results

    def draw_annotations(self, image_path: str, results: list, out_path: str):
        image = cv2.imread(image_path)
        for r in results:
            x, y, w, h = r.box
            color = (43, 183, 163) if r.is_known else (60, 150, 232)  # teal if known, amber if unknown
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            tag = f"{r.label} ({r.distance:.0f})" if r.is_known else "Unknown Visitor"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - th - 10), (x + tw + 6, y), color, -1)
            cv2.putText(image, tag, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(out_path, image)
        return out_path
