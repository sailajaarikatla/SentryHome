# SentryHome — Computer Vision Backend

A real, working computer vision pipeline implementing the three features
from the SentryHome prototype deck (Week 6 assignment). This is not a UI
simulation — every bounding box, confidence score, face match, and
segmentation mask below is produced by an actual model running on a real
image.

| Storyboard feature | File | Real technique used |
|---|---|---|
| Screen 1/2 — Object Detection & zone filtering | `object_detection.py` | YOLOv8n (Ultralytics), pretrained on COCO |
| Screen 3/4 — Face Recognition & correction loop | `face_recognition_module.py` | Haar cascade face detection + LBPH face recognizer (OpenCV) |
| Screen 2/5 — Image Segmentation | `segmentation.py` | GrabCut (OpenCV) |
| Orchestration / alert formatting | `pipeline.py` | Ties the three together into one SentryHome "detection event" |

## Setup

```bash
pip install -r requirements.txt
```

The first run of `pipeline.py` downloads the ~6MB YOLOv8n weight file from
Ultralytics' GitHub releases automatically. No other downloads are needed —
the demo uses two sample photos (`zidane.jpg`, `bus.jpg`) that ship inside
the `ultralytics` package itself.

**Note on `cv2.face`:** some `opencv-contrib-python` wheels get shadowed by a
plain `opencv-python`/`opencv-python-headless` install pulled in by another
dependency (e.g. `mediapipe`), which silently removes the `cv2.face` module
and the bundled Haar cascade XML files. If you hit
`AttributeError: module 'cv2.face' has no attribute 'LBPHFaceRecognizer_create'`,
uninstall all three opencv variants and reinstall only
`opencv-contrib-python`:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python
```

If the cascade XML still isn't present at `cv2.data.haarcascades`, this repo
ships a local fallback copy in `cascades/` (fetched from OpenCV's official
GitHub repo) that `face_recognition_module.py` uses automatically.

## Run the demo

```bash
python3 pipeline.py
```

This will:
1. Enroll "Zidane" as a known resident from `zidane.jpg` (one reference photo).
2. Run a full detection event on `zidane.jpg` — detects 2 people, segments
   each one out of the background, and recognizes one of them as the
   enrolled "Zidane" while flagging the other as an "Unknown Visitor" —
   exactly like Screen 3 of the app.
3. Run a second detection event on `bus.jpg` — detects a vehicle and three
   people, none enrolled, all correctly flagged unknown.
4. Print each event's SentryHome-style notification text (what Screen 2's
   banner would say) and the full structured alert JSON.
5. Write annotated images and segmentation masks/cutouts to `output/`.

## Using your own photos

```python
from face_recognition_module import SentryHomeFaceRecognizer
from pipeline import run_detection_event, format_notification

recognizer = SentryHomeFaceRecognizer()
recognizer.enroll("Mom", ["photos/mom_1.jpg", "photos/mom_2.jpg"])  # more photos = more robust

zone = (100, 150, 900, 700)  # the detection zone drawn on Screen 1, in pixels
alert = run_detection_event("photos/front_door_frame.jpg", zone, recognizer)
print(format_notification(alert))
```

To simulate Screen 4's correction flow (a known person misflagged as
unknown, e.g. because they're wearing sunglasses), call:

```python
recognizer.add_correction("Mom", "photos/front_door_frame.jpg", box=(x, y, w, h))
```

where `box` is the face box from the misclassified `FaceResult`. This adds
that frame as another reference photo for "Mom" and retrains, the same
active-learning loop described in the deck.

## Known limitations 
- **LBPH vs. CNN embeddings:** LBPH is a real, classic (non-deep-learning)
  face recognition algorithm — texture-pattern matching rather than a
  learned embedding space. It demonstrates the same enroll → compare →
  threshold pipeline without needing a large pretrained face-embedding model,
  but a production SentryHome would likely use a CNN-based embedding model
  (e.g. FaceNet/ArcFace) for better accuracy, especially across lighting,
  angle, and demographic variation — which ties directly into the bias
  discussion on the deck's Reflection slide.
- **"Package" has no COCO class:** YOLO's COCO training set has no literal
  "package" label, so `backpack`/`handbag`/`suitcase` are used as a proxy.
  A real deployment would fine-tune on delivery-box imagery.
- **GrabCut needs a decent bounding box:** segmentation quality depends on
  the object detector's box being reasonably tight. It can leave ragged
  edges around hair/limbs, as you'll see in the sample output.

## Browser UI (HTML + Python service)

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**. The page (`static/index.html`) lets you:

1. Enroll a known resident (name + face photo(s)).
2. Upload a "camera frame" and run detection. The Python service runs YOLOv8
   object detection, LBPH face recognition and GrabCut segmentation, and the
   page shows the annotated images, cutouts and notification text.
3. Tick "after hours" to see the Screen 6 behavior: an unrecognized person
   outside scheduled hours escalates to an **Intruder Alert**. That escalation
   is a rule in `app.py` layered on the CV output, not a separate model.

Enrolled residents live in memory, so they reset when you restart the server.
