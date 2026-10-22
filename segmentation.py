"""
segmentation.py
-----------------
Real image segmentation for SentryHome's "separate the person from
background clutter" feature, referenced on Screens 2 and 5.

Uses OpenCV's GrabCut algorithm -- a classic (non-deep-learning) interactive
segmentation technique that iteratively models foreground/background color
distributions to carve a precise pixel mask out of a bounding box. We seed
GrabCut with the object-detector's bounding box (from object_detection.py),
exactly matching the pipeline described in the deck: detect the box first,
then segment inside it to get the object's exact outline rather than just a
rectangle.
"""

from pathlib import Path

import cv2
import numpy as np


def segment_box(image_path: str, box: tuple, out_dir: str, iterations: int = 5, tag: str = ""):
    """
    Run GrabCut inside (and slightly padded around) a detection box to
    produce a binary mask isolating that object from the background.

    Returns (mask_path, cutout_path, mask_coverage_ratio).
    mask_coverage_ratio is the fraction of the padded box classified as
    foreground -- a cheap way to sanity-check the segmentation (a person
    standing in an open doorway should cover roughly 30-60% of their box;
    near 0% or 100% usually means GrabCut failed to separate anything).
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)

    pad = 10
    x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    x2, y2 = min(x2 + pad, w), min(y2 + pad, h)
    rect = (x1, y1, x2 - x1, y2 - y1)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    stem = Path(image_path).stem
    suffix = f"_{tag}" if tag else ""
    mask_path = str(Path(out_dir) / f"{stem}{suffix}_mask.png")
    cv2.imwrite(mask_path, binary_mask)

    # cutout: original pixels where mask is foreground, transparent/black elsewhere
    cutout = cv2.bitwise_and(image, image, mask=binary_mask)
    cutout_path = str(Path(out_dir) / f"{stem}{suffix}_cutout.png")
    cv2.imwrite(cutout_path, cutout)

    box_area = max(rect[2] * rect[3], 1)
    fg_pixels = int(np.count_nonzero(binary_mask[y1:y2, x1:x2]))
    coverage_ratio = fg_pixels / box_area

    return mask_path, cutout_path, coverage_ratio
