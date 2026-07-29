"""
iris_masking.py
---------------
Standalone, importable iris/pupil masking module.

Primary API
-----------
    from iris_masking import mask_iris

    masked_img, meta = mask_iris(image_rgb)   # image_rgb: H×W×3 uint8 RGB numpy array

    masked_img  — same shape/dtype as input, iris region filled with surrounding
                  periocular skin tone (annular mean) — no black or constant fill.
    meta        — dict:
                    "method"      : "mediapipe" | "hough" | "fallback"
                    "confidence"  : float  (MediaPipe face detection score; 0.0 for others)
                    "iris_center" : (cx_px, cy_px) in input image coords
                    "iris_radius" : r_px (int)
                    "fallback"    : bool

Detection cascade
-----------------
1. MediaPipe FaceLandmarker (new Tasks API, >= 0.10.5).
   Iris ring landmarks: right eye 468-472, left eye 473-477.
   Image is padded 1.5× before detection to improve hit rate on tight crops.
2. Hough Circle Transform on CLAHE-enhanced grayscale (fallback 1).
3. Central circular mask covering inner 25% of min(H, W) (fallback 2).
   Failures from fallbacks 2 and 3 are appended to masking_failures.log.

Mask fill
---------
Pixels inside the iris ellipse are replaced with the mean colour of the
annular region at 1.2× to 2.0× the iris radius (periocular skin tone).
If the annulus falls outside the image boundary, the global image mean is used.
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── MediaPipe import (graceful degradation) ─────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────

# MediaPipe 478-landmark model: iris ring indices
_RIGHT_IRIS = [468, 469, 470, 471, 472]   # [centre, right, bottom, left, top]
_LEFT_IRIS  = [473, 474, 475, 476, 477]

_ANNULUS_INNER = 1.2   # inner ring multiplier for skin-tone sampling
_ANNULUS_OUTER = 2.0   # outer ring multiplier

_FALLBACK_FRACTION = 0.25   # fallback mask radius = min(H,W) * fraction / 2

# Pupil darkness gate (used only when skip_if_no_iris=True).
# A real iris always contains a dark pupil; the minimum pixel inside the
# detected circle should be below this threshold.
# Pupil: min ≈ 10-40.  Eyelid skin (closed eye): min ≈ 60-150.
_PUPIL_MIN_THRESHOLD = 55

_FAILURES_LOG = Path(__file__).parent / "masking_failures.log"

# ── Locate the MediaPipe .task file ─────────────────────────────────────────

def _find_task_file() -> Optional[str]:
    candidates = [
        Path(__file__).parent / "face_landmarker.task",
        Path(__file__).parent.parent / "face_landmarker.task",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

_LANDMARKER: Optional[object] = None   # FaceLandmarker instance
_LANDMARKER_INIT_ATTEMPTED = False


def _get_landmarker() -> Optional[object]:
    global _LANDMARKER, _LANDMARKER_INIT_ATTEMPTED
    if _LANDMARKER_INIT_ATTEMPTED:
        return _LANDMARKER
    _LANDMARKER_INIT_ATTEMPTED = True

    if not _MP_AVAILABLE:
        return None

    task_path = _find_task_file()
    if task_path is None:
        print("[iris_masking] WARNING: face_landmarker.task not found — "
              "MediaPipe iris detection disabled, falling back to Hough.",
              file=sys.stderr)
        return None

    try:
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        _LANDMARKER = FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[iris_masking] WARNING: MediaPipe init failed ({e}) — "
              "falling back to Hough.", file=sys.stderr)
        _LANDMARKER = None

    return _LANDMARKER

def _annular_mean(image_rgb: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """Mean colour of pixels in the annulus [inner_r, outer_r] around (cx, cy)."""
    h, w = image_rgb.shape[:2]
    inner_r = r * _ANNULUS_INNER
    outer_r = r * _ANNULUS_OUTER

    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    annulus_mask = (dist >= inner_r) & (dist <= outer_r)

    pixels = image_rgb[annulus_mask]
    if len(pixels) == 0:
        # fallback: global image mean
        return image_rgb.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return pixels.mean(axis=0).astype(np.uint8)


def _fill_circle(image_rgb: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """Return a copy of image_rgb with the circle (cx,cy,r) filled with annular mean."""
    fill_color = _annular_mean(image_rgb, cx, cy, r)
    result = image_rgb.copy()
    h, w = result.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    result[mask] = fill_color
    return result

def _log_failure(image_path: Optional[str]) -> None:
    if image_path is None:
        return
    with open(_FAILURES_LOG, "a", encoding="utf-8") as f:
        f.write(str(image_path) + "\n")

def _detect_iris_mediapipe(image_rgb: np.ndarray):
    landmarker = _get_landmarker()
    if landmarker is None:
        return None

    h, w = image_rgb.shape[:2]
    pad_h, pad_w = int(h * 0.75), int(w * 0.75)
    padded = np.full((h + 2 * pad_h, w + 2 * pad_w, 3), 128, dtype=np.uint8)
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = image_rgb

    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
        result = landmarker.detect(mp_img)
    except Exception:
        return None

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    ph, pw = padded.shape[:2]

    best = None
    best_r = -1

    for iris_pts in (_RIGHT_IRIS, _LEFT_IRIS):
        if iris_pts[-1] >= len(lms):
            continue
        centre = lms[iris_pts[0]]
        cx_pad = centre.x * pw
        cy_pad = centre.y * ph

        # Radius: mean distance from centre to the 4 edge points
        radii = []
        for idx in iris_pts[1:]:
            ep = lms[idx]
            dx = (ep.x - centre.x) * pw
            dy = (ep.y - centre.y) * ph
            radii.append(math.sqrt(dx * dx + dy * dy))
        r_pad = float(np.mean(radii)) if radii else 0.0

        if r_pad < 15:
            continue

        cx_orig = cx_pad - pad_w
        cy_orig = cy_pad - pad_h

        dist_to_centre = abs(cx_orig - w / 2)
        if best is None or dist_to_centre < abs(best[0] - w / 2):
            best = (cx_orig, cy_orig, r_pad)
            best_r = r_pad

    if best is None:
        return None

    cx, cy, r = best
    confidence = result.face_landmarks[0][0].visibility if hasattr(result.face_landmarks[0][0], 'visibility') else 0.5
    return int(cx), int(cy), int(r), float(confidence)


def _detect_iris_hough(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cv2.GaussianBlur(gray, (5, 5), 0))

    h, w = gray.shape
    min_r = max(8,  int(min(h, w) * 0.05))         # ~11 px  — below this is pupil-only
    max_r = max(min_r + 1, int(min(h, w) * 0.14))  # ~31 px  — iris only; eye socket arcs are ~45 px

    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(min(h, w) / 4),
        param1=50, param2=25,          # lower threshold → more candidates
        minRadius=min_r, maxRadius=max_r,
    )

    if circles is None:
        return None

    circles_arr = np.round(circles[0]).astype(int)

    eyebrow_y = int(h * 0.28)
    iris_min_r = max(12, int(min(h, w) * 0.07)) 
    valid = [
        (int(cx), int(cy), int(r)) for cx, cy, r in circles_arr
        if cy >= eyebrow_y
        and abs(cx - w / 2) < w * 0.40  
        and r >= iris_min_r       
    ]

    if not valid:
        return None
    
    roi_blurred = cv2.GaussianBlur(gray[eyebrow_y:, :], (11, 11), 0)
    _, _, min_loc, _ = cv2.minMaxLoc(roi_blurred)
    prior_x = min_loc[0]
    prior_y = min_loc[1] + eyebrow_y   

    best, best_score = None, float("inf")
    for cx, cy, r in valid:
        roi_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        mean_val        = cv2.mean(gray, mask=roi_mask)[0]
        darkness_score  = mean_val / 255.0
        proximity_score = float(np.sqrt((cx - prior_x) ** 2 + (cy - prior_y) ** 2)) / w
        score = 0.6 * darkness_score + 0.4 * proximity_score
        if score < best_score:
            best_score, best = score, (cx, cy, r)

    return best

def mask_iris(
    image: np.ndarray,
    image_path: Optional[str] = None,
    skip_if_no_iris: bool = False,
) -> tuple:
   
    assert image.ndim == 3 and image.shape[2] == 3, \
        "image must be H×W×3 RGB uint8"
    h, w = image.shape[:2]

    mp_result = _detect_iris_mediapipe(image)
    if mp_result is not None:
        cx, cy, r, conf = mp_result
        if r > 0:
            masked = _fill_circle(image, cx, cy, r)
            meta = {
                "method": "mediapipe",
                "confidence": conf,
                "iris_center": (cx, cy),
                "iris_radius": r,
                "fallback": False,
            }
            return masked, meta

    hough_result = _detect_iris_hough(image)

    if hough_result is not None and skip_if_no_iris:
        cx, cy, r = hough_result
        _gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _roi  = np.zeros(_gray.shape, dtype=np.uint8)
        cv2.circle(_roi, (cx, cy), r, 255, -1)
        min_val = cv2.minMaxLoc(_gray, mask=_roi)[0]
        if min_val >= _PUPIL_MIN_THRESHOLD:
            hough_result = None   # no pupil detected so eye is likely closed

    if hough_result is not None:
        cx, cy, r = hough_result
        masked = _fill_circle(image, cx, cy, r)
        meta = {
            "method": "hough",
            "confidence": 0.0,
            "iris_center": (cx, cy),
            "iris_radius": r,
            "fallback": False,
        }
        return masked, meta

    if skip_if_no_iris:
        # Eye is closed or iris not visible — return original image untouched.
        meta = {
            "method": "skipped",
            "confidence": 0.0,
            "iris_center": (w // 2, h // 2),
            "iris_radius": 0,
            "fallback": False,
        }
        return image.copy(), meta

    _log_failure(image_path)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * _FALLBACK_FRACTION / 2)
    masked = _fill_circle(image, cx, cy, r)
    meta = {
        "method": "fallback",
        "confidence": 0.0,
        "iris_center": (cx, cy),
        "iris_radius": r,
        "fallback": True,
    }
    return masked, meta
