
from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# MediaPipe landmark indices (FaceMesh/FaceLandmarker topology, 468/478 pts)
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # p1..p6
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]   # p1..p6
RIGHT_EYE_INNER_CANTHUS = 133
LEFT_EYE_INNER_CANTHUS = 362
RIGHT_EYE_OUTER_CANTHUS = 33
LEFT_EYE_OUTER_CANTHUS = 263
RIGHT_EYE_UPPER_LID = [160, 159, 158]
RIGHT_EYE_LOWER_LID = [144, 145, 153]
LEFT_EYE_UPPER_LID = [385, 386, 387]
LEFT_EYE_LOWER_LID = [380, 374, 373]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# Data structures
@dataclass
class EyeMetrics:
    side: str          # "left" or "right" (subject's anatomical side)
    method: str         # "mediapipe", "contour (iris-anchored)", or "contour (whole-half, coarse)"
    ear: float
    fissure_width_px: float
    fissure_height_px: float
    fissure_width_norm: float
    fissure_height_norm: float
    ear_points: list    # 6 (x, y) pixel points [p1..p6] used for EAR, full-image coords
    outer_canthus: tuple
    inner_canthus: tuple
    upper_lid_pt: tuple  # point used for max height (upper edge)
    lower_lid_pt: tuple  # point used for max height (lower edge)
    contour: Optional[np.ndarray] = None  # full-image-coords contour, contour method only


# Geometry helpers
def euclidean(p: tuple, q: tuple) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def compute_ear(points: list) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    points ordered [p1, p2, p3, p4, p5, p6].
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def compute_osi(left: EyeMetrics, right: EyeMetrics) -> float:
    """
    Ocular Symmetry Index (OSI) - one reasonable composite definition: mean
    of the normalized absolute differences across three paired, scale-
    invariant metrics (EAR, normalized fissure width, normalized fissure
    height). 0 = perfectly symmetric, larger = more asymmetric.

    NOTE: illustrative definition; swap in the thesis's exact OSI formula
    if it differs (e.g. different metric set/weights or normalization).
    """
    def rel_diff(a: float, b: float) -> float:
        denom = (abs(a) + abs(b)) / 2.0
        return abs(a - b) / denom if denom else 0.0

    ear_diff = rel_diff(left.ear, right.ear)
    width_diff = rel_diff(left.fissure_width_norm, right.fissure_width_norm)
    height_diff = rel_diff(left.fissure_height_norm, right.fissure_height_norm)
    return float(np.clip((ear_diff + width_diff + height_diff) / 3.0, 0.0, 1.0))


def flag_from_ear_diff(ear_diff: float) -> str:
    if ear_diff < 0.05:
        return "SYMMETRIC (normal range)"
    if ear_diff <= 0.10:
        return "BORDERLINE"
    return "ASYMMETRY FLAGGED - structural distortion detected"


# Tier 1: MediaPipe FaceLandmarker
def ensure_model(model_path: Path) -> Path:
    """Download the FaceLandmarker task model if it isn't present locally."""
    if model_path.exists():
        return model_path
    print(f"[info] Model not found at {model_path}, downloading...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"[info] Model downloaded to {model_path}")
    return model_path


def build_landmarker(model_path: Path) -> "mp_vision.FaceLandmarker":
    # model_asset_path must be a plain str - a Path object silently breaks
    # BaseOptions on some mediapipe builds.
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def landmark_to_px(landmark, width: int, height: int) -> tuple:
    return (landmark.x * width, landmark.y * height)


def try_mediapipe(image_bgr: np.ndarray, model_path: Path):
    """
    Attempt landmark-based extraction with MediaPipe FaceLandmarker.
    Returns (left_metrics, right_metrics) on success, or None if no usable
    face/landmarks were found (expected on tight eyes-only crops).
    """
    if not MEDIAPIPE_AVAILABLE:
        print("[warn] mediapipe not installed; skipping landmark tier.")
        return None

    try:
        model_path = ensure_model(model_path)
        landmarker = build_landmarker(model_path)
    except Exception as exc:
        print(f"[warn] Could not initialize MediaPipe FaceLandmarker ({exc}); "
              f"falling back to contour method.")
        return None

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    landmarker.close()

    if not result.face_landmarks:
        print("[info] MediaPipe found no face on this crop (expected for "
              "tight eyes-only images) - falling back to contour method.")
        return None

    landmarks = result.face_landmarks[0]
    needed = set(
        RIGHT_EYE_EAR + LEFT_EYE_EAR
        + [RIGHT_EYE_INNER_CANTHUS, LEFT_EYE_INNER_CANTHUS]
        + [RIGHT_EYE_OUTER_CANTHUS, LEFT_EYE_OUTER_CANTHUS]
        + RIGHT_EYE_UPPER_LID + RIGHT_EYE_LOWER_LID
        + LEFT_EYE_UPPER_LID + LEFT_EYE_LOWER_LID
    )
    if len(landmarks) <= max(needed):
        print(f"[warn] MediaPipe returned only {len(landmarks)} landmarks "
              f"(need indices up to {max(needed)}) - falling back to contour method.")
        return None

    # Guard against landmark.visibility being None (not always populated).
    for i in needed:
        _ = getattr(landmarks[i], "visibility", 1.0)

    right_inner_px = landmark_to_px(landmarks[RIGHT_EYE_INNER_CANTHUS], width, height)
    left_inner_px = landmark_to_px(landmarks[LEFT_EYE_INNER_CANTHUS], width, height)
    intercanthal_px = euclidean(right_inner_px, left_inner_px)
    if intercanthal_px == 0:
        print("[warn] Degenerate MediaPipe landmark geometry - falling back to contour method.")
        return None

    def extract(side, ear_idx, outer_idx, inner_idx, upper_idx, lower_idx):
        def pt(i):
            return landmark_to_px(landmarks[i], width, height)

        ear_points = [pt(i) for i in ear_idx]
        ear = compute_ear(ear_points)
        outer_canthus = pt(outer_idx)
        inner_canthus = pt(inner_idx)
        fissure_width_px = euclidean(outer_canthus, inner_canthus)

        upper_pts = [pt(i) for i in upper_idx]
        lower_pts = [pt(i) for i in lower_idx]
        best_h, best_pair = -1.0, (upper_pts[0], lower_pts[0])
        for up in upper_pts:
            for lo in lower_pts:
                h = euclidean(up, lo)
                if h > best_h:
                    best_h, best_pair = h, (up, lo)

        return EyeMetrics(
            side=side, method="mediapipe", ear=ear,
            fissure_width_px=fissure_width_px, fissure_height_px=best_h,
            fissure_width_norm=fissure_width_px / intercanthal_px,
            fissure_height_norm=best_h / intercanthal_px,
            ear_points=ear_points, outer_canthus=outer_canthus, inner_canthus=inner_canthus,
            upper_lid_pt=best_pair[0], lower_lid_pt=best_pair[1],
        )

    right_metrics = extract("right", RIGHT_EYE_EAR, RIGHT_EYE_OUTER_CANTHUS,
                             RIGHT_EYE_INNER_CANTHUS, RIGHT_EYE_UPPER_LID, RIGHT_EYE_LOWER_LID)
    left_metrics = extract("left", LEFT_EYE_EAR, LEFT_EYE_OUTER_CANTHUS,
                            LEFT_EYE_INNER_CANTHUS, LEFT_EYE_UPPER_LID, LEFT_EYE_LOWER_LID)
    print("[info] MediaPipe FaceLandmarker succeeded on this crop.")
    return left_metrics, right_metrics


# Tier 2: classical CV contour fallback (for eyes-only crops with no nose/mouth)
IRIS_ROI_HALF_WIDTH_SCALE = 3.2   # ROI half-width as a multiple of iris radius
IRIS_ROI_HALF_HEIGHT_SCALE = 2.0  # ROI half-height as a multiple of iris radius


def detect_iris(half_gray: np.ndarray) -> Optional[tuple]:
    """Locate the iris in one eye's half-crop via Hough circle detection.
    Returns (cx, cy, r) in half-local pixel coordinates, or None."""
    hh, hw = half_gray.shape[:2]
    blurred = cv2.medianBlur(half_gray, 5)
    min_r = max(3, int(0.04 * min(hh, hw)))
    max_r = max(min_r + 1, int(0.22 * min(hh, hw)))

    # Progressively relax the accumulator threshold until a circle is found.
    for param2 in (35, 30, 25, 20, 15):
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=hw,
            param1=80, param2=param2, minRadius=min_r, maxRadius=max_r,
        )
        if circles is not None and len(circles[0]) > 0:
            cx, cy, r = circles[0][0]
            return float(cx), float(cy), float(r)
    return None


def find_eye_contour_iris_anchored(half_bgr: np.ndarray, iris: tuple) -> Optional[np.ndarray]:
    """
    Segment the palpebral fissure within a small ROI centered on the
    already-detected iris, using saturation thresholding (skin is markedly
    more saturated than sclera/iris). Returns a contour in half-local
    coordinates (already offset back from ROI-local), or None.
    """
    hh, hw = half_bgr.shape[:2]
    cx, cy, r = iris
    rx, ry = int(r * IRIS_ROI_HALF_WIDTH_SCALE), int(r * IRIS_ROI_HALF_HEIGHT_SCALE)
    x0, x1 = max(0, int(cx - rx)), min(hw, int(cx + rx))
    y0, y1 = max(0, int(cy - ry)), min(hh, int(cy + ry))
    roi = half_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    smooth = cv2.bilateralFilter(roi, d=0, sigmaColor=50, sigmaSpace=15)
    saturation = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Prefer whichever contour actually contains the iris center; otherwise
    # fall back to the largest contour in the ROI.
    iris_local = (cx - x0, cy - y0)
    containing = [c for c in contours if cv2.pointPolygonTest(c, iris_local, False) >= 0]
    best = max(containing, key=cv2.contourArea) if containing else max(contours, key=cv2.contourArea)

    offset = np.array([x0, y0])
    result = (best.reshape(-1, 2) + offset).reshape(-1, 1, 2).astype(np.int32)

    # Reject implausibly small contours that are really just the iris/pupil,
    # not the palpebral fissure. This happens whenever there isn't enough
    # saturation contrast between sclera and skin for the threshold above to
    # expand past the iris - e.g. bloodshot sclera, crusting/discharge on
    # the affected eye, or simply a narrow eye opening with little visible
    # sclera. A real fissure opening is wider than the iris that sits
    # inside it, so use that as a plausibility gate and let the caller fall
    # back to a method that doesn't depend on sclera visibility.
    _, _, bw, bh = cv2.boundingRect(result)
    if bw < 1.3 * (2 * r):
        return None

    return result


LASH_LINE_ROI_HALF_HEIGHT_SCALE = 0.9  # deliberately tighter than IRIS_ROI_HALF_HEIGHT_SCALE


def find_eye_contour_lash_line(half_bgr: np.ndarray, iris: tuple) -> Optional[np.ndarray]:
    """
    Fallback for when find_eye_contour_iris_anchored's sclera-ring
    assumption breaks down and its result collapses to roughly the iris
    itself. Thresholds on brightness (V channel) instead of saturation -
    the iris, pupil, and lash line are all notably darker than surrounding
    eyelid skin regardless of how much sclera is visible - within a
    vertically tight ROI around the iris.

    The tight vertical window (LASH_LINE_ROI_HALF_HEIGHT_SCALE, vs. the
    wider IRIS_ROI_HALF_HEIGHT_SCALE used above) is deliberate: brightness
    thresholding with the wider window reliably bridges into the eyebrow
    shadow above and the tear-trough shadow below, merging them with the
    eye opening into one oversized blob. Returns a contour in half-local
    coordinates, or None.
    """
    hh, hw = half_bgr.shape[:2]
    cx, cy, r = iris
    rx, ry = int(r * IRIS_ROI_HALF_WIDTH_SCALE), int(r * LASH_LINE_ROI_HALF_HEIGHT_SCALE)
    x0, x1 = max(0, int(cx - rx)), min(hw, int(cx + rx))
    y0, y1 = max(0, int(cy - ry)), min(hh, int(cy + ry))
    roi = half_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    smooth = cv2.bilateralFilter(roi, d=0, sigmaColor=50, sigmaSpace=15)
    value = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)[:, :, 2]
    _, mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    iris_local = (cx - x0, cy - y0)
    containing = [c for c in contours if cv2.pointPolygonTest(c, iris_local, False) >= 0]
    if not containing:
        return None
    best = max(containing, key=cv2.contourArea)

    bx, by, bw, bh = cv2.boundingRect(best)
    # Reject results that still bled out to the ROI edges (eyebrow/cheek
    # shadow merged in) or aren't eye-shaped - same reasoning as the
    # plausibility gate above, applied to this method's own failure mode.
    touched_edges = (bx == 0) + (by == 0) + (bx + bw >= roi.shape[1]) + (by + bh >= roi.shape[0])
    aspect = bw / max(bh, 1)
    if touched_edges >= 2 or not (1.2 <= aspect <= 4.0):
        return None

    offset = np.array([x0, y0])
    return (best.reshape(-1, 2) + offset).reshape(-1, 1, 2).astype(np.int32)


def find_eye_contour(half_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Last-resort fallback when iris detection fails: blind whole-half
    edge search. Coarser and more prone to false positives on textured
    skin, but better than nothing."""
    gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h_img, w_img = gray.shape
    min_area = 0.005 * h_img * w_img

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / h
        cy = y + h / 2.0
        # Reject contours dominated by eyebrow (too high) or cheek (too low);
        # keep eye-like aspect ratios (wide ellipse, not round or sliver).
        if not (0.15 * h_img < cy < 0.90 * h_img):
            continue
        if not (1.2 <= aspect <= 6.0):
            continue
        candidates.append((area, c))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def locate_eye_contour(half_bgr: np.ndarray) -> tuple:
    """Try the iris-anchored method first, then lash-line tracing (for
    narrow/swollen eyes with no visible sclera ring), then the blind
    whole-half fallback. Returns (contour_or_None, method_str)."""
    gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)
    iris = detect_iris(gray)
    if iris is not None:
        contour = find_eye_contour_iris_anchored(half_bgr, iris)
        if contour is not None:
            return contour, "contour (iris-anchored)"

        contour = find_eye_contour_lash_line(half_bgr, iris)
        if contour is not None:
            return contour, "contour (lash-line)"

    contour = find_eye_contour(half_bgr)
    if contour is not None:
        return contour, "contour (whole-half, coarse)"
    return None, "contour (failed)"


def eye_metrics_from_contour(
    half_bgr: np.ndarray,
    contour: np.ndarray,
    offset: tuple,
    side: str,
    inner_side: str,  # "left" or "right" edge of the contour that faces the nose/midline
    method: str,
) -> EyeMetrics:
    """
    Sample a segmented eye-opening contour the same way the 6-point EAR
    formula samples real landmarks: horizontal span (p1-p4) plus two
    vertical thickness measurements at the 1/3 and 2/3 x-positions
    (p2-p6, p3-p5).
    """
    mask = np.zeros(half_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(contour)
    pts = contour.reshape(-1, 2)
    leftmost = tuple(pts[pts[:, 0].argmin()])
    rightmost = tuple(pts[pts[:, 0].argmax()])

    def col_extent(x_col: int):
        x_col = int(np.clip(x_col, 0, mask.shape[1] - 1))
        ys = np.where(mask[:, x_col] > 0)[0]
        if ys.size == 0:
            return None
        return (x_col, int(ys.min())), (x_col, int(ys.max()))

    ext1 = col_extent(x + w // 3)
    ext2 = col_extent(x + 2 * w // 3)
    fallback = col_extent(x + w // 2) or ((x + w // 2, y), (x + w // 2, y + h))
    ext1 = ext1 or fallback
    ext2 = ext2 or fallback

    p1, p4 = leftmost, rightmost
    p2, p6 = ext1
    p3, p5 = ext2

    def off(pt):
        return (pt[0] + offset[0], pt[1] + offset[1])

    p1, p2, p3, p4, p5, p6 = (off(p) for p in (p1, p2, p3, p4, p5, p6))
    ear_points = [p1, p2, p3, p4, p5, p6]
    ear = compute_ear(ear_points)

    outer_canthus, inner_canthus = (p1, p4) if inner_side == "right" else (p4, p1)
    fissure_width_px = euclidean(p1, p4)

    h1, h2 = euclidean(p2, p6), euclidean(p3, p5)
    if h1 >= h2:
        fissure_height_px, upper_lid_pt, lower_lid_pt = h1, p2, p6
    else:
        fissure_height_px, upper_lid_pt, lower_lid_pt = h2, p3, p5

    offset_contour = contour.reshape(-1, 2) + np.array(offset)

    return EyeMetrics(
        side=side, method=method, ear=ear,
        fissure_width_px=fissure_width_px, fissure_height_px=fissure_height_px,
        fissure_width_norm=0.0, fissure_height_norm=0.0,  # normalized after intercanthal is known
        ear_points=ear_points, outer_canthus=outer_canthus, inner_canthus=inner_canthus,
        upper_lid_pt=upper_lid_pt, lower_lid_pt=lower_lid_pt,
        contour=offset_contour.reshape(-1, 1, 2).astype(np.int32),
    )


def try_contour(image_bgr: np.ndarray, split_x: Optional[int], swap_sides: bool):
    """
    Classical CV fallback for eyes-only crops. Splits the image at split_x
    (default: horizontal midpoint) into a "right eye" half and "left eye"
    half, per the front-facing-photo convention documented in the module
    docstring (overridable with swap_sides).
    """
    height, width = image_bgr.shape[:2]
    split_x = split_x if split_x is not None else width // 2

    right_half = image_bgr[:, :split_x]   # subject's right eye (viewer's left)
    left_half = image_bgr[:, split_x:]    # subject's left eye (viewer's right)
    right_offset = (0, 0)
    left_offset = (split_x, 0)

    if swap_sides:
        right_half, left_half = left_half, right_half
        right_offset, left_offset = left_offset, right_offset

    right_contour, right_method = locate_eye_contour(right_half)
    left_contour, left_method = locate_eye_contour(left_half)
    if right_contour is None or left_contour is None:
        return None

    # For the right eye (left side of image) the nose/midline is to its
    # right (inner_side="right"); mirror for the left eye.
    right_metrics = eye_metrics_from_contour(right_half, right_contour, right_offset, "right",
                                              inner_side="right", method=right_method)
    left_metrics = eye_metrics_from_contour(left_half, left_contour, left_offset, "left",
                                             inner_side="left", method=left_method)

    intercanthal_px = euclidean(right_metrics.inner_canthus, left_metrics.inner_canthus)
    if intercanthal_px == 0:
        return None

    right_metrics.fissure_width_norm = right_metrics.fissure_width_px / intercanthal_px
    right_metrics.fissure_height_norm = right_metrics.fissure_height_px / intercanthal_px
    left_metrics.fissure_width_norm = left_metrics.fissure_width_px / intercanthal_px
    left_metrics.fissure_height_norm = left_metrics.fissure_height_px / intercanthal_px

    print(f"[info] Contour fallback used - right eye: {right_method}, left eye: {left_method}.")
    return left_metrics, right_metrics


# Visualization
def draw_eye_annotations(image: np.ndarray, metrics: EyeMetrics, color: tuple, label: str) -> None:
    if metrics.contour is not None:
        cv2.drawContours(image, [metrics.contour], -1, color, 1, lineType=cv2.LINE_AA)

    for p in metrics.ear_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 2, color, -1, lineType=cv2.LINE_AA)

    cv2.line(image, tuple(map(int, metrics.outer_canthus)), tuple(map(int, metrics.inner_canthus)),
              color, 1, lineType=cv2.LINE_AA)
    cv2.line(image, tuple(map(int, metrics.upper_lid_pt)), tuple(map(int, metrics.lower_lid_pt)),
              color, 1, lineType=cv2.LINE_AA)

    text = f"{label} EAR={metrics.ear:.3f}"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

    img_h, img_w = image.shape[:2]
    margin = 4
    x = int(metrics.outer_canthus[0]) - 10
    # Anchor rightward-overflowing labels by their right edge instead of
    # their left edge, and clamp so the string never runs off either side.
    if x + text_w > img_w - margin:
        x = int(metrics.outer_canthus[0]) - text_w + 10
    x = max(margin, min(x, img_w - text_w - margin))
    y = max(text_h + margin, int(metrics.outer_canthus[1]) - 15)
    text_origin = (x, y)
    cv2.putText(image, text, text_origin,
                font, scale, color, thickness, lineType=cv2.LINE_AA)


def annotate_image(image: np.ndarray, left: EyeMetrics, right: EyeMetrics, affected_side: str) -> np.ndarray:
    annotated = image.copy()
    red, green = (0, 0, 255), (0, 200, 0)
    left_color = red if affected_side == "left" else green
    right_color = red if affected_side == "right" else green
    left_label = "L (affected)" if affected_side == "left" else "L (normal)"
    right_label = "R (affected)" if affected_side == "right" else "R (normal)"

    draw_eye_annotations(annotated, left, left_color, left_label)
    draw_eye_annotations(annotated, right, right_color, right_label)

    cv2.line(annotated, tuple(map(int, right.inner_canthus)), tuple(map(int, left.inner_canthus)),
              (255, 255, 0), 1, lineType=cv2.LINE_AA)

    method_tag = (f"method: {left.method}" if left.method == right.method
                  else f"L: {left.method} | R: {right.method}")
    cv2.putText(annotated, method_tag, (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return annotated


# Console report
def print_comparison_table(left: EyeMetrics, right: EyeMetrics, intercanthal_px: float,
                             osi: float, ear_diff: float, flag: str) -> None:
    rows = [
        ("EAR", f"{left.ear:.4f}", f"{right.ear:.4f}"),
        ("Fissure width (norm.)", f"{left.fissure_width_norm:.4f}", f"{right.fissure_width_norm:.4f}"),
        ("Fissure height (norm.)", f"{left.fissure_height_norm:.4f}", f"{right.fissure_height_norm:.4f}"),
        ("Fissure width (px)", f"{left.fissure_width_px:.1f}", f"{right.fissure_width_px:.1f}"),
        ("Fissure height (px)", f"{left.fissure_height_px:.1f}", f"{right.fissure_height_px:.1f}"),
    ]
    col1_w = max(len(r[0]) for r in rows) + 2
    col_w = 14

    print("\n" + "=" * (col1_w + 2 * col_w + 2))
    print("BILATERAL PERIOCULAR SYMMETRY ANALYSIS (illustrative)")
    if left.method == right.method:
        print(f"Landmark method: {left.method}")
    else:
        print(f"Landmark method: L={left.method}, R={right.method}")
    print("=" * (col1_w + 2 * col_w + 2))
    print(f"{'Metric':<{col1_w}}{'Left eye':>{col_w}}{'Right eye':>{col_w}}")
    print("-" * (col1_w + 2 * col_w + 2))
    for name, lval, rval in rows:
        print(f"{name:<{col1_w}}{lval:>{col_w}}{rval:>{col_w}}")
    print("-" * (col1_w + 2 * col_w + 2))
    print(f"Intercanthal distance (px): {intercanthal_px:.1f}")
    print(f"Bilateral EAR difference:   {ear_diff:.4f}")
    print(f"Ocular Symmetry Index (OSI): {osi:.4f}  (0 = perfectly symmetric)")
    print(f"Verdict: {flag}")
    print("=" * (col1_w + 2 * col_w + 2) + "\n")


# Main pipeline
def run(image_path: Path, output_path: Path, model_path: Path,
        method: str, split_x: Optional[int], swap_sides: bool,
        affected_side_arg: str) -> None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Could not read image at: {image_path}")

    result = None
    if method in ("auto", "mediapipe"):
        result = try_mediapipe(image_bgr, model_path)
        if result is None and method == "mediapipe":
            raise SystemExit(
                "MediaPipe FaceLandmarker found no usable face/landmarks on this "
                "image. Eyes-only crops often fail full-face detection - try "
                "--method contour instead."
            )

    if result is None and method in ("auto", "contour"):
        result = try_contour(image_bgr, split_x, swap_sides)

    if result is None:
        raise SystemExit(
            "Could not extract eye metrics with either method. For eyes-only "
            "crops, ensure both eyes are clearly visible with sufficient "
            "contrast between the eye opening and surrounding skin, and "
            "consider passing --split-x to correct the left/right split point."
        )

    left_metrics, right_metrics = result

    ear_diff = abs(left_metrics.ear - right_metrics.ear)
    osi = compute_osi(left_metrics, right_metrics)
    flag = flag_from_ear_diff(ear_diff)
    intercanthal_px = euclidean(right_metrics.inner_canthus, left_metrics.inner_canthus)

    if affected_side_arg == "auto":
        # Heuristic only: assumes a lesion restricts lid opening (lower
        # EAR). Real photos don't always match this - e.g. crusting/
        # discharge at the lesion can inflate the segmented contour and
        # raise EAR instead. Prefer --affected-side left/right when you
        # already know which eye is affected.
        affected_side = "left" if left_metrics.ear < right_metrics.ear else "right"
        print(f"[info] Affected side auto-detected as '{affected_side}' (lower EAR) - "
              f"verify against the photo, or pass --affected-side to override.")
    else:
        affected_side = affected_side_arg

    print_comparison_table(left_metrics, right_metrics, intercanthal_px, osi, ear_diff, flag)

    annotated = annotate_image(image_bgr, left_metrics, right_metrics, affected_side)
    if not cv2.imwrite(str(output_path), annotated):
        raise SystemExit(f"Failed to write annotated image to: {output_path}")
    print(f"[info] Annotated figure saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Illustrative bilateral periocular symmetry analysis on a single "
            "image containing both eyes (not a full face). Conceptual "
            "methodology figure, not a clinical result."
        )
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to the periocular crop (both eyes).")
    parser.add_argument("--output", type=Path, default=Path("annotated_symmetry.png"),
                         help="Path to save the annotated output image (default: annotated_symmetry.png).")
    parser.add_argument("--model", type=Path, default=Path("face_landmarker.task"),
                         help="Path to the MediaPipe face_landmarker.task model (downloaded if missing).")
    parser.add_argument("--method", choices=["auto", "mediapipe", "contour"], default="auto",
                         help="Landmark strategy: auto tries MediaPipe then falls back to contour "
                              "detection (default); force one tier with mediapipe/contour.")
    parser.add_argument("--split-x", type=int, default=None,
                         help="Pixel x-coordinate to split left/right eye halves for the contour "
                              "method (default: image midpoint).")
    parser.add_argument("--swap-sides", action="store_true",
                         help="Swap which half of the image is treated as the subject's left vs "
                              "right eye (use if the crop is mirrored).")
    parser.add_argument("--affected-side", choices=["auto", "left", "right"], default="auto",
                         help="Which eye to label as affected in the annotated figure. Default "
                              "'auto' guesses from the lower EAR, which can be wrong on real "
                              "photos (e.g. lesion crusting can inflate rather than shrink the "
                              "segmented opening) - pass left/right explicitly when known.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Input image not found: {args.image}")
    run(args.image, args.output, args.model, args.method, args.split_x, args.swap_sides,
        args.affected_side)


if __name__ == "__main__":
    main()
