"""
Geometric periocular landmark analysis (pure OpenCV, no trained model).

Restored from the previous frontend/app.py. Works on a 224x224 BGR image and
returns (annotated_bgr, measurements); all measurements are in pixels of that
224x224 image.
"""

import math

import cv2
import numpy as np


def run_landmark_analysis(image_bgr: np.ndarray):
    """
    Detects periocular landmarks on a 224×224 BGR image using pure OpenCV:
      • Iris / pupil  — Hough Circle Transform (picks darkest circle)
      • Canthi        — estimated at ±2.2× iris radius from pupil centre
      • Eyelid margins — dominant horizontal Canny edge in upper/lower halves
      • Lesion region  — largest dark blob in the upper 60 % of the image

    Returns (annotated_bgr, measurements_dict).
    All pixel measurements are in the 224×224 coordinate space.
    """
    h, w = image_bgr.shape[:2]
    gray      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    annotated = image_bgr.copy()
    coords    = {}
    meas      = {}

    # ── Iris via Hough circles ────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=50, param1=50, param2=30,
        minRadius=20, maxRadius=90,
    )
    if circles is not None:
        best, best_val = None, 999.0
        for cx, cy, r in np.round(circles[0]).astype(int):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            val = cv2.mean(gray, mask=mask)[0]
            if val < best_val:
                best_val, best = val, (int(cx), int(cy), int(r))
        if best:
            cx, cy, r = best
            coords["pupil_centre"] = (cx, cy)
            meas["iris_radius"] = r
            cv2.circle(annotated, (cx, cy), r, (0, 215, 255), 2)
            cv2.circle(annotated, (cx, cy), 3, (0, 255,   0), -1)
            cv2.putText(annotated, f"iris r={r}",
                        (max(0, cx - r), max(10, cy - r - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 215, 255), 1)
            mc = (max(0,   cx - int(r * 2.2)), cy)
            lc = (min(w-1, cx + int(r * 2.2)), cy)
            coords["medial_canthus"]  = mc
            coords["lateral_canthus"] = lc
            for label, pt, col in [("med", mc, (0, 80, 255)), ("lat", lc, (0, 80, 255))]:
                cv2.circle(annotated, pt, 5, col, -1)
                cv2.putText(annotated, label, (pt[0]+4, pt[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

    # ── Eyelid margins via Canny ──────────────────────────────────────────────
    def find_lid(roi, is_upper, offset_y=0):
        edges = cv2.Canny(roi, 30, 80)
        edges = cv2.dilate(edges, np.ones((1, 20), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        pts = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        idx = np.argmax(pts[:, 1]) if is_upper else np.argmin(pts[:, 1])
        return (int(pts[idx][0]), int(pts[idx][1] + offset_y))

    upper_lid = find_lid(gray[0:h//2, :],   is_upper=True,  offset_y=0)
    lower_lid = find_lid(gray[h//2:h, :],   is_upper=False, offset_y=h//2)

    lid_col = (255, 120, 0)
    if upper_lid:
        coords["upper_lid"] = upper_lid
        cv2.circle(annotated, upper_lid, 5, lid_col, -1)
        cv2.putText(annotated, "upper", (upper_lid[0]+4, upper_lid[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, lid_col, 1)
    if lower_lid:
        coords["lower_lid"] = lower_lid
        cv2.circle(annotated, lower_lid, 5, lid_col, -1)
        cv2.putText(annotated, "lower", (lower_lid[0]+4, lower_lid[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, lid_col, 1)
    if upper_lid and lower_lid:
        cv2.line(annotated, upper_lid, lower_lid, lid_col, 1)

    # ── Lesion region via dark-blob thresholding (upper 60 %) ────────────────
    upper_region = image_bgr[0:int(h * 0.6), :]
    gray_upper   = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
    _, thresh    = cv2.threshold(gray_upper, 90, 255, cv2.THRESH_BINARY_INV)
    thresh       = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _      = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        lx, ly, lw2, lh2 = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        coords["lesion_tl"]     = (lx, ly)
        coords["lesion_br"]     = (lx + lw2, ly + lh2)
        coords["lesion_centre"] = (lx + lw2 // 2, ly + lh2 // 2)
        meas["lesion_width"]    = int(lw2)
        meas["lesion_height"]   = int(lh2)
        cv2.rectangle(annotated, (lx, ly), (lx + lw2, ly + lh2), (0, 50, 255), 2)
        cv2.putText(annotated, f"LESION {lw2}x{lh2}",
                    (lx, max(8, ly - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 50, 255), 1)

    # ── Distance metrics ──────────────────────────────────────────────────────
    def dist(p1, p2):
        return round(math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2), 1)

    if upper_lid and lower_lid:
        meas["palpebral_fissure_height"] = dist(upper_lid, lower_lid)
    if "medial_canthus" in coords and "lateral_canthus" in coords:
        meas["palpebral_fissure_length"] = dist(
            coords["medial_canthus"], coords["lateral_canthus"])
    if "palpebral_fissure_height" in meas and "palpebral_fissure_length" in meas:
        meas["aspect_ratio"] = round(
            meas["palpebral_fissure_height"] / (meas["palpebral_fissure_length"] + 1e-6), 3)
    if "lesion_centre" in coords and "pupil_centre" in coords:
        meas["lesion_to_pupil_dist"] = dist(
            coords["lesion_centre"], coords["pupil_centre"])

    return annotated, meas
