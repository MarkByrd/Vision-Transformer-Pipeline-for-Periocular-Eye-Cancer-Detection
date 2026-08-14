import cv2
import numpy as np
import argparse
import os


def detect_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    # Try on upper half first (eyes are in upper portion of face)
    upper = gray[:h // 2, :]

    for min_neighbors in [8, 6, 4]:
        eyes = eye_cascade.detectMultiScale(
            upper,
            scaleFactor=1.05,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        if len(eyes) >= 2:
            break

    if len(eyes) < 2:
        raise RuntimeError(
            f"Could not detect 2 eyes (found {len(eyes)}). "
            "Try a clearer frontal face image."
        )

    # Sort left-to-right, take the two most likely eyes
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    return img, eyes


def compute_measurements(img, eyes):
    h, w = img.shape[:2]

    e1x, e1y, e1w, e1h = eyes[0]  # image-left eye  (person's right)
    e2x, e2y, e2w, e2h = eyes[1]  # image-right eye (person's left)

    # Eye centres
    c1x, c1y = e1x + e1w // 2, e1y + e1h // 2
    c2x, c2y = e2x + e2w // 2, e2y + e2h // 2

    # Approximate canthi from bounding box edges
    right_outer = np.array([e1x,       c1y])
    right_inner = np.array([e1x + e1w, c1y])
    left_inner  = np.array([e2x,       c2y])
    left_outer  = np.array([e2x + e2w, c2y])

    # Palpebral fissure dimensions
    # Height estimated at 55% of bbox (Haar bbox includes brow region)
    fissure_ratio = 0.55
    PFW_right = float(e1w)
    PFW_left  = float(e2w)
    PFH_right = e1h * fissure_ratio
    PFH_left  = e2h * fissure_ratio

    # Intercanthal distance (inner canthus to inner canthus)
    ICD = np.linalg.norm(left_inner - right_inner)

    # Eye Aspect Ratio (height / width)
    EAR_right = PFH_right / PFW_right
    EAR_left  = PFH_left  / PFW_left

    # Orbital Symmetry Indices (1.0 = perfect symmetry)
    OSI_width  = min(PFW_right, PFW_left) / max(PFW_right, PFW_left)
    OSI_height = min(PFH_right, PFH_left) / max(PFH_right, PFH_left)
    OSI_EAR    = min(EAR_right, EAR_left) / max(EAR_right, EAR_left)

    # Facial Symmetry Index (inner canthi equidistance from midline)
    midline_x = (right_inner[0] + left_inner[0]) / 2.0
    r_to_mid  = abs(right_inner[0] - midline_x)
    l_to_mid  = abs(left_inner[0]  - midline_x)
    FSI = (min(r_to_mid, l_to_mid) / max(r_to_mid, l_to_mid)
           if max(r_to_mid, l_to_mid) > 0 else 1.0)

    # Scale normalisation: average human ICD = 32mm
    px_per_mm = 32.0 / ICD

    measurements = {
        "image_size":        (w, h),
        "px_per_mm":         px_per_mm,
        "ICD_px":            ICD,
        "ICD_mm":            32.0,  # normalised
        "PFW_right_px":      PFW_right,
        "PFW_left_px":       PFW_left,
        "PFW_right_mm":      PFW_right * px_per_mm,
        "PFW_left_mm":       PFW_left  * px_per_mm,
        "PFW_asym_px":       abs(PFW_right - PFW_left),
        "PFW_asym_mm":       abs(PFW_right - PFW_left) * px_per_mm,
        "PFH_right_px":      PFH_right,
        "PFH_left_px":       PFH_left,
        "PFH_right_mm":      PFH_right * px_per_mm,
        "PFH_left_mm":       PFH_left  * px_per_mm,
        "PFH_asym_px":       abs(PFH_right - PFH_left),
        "PFH_asym_mm":       abs(PFH_right - PFH_left) * px_per_mm,
        "EAR_right":         EAR_right,
        "EAR_left":          EAR_left,
        "EAR_asym":          abs(EAR_right - EAR_left),
        "OSI_width":         OSI_width,
        "OSI_height":        OSI_height,
        "OSI_EAR":           OSI_EAR,
        "OSI_avg":           (OSI_width + OSI_height + OSI_EAR) / 3,
        "FSI":               FSI,
        "midline_x":         midline_x,
        "r_to_mid_px":       r_to_mid,
        "l_to_mid_px":       l_to_mid,
        # Store for annotation
        "_eyes":             eyes,
        "_centres":          ((c1x, c1y), (c2x, c2y)),
        "_right_inner":      right_inner,
        "_left_inner":       left_inner,
    }
    return measurements


def print_report(measurements):
    """Print a formatted measurement report to console."""
    m = measurements

    print(f"\nImage: {m['image_size'][0]}x{m['image_size'][1]} px")
    print(f"Scale: {m['px_per_mm']:.4f} mm/px (ICD normalised to 32mm)\n")

    print("=" * 50)
    print("INTERCANTHAL DISTANCE")
    print("=" * 50)
    print(f"ICD: {m['ICD_px']:.1f} px  (~32.0 mm)")

    print("\n" + "=" * 50)
    print("PALPEBRAL FISSURE WIDTH")
    print("=" * 50)
    print(f"Right eye: {m['PFW_right_px']:.1f} px (~{m['PFW_right_mm']:.1f} mm)")
    print(f"Left eye: {m['PFW_left_px']:.1f} px (~{m['PFW_left_mm']:.1f} mm)")
    print(f"Asymmetry |R-L|:{m['PFW_asym_px']:.1f} px (~{m['PFW_asym_mm']:.1f} mm)")

    print("\n" + "=" * 50)
    print("PALPEBRAL FISSURE HEIGHT (estimated)")
    print("=" * 50)
    print(f"Right eye:{m['PFH_right_px']:.1f} px  (~{m['PFH_right_mm']:.1f} mm)")
    print(f"Left eye: {m['PFH_left_px']:.1f} px  (~{m['PFH_left_mm']:.1f} mm)")
    print(f"Asymmetry |R-L|:{m['PFH_asym_px']:.1f} px  (~{m['PFH_asym_mm']:.1f} mm)")

    print("\n" + "=" * 50)
    print("EYE ASPECT RATIO  (EAR = height / width)")
    print("=" * 50)
    print(f"Right EAR:{m['EAR_right']:.4f}")
    print(f"Left EAR:{m['EAR_left']:.4f}")
    print(f"|EAR_R - EAR_L|: {m['EAR_asym']:.4f}")
    if m['EAR_asym'] < 0.05:
        print("EAR verdict: Normal variation")
    elif m['EAR_asym'] < 0.10:
        print("EAR verdict:Mild asymmetry")
    else:
        print("EAR verdict: Notable asymmetry - FLAG")

    print("\n" + "=" * 50)
    print("ORBITAL SYMMETRY INDEX (1.0 = perfect)")
    print("=" * 50)
    print(f"OSI width:{m['OSI_width']:.4f}")
    print(f"OSI height: {m['OSI_height']:.4f}")
    print(f"OSI EAR: {m['OSI_EAR']:.4f}")
    print(f"OSI average: {m['OSI_avg']:.4f}")

    print("\n" + "=" * 50)
    print("FACIAL SYMMETRY INDEX")
    print("=" * 50)
    print(f"Right inner→midline:{m['r_to_mid_px']:.1f} px")
    print(f"Left inner→midline: {m['l_to_mid_px']:.1f} px")
    print(f"FSI (1.0 = perfect):{m['FSI']:.4f}")

    print("\n" + "=" * 50)
    print("Z2 BILATERAL SYMMETRY VERDICT")
    print("=" * 50)
    avg = m['OSI_avg']
    print(f"Average OSI:{avg:.4f}")
    if avg > 0.95:
        print(" Result: High symmetry - normal healthy range")
    elif avg > 0.85:
        print("Result: Mild asymmetry - within natural variation")
    else:
        print("Result: Notable asymmetry - FLAG for clinical review")
    print()


def annotate_image(img, measurements, save_path):
    """Draw landmarks, measurements and verdict onto image and save."""
    m   = measurements
    out = img.copy()

    eyes      = m['_eyes']
    c1, c2    = m['_centres']
    r_inner   = m['_right_inner'].astype(int)
    l_inner   = m['_left_inner'].astype(int)
    mid_x     = int(m['midline_x'])

    e1x, e1y, e1w, e1h = eyes[0]
    e2x, e2y, e2w, e2h = eyes[1]

    h = out.shape[0]

    # Eye bounding boxes
    cv2.rectangle(out, (e1x, e1y), (e1x+e1w, e1y+e1h), (0, 200, 0), 2)   # green
    cv2.rectangle(out, (e2x, e2y), (e2x+e2w, e2y+e2h), (0, 100, 255), 2) # orange

    # Eye centres
    cv2.circle(out, c1, 5, (0, 200, 0),   -1)
    cv2.circle(out, c2, 5, (0, 100, 255), -1)

    # Inner canthi
    cv2.circle(out, tuple(r_inner), 5, (255, 255, 0), -1)
    cv2.circle(out, tuple(l_inner), 5, (255, 255, 0), -1)

    # ICD line
    cv2.line(out, tuple(r_inner), tuple(l_inner), (0, 255, 255), 2)

    # Facial midline
    cv2.line(out, (mid_x, 40), (mid_x, h - 40), (255, 255, 0), 1)

    # Labels
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, out.shape[1] / 1800)

    cv2.putText(out, f"ICD:{m['ICD_px']:.0f}px",
                (mid_x - 40, c1[1] - 25), font, scale, (0, 255, 255), 1)
    cv2.putText(out, f"EAR:{m['EAR_right']:.3f}",
                (e1x, e1y - 8), font, scale, (0, 200, 0), 1)
    cv2.putText(out, f"EAR:{m['EAR_left']:.3f}",
                (e2x, e2y - 8), font, scale, (0, 100, 255), 1)

    verdict_colour = (0, 200, 0) if m['OSI_avg'] > 0.95 else \
                     (0, 165, 255) if m['OSI_avg'] > 0.85 else (0, 0, 255)
    cv2.putText(out, f"OSI:{m['OSI_avg']:.3f}",
                (mid_x - 35, c1[1] + 45), font, scale + 0.05, verdict_colour, 1)

    cv2.imwrite(save_path, out)
    print(f"Annotated image saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Periocular Geometric Measurement")
    parser.add_argument("--image", required=True, help="Path to face image")
    parser.add_argument("--save",  default=None,  help="Path to save annotated output")
    args = parser.parse_args()

    img, eyes = detect_eyes(args.image)
    measurements = compute_measurements(img, eyes)
    print_report(measurements)

    if args.save:
        annotate_image(img, measurements, args.save)
    else:
        # Auto-save next to input image
        base, ext = os.path.splitext(args.image)
        auto_save = base + "_measured" + ext
        annotate_image(img, measurements, auto_save)


if __name__ == "__main__":
    main()