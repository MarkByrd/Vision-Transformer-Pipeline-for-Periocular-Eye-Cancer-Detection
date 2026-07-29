import argparse
import random
import sys
import threading
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from iris_masking import mask_iris

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_BASE     = Path(__file__).parent.parent   # ViT-implementation/

SOURCES = {
    "periocular_normal": [
        _BASE / "periculor_dataset" / "LEFT",
        _BASE / "periculor_dataset" / "RIGHT",
    ],
    "eyelid_cancer": [
        _BASE / "Eyelid_Cancer_Dataset",
    ],
}


def collect_images(dirs: list) -> list:
    paths = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  [skip] {d} does not exist", flush=True)
            continue
        for item in d.iterdir():
            if item.is_dir():
                paths.extend(p for p in item.iterdir()
                             if p.suffix.lower() in VALID_EXT)
            elif item.suffix.lower() in VALID_EXT:
                paths.append(item)
    return paths


def sample(paths: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


def mask_with_timeout(img_rgb, image_path, timeout_sec, skip_if_no_iris=False):
    result = [None]
    exc    = [None]

    def _run():
        try:
            result[0] = mask_iris(img_rgb, image_path=image_path,
                                  skip_if_no_iris=skip_if_no_iris)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if result[0] is not None:
        return result[0]

    # Timed out or errored - run Hough directly (no brightness gate on timeout path)
    from iris_masking import _detect_iris_hough, _fill_circle, _log_failure, _FALLBACK_FRACTION
    hough = _detect_iris_hough(img_rgb)
    if hough:
        cx, cy, r = hough
        masked = _fill_circle(img_rgb, cx, cy, r)
        return masked, {"method": "hough", "confidence": 0.0,
                        "iris_center": (cx, cy), "iris_radius": r, "fallback": False}

    if skip_if_no_iris:
        h, w = img_rgb.shape[:2]
        return img_rgb.copy(), {"method": "skipped", "confidence": 0.0,
                                "iris_center": (w // 2, h // 2), "iris_radius": 0,
                                "fallback": False}

    _log_failure(image_path)
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * _FALLBACK_FRACTION / 2)
    masked = _fill_circle(img_rgb, cx, cy, r)
    return masked, {"method": "fallback", "confidence": 0.0,
                    "iris_center": (cx, cy), "iris_radius": r, "fallback": True}


def save_panel(orig_rgb, masked_rgb, meta, fname, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_rgb);  axes[0].set_title("Original", fontsize=12);  axes[0].axis("off")
    axes[1].imshow(masked_rgb)
    if meta["method"] == "skipped":
        axes[1].set_title("Original (eye closed — skipped)", fontsize=10)
    else:
        axes[1].set_title(
            f"Masked  [{meta['method']}]\n"
            f"centre={meta['iris_center']}  r={meta['iris_radius']}px",
            fontsize=10,
        )
    axes[1].axis("off")
    fig.suptitle(fname, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_source(source_name, dirs, n, seed, preview_dir, timeout_mp):
    print(f"\n{'='*60}", flush=True)
    print(f"  Source: {source_name}", flush=True)
    print(f"{'='*60}", flush=True)

    all_paths = collect_images(dirs)
    if not all_paths:
        print(f"  ERROR: no images found", flush=True)
        return {"source": source_name, "total": 0, "detected": 0,
                "success_rate": 0, "method_counts": {}}

    sampled = sample(all_paths, n, seed)
    print(f"  Pool: {len(all_paths)} images — sampling {len(sampled)}", flush=True)

    method_counts = Counter()

    for i, img_path in enumerate(sampled):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            method_counts["load_fail"] += 1
            print(f"  [{i+1:>2}/{len(sampled)}] LOAD FAIL: {img_path.name}", flush=True)
            continue

        # Resize to 224x224 — matching the actual inference pipeline
        rgb_224 = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)

        skip = (source_name == "eyelid_cancer")
        masked_rgb, meta = mask_with_timeout(rgb_224, str(img_path), timeout_mp,
                                             skip_if_no_iris=skip)
        method_counts[meta["method"]] += 1

        tag = {"mediapipe": "[MP]", "hough": "[H] ",
               "fallback": "[FB]", "skipped": "[SK]"}.get(meta["method"], "[??]")
        print(f"  [{i+1:>2}/{len(sampled)}] {tag} {img_path.name:40s}  "
              f"r={meta['iris_radius']:>3}px  c={meta['iris_center']}", flush=True)

        out_name = f"{source_name}_{i+1:02d}_{img_path.stem}.png"
        save_panel(rgb_224, masked_rgb, meta, img_path.name, preview_dir / out_name)

    total    = len(sampled)
    # skipped = intentional (closed eye) — counts as handled, not a failure
    failed   = method_counts.get("fallback", 0) + method_counts.get("load_fail", 0)
    detected = total - failed
    rate     = detected / total * 100 if total else 0

    print(f"\n  --- {source_name} results ---", flush=True)
    for m, c in sorted(method_counts.items()):
        print(f"    {m:<12}: {c}", flush=True)
    print(f"  Success (non-fallback): {detected}/{total} = {rate:.1f}%", flush=True)

    return {"source": source_name, "total": total, "detected": detected,
            "success_rate": rate, "method_counts": dict(method_counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=30)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--timeout_mp", type=float, default=5.0,
                        help="Seconds before abandoning MediaPipe and using Hough")
    args = parser.parse_args()

    preview_dir = Path(__file__).parent / "diagnostics" / "masking_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for source_name, dirs in SOURCES.items():
        r = run_source(source_name, dirs, args.n, args.seed, preview_dir, args.timeout_mp)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("  OVERALL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    report_lines = ["IRIS MASKING DIAGNOSTIC REPORT", "=" * 60, ""]
    all_ok = True

    for r in results:
        if r["total"] == 0:
            continue
        methods_str = ", ".join(f"{k}:{v}" for k, v in r["method_counts"].items())
        line = (f"  {r['source']:<28}: {r['detected']:>3}/{r['total']:>3} "
                f"= {r['success_rate']:5.1f}%  [{methods_str}]")
        print(line, flush=True)
        report_lines.append(line)
        if r["success_rate"] < 70.0:
            warn = "  *** BELOW 70% — fix detection before retraining ***"
            print(warn, flush=True)
            report_lines.append(warn)
            all_ok = False

    verdict = "PROCEED — detection looks good" if all_ok else "STOP — fix detection first"
    report_lines += [
        "",
        f"Before/after previews: diagnostics/masking_preview/",
        f"VERDICT: {verdict}",
    ]

    report_path = Path(__file__).parent / "diagnostics" / "masking_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n  Report : {report_path}", flush=True)
    print(f"  Preview: {preview_dir}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
