"""
Unified Eyelid Cancer Classification Backend
Serves the shared frontend and routes predictions to whichever model the user selects.

Models loaded at startup:
  jay_cnn          — ResNet-18 CNN  (backend_jay)
  mark_cnn         — ResNet-152 CNN (backend_mark_system)
  aparna_vit       — ViT-B/16 MAE unmasked (backend_Aparna_ViT) — ablation baseline
  aparna_vit_masked — ViT-B/16 MAE iris-masked (ViTbackend)  — improved model

POST /predict  { file, model_type: "jay_cnn"|"mark_cnn"|"aparna_vit"|"aparna_vit_masked" }
Response       { prediction, confidence, model, original_image, heatmap | null }
"""

import base64
import io
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

# ── Directory layout ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent        # .../frontend/
REPO_DIR = BASE_DIR.parent             # .../ViT-implementation/
STATIC_DIR = BASE_DIR / "static"

APARNA_SRC    = REPO_DIR / "backend_Aparna_ViT"
NEW_VIT_SRC   = REPO_DIR / "ViTbackend"
JAY_MODEL     = REPO_DIR / "backend_jay" / "Pipeline" / "GrabCut" / "saved_models" / "cnn_b.pth"
MARK_MODEL    = REPO_DIR / "backend_mark_system" / "model" / "final_model.pth"
APARNA_MODEL  = APARNA_SRC / "outputs" / "aspect_fix_only_no_iris_mask" / "best_classifier.pt"
MASKED_MODEL  = NEW_VIT_SRC / "outputs" / "vit_finetuned_masked.pth"

# Shared architecture classes come from backend_Aparna_ViT
sys.path.insert(0, str(APARNA_SRC))
from finetune import FineTuneModel, MLPHead          # noqa: E402
from gmar import ViTGMAR, overlay_cam_on_image       # noqa: E402

# iris_masking lives in ViTbackend
sys.path.insert(0, str(NEW_VIT_SRC))
from iris_masking import mask_iris                   # noqa: E402

# ── Device ──────────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Benign", "Malignant"]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

print(f"\n{'='*55}")
print(f"  Unified Eyelid Cancer Backend")
print(f"  Device : {DEVICE}")
print(f"{'='*55}")

# ── Load Tumour Locator ResNet-18 CNN ─────────────────────────────────────────────────────
jay_model = None
if JAY_MODEL.exists():
    jay_model = models.resnet18(weights=None)
    jay_model.fc = nn.Linear(jay_model.fc.in_features, 2)
    jay_model.load_state_dict(torch.load(JAY_MODEL, map_location=DEVICE))
    jay_model.to(DEVICE).eval()
    print(f"  [✓] Tumour Locator CNN      loaded  ({JAY_MODEL.name})")
else:
    print(f"  [✗] Tumour Locator CNN      NOT FOUND: {JAY_MODEL}")

# ── Load Periocular Analysis system ResNet-152 CNN ───────────────────────────────────────────────────
mark_model = None
if MARK_MODEL.exists():
    mark_model = models.resnet152(weights=None)
    mark_model.fc = nn.Linear(mark_model.fc.in_features, 2)
    mark_model.load_state_dict(torch.load(MARK_MODEL, map_location=DEVICE))
    mark_model.to(DEVICE).eval()
    print(f"  [✓] Periocular Analysis system CNN     loaded  ({MARK_MODEL.name})")
else:
    print(f"  [✗] Periocular Analysis system CNN     NOT FOUND: {MARK_MODEL}")

# ── Load Periocular Transformer - ViT-B/16 ──────────────────────────────────────────────────────
aparna_model = None
aparna_gmar = None
if APARNA_MODEL.exists():
    backbone = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token"
    )
    head = MLPHead(in_dim=backbone.embed_dim)
    aparna_model = FineTuneModel(backbone, head).to(DEVICE)
    aparna_model.load_state_dict(
        torch.load(APARNA_MODEL, map_location=DEVICE, weights_only=True)
    )
    aparna_model.eval()
    aparna_gmar = ViTGMAR(aparna_model)
    print(f"  [✓] Periocular Transformer - ViT   loaded  ({APARNA_MODEL.name})")
else:
    print(f"  [✗] Periocular Transformer - ViT   NOT FOUND: {APARNA_MODEL}")

# ── Load iris-masked ViT (ViTbackend) ─────────────────────────────────────
masked_vit_model = None
masked_vit_gmar  = None
if MASKED_MODEL.exists():
    _backbone2 = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token"
    )
    _head2 = MLPHead(in_dim=_backbone2.embed_dim)
    masked_vit_model = FineTuneModel(_backbone2, _head2).to(DEVICE)
    masked_vit_model.load_state_dict(
        torch.load(MASKED_MODEL, map_location=DEVICE, weights_only=True)
    )
    masked_vit_model.eval()
    masked_vit_gmar = ViTGMAR(masked_vit_model)
    print(f"  [✓] ViT (iris-masked)              loaded  ({MASKED_MODEL.name})")
else:
    print(f"  [✗] ViT (iris-masked)              NOT FOUND: {MASKED_MODEL}")
    print(f"       Run ViTbackend/finetune.py to generate it.")

print(f"{'='*55}\n")

# ── Transforms ──────────────────────────────────────────────────────────────────
# Tumour Locator CNN was trained on raw pixels → ToTensor only, no normalisation
jay_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Periocular Analysis system ResNet-152 and Periocular Transformer - ViT → ImageNet normalisation
imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Iris-masked ViT: normalise only (iris masking + resize applied manually before)
_masked_vit_normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _preprocess_for_masked_vit(pil_image: Image.Image):
    """Resize to 224x224, apply iris masking, then normalise.

    Returns (tensor, masked_img_np) so the caller can also visualise the
    masked input that was actually fed to the model.
    """
    img_224 = pil_image.resize((224, 224), Image.BILINEAR)
    img_np  = np.array(img_224, dtype=np.uint8)
    masked, _ = mask_iris(img_np)
    return _masked_vit_normalise(Image.fromarray(masked)), masked

# ── Grad-CAM for ResNet (used by Tumour Locator CNN) ─────────────────────────────────────
class ResNetGradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    @torch.enable_grad()
    def __call__(self, tensor):
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        out    = self.model(tensor)
        self.model.zero_grad()
        out[0, out.argmax()].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()
        lo, hi  = cam.min(), cam.max()
        return (cam - lo) / (hi - lo + 1e-8)


jay_gradcam  = ResNetGradCAM(jay_model,  jay_model.layer4[-1].conv2)  if jay_model  else None
mark_gradcam = ResNetGradCAM(mark_model, mark_model.layer4[-1].conv2) if mark_model else None

# ── Helpers ──────────────────────────────────────────────────────────────────────
def encode_rgb(img_np: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()

def make_overlay(img_np: np.ndarray, cam: np.ndarray) -> str:
    cam_up  = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 255).astype(np.uint8)
    return encode_rgb(overlay)

# ── FastAPI ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Eyelid Cancer Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Geometric landmark analysis (OpenCV, no external model required) ──────────
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


@app.post("/analyse_landmarks")
async def analyse_landmarks(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    img_np    = np.array(image.resize((224, 224)))
    image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    annotated_bgr, measurements = run_landmark_analysis(image_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "annotated_image":    encode_rgb(annotated_rgb),
        "measurements":       measurements,
        "landmarks_detected": len(measurements) > 0,
    }


@app.post("/predict")
async def predict(
    file:       UploadFile = File(...),
    model_type: str        = Form(...),
):
    # ── Validate file ──────────────────────────────────────────────────────────
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    img_np = np.array(image.resize((224, 224)))

    # ── Route to the selected model ────────────────────────────────────────────
    if model_type == "jay_cnn":
        if jay_model is None:
            raise HTTPException(503, "Tumour Locator CNN model is not loaded.")

        tensor = jay_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(jay_model(tensor), dim=1).squeeze().tolist()
        idx  = int(probs[1] > probs[0])
        conf = round(probs[idx] * 100, 2)

        heatmap_b64 = make_overlay(img_np, jay_gradcam(tensor)) if jay_gradcam else None
        return {
            "prediction":    CLASSES[idx],
            "confidence":    conf,
            "model":         "Tumour Locator CNN",
            "original_image": encode_rgb(img_np),
            "heatmap":       heatmap_b64,
        }

    elif model_type == "mark_cnn":
        if mark_model is None:
            raise HTTPException(503, "Periocular Analysis system CNN model is not loaded. Place final_model.pth in backend_mark_system/model/.")

        tensor = imagenet_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(mark_model(tensor), dim=1).squeeze().tolist()
        idx  = int(probs[1] > probs[0])
        conf = round(probs[idx] * 100, 2)

        heatmap_b64 = make_overlay(img_np, mark_gradcam(tensor)) if mark_gradcam else None
        return {
            "prediction":    CLASSES[idx],
            "confidence":    conf,
            "model":         "Periocular Analysis system CNN",
            "original_image": encode_rgb(img_np),
            "heatmap":       heatmap_b64,
        }

    elif model_type == "aparna_vit":
        if aparna_model is None:
            raise HTTPException(503, "Periocular Transformer - ViT model is not loaded.")

        tensor = imagenet_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(aparna_model(tensor)).item()
        idx  = int(prob >= 0.5)
        conf = round((prob if idx == 1 else 1 - prob) * 100, 2)

        cam     = aparna_gmar(tensor)
        overlay = overlay_cam_on_image(img_np, cam, alpha=0.45)
        return {
            "prediction": CLASSES[idx],
            "confidence": conf,
            "model": "Periocular Transformer - ViT",
            "original_image": encode_rgb(img_np),
            "heatmap":       encode_rgb(overlay),
        }

    elif model_type == "aparna_vit_masked":
        if masked_vit_model is None:
            raise HTTPException(
                503,
                "Iris-masked ViT model is not loaded. "
                "Run ViTbackend/finetune.py to train it first."
            )

        tensor, masked_img_np = _preprocess_for_masked_vit(image)
        tensor = tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(masked_vit_model(tensor)).item()
        idx  = int(prob >= 0.5)
        conf = round((prob if idx == 1 else 1 - prob) * 100, 2)

        cam     = masked_vit_gmar(tensor)
        overlay = overlay_cam_on_image(masked_img_np, cam, alpha=0.45)
        return {
            "prediction":     CLASSES[idx],
            "confidence":     conf,
            "model":          "Periocular Transformer - ViT (iris-masked)",
            "original_image": encode_rgb(img_np),
            "masked_image":   encode_rgb(masked_img_np),
            "heatmap":        encode_rgb(overlay),
        }

    else:
        raise HTTPException(400, f"Unknown model_type '{model_type}'.")


# ── Serve frontend ────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
