"""
GAOIS - Eyelid Cancer Detection: local web app for the Iris-Masked Periocular
Transformer (ViT-B/16), with geometric landmark analysis and clinical evaluation
records.

    uv run main.py            # then open http://127.0.0.1:8000

Environment variables:
    EYELID_MODEL     path to the checkpoint (default: first vit_finetuned_masked.pth
                     found in MODEL_SEARCH_DIRS below)
    GAOIS_DATA_DIR   where evaluation records are saved (default: app/evaluations/)
    PORT             port to listen on (default: 8000)
"""

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import ValidationError

from evaluations import EvaluationIn, ensure_data_dir, save_evaluation
from inference import Classifier, pick_device
from landmarks import run_landmark_analysis

MODEL_NAME = "Iris-Masked Periocular Transformer"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("GAOIS_DATA_DIR", APP_DIR / "evaluations"))
STATIC_DIR = APP_DIR / "static"
MODEL_FILE = "vit_finetuned_masked.pth"
MODEL_SEARCH_DIRS = [
    APP_DIR / "models",                        # app/models/
    APP_DIR.parent / "ViTbackend" / "outputs",  # where finetune.py writes it
    APP_DIR.parent.parent / "outputs",          # outputs/ next to the repo folder
]


def find_model() -> Path:
    if os.environ.get("EYELID_MODEL"):
        return Path(os.environ["EYELID_MODEL"])
    for d in MODEL_SEARCH_DIRS:
        if (d / MODEL_FILE).exists():
            return d / MODEL_FILE
    return MODEL_SEARCH_DIRS[0] / MODEL_FILE


MODEL_PATH = find_model()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

classifier: Classifier | None = None
load_error: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global classifier, load_error
    device = pick_device()
    print(f"Device: {device}")
    ensure_data_dir(DATA_DIR)
    print(f"Evaluation records: {DATA_DIR}")
    if not MODEL_PATH.exists():
        load_error = f"Model checkpoint not found: {MODEL_PATH}"
        print(f"[!] {load_error}")
    else:
        try:
            classifier = Classifier.load(MODEL_PATH, device)
            print(f"[ok] Loaded {MODEL_PATH}")
        except Exception as e:  # report in the UI instead of refusing to start
            load_error = f"Could not load {MODEL_PATH.name}: {e}"
            print(f"[!] {load_error}")
    yield


app = FastAPI(title="Eyelid Cancer Classifier", lifespan=lifespan)


def encode_jpeg(img_rgb: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


@app.get("/api/status")
def status():
    return {"ready": classifier is not None, "error": load_error}


def read_image(file: UploadFile) -> Image.Image:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        return image.convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")


# Endpoints are sync (def, not async def): FastAPI runs them in a worker
# thread, so image processing doesn't block the event loop.

@app.post("/api/landmarks")
def landmarks(file: UploadFile = File(...)):
    img_rgb = np.array(read_image(file).resize((224, 224)))
    annotated_bgr, measurements = run_landmark_analysis(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return {
        "annotated_image": encode_jpeg(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)),
        "measurements": measurements,
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(503, load_error or "Model is not loaded.")
    result = classifier.predict(read_image(file))
    return {
        "model": MODEL_NAME,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probability_malignant": result["probability_malignant"],
        "iris_method": result["iris_method"],
        "original_image": encode_jpeg(result["original"]),
        "masked_image": encode_jpeg(result["masked"]),
        "heatmap": encode_jpeg(result["heatmap"]),
    }


def read_limited(file: UploadFile, limit: int, label: str) -> bytes:
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(400, f"{label} is too large.")
    return data


@app.post("/api/evaluations")
def create_evaluation(
    record: str = Form(...),
    original: UploadFile = File(...),
    gmar: UploadFile = File(...),
    annotated: UploadFile | None = File(None),
):
    """Save one evaluation record: the form fields as JSON plus the three images."""
    try:
        evaluation = EvaluationIn.model_validate_json(record)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise HTTPException(422, f"Invalid evaluation: {problems}")

    ext = Path(original.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported original image type '{ext}'.")
    original_bytes = read_limited(original, MAX_FILE_SIZE, "Original image")

    gmar_bytes = read_limited(gmar, 5 * 1024 * 1024, "GMAR image")
    if not gmar_bytes.startswith(b"\xff\xd8"):
        raise HTTPException(400, "GMAR image must be a JPEG.")

    annotated_bytes = None
    if annotated is not None:
        annotated_bytes = read_limited(annotated, 40 * 1024 * 1024, "Annotated image")
        if not annotated_bytes.startswith(b"\x89PNG"):
            raise HTTPException(400, "Annotated image must be a PNG.")

    try:
        return save_evaluation(
            DATA_DIR, evaluation, MODEL_PATH.name,
            (original_bytes, ext), gmar_bytes, annotated_bytes,
        )
    except OSError as e:
        raise HTTPException(500, f"Could not write the evaluation record: {e}")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
