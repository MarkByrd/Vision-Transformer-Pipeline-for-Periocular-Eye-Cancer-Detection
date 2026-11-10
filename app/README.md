# app

Local web app that classifies eyelid lesion images as **benign** or **malignant** with the
iris-masked ViT-B/16 model trained in `../ViTbackend/`, and shows a GMAR attention heatmap
for each prediction. It also runs a geometric landmark analysis (OpenCV, no model) on each
image: iris, eye corners, eyelid margins, lesion bounding box, and distances between them.

This folder is self-contained: it does not import anything from `ViTbackend/`.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI server: serves the UI and the `/api/*` endpoints. Run this. |
| `inference.py` | `Classifier`: loads the checkpoint and runs one image through the same preprocessing as `ViTbackend/cancer_dataset.py` (CLAHE → centre-crop → resize → iris mask, unmasked if no iris is found) → ViT → GMAR. No web code. |
| `landmarks.py` | `run_landmark_analysis()`: Hough-circle iris, estimated canthi, Canny eyelid margins, dark-blob lesion box, and pixel measurements on a 224×224 image. |
| `model.py` | `MLPHead`, `FineTuneModel`, `ViTGMAR`, `overlay_cam_on_image`, copied from `ViTbackend/finetune.py` and `ViTbackend/gmar.py`. |
| `iris_masking.py`, `face_landmarker.task` | Copies of the files in `ViTbackend/`. Inference must mask the iris exactly as training did, so keep these in sync with `ViTbackend/` if either changes. |
| `static/` | The UI (`index.html`, `script.js`, `style.css`). |
| `models/` | Optional place for `vit_finetuned_masked.pth` (git-ignored). |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies and the exact locked versions for Windows, macOS and Linux. After editing dependencies, run `uv lock`. On Linux, torch comes from PyTorch's CPU-only index to avoid a ~2.5 GB CUDA download. |

## Run

1. Make sure `vit_finetuned_masked.pth` is in one of these folders (the first match is used):
   - `app/models/`
   - `ViTbackend/outputs/` (where `finetune.py` saves it)
   - `outputs/` in the folder that contains this repo (e.g. `Eyelidcancer/outputs/`)

   Or set `EYELID_MODEL` to its full path. The startup log prints which file was loaded.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then start the server:

   ```bash
   cd app
   uv run main.py
   ```

   The first run downloads Python 3.12 if needed and installs the exact package versions
   from `uv.lock` into `app/.venv`; later runs start straight away.

   Without uv: create a Python 3.10–3.12 virtual environment, run `pip install .`
   (installs the dependencies from `pyproject.toml`, unpinned beyond what it lists), then
   `python main.py`.

3. Open <http://127.0.0.1:8000>. Set `PORT` to use a different port.

The app uses CUDA or Apple Silicon (MPS) if available, otherwise the CPU. If the
checkpoint is missing or fails to load, the server still starts, the page shows why, and
landmark analysis still works.

## API

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/api/status` | GET | — | `{ ready, error }` |
| `/api/predict` | POST | `file` (image, ≤ 10 MB) | `{ prediction, confidence, probability_malignant, iris_method, original_image, masked_image, heatmap }` (images are base64 JPEG, 224×224) |
| `/api/landmarks` | POST | `file` (image, ≤ 10 MB) | `{ annotated_image, measurements }` (measurements in pixels of the 224×224 image) |
