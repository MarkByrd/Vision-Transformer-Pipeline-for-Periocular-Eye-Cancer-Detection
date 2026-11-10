# app — GAOIS Eyelid Cancer Detection

Local web app for the hospital evaluation of the **Iris-Masked Periocular Transformer**
(the ViT-B/16 trained in `../ViTbackend/`). For each uploaded image it shows:

1. the model's prediction (benign / malignant), confidence and GMAR attention heatmap;
2. a geometric landmark analysis (OpenCV, no model): iris, eye corners, eyelid margins,
   lesion bounding box and the distances between them;
3. an **Evaluation** form for the Clinical User;
4. an **Image Annotation** area where the user draws lines over the original image to mark
   the region that should influence the model's decision;
5. a **Medical Assessment** form for the Consultant.

One **Save Evaluation** button at the bottom saves one record per image (see
[Evaluation records](#evaluation-records)). This implements the *GAOIS Functional
Specification — Hospital Evaluation Prototype v1.0* (FR-01 to FR-12).

This folder is self-contained: it does not import anything from `ViTbackend/`.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI server: serves the UI and the `/api/*` endpoints. Run this. |
| `inference.py` | `Classifier`: loads the checkpoint and runs one image through the same preprocessing as `ViTbackend/cancer_dataset.py` (CLAHE → centre-crop → resize → iris mask, unmasked if no iris is found) → ViT → GMAR. No web code. |
| `landmarks.py` | `run_landmark_analysis()`: Hough-circle iris, estimated canthi, Canny eyelid margins, dark-blob lesion box, and pixel measurements on a 224×224 image. |
| `evaluations.py` | Validates and writes evaluation records to the data folder. |
| `model.py` | `MLPHead`, `FineTuneModel`, `ViTGMAR`, `overlay_cam_on_image`, copied from `ViTbackend/finetune.py` and `ViTbackend/gmar.py`. |
| `iris_masking.py`, `face_landmarker.task` | Copies of the files in `ViTbackend/`. Inference must mask the iris exactly as training did, so keep these in sync with `ViTbackend/` if either changes. |
| `static/` | The UI: `index.html`, `script.js` (page logic and forms), `annotator.js` (line drawing), `style.css`. |
| `models/` | Optional place for `vit_finetuned_masked.pth` (git-ignored). |
| `evaluations/` | Default location of saved evaluation records (git-ignored). |
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

3. Open <http://127.0.0.1:8000>. Set `PORT` to use a different port.

The app uses CUDA or Apple Silicon (MPS) if available, otherwise the CPU. If the
checkpoint is missing or fails to load, the server still starts, the page shows why, and
landmark analysis still works.

## Evaluation records

Records are written to `app/evaluations/`, or to the folder in the `GAOIS_DATA_DIR`
environment variable. **For the hospital evaluation, set `GAOIS_DATA_DIR` to a folder that
only authorised personnel can access (FR-12).** On macOS/Linux the app creates a new data
folder readable by the current user only; on Windows, restrict the folder's permissions
yourself. The startup log prints the folder in use.

```
<data folder>/
├── evaluations.csv                 one row per record — open in Excel for review
└── EV-20261110-142233-7F3A/        one folder per record (the record ID)
    ├── record.json                 every form field, the AI result and the image references
    ├── original.jpg                the uploaded file, byte-for-byte (original extension)
    ├── annotated.png               original + the user's lines (only if lines were drawn)
    └── gmar.jpg                    the GMAR heatmap shown to the user
```

`record.json` holds the record ID; the Clinical User's name, date/time, performance rating
and comments; the Consultant's name, medical assessment and comments; the model name,
prediction, confidence, P(malignant), iris-detection method and checkpoint file; the
original filename; references to the three images; and the drawn lines as point lists
normalised to 0–1 of the image size.

Required fields: Name of Clinical User and Performance. The Medical Assessment fields
are optional, because the Consultant may not be present. Comments are limited to 500
characters. The date and time are filled in when the result appears and can't be edited.

Saving writes each record into a temporary folder and renames it when complete, so a
failed save never leaves a partial record. If a save fails, the page says so, keeps
everything entered, and offers **Retry saving** for the records that weren't saved.

**Retention (FR-12):** records are kept for 18 months from the date of evaluation and then
deleted by hand. The record ID starts with the date (`EV-YYYYMMDD-…`), so expired folders
are easy to find. Remove their rows from `evaluations.csv` as well.

## API

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/api/status` | GET | — | `{ ready, error }` |
| `/api/predict` | POST | `file` (image, ≤ 10 MB) | `{ model, prediction, confidence, probability_malignant, iris_method, original_image, masked_image, heatmap }` (images are base64 JPEG, 224×224) |
| `/api/landmarks` | POST | `file` (image, ≤ 10 MB) | `{ annotated_image, measurements }` (measurements in pixels of the 224×224 image) |
| `/api/evaluations` | POST | `record` (JSON), `original` (image file), `gmar` (JPEG), `annotated` (PNG, optional) | `{ record_id, saved_at }`; 422 with the reason if a field is invalid |
