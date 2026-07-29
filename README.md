# Vision Transformer Pipeline for Periocular Eyelid Cancer Detection

A dissertation project (Trinity College Dublin, School of Computer Science & Statistics,
supervised by Prof. Khurshid Ahmad) comparing three independent computer-vision approaches
to classifying eyelid lesion images as **benign** or **malignant**, served from a single
unified web app:

| Prototype | Models | Author | Source |
|---|---|---|---|
| **A - Dual CNN Analysis** | ResNet-18 ("Tumour Locator") + ResNet-152 ("Periocular Analysis") | Jay Birthariya / Mark Byrd | `backend_jay/`, `backend_mark_system/` |
| **B - Periocular Transformer** | ViT-B/16, MAE-pretrained — baseline (unmasked) + iris-masked variant | Aparna | `ViTbackend/` |

Both prototypes are reachable from one page, with Grad-CAM (CNNs) or GMAR attention-rollout
(ViTs) heatmaps overlaid on every prediction.

---

## Architecture

This is **not** a microservices setup — there is a single FastAPI process
(`frontend/app.py`) that imports each contributor's model-building code directly and loads
all four trained checkpoints into memory at startup. The static UI is served from the same
process.

```
frontend/app.py  (single process, http://localhost:8000)
 ├── imports ResNet-18 arch from backend_jay/           → loads backend_jay/Pipeline/GrabCut/saved_models/cnn_b.pth
 ├── imports ResNet-152 arch from backend_mark_system/  → loads backend_mark_system/model/final_model.pth
 └── imports ViT-B/16 arch from ViTbackend/              → loads ViTbackend/outputs/vit_finetuned_unmasked.pth and ViTbackend/outputs/vit_finetuned_masked.pth
```

`ViTbackend/` is self-contained for both ViT variants:

```
ViTbackend/
├── mae_dataset.py, mae_model.py, train_mae.py     # Phase 1: MAE pretraining (iris-masked)
├── cancer_dataset.py, finetune.py, gmar.py         # Phase 2/3: fine-tuning + GMAR viz (iris-masked)
├── iris_masking.py, face_landmarker.task           # MediaPipe-based iris/pupil masking
├── outputs/                                        # trained checkpoints (not in git — see below)
└── unmasked/                                       # the no-masking ablation baseline pipeline
    ├── clahe_preprocess.py                         # Phase 0: CLAHE preprocessing
    ├── mae_dataset.py, mae_model.py, train_mae.py   # Phase 1: MAE pretraining (unmasked)
    └── cancer_dataset.py, finetune.py, evaluate.py  # Phase 2/3: fine-tuning + evaluation
```

`ViTbackend/finetune.py` and `ViTbackend/gmar.py` define the shared `FineTuneModel`,
`MLPHead`, and `ViTGMAR` classes used to load **both** the masked and unmasked checkpoints —
the two variants only differ in which dataset pipeline produced the weights.

---

## Repository layout

```
ViT-implementation/
├── frontend/              # unified FastAPI backend + static web UI (this repo)
├── ViTbackend/            # ViT-B/16 MAE pipeline: masked + unmasked (this repo)
├── backend_jay/           # Jay's ResNet-18 CNN pipeline        (clone separately, see below)
├── backend_mark_system/   # Mark's ResNet-152 CNN pipeline      (clone separately, see below)
├── periculor_dataset/     # normal periocular images for MAE pretraining (download separately)
└── Eyelid_Cancer_Dataset/ # labelled eyelid lesion images for fine-tuning (supply separately)
```

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/AparnaGayathri16/Vision-Transformer-Pipeline-for-Periocular-Eye-Cancer-Detection.git
cd Vision-Transformer-Pipeline-for-Periocular-Eye-Cancer-Detection
```

### 2. Clone Jay's and Mark's backends into place

The unified frontend expects their code at `backend_jay/` and `backend_mark_system/`
alongside `ViTbackend/` and `frontend/`:

```bash
git clone https://github.com/JayBirthariya581/CV-ML-PIPELINE-FOR-DETECTING-EYELID-CANCER.git backend_jay
git clone https://github.com/MarkByrd/New_Eyelid_Cancer_Classifier.git backend_mark_system
```

### 3. Download the periocular dataset (MAE pretraining)

Used to pretrain the ViT-B/16 encoder in `ViTbackend/train_mae.py` /
`ViTbackend/unmasked/train_mae.py`. Download from Kaggle:

```
https://www.kaggle.com/datasets/shriaman/periocular
```

Extract it to `periculor_dataset/` at the repo root, arranged as:

```
periculor_dataset/
├── LEFT/   (left-eye periocular images, flat or one subfolder per subject)
└── RIGHT/  (right-eye periocular images, same layout)
```

### 4. Supply the eyelid cancer dataset (fine-tuning)

This is the labelled clinical dataset used for Phase 2 fine-tuning
(`ViTbackend/cancer_dataset.py`, `ViTbackend/unmasked/cancer_dataset.py`, and Jay's/Mark's
own training scripts). It is not publicly hosted — place your copy at
`Eyelid_Cancer_Dataset/` at the repo root, with filenames prefixed `B` for benign or `M`/`m`
for malignant (e.g. `B1.jpg`, `M12.jpg`).

### 5. Install dependencies

Python 3.10+ recommended. A CUDA-capable GPU is optional but strongly recommended for
training (inference runs fine on CPU).

```bash
pip install -r backend_jay/requirements.txt
pip install -r backend_mark_system/requirements.txt
pip install torch torchvision timm opencv-python mediapipe scikit-learn pillow numpy matplotlib fastapi uvicorn python-multipart
```

> Versions verified working together during development: `torch==2.5.1+cu121`,
> `torchvision==0.20.1+cu121`, `timm==1.0.26`, `fastapi==0.115.0`, `uvicorn==0.30.6`,
> `opencv-python==4.11.0`, `mediapipe==0.10.13`, `scikit-learn==1.7.2`.

---

## Training the ViT pipeline (`ViTbackend/`)

Both variants follow the same three phases; run each from inside its own directory so the
default relative paths resolve correctly.

### Iris-masked (main model)

```bash
cd ViTbackend
python train_mae.py                 # Phase 1: MAE pretrain  -> outputs/mae_normal_maskediris.pth
python finetune.py                  # Phase 2: fine-tune      -> outputs/vit_finetuned_masked.pth
```

### Unmasked (ablation baseline)

```bash
cd ViTbackend/unmasked
python clahe_preprocess.py          # Phase 0: CLAHE-preprocess Eyelid_Cancer_Dataset -> outputs/clahe_cancer/
python train_mae.py                 # Phase 1: MAE pretrain   -> outputs/best_encoder.pt
python finetune.py                  # Phase 2: fine-tune      -> outputs/freeze7/best_classifier.pt
python evaluate.py                  # Phase 3: evaluate on a held-out test set
```

Copy (or symlink) whichever checkpoint you want the frontend to serve to
`ViTbackend/outputs/vit_finetuned_masked.pth` and
`ViTbackend/outputs/vit_finetuned_unmasked.pth` respectively — that's what `frontend/app.py`
loads at startup.

Training checkpoints are large (300+ MB) and are **not** committed to git — see
`.gitignore`.

---

## Running the app

Once `backend_jay/`, `backend_mark_system/`, and `ViTbackend/outputs/*.pth` are all in
place:

```bash
cd frontend
python app.py
```

This starts the unified backend + UI at **http://localhost:8000**. The startup log reports
which of the four models loaded successfully — any model whose checkpoint is missing is
skipped (its button will return a 503 rather than crashing the app):

```
=======================================================
  Unified Eyelid Cancer Backend
  Device : cuda
=======================================================
  [✓] Tumour Locator CNN      loaded  (cnn_b.pth)
  [✓] Periocular Analysis system CNN     loaded  (final_model.pth)
  [✓] Periocular Transformer - ViT   loaded  (vit_finetuned_unmasked.pth)
  [✓] ViT (iris-masked)              loaded  (vit_finetuned_masked.pth)
=======================================================
```

Open the page, choose **Prototype A** (dual CNN) or **Prototype B** (periocular
transformer), upload one or more eyelid images, and view predictions with heatmap overlays.

### API reference

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/predict` | POST | `file` (image), `model_type` (`jay_cnn` \| `mark_cnn` \| `aparna_vit` \| `aparna_vit_masked`) | `{ prediction, confidence, model, original_image, heatmap }` |
| `/analyse_landmarks` | POST | `file` (image) | OpenCV-based periocular landmark/lesion geometry analysis, no model required |

---

## Notes

- All Windows-style paths (`..\Eyelid_Cancer_Dataset`, `num_workers=0` for MediaPipe) reflect
  the development environment; adjust for other platforms as needed.
- `ViTbackend/iris_masking.py` uses MediaPipe FaceLandmarker (`face_landmarker.task`, bundled
  in `ViTbackend/`) with a Hough-circle fallback when no face/iris is detected.
