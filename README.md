# Vision Transformer Pipeline for Periocular Eyelid Cancer Detection

A dissertation project (Trinity College Dublin, School of Computer Science & Statistics,
supervised by Prof. Khurshid Ahmad) that classifies eyelid lesion images as **benign** or
**malignant** with a ViT-B/16, pretrained as a Masked Autoencoder (MAE) on normal periocular
images and fine-tuned on labelled lesion images. The iris/pupil is masked out before every
training and inference step so the model can't key on iris colour or reflections.

> Research prototype — not a medical device and not for clinical diagnosis.

## Repository layout

```
.
├── app/                   # the application: local web UI + inference (no training code)
├── ViTbackend/            # training pipeline: MAE pretraining, fine-tuning, GMAR, evaluation
│   └── unmasked/          #   the no-masking ablation baseline
├── periculor_dataset/     # normal periocular images for MAE pretraining (download separately)
└── Eyelid_Cancer_Dataset/ # labelled eyelid lesion images for fine-tuning (supply separately)
```

`app/` and `ViTbackend/` are independent: the app has its own copy of the model classes and
the iris-masking code, and needs only the trained checkpoint.

---

## Running the app

1. Train the model (below), or obtain `vit_finetuned_masked.pth`. The app finds it in
   `app/models/`, `ViTbackend/outputs/`, or an `outputs/` folder next to this repo.
2. Start it with [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs
   Python and the locked dependencies on first run:

   ```bash
   cd app
   uv run main.py
   ```

3. Open **http://127.0.0.1:8000**, upload one or more eyelid images, and click **Analyse**.
   Each result shows the prediction and confidence, the iris-masked image the model saw,
   a GMAR attention heatmap, and a geometric landmark analysis (iris, eye corners, eyelid
   margins, lesion box and their measurements).

See [`app/README.md`](app/README.md) for details and the API.

---

## Training (`ViTbackend/`)

### Datasets

- **Periocular dataset** (MAE pretraining): download from
  <https://www.kaggle.com/datasets/shriaman/periocular> and extract to `periculor_dataset/`
  with `LEFT/` and `RIGHT/` subfolders.
- **Eyelid cancer dataset** (fine-tuning): not publicly hosted. Place your copy at
  `Eyelid_Cancer_Dataset/`, with filenames prefixed `B` for benign or `M`/`m` for malignant
  (e.g. `B1.jpg`, `M12.jpg`).

### Dependencies

Python 3.10+. A CUDA GPU is strongly recommended for training.

```bash
pip install torch torchvision timm opencv-python mediapipe scikit-learn pillow numpy matplotlib
```

> Versions verified working together: `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`,
> `timm==1.0.26`, `opencv-python==4.11.0`, `mediapipe==0.10.13`, `scikit-learn==1.7.2`.

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

Run each script from inside its own directory so the default relative paths resolve.
Checkpoints are large (300+ MB) and are **not** committed to git. See
[`ViTbackend/README.md`](ViTbackend/README.md) for every script and flag.

## Notes

- Windows-style default paths (`..\Eyelid_Cancer_Dataset`, `num_workers=0` for MediaPipe)
  reflect the development environment; adjust for other platforms as needed.
- `iris_masking.py` uses MediaPipe FaceLandmarker (`face_landmarker.task`) with a
  Hough-circle fallback, then a fixed central circle if no iris is found.
