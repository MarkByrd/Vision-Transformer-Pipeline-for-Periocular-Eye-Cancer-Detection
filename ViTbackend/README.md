# ViTbackend

ViT-B/16 periocular eyelid-cancer classifier, pretrained with a Masked Autoencoder (MAE) on
normal periocular images and fine-tuned on labelled eyelid lesion images. This directory is
self-contained for **two** ablation variants:

- **Iris-masked** (top level of this directory) - the main model. The iris/pupil region is
  removed (`iris_masking.py`) before every training and inference step, so the classifier
  cannot key on iris colour/reflection artefacts instead of periocular skin.
- **Unmasked baseline** (`unmasked/`) - the same three-phase pipeline without iris masking,
  used to measure how much the masking actually helps.

Both variants share the same `FineTuneModel` / `MLPHead` / `ViTGMAR` classes (defined in
`finetune.py` and `gmar.py` at the top level) - they only differ in which dataset pipeline
produced the checkpoint. The app in `../app/` keeps its own copy of these classes and of
`iris_masking.py`, so it serves the iris-masked checkpoint without importing this folder.

Run everything below from **inside the directory that contains the script** (`ViTbackend/`
or `ViTbackend/unmasked/`) — all default paths are relative to that location.

---

## Files (iris-masked pipeline - top level)

| File | Purpose |
|---|---|
| `iris_masking.py` | `mask_iris()`: detects the iris/pupil via MediaPipe FaceLandmarker (falls back to Hough circle detection, then a fixed-fraction fallback circle if no face/iris is found) and fills it with a skin-tone estimate sampled from the surrounding annulus. Used by every other script here. |
| `face_landmarker.task` | Bundled MediaPipe face-landmark model asset used by `iris_masking.py`. |
| `mae_dataset.py` | `PeriocularDataset`: Phase 1 dataset. Loads `periculor_dataset/LEFT` and `RIGHT`, applies geometric augmentation (resize/crop/flip), then `mask_iris()`, then photometric augmentation + normalisation. Also returns a `(196,)` patch-mask marking which ViT patches fall inside the iris circle, so the MAE loss can exclude them. |
| `mae_model.py` | `MAEModel`: ViT-B/16 encoder + lightweight transformer decoder, sin-cos positional embeddings, random patch masking, MSE reconstruction loss (accepts the iris `iris_patch_mask` to exclude iris patches from the loss). |
| `train_mae.py` | Phase 1 training script- pretrains the MAE encoder on periocular images. |
| `cancer_dataset.py` | `CancerDataset` / `TestDataset`: Phase 2 dataset. CLAHE → centre-crop-square → resize → `mask_iris()` → augmentation (train only) → normalisation. Labels: filename prefix `B` = benign, `M`/`m` = malignant. |
| `finetune.py` | Phase 2 training script. Defines `MLPHead` (768→256→1) and `FineTuneModel` (backbone + head) - the shared architecture classes. Trains with **gradual unfreezing**: epochs 1-5 only block 12 + head, epochs 6-10 blocks 10-12 + head, epochs 11-20 blocks 8-12 + head with layer-wise LR decay (0.75×/block). AdamW + CosineAnnealingLR per phase. |
| `gmar.py` | Defines `ViTGMAR` (Gradient-weighted Multi-head Attention Rollout - patches the ViT attention modules to retain gradients, then rolls out gradient-weighted attention across all 12 blocks for a 14×14 saliency map) and `overlay_cam_on_image()`. Run standalone to generate GMAR heatmaps + predictions over a test set. |
| `diagnostics_masking.py` | Sanity-checks `mask_iris()` across a random sample of both `periculor_dataset` and `Eyelid_Cancer_Dataset` images; saves before/after preview panels and a pass/fail report (flags if success rate < 70%). |
| `outputs/` | Checkpoints and logs (git-ignored - see below). |

## Files (unmasked baseline - `unmasked/`)

Same three-phase pipeline, copied from the original ablation baseline, **without** any iris
masking:

| File | Purpose |
|---|---|
| `clahe_preprocess.py` | Phase 0 - applies CLAHE to the L-channel of `Eyelid_Cancer_Dataset` and writes the result to `outputs/clahe_cancer/` (this is a prerequisite `cancer_dataset.py` here expects pre-CLAHE'd input, unlike the masked variant which applies CLAHE on the fly). |
| `mae_dataset.py`, `mae_model.py`, `train_mae.py` | Phase 1 - MAE pretraining, identical in spirit to the masked variant but with no iris masking or patch-mask exclusion. |
| `cancer_dataset.py` | Phase 2 dataset - reads from `outputs/clahe_cancer/`, centre-crop-square, resize, augmentation, normalisation. No masking step. |
| `finetune.py` | Phase 2 training - simpler **tri-level LR** scheme (not gradual unfreezing): block 7 gets a conservative LR, blocks 8-11+norm a standard LR, the head the fastest LR; `StepLR` schedule. |
| `evaluate.py` | Phase 3 - evaluates a trained classifier on a held-out test set: ROC-AUC curve, confusion matrix, F1/sensitivity/specificity, per-image predictions. |

---

## Dataset locations

Both variants expect these two folders at the **repo root** (one level above `ViTbackend/`):

```
periculor_dataset/          # normal periocular images (Phase 1) - LEFT/ and RIGHT/ subfolders
Eyelid_Cancer_Dataset/       # labelled lesion images (Phase 2) - filenames prefixed B*/M*
```

See the top-level `README.md` for where to get/place these.

---

## Training - iris-masked (main model)

```bash
cd ViTbackend

# Phase 1: MAE pretraining
python train_mae.py --epochs 200 --batch_size 16 --num_workers 0  
# -> outputs/mae_normal_maskediris.pth

# Phase 2: fine-tuning (gradual unfreezing + LLRD)
python finetune.py --dataset "..\Eyelid_Cancer_Dataset" --encoder "outputs\mae_normal_maskediris.pth" --output "outputs"   
# -> outputs/vit_finetuned_masked.pth, outputs/finetune_masked_log.csv
```

Bracketed flags show the defaults — all are optional overrides.

## Testing / visualisation — iris-masked

`gmar.py` runs the fine-tuned classifier over a test folder and saves a 3-panel figure
(original / GMAR map / overlay) per image, printing true vs. predicted label:

```bash
cd ViTbackend
python gmar.py --classifier outputs\vit_finetuned_masked.pth --test_dir "path\to\test_images" --output outputs\gmar
```

> The default `--classifier` value (`outputs/best_classifier.pt`) does **not** match the
> filename `finetune.py` actually saves (`vit_finetuned_masked.pth`) — always pass
> `--classifier` explicitly, as above.

## Masking sanity check

Run before (re)training if you've changed `iris_masking.py` or added new source images:

```bash
cd ViTbackend
python diagnostics_masking.py
```

Outputs `diagnostics/masking_preview/*.png` and `diagnostics/masking_report.txt`, and exits
non-zero if either dataset's detection success rate drops below 70%.

---

## Training - unmasked baseline

```bash
cd ViTbackend\unmasked

# Phase 0: CLAHE preprocessing (required before Phase 2 — cancer_dataset.py reads its output)
python clahe_preprocess.py --dataset ..\..\Eyelid_Cancer_Dataset --output outputs\clahe_cancer

# Phase 1: MAE pretraining
python train_mae.py --root ..\..\periculor_dataset --output outputs 
# -> outputs/best_encoder.pt

# Phase 2: fine-tuning (tri-level LR, block 7 conservative)
python finetune.py --dataset outputs\clahe_cancer --encoder outputs\best_encoder.pt
# -> outputs/freeze7/best_classifier.pt, outputs/freeze7/finetune_log.csv
```

## Testing / evaluation - unmasked baseline

```bash
cd ViTbackend\unmasked
python evaluate.py --test_dir "path\to\test_images" --classifier outputs\freeze7\best_classifier.pt
                    [--output outputs] [--batch_size 8] [--freeze_blocks 8]
```

Produces `outputs/roc_curve.png`, `outputs/confusion_matrix.png`, and prints AUC / F1 /
sensitivity / specificity plus a per-image prediction table.

---

## Serving the model in the app

The app (`../app/`) serves only the iris-masked model and picks up
`outputs/vit_finetuned_masked.pth` from here automatically — see `../app/README.md`. If you change `iris_masking.py`, `finetune.py`'s model classes or
`gmar.py`, update the copies in `../app/` to match.

