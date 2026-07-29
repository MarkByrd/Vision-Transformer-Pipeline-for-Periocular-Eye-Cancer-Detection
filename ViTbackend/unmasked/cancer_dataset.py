"""
cancer_dataset.py
-----------------
PyTorch Dataset for Phase 2 fine-tuning on the eyelid cancer dataset.

Label convention:
  - Filename starts with 'B' → benign  (label 0)
  - Filename starts with 'm' or 'M' → malignant (label 1)

Reads from outputs/clahe_cancer/ (CLAHE-processed).
Provides stratified train/val splits.
The held-out test set comes from "new testing dataset/".

Usage (imported by finetune.py):
    from cancer_dataset import CancerDataset, TestDataset
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(pil_img: Image.Image) -> Image.Image:
    """Apply CLAHE to the L channel of LAB, matching clahe_preprocess.py."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_eq = cv2.merge([_CLAHE.apply(l), a, b])
    rgb_eq = cv2.cvtColor(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_eq)


def _get_label_from_filename(filename: str) -> int:
    """B* → 0 (benign), m*/M* → 1 (malignant)."""
    base = Path(filename).stem
    if base[0].upper() == "B":
        return 0
    elif base[0].upper() == "M":
        return 1
    else:
        raise ValueError(f"Cannot determine label for: {filename}")


class CenterCropSquare:
    """
    Crop to a centred square using the shorter side, before Resize.

    Benign and malignant source images have systematically different aspect
    ratios (most malignant crops are ~square, most benign crops are wide
    rectangles). A plain Resize((224,224)) stretches each image by a
    different amount depending on its original aspect ratio, which leaks
    class information through image geometry rather than lesion content.
    Cropping to a square first (instead of padding) removes that confound
    without adding new synthetic border content whose extent would itself
    correlate with the original aspect ratio.
    """

    def __call__(self, img):
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return img.crop((left, top, left + s, top + s))


def build_train_transform() -> transforms.Compose:
    """Clinically justified augmentation for fine-tuning."""
    return transforms.Compose([
        CenterCropSquare(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform() -> transforms.Compose:
    """Deterministic transform for val/test."""
    return transforms.Compose([
        CenterCropSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CancerDataset(Dataset):
    """
    Eyelid cancer dataset for fine-tuning.

    Args:
        root:         Path to CLAHE-processed cancer images (outputs/clahe_cancer/)
        split:        'train' or 'val'
        val_fraction: Fraction reserved for validation (default 0.10)
        seed:         Random seed for reproducible split
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.10,
        seed: int = 42,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
        self.root = Path(root)
        self.split = split

        # Collect all images
        all_files = sorted([
            f for f in self.root.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ])

        if not all_files:
            raise RuntimeError(f"No images found in {self.root}")

        all_labels = [_get_label_from_filename(f.name) for f in all_files]

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(sss.split(all_files, all_labels))

        if split == "train":
            indices = train_idx
            self.transform = build_train_transform()
        else:
            indices = val_idx
            self.transform = build_val_transform()

        self.paths = [all_files[i] for i in indices]
        self.labels = [all_labels[i] for i in indices]

        # Class counts for weighted loss
        n_benign = sum(1 for l in self.labels if l == 0)
        n_malignant = sum(1 for l in self.labels if l == 1)
        total = len(self.labels)

        print(f"[CancerDataset] {split.upper()} — {total} images "
              f"(benign={n_benign}, malignant={n_malignant})")

        self.class_counts = {0: n_benign, 1: n_malignant}

    def get_pos_weight(self) -> float:
        """Return pos_weight for BCEWithLogitsLoss (ratio of negatives/positives)."""
        n_neg = self.class_counts.get(0, 1)
        n_pos = self.class_counts.get(1, 1)
        return n_neg / n_pos

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class TestDataset(Dataset):
    """
    Held-out test dataset from 'new testing dataset/' folder.

    Filename convention:
      - B{n} (2).jpg → benign (label 0)
      - M{n}.jpg     → malignant (label 1)
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.transform = build_val_transform()

        all_files = sorted([
            f for f in self.root.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ])

        if not all_files:
            raise RuntimeError(f"No images found in {self.root}")

        self.paths = all_files
        self.labels = [_get_label_from_filename(f.name) for f in all_files]

        n_benign = sum(1 for l in self.labels if l == 0)
        n_malignant = sum(1 for l in self.labels if l == 1)
        print(f"[TestDataset] {len(all_files)} images "
              f"(benign={n_benign}, malignant={n_malignant})")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            img = _apply_clahe(img)
        except Exception:
            img = Image.new("RGB", (224, 224), 0)

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    def get_filenames(self):
        """Return list of filenames for per-image reporting."""
        return [p.name for p in self.paths]
