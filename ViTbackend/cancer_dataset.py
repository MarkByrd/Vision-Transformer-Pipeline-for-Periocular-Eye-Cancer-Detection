import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from iris_masking import mask_iris

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(pil_img: Image.Image) -> Image.Image:
    bgr   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_eq  = cv2.merge([_CLAHE.apply(l), a, b])
    rgb_eq  = cv2.cvtColor(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_eq)


def _get_label(filename: str) -> int:
    base = Path(filename).stem
    if base[0].upper() == "B":
        return 0
    elif base[0].upper() == "M":
        return 1
    raise ValueError(f"Cannot determine label for: {filename}")


class CenterCropSquare:
    def __call__(self, img):
        w, h = img.size
        s = min(w, h)
        return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def _apply_iris_mask(pil_img: Image.Image, path: str = "") -> Image.Image:
    """Apply iris masking to a PIL RGB image; return PIL RGB.
    skip_if_no_iris=True: closed-eye cancer images are returned unchanged."""
    arr = np.array(pil_img, dtype=np.uint8)
    masked, _ = mask_iris(arr, image_path=path, skip_if_no_iris=True)
    return Image.fromarray(masked)


def build_train_transform() -> transforms.Compose:
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
    return transforms.Compose([
        CenterCropSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CancerDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.10,
        seed: int = 42,
    ):
        assert split in ("train", "val")
        self.root  = Path(root)
        self.split = split

        all_files  = sorted(f for f in self.root.iterdir()
                            if f.suffix.lower() in VALID_EXTENSIONS)
        if not all_files:
            raise RuntimeError(f"No images found in {self.root}")

        all_labels = [_get_label(f.name) for f in all_files]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(sss.split(all_files, all_labels))

        indices   = train_idx if split == "train" else val_idx
        self.paths  = [all_files[i] for i in indices]
        self.labels = [all_labels[i] for i in indices]

        n_b = sum(1 for l in self.labels if l == 0)
        n_m = sum(1 for l in self.labels if l == 1)
        print(f"[CancerDataset] {split.upper()} - {len(self.labels)} images "
              f"(benign={n_b}, malignant={n_m})  [iris masking ON]")

        self.class_counts = {0: n_b, 1: n_m}
        self.transform    = (build_train_transform() if split == "train"
                             else build_val_transform())

    def get_pos_weight(self) -> float:
        return self.class_counts.get(0, 1) / max(self.class_counts.get(1, 1), 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
            img = _apply_clahe(img)
            img = CenterCropSquare()(img)
            img = img.resize((224, 224), Image.BILINEAR)
            img = _apply_iris_mask(img, str(path))
        except Exception:
            img = Image.new("RGB", (224, 224), 128)

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class TestDataset(Dataset):

    def __init__(self, root: str):
        self.root      = Path(root)
        self.transform = build_val_transform()

        all_files = sorted(f for f in self.root.iterdir()
                           if f.suffix.lower() in VALID_EXTENSIONS)
        if not all_files:
            raise RuntimeError(f"No images found in {self.root}")

        self.paths  = all_files
        self.labels = [_get_label(f.name) for f in all_files]

        n_b = sum(1 for l in self.labels if l == 0)
        n_m = sum(1 for l in self.labels if l == 1)
        print(f"[TestDataset] {len(all_files)} images "
              f"(benign={n_b}, malignant={n_m})  [iris masking ON]")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path  = self.paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
            img = _apply_clahe(img)
            img = CenterCropSquare()(img)
            img = img.resize((224, 224), Image.BILINEAR)
            img = _apply_iris_mask(img, str(path))
        except Exception:
            img = Image.new("RGB", (224, 224), 128)

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    def get_filenames(self):
        return [p.name for p in self.paths]
