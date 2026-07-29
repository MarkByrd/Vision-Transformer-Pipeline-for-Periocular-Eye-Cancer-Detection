"""
mae_dataset.py
--------------
PyTorch Dataset for Phase 1 MAE pretraining on the normal periocular dataset.

Recursively collects all JPEG/PNG images from LEFT/ and RIGHT/ subject folders.
No labels are needed — returns image tensors only.

Usage (imported by train_mae.py):
    from mae_dataset import PeriocularDataset
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ImageNet normalisation (standard for ViT-B/16)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool = True) -> transforms.Compose:
    """Build the augmentation pipeline for MAE pretraining."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class PeriocularDataset(Dataset):
    """
    Dataset for periocular MAE pretraining.

    Args:
        root: Path to periculor_dataset/ (contains LEFT/ and RIGHT/)
        train: If True, use training augmentations; else validation transforms
        val_fraction: Fraction reserved for validation (default 0.10)
        seed: Random seed for reproducible split
        split: 'train' or 'val' — which split this instance represents
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
        self.transform = build_transform(train=(split == "train"))

        # Collect all image paths from both LEFT and RIGHT
        all_paths = []
        for side in ("LEFT", "RIGHT"):
            side_dir = self.root / side
            if not side_dir.exists():
                print(f"[WARN] Missing directory: {side_dir}")
                continue
            for subject_dir in sorted(side_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                for img_file in sorted(subject_dir.iterdir()):
                    if img_file.suffix.lower() in VALID_EXTENSIONS:
                        all_paths.append(img_file)

        if not all_paths:
            raise RuntimeError(f"No images found under {self.root}")

        # Reproducible train/val split
        rng = random.Random(seed)
        all_paths_shuffled = all_paths.copy()
        rng.shuffle(all_paths_shuffled)

        n_val = max(1, int(len(all_paths_shuffled) * val_fraction))
        if split == "val":
            self.paths = all_paths_shuffled[:n_val]
        else:
            self.paths = all_paths_shuffled[n_val:]

        print(f"[PeriocularDataset] {split.upper()} — {len(self.paths)} images")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # On rare corrupt file: return black image
            img = Image.new("RGB", (224, 224), 0)
        return self.transform(img)
