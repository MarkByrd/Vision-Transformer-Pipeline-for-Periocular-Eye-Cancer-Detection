import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from iris_masking import mask_iris

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

PATCH_SIZE = 16
GRID       = 14   # 224 / 16


def _compute_iris_patch_mask(iris_center, iris_radius: int) -> torch.Tensor:
    cx, cy = iris_center
    mask = torch.zeros(GRID * GRID, dtype=torch.float32)
    for row in range(GRID):
        for col in range(GRID):
            patch_cx = (col + 0.5) * PATCH_SIZE
            patch_cy = (row + 0.5) * PATCH_SIZE
            dist = ((patch_cx - cx) ** 2 + (patch_cy - cy) ** 2) ** 0.5
            if dist <= iris_radius:
                mask[row * GRID + col] = 1.0
    return mask


def _build_geo_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])


def _build_photo_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


_PHOTO_VAL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PeriocularDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.10,
        seed: int = 42,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
        self.root  = Path(root)
        self.split = split

        self.geo_transform   = _build_geo_transform(train=(split == "train"))
        self.photo_transform = _build_photo_transform() if split == "train" else _PHOTO_VAL

        all_paths = []
        for side in ("LEFT", "RIGHT"):
            side_dir = self.root / side
            if not side_dir.exists():
                print(f"[WARN] Missing directory: {side_dir}")
                continue
            for item in sorted(side_dir.iterdir()):
                if item.is_dir():
                    # Nested structure: side/subject/img.jpg
                    for img_file in sorted(item.iterdir()):
                        if img_file.suffix.lower() in VALID_EXTENSIONS:
                            all_paths.append(img_file)
                elif item.suffix.lower() in VALID_EXTENSIONS:
                    # Flat structure: side/img.jpg
                    all_paths.append(item)

        if not all_paths:
            raise RuntimeError(f"No images found under {self.root}")

        rng = random.Random(seed)
        shuffled = all_paths.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_fraction))
        self.paths = shuffled[:n_val] if split == "val" else shuffled[n_val:]

        print(f"[PeriocularDataset] {split.upper()} — {len(self.paths)} images "
              f"(iris masking enabled)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            img_pil = Image.new("RGB", (224, 224), 128)

        # Stage 1: geometric transforms → 224×224 PIL
        img_pil = self.geo_transform(img_pil)

        # Stage 2: iris masking on the 224×224 RGB array
        img_np = np.array(img_pil, dtype=np.uint8)
        masked_np, meta = mask_iris(img_np, image_path=str(img_path))

        # Stage 3: compute which patches are inside the iris region
        iris_patch_mask = _compute_iris_patch_mask(
            meta["iris_center"], meta["iris_radius"]
        )

        # Stage 4: photometric transforms + normalise
        masked_pil  = Image.fromarray(masked_np)
        img_tensor  = self.photo_transform(masked_pil)

        return img_tensor, iris_patch_mask
