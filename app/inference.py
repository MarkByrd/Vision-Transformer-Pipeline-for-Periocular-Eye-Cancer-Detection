"""
Inference for the iris-masked ViT classifier, independent of any web framework.

    clf = Classifier.load("models/vit_finetuned_masked.pth")
    result = clf.predict(pil_image)

Preprocessing matches ViTbackend/cancer_dataset.py (TestDataset), i.e. what the
model was fine-tuned and evaluated on:
CLAHE on L channel -> centre-crop to square -> resize 224x224 -> iris masking
(left unmasked if no iris is found) -> ImageNet normalisation.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from iris_masking import mask_iris
from model import ViTGMAR, build_model, overlay_cam_on_image

CLASSES = ("Benign", "Malignant")
INPUT_SIZE = 224

_normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(rgb: np.ndarray) -> np.ndarray:
    """Contrast-limited histogram equalisation on the L channel of LAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_CLAHE.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    s = min(w, h)
    return image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def preprocess(image: Image.Image):
    """Returns (cropped_original, masked_input, mask_meta); images are 224x224 RGB uint8.

    cropped_original is for display; masked_input is exactly what the model sees
    before normalisation. Same steps and order as ViTbackend/cancer_dataset.py.
    """
    image = image.convert("RGB")
    size = (INPUT_SIZE, INPUT_SIZE)
    original = np.array(center_crop_square(image).resize(size, Image.BILINEAR))

    enhanced = Image.fromarray(apply_clahe(np.array(image)))
    enhanced = np.array(center_crop_square(enhanced).resize(size, Image.BILINEAR))
    masked, meta = mask_iris(enhanced, skip_if_no_iris=True)
    return original, masked, meta


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Classifier:
    def __init__(self, model, device: torch.device):
        self.device = device
        self.model = model
        self.gmar = ViTGMAR(model)
        # GMAR stores per-request attention maps on the model's modules, so
        # only one prediction may run at a time.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device | None = None) -> "Classifier":
        device = device or pick_device()
        model = build_model()
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return cls(model, device)

    def predict(self, image: Image.Image) -> dict:
        """Classify one image.

        Returns a dict with the predicted class, its confidence (0-100), the
        malignancy probability (0-1), how the iris was found ("mediapipe",
        "hough", or "skipped" if none was found and the image was left unmasked),
        and three 224x224 RGB uint8 arrays: the cropped original, the
        preprocessed model input, and the GMAR overlay.
        """
        original, masked, mask_meta = preprocess(image)
        tensor = _normalise(Image.fromarray(masked)).unsqueeze(0).to(self.device)

        with self._lock:
            logit, cam = self.gmar(tensor)

        p_malignant = torch.sigmoid(logit).item()
        idx = int(p_malignant >= 0.5)
        confidence = p_malignant if idx == 1 else 1 - p_malignant

        return {
            "prediction": CLASSES[idx],
            "confidence": round(confidence * 100, 2),
            "probability_malignant": round(p_malignant, 4),
            "iris_method": mask_meta["method"],
            "original": original,
            "masked": masked,
            "heatmap": overlay_cam_on_image(masked, cam, alpha=0.45),
        }
