"""
evaluate.py
-----------
Phase 3 — Evaluate the fine-tuned classifier on the held-out test set.

Produces:
  - AUC-ROC curve         → outputs/roc_curve.png
  - Confusion matrix      → outputs/confusion_matrix.png
  - Per-image predictions → printed table
  - Comparison table vs Jiang et al. benchmarks

Usage:
    python evaluate.py
    python evaluate.py --test_dir "new testing dataset" --classifier outputs/best_classifier.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from cancer_dataset import TestDataset
from finetune import FineTuneModel, MLPHead


DEFAULTS = dict(
    test_dir="test_dataset",
    classifier="outputs/best_classifier.pt",
    output="outputs",
    batch_size=8,
    freeze_blocks=8,
)


def build_model(classifier_path: str, device: torch.device, freeze_blocks: int = 8):
    """Rebuild the same architecture used in finetune.py and load weights."""
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="token",
    )
    head = MLPHead(in_dim=backbone.embed_dim)
    model = FineTuneModel(backbone, head).to(device)

    state_dict = torch.load(classifier_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded classifier from: {classifier_path}")
    return model


@torch.no_grad()
def predict(model, loader, device):
    """Run inference, return labels, probs, and logits as numpy arrays."""
    all_labels = []
    all_probs = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_logits.extend(logits.cpu().numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_logits)


def plot_roc(labels, probs, output_dir: Path):
    """Generate and save the AUC-ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5,
            label=f"MAE-ViT  AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)

    # Benchmark lines
    ax.axhline(y=0.945, color="#10b981", ls=":", lw=1.5,
               label="Jiang et al. internal AUC = 0.945")
    ax.axhline(y=0.915, color="#f59e0b", ls=":", lw=1.5,
               label="Jiang et al. external AUC = 0.915")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve - Eyelid Cancer Classification", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = output_dir / "roc_curve.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved → {save_path}")
    return roc_auc


def plot_confusion(labels, preds, output_dir: Path):
    """Generate and save the confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = output_dir / "confusion_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")
    return cm


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Evaluation")
    parser.add_argument("--test_dir", default=DEFAULTS["test_dir"])
    parser.add_argument("--classifier", default=DEFAULTS["classifier"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--freeze_blocks", type=int, default=DEFAULTS["freeze_blocks"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Phase 3 — Evaluation")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load model
    if not Path(args.classifier).exists():
        print(f"ERROR: Classifier not found: {args.classifier}", file=sys.stderr)
        sys.exit(1)

    model = build_model(args.classifier, device, args.freeze_blocks)

    # Load test set
    test_ds = TestDataset(root=args.test_dir)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Predict
    labels, probs, logits = predict(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    # ── Per-image results ──
    filenames = test_ds.get_filenames()
    print("\n  Per-image predictions:")
    print(f"  {'File':<25} {'True':>6} {'Pred':>6} {'Prob':>8}")
    print(f"  {'─'*50}")
    for fname, true, pred, prob in zip(filenames, labels, preds, probs):
        label_str = "MAL" if true == 1 else "BEN"
        pred_str = "MAL" if pred == 1 else "BEN"
        correct = "✓" if true == pred else "✗"
        print(f"  {fname:<25} {label_str:>6} {pred_str:>6} {prob:>8.4f}  {correct}")

    # ── Aggregate metrics ──
    try:
        test_auc = roc_auc_score(labels, probs)
    except ValueError:
        test_auc = float("nan")
    test_f1 = f1_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"AUC-ROC     : {test_auc:.4f}")
    print(f"F1 Score    : {test_f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

    # ── Benchmark comparison ──
    print(f"\n{'─'*60}")
    print(f"BENCHMARK COMPARISON")
    print(f"{'─'*60}")
    print(f"{'Method':<35} {'AUC':>8}")
    print(f"{'─'*45}")
    print(f"{'Jiang et al. (internal)':<35} {'0.945':>8}")
    print(f"{'Jiang et al. (external)':<35} {'0.915':>8}")
    print(f"{'MAE-ViT, context-aware':<35} {test_auc:>8.4f}")
    print(f"{'='*60}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_roc(labels, probs, output_dir)
    plot_confusion(labels, preds, output_dir)

    print(f"\n  All outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
