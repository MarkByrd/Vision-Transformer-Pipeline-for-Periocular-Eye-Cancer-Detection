"""
finetune.py
-----------
Phase 2 — Fine-tune the MAE-pretrained ViT-B/16 on the eyelid cancer dataset.

Strategy:
  - Load best_encoder.pt into ViT-B/16 backbone
  - Freeze blocks 0–6 + patch_embed + pos_embed (preserve periocular anatomy)
  - Unfreeze blocks 7–11 + final layer norm (5 trainable blocks)
  - Replace classification head with MLP: 768→256(ReLU, Dropout 0.4)→1(sigmoid)
  - Tri-level differential LR:
      block 7 (newly unfrozen)  3e-5   — conservative, protects mid-level features
      blocks 8–11 + norm        1e-4   — standard backbone LR
      MLP head                  1e-3   — fast head learning
  - Loss: BCEWithLogitsLoss with pos_weight (handles class imbalance)
  - Scheduler: StepLR, step_size=15, gamma=0.3  (gentler than previous 10/0.1)
  - Batch size 32, tracks AUC/F1/sensitivity/specificity per epoch

Output:
  - outputs/freeze7/best_classifier.pt
  - outputs/freeze7/finetune_log.csv
  - outputs/freeze7/finetune_checkpoint_epoch_*.pt

Usage:
    python finetune.py
    python finetune.py --epochs 50 --batch_size 32
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from cancer_dataset import CancerDataset, TestDataset


# ──────────────────────────────────────────────────────────────
#  Classifier head
# ──────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """768 → 256 (ReLU, Dropout) → 1"""

    def __init__(self, in_dim: int = 768, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────
#  Build model
# ──────────────────────────────────────────────────────────────

def build_classifier(encoder_path: str, device: torch.device, freeze_blocks: int = 8):
    """
    Build the fine-tuning classifier:
      1. Create ViT-B/16 (pretrained=False)
      2. Load MAE encoder weights
      3. Freeze blocks 0..freeze_blocks-1
      4. Replace head with MLPHead
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,       # remove default head
        global_pool="token", # use CLS token
    )

    # Load MAE encoder weights
    enc_path = Path(encoder_path)
    if enc_path.exists():
        state_dict = torch.load(enc_path, map_location=device, weights_only=True)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded encoder from {enc_path}")
        print(f"  missing keys : {len(msg.missing_keys)}")
        print(f"  unexpected   : {len(msg.unexpected_keys)}")
    else:
        print(f"[WARN] Encoder not found: {enc_path} — using random init")

    # Freeze: patch_embed, pos_embed, blocks 0..freeze_blocks-1
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    model.pos_embed.requires_grad = False
    model.cls_token.requires_grad = False

    for i in range(freeze_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    # Unfreeze: blocks freeze_blocks..11, norm
    for i in range(freeze_blocks, len(model.blocks)):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True

    # Replace head
    head = MLPHead(in_dim=model.embed_dim, hidden=256, dropout=0.4)

    # Count trainable params
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"\nEncoder: {n_frozen/1e6:.1f}M frozen, {n_unfrozen/1e6:.1f}M unfrozen")
    print(f"Head   : {n_head/1e6:.2f}M parameters")

    return model, head


class FineTuneModel(nn.Module):
    """Wrapper combining ViT backbone + MLP head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)   # (B, 768) — CLS token via global_pool="token"
        logits = self.head(features)   # (B,)
        return logits


# ──────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    """Compute AUC, F1, sensitivity, specificity."""
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    f1 = f1_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ──────────────────────────────────────────────────────────────
#  Training & validation loops
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f"  Epoch [{epoch}/{total}] Batch [{batch_idx}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / max(len(loader), 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(len(loader), 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = avg_loss
    return metrics


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

DEFAULTS = dict(
    dataset=r"outputs\clahe_cancer",
    encoder=r"outputs\best_encoder.pt",
    output=r"outputs\freeze7",
    epochs=50,
    batch_size=8,      # reduced for 6 GB GPU (5 unfrozen blocks need more grad memory)
    lr_block7=3e-5,    # newly unfrozen block — conservative rate
    lr_backbone=1e-4,  # blocks 8-11 + norm
    lr_head=1e-3,
    weight_decay=0.01,
    freeze_blocks=7,
    num_workers=0,     # 0 on Windows — multiprocessing workers hold extra GPU memory
    seed=42,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tuning")
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--encoder", default=DEFAULTS["encoder"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr_block7", type=float, default=DEFAULTS["lr_block7"])
    parser.add_argument("--lr_backbone", type=float, default=DEFAULTS["lr_backbone"])
    parser.add_argument("--lr_head", type=float, default=DEFAULTS["lr_head"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--freeze_blocks", type=int, default=DEFAULTS["freeze_blocks"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Phase 2 — Cancer Fine-tuning")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # ── Datasets ──
    train_ds = CancerDataset(root=args.dataset, split="train", seed=args.seed)
    val_ds = CancerDataset(root=args.dataset, split="val", seed=args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    backbone, head = build_classifier(args.encoder, device, args.freeze_blocks)
    model = FineTuneModel(backbone, head).to(device)

    # ── Loss (weighted for imbalance) ──
    pos_weight = torch.tensor([train_ds.get_pos_weight()], device=device)
    print(f"\npos_weight for BCEWithLogitsLoss: {pos_weight.item():.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Tri-level optimizer ──
    # Group 0: block freeze_blocks (newly unfrozen) — lower LR to avoid disruption
    # Group 1: blocks (freeze_blocks+1)..11 + norm  — standard backbone LR
    # Group 2: MLP head                              — fastest LR
    block7_params    = list(model.backbone.blocks[args.freeze_blocks].parameters())
    block7_param_ids = {id(p) for p in block7_params}
    upper_params     = [
        p for p in model.backbone.parameters()
        if p.requires_grad and id(p) not in block7_param_ids
    ]
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": block7_params, "lr": args.lr_block7},
        {"params": upper_params,  "lr": args.lr_backbone},
        {"params": head_params,   "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # Gentler schedule: LR halved every 15 epochs (was /10 every 10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

    # ── Resume ──
    start_epoch = 1
    best_val_auc = 0.0
    log_path = output_dir / "finetune_log.csv"

    if args.resume:
        ckpts = sorted(output_dir.glob("finetune_checkpoint_epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"\nResuming from: {latest}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_auc = ckpt.get("best_val_auc", 0.0)
            print(f"Resuming from epoch {start_epoch}, best_val_auc={best_val_auc:.4f}\n")

    # ── CSV logger ──
    write_header = not log_path.exists() or not args.resume
    log_file = open(log_path, "a" if args.resume else "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "epoch", "train_loss", "val_loss",
            "train_auc", "val_auc",
            "train_f1", "val_f1",
            "val_sensitivity", "val_specificity",
            "lr_block7", "lr_backbone", "lr_head", "time_sec",
        ])

    # ── Training ──
    print(f"\nTraining epochs {start_epoch}–{args.epochs}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_b7 = optimizer.param_groups[0]["lr"]
        lr_bb = optimizer.param_groups[1]["lr"]
        lr_hd = optimizer.param_groups[2]["lr"]

        print(f"Epoch {epoch:>3d}/{args.epochs}  │  "
              f"loss={val_m['loss']:.4f}  AUC={val_m['auc']:.4f}  "
              f"F1={val_m['f1']:.4f}  sens={val_m['sensitivity']:.3f}  "
              f"spec={val_m['specificity']:.3f}  time={elapsed:.0f}s")

        log_writer.writerow([
            epoch,
            f"{train_m['loss']:.4f}", f"{val_m['loss']:.4f}",
            f"{train_m['auc']:.4f}", f"{val_m['auc']:.4f}",
            f"{train_m['f1']:.4f}", f"{val_m['f1']:.4f}",
            f"{val_m['sensitivity']:.4f}", f"{val_m['specificity']:.4f}",
            f"{lr_b7:.2e}", f"{lr_bb:.2e}", f"{lr_hd:.2e}", f"{elapsed:.1f}",
        ])
        log_file.flush()

        # Save best by val AUC
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_path = output_dir / "best_classifier.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best val AUC={val_m['auc']:.4f} — saved → {best_path}")

        # Periodic checkpoint
        if epoch % 5 == 0:
            ckpt_path = output_dir / f"finetune_checkpoint_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_auc": best_val_auc,
            }, ckpt_path)
            print(f"  ✓ Checkpoint → {ckpt_path}")

    log_file.close()

    print(f"\n{'='*60}")
    print(f"  Fine-tuning Complete")
    print(f"  Best val AUC : {best_val_auc:.4f}")
    print(f"  Model saved  : {output_dir / 'best_classifier.pt'}")
    print(f"  Log          : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
