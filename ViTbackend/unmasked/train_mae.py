"""
train_mae.py
------------
Phase 1 — MAE pretraining on the normal periocular dataset.

Trains the MAE encoder to reconstruct masked patches of healthy eye images,
teaching it periocular anatomy without labels.

Key hyperparameters (matching plan):
  - Optimizer : AdamW, LR 1.5e-4, weight_decay 0.05, betas (0.9, 0.95)
  - Scheduler : CosineAnnealingLR, 200 epochs, min LR 1e-6
  - Batch size: 32
  - Mask ratio: 75%
  - Validation: 10% random split

Output:
  - outputs/best_encoder.pt           — encoder-only weights (best val loss)
  - outputs/mae_checkpoint_epoch_*.pt — full checkpoint every 10 epochs
  - outputs/mae_loss_log.csv          — per-epoch train & val loss

Usage:
    python train_mae.py
    python train_mae.py --epochs 200 --batch_size 32 --resume
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mae_dataset import PeriocularDataset
from mae_model import MAEModel

DEFAULTS = dict(
    root=r"..\..\periculor_dataset",
    output=r"outputs",
    epochs=200,
    batch_size=32,
    lr=1.5e-4, 
    
    weight_decay=0.05,
    min_lr=1e-6,
    mask_ratio=0.75,
    checkpoint_every=10,
    num_workers=4,
    seed=42,
)


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        loss, _, _ = model(images)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.6f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images in dataloader:
        images = images.to(device, non_blocking=True)
        loss, _, _ = model(images)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: MAE Pretraining")
    parser.add_argument("--root", default=DEFAULTS["root"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--min_lr", type=float, default=DEFAULTS["min_lr"])
    parser.add_argument("--mask_ratio", type=float, default=DEFAULTS["mask_ratio"])
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULTS["checkpoint_every"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Phase 1 — MAE Pretraining")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ── Datasets ──
    print("Loading datasets...")
    train_dataset = PeriocularDataset(root=args.root, split="train", seed=args.seed)
    val_dataset = PeriocularDataset(root=args.root, split="val", seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model ──
    model = MAEModel(mask_ratio=args.mask_ratio).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MAE model: {n_params:.1f}M parameters\n")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    # Resume logic
    start_epoch = 1
    best_val_loss = float("inf")
    log_path = output_dir / "mae_loss_log.csv"

    if args.resume:
        # Find latest checkpoint
        ckpts = sorted(output_dir.glob("mae_checkpoint_epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"Resuming from: {latest}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}\n")
        else:
            print("No checkpoint found, starting from scratch.\n")

    # CSV logger
    write_header = not log_path.exists() or not args.resume
    log_file = open(log_path, "a" if args.resume else "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_sec"])

    # Training Loop
    print(f"Training for epochs {start_epoch}–{args.epochs} "
          f"({len(train_loader)} train batches, {len(val_loader)} val batches)\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:>3d}/{args.epochs}  │  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={current_lr:.2e}  time={elapsed:.0f}s")

        # Log
        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{current_lr:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        # Save best encoder
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            encoder_path = output_dir / "best_encoder.pt"
            torch.save(model.get_encoder_state_dict(), encoder_path)
            print(f"  ★ New best val_loss={val_loss:.6f} — saved encoder → {encoder_path}")

        # Periodic checkpoint
        if epoch % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"mae_checkpoint_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

    log_file.close()

    print(f"\n{'='*60}")
    print(f"MAE Pretraining Complete")
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"Encoder saved : {output_dir / 'best_encoder.pt'}")
    print(f"Loss log : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
