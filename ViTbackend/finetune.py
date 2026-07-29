"""
finetune.py  (ViTbackend — iris-masked + gradual unfreezing)
---------------------------------------------------------------
Phase 2 — Fine-tuning the MAE-pretrained ViT-B/16 classifier on the
iris-masked eyelid cancer dataset.

Gradual unfreezing schedule (per spec):
  Epochs  1- 5:  only block 12 + head trainable
  Epochs  6-10:  blocks 10-12 + head trainable
  Epochs 11-20:  blocks 8-12 + head, with layer-wise LR decay
                 decay factor 0.75 per block going down:
                   block 12 -> base_lr
                   block 11 -> base_lr * 0.75
                   block 10 -> base_lr * 0.75^2
                   block  9 -> base_lr * 0.75^3
                   block  8 -> base_lr * 0.75^4

Optimizer:  AdamW
Scheduler:  CosineAnnealingLR (restarts at each phase boundary)
Base LR:    5e-5 for backbone, 1e-4 for classification head
Weight decay: 0.05

Output:  outputs/vit_finetuned_masked.pth   (best val AUC)
         outputs/finetune_masked_log.csv

Usage:
    python finetune.py
    python finetune.py --dataset D:/path/to/Eyelid_Cancer_Dataset
                       --encoder outputs/mae_normal_maskediris.pth
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from cancer_dataset import CancerDataset, TestDataset
from torch.utils.data import DataLoader

DEFAULTS = dict(
    dataset=r"..\Eyelid_Cancer_Dataset",
    encoder=r"outputs\mae_normal_maskediris.pth",
    output=r"outputs",
    batch_size=8,
    base_lr_backbone=5e-5,
    base_lr_head=1e-4,
    llrd_factor=0.75,
    weight_decay=0.05,
    num_workers=0,
    seed=42,
)

PHASE_SCHEDULE = [
    (1,   5,  12),
    (6,  10,  10),
    (11, 20,   8),
]

class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FineTuneModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_classifier(encoder_path: str, device: torch.device) -> FineTuneModel:
    backbone = timm.create_model(
        "vit_base_patch16_224", pretrained=False,
        num_classes=0, global_pool="token",
    )
    enc_p = Path(encoder_path)
    if enc_p.exists():
        sd  = torch.load(enc_p, map_location=device, weights_only=True)
        msg = backbone.load_state_dict(sd, strict=False)
        print(f"  Loaded encoder from {enc_p} "
              f"({len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected)")
    else:
        print(f"  [WARN] Encoder not found: {enc_p} — using random init")

    head  = MLPHead(in_dim=backbone.embed_dim)
    model = FineTuneModel(backbone, head).to(device)
    return model

def _freeze_all(model: FineTuneModel):
    for p in model.backbone.parameters():
        p.requires_grad = False


def _unfreeze_from(model: FineTuneModel, first_block: int):
    """Unfreeze blocks [first_block .. 11] + final LayerNorm."""
    _freeze_all(model)
    for i in range(first_block, len(model.backbone.blocks)):
        for p in model.backbone.blocks[i].parameters():
            p.requires_grad = True
    for p in model.backbone.norm.parameters():
        p.requires_grad = True


def build_optimizer(
    model: FineTuneModel,
    first_block: int,
    base_lr_backbone: float,
    base_lr_head: float,
    llrd: float,
    weight_decay: float,
    apply_llrd: bool,
) -> torch.optim.Optimizer:
    
    _unfreeze_from(model, first_block)

    param_groups = []
    n_blocks = len(model.backbone.blocks)  # 12 for ViT-B

    # Backbone blocks (from first_block to 11)
    for i in range(first_block, n_blocks):
        if apply_llrd:
            # Block 12 (index 11) gets base_lr; lower blocks decay by llrd
            exponent = (n_blocks - 1) - i
            lr_i = base_lr_backbone * (llrd ** exponent)
        else:
            lr_i = base_lr_backbone

        block_params = [p for p in model.backbone.blocks[i].parameters() if p.requires_grad]
        if block_params:
            param_groups.append({"params": block_params, "lr": lr_i,
                                 "name": f"block_{i}"})

    # Final LayerNorm — same LR as block 11 (highest backbone LR)
    norm_params = [p for p in model.backbone.norm.parameters() if p.requires_grad]
    if norm_params:
        param_groups.append({"params": norm_params, "lr": base_lr_backbone,
                             "name": "norm"})

    # Head — always highest LR
    head_params = list(model.head.parameters())
    param_groups.append({"params": head_params, "lr": base_lr_head, "name": "head"})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    f1    = f1_score(labels, preds, zero_division=0)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"auc": auc, "f1": f1, "sensitivity": sens, "specificity": spec}


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train(is_train)
    labels_all, probs_all = [], []
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    metrics  = compute_metrics(np.array(labels_all), np.array(probs_all))
    return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tuning (iris-masked)")
    parser.add_argument("--dataset",default=DEFAULTS["dataset"])
    parser.add_argument("--encoder", default=DEFAULTS["encoder"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--base_lr_backbone", type=float, default=DEFAULTS["base_lr_backbone"])
    parser.add_argument("--base_lr_head",type=float, default=DEFAULTS["base_lr_head"])
    parser.add_argument("--llrd_factor",type=float, default=DEFAULTS["llrd_factor"])
    parser.add_argument("--weight_decay",type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed",type=int, default=DEFAULTS["seed"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = PHASE_SCHEDULE[-1][1]

    print(f"\n{'='*65}")
    print(f"Phase 2 - Fine-tuning (iris-masked, gradual unfreezing)")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Encoder: {args.encoder}")
    print(f"Output : {output_dir}")
    print(f"Schedule : {PHASE_SCHEDULE}")
    print(f"{'='*65}\n")

    train_ds = CancerDataset(root=args.dataset, split="train", seed=args.seed)
    val_ds   = CancerDataset(root=args.dataset, split="val", seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    pos_weight = torch.tensor([train_ds.get_pos_weight()], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_classifier(args.encoder, device)

    log_path = output_dir / "finetune_masked_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "phase", "first_block",
            "train_loss", "val_loss",
            "train_auc", "val_auc",
            "train_f1", "val_f1",
            "val_sensitivity", "val_specificity",
            "time_sec",
        ])

    best_val_auc  = 0.0
    optimizer = None
    scheduler = None
    current_phase = None

    for epoch in range(1, total_epochs + 1):

        # Determine current phase
        for phase_idx, (start, end, first_block) in enumerate(PHASE_SCHEDULE):
            if start <= epoch <= end:
                break

        phase_name  = f"phase{phase_idx + 1}"
        apply_llrd  = (phase_idx == 2)   # only in final phase

        # Rebuild optimizer/scheduler at phase transitions
        if current_phase != phase_idx:
            current_phase = phase_idx
            phase_epochs  = PHASE_SCHEDULE[phase_idx][1] - PHASE_SCHEDULE[phase_idx][0] + 1

            optimizer = build_optimizer(
                model, first_block,
                args.base_lr_backbone, args.base_lr_head,
                args.llrd_factor, args.weight_decay, apply_llrd,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase_epochs, eta_min=1e-7,
            )

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n[Phase {phase_idx+1}] epochs {start}-{end}  "
                  f"unfreezing from block {first_block}  "
                  f"LLRD={'yes' if apply_llrd else 'no'}  "
                  f"trainable_params={trainable/1e6:.2f}M")
            if apply_llrd:
                for g in optimizer.param_groups:
                    print(f"  {g.get('name','?'):<10}  lr={g['lr']:.2e}")

        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_m   = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>2d}/{total_epochs}  [{phase_name}]  "
              f"train_auc={train_m['auc']:.4f}  val_auc={val_m['auc']:.4f}  "
              f"val_f1={val_m['f1']:.4f}  time={elapsed:.0f}s")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, phase_name, first_block,
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{train_m['auc']:.4f}", f"{val_m['auc']:.4f}",
                f"{train_m['f1']:.4f}",  f"{val_m['f1']:.4f}",
                f"{val_m['sensitivity']:.4f}", f"{val_m['specificity']:.4f}",
                f"{elapsed:.1f}",
            ])

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            save_path    = output_dir / "vit_finetuned_masked.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  * Best val_auc={best_val_auc:.4f} -> saved {save_path}")

    print(f"\n{'='*65}")
    print(f"Fine-tuning complete")
    print(f"Best val AUC : {best_val_auc:.4f}")
    print(f"Model saved  : {output_dir / 'vit_finetuned_masked.pth'}")
    print(f"Log: {log_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
