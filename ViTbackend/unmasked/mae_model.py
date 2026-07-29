"""
mae_model.py
------------
Masked Autoencoder (MAE) built on ViT-B/16 from timm.

Architecture:
  - Encoder: ViT-B/16 (768-dim, 12 blocks, 12 heads, patch size 16)
  - Decoder: lightweight 4-layer transformer (512-dim, 8 heads)
  - Masking: random 75% of 196 patch tokens
  - Loss:    MSE on masked patches only, with per-patch normalisation

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
"""

import math
from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
  
#  Positional embedding helpers

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True):
    """Generate 2-D sinusoidal positional embedding (numpy → tensor)."""
    import numpy as np

    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)           # (2, G, G)
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, G*G)

    half_dim = embed_dim // 2
    emb_h = _get_1d_sincos(half_dim, grid[0])    # (G*G, D/2)
    emb_w = _get_1d_sincos(half_dim, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (G*G, D)

    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim)), emb], axis=0)
    return torch.from_numpy(emb).float()


def _get_1d_sincos(dim: int, pos: "np.ndarray"):
    import numpy as np
    omega = np.arange(dim // 2, dtype=float) / (dim / 2)
    omega = 1.0 / (10000.0 ** omega)
    out = pos[:, None] * omega[None, :]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ──────────────────────────────────────────────────────────────
#  Lightweight Transformer Decoder
# ──────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Standard pre-norm transformer block for the MAE decoder."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────
#  MAE Model
# ──────────────────────────────────────────────────────────────

class MAEModel(nn.Module):
    """

    
    Masked Autoencoder with ViT-B/16 encoder.

    Args:
        mask_ratio:   fraction of patches to mask (default 0.75)
        decoder_dim:  decoder hidden dimension (default 512)
        decoder_depth: number of decoder transformer blocks (default 4)
        decoder_heads: number of decoder attention heads (default 8)
    """

    PATCH_SIZE = 16
    IMG_SIZE = 224
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
    PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * 3    # 768

    def __init__(
        self,
        mask_ratio: float = 0.75,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio

        # ── encoder (ViT-B/16 from timm, no pretrained weights) ──
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,       # remove classification head
            global_pool="",      # keep all tokens
        )
        encoder_dim = self.encoder.embed_dim  # 768

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # ── decoder ──
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # Fixed sincos positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(decoder_dim, self.IMG_SIZE // self.PATCH_SIZE, cls_token=True),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.PATCH_PIXELS, bias=True)

    # ── helpers ──────────────────────────────────────────────

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, patch_pixels)"""
        p = self.PATCH_SIZE
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_pixels) → (B, 3, H, W)"""
        p = self.PATCH_SIZE
        h = w = self.IMG_SIZE // p
        B = x.shape[0]
        x = x.reshape(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, h * p, w * p)
        return x

    def random_masking(self, x: torch.Tensor):
        """
        Randomly mask patches.

        Args:
            x: (B, N, D)  — patch embeddings (no CLS token)

        Returns:
            x_visible: (B, N_vis, D)
            mask:      (B, N)  — 1 = masked, 0 = visible
            ids_restore: (B, N) — indices to unshuffle
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    # ── forward ──────────────────────────────────────────────

    def forward_encoder(self, x: torch.Tensor):
        """
        Encode with masking.

        Args:
            x: (B, 3, 224, 224)

        Returns:
            latent:      (B, N_vis+1, encoder_dim)   — CLS + visible tokens
            mask:        (B, N)
            ids_restore: (B, N)
        """
        # Patch embed (gives CLS + 196 patch tokens)
        x = self.encoder.patch_embed(x)                # (B, N, D)
        # Add positional embedding (skip CLS pos at index 0)
        x = x + self.encoder.pos_embed[:, 1:, :]

        # Random mask before transformer
        x, mask, ids_restore = self.random_masking(x)  # (B, N_vis, D)

        # Prepend CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, N_vis+1, D)

        # Transformer blocks
        x = self.encoder.pos_drop(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        """
        Decode: re-insert mask tokens, apply decoder, predict pixel values.

        Args:
            x: (B, N_vis+1, encoder_dim)
            ids_restore: (B, N)

        Returns:
            pred: (B, N, patch_pixels) — predicted pixel values for ALL patches
        """
        # Project encoder → decoder dim
        x = self.enc_to_dec(x)   # (B, N_vis+1, decoder_dim)

        # Append mask tokens for masked positions
        B = x.shape[0]
        N = ids_restore.shape[1]
        n_vis = x.shape[1] - 1   # subtract CLS

        mask_tokens = self.mask_token.expand(B, N - n_vis, -1)
        # Separate CLS, combine visible + mask, unshuffle
        x_no_cls = x[:, 1:, :]                                    # (B, N_vis, D_dec)
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)        # (B, N, D_dec)
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]),
        )                                                           # (B, N, D_dec)

        # Re-add CLS
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)          # (B, N+1, D_dec)

        # Add decoder positional embedding
        x_full = x_full + self.decoder_pos_embed

        # Decoder transformer
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)

        # Predict pixel values (exclude CLS)
        pred = self.decoder_pred(x_full[:, 1:, :])    # (B, N, patch_pixels)
        return pred

    def forward(self, imgs: torch.Tensor):
        """
        Full MAE forward pass.

        Args:
            imgs: (B, 3, 224, 224)

        Returns:
            loss:         scalar MSE loss on masked patches (per-patch normalised)
            pred_patches: (B, N, patch_pixels) — predicted pixel values
            mask:         (B, N) — 1 = masked, 0 = visible
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)

        # Target: patchified original pixels
        target = self.patchify(imgs)   # (B, N, patch_pixels)

        # Per-patch normalisation
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var.sqrt() + 1e-6)

        # MSE loss on masked patches only
        loss = (pred - target_norm) ** 2
        loss = loss.mean(dim=-1)                    # (B, N)
        loss = (loss * mask).sum() / mask.sum()     # mean over masked patches

        return loss, pred, mask

    def get_encoder_state_dict(self):
        """Return only the encoder state dict (for saving best_encoder.pt)."""
        return self.encoder.state_dict()
