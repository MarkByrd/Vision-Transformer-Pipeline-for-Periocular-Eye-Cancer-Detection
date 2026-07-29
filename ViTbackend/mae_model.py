import math
from typing import Optional

import timm
import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True):
    import numpy as np
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)
    half_dim = embed_dim // 2
    emb_h = _get_1d_sincos(half_dim, grid[0])
    emb_w = _get_1d_sincos(half_dim, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim)), emb], axis=0)
    return torch.from_numpy(emb).float()


def _get_1d_sincos(dim: int, pos):
    import numpy as np
    omega = np.arange(dim // 2, dtype=float) / (dim / 2)
    omega = 1.0 / (10000.0 ** omega)
    out = pos[:, None] * omega[None, :]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DecoderBlock(nn.Module):
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


class MAEModel(nn.Module):

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

        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        encoder_dim = self.encoder.embed_dim  # 768

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(decoder_dim, self.IMG_SIZE // self.PATCH_SIZE, cls_token=True),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.PATCH_PIXELS, bias=True)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, patch_pixels)"""
        p = self.PATCH_SIZE
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_pixels) -> (B, 3, H, W)"""
        p = self.PATCH_SIZE
        h = w = self.IMG_SIZE // p
        B = x.shape[0]
        x = x.reshape(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, h * p, w * p)
        return x

    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor):
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x)
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.encoder.pos_drop(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.enc_to_dec(x)
        B = x.shape[0]
        N = ids_restore.shape[1]
        n_vis = x.shape[1] - 1
        mask_tokens = self.mask_token.expand(B, N - n_vis, -1)
        x_no_cls = x[:, 1:, :]
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1,
                              index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]))
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)
        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full[:, 1:, :])
        return pred

    def forward(self, imgs: torch.Tensor, iris_patch_mask: Optional[torch.Tensor] = None, ):
        
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)

        target = self.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var  = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var.sqrt() + 1e-6)

        loss_per_patch = (pred - target_norm) ** 2
        loss_per_patch = loss_per_patch.mean(dim=-1)    # (B, N)

        if iris_patch_mask is not None:
            effective_mask = mask * (1.0 - iris_patch_mask.to(mask.device, mask.dtype))
        else:
            effective_mask = mask

        denom = effective_mask.sum()
        if denom < 1:
            # Entire batch is inside iris patches- fall back to standard mask
            loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = (loss_per_patch * effective_mask).sum() / denom

        return loss, pred, mask

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()
