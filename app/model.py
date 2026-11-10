"""
Iris-masked ViT-B/16 classifier architecture + GMAR attention visualisation.

These classes mirror ViTbackend/finetune.py (MLPHead, FineTuneModel) and
ViTbackend/gmar.py (ViTGMAR, overlay_cam_on_image). They are duplicated here so
the app does not import the training code (and its sklearn / matplotlib /
dataset dependencies). The layer names must stay identical to the training
versions, or saved checkpoints will no longer load.
"""

import types

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.attention import maybe_add_mask, resolve_self_attn_mask


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
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model() -> FineTuneModel:
    """ViT-B/16 backbone (CLS token, no classifier) + 768->256->1 MLP head."""
    backbone = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token"
    )
    return FineTuneModel(backbone, MLPHead(in_dim=backbone.embed_dim))


class ViTGMAR:
    """Gradient-weighted Multi-head Attention Rollout.

    Patches every attention block so its softmax attention map keeps its
    gradient, then rolls the gradient-weighted attention out across all 12
    blocks to get a 14x14 saliency map for the CLS token.
    """

    def __init__(self, model: FineTuneModel):
        self.model = model
        self.attn_modules = []
        for block in self.model.backbone.blocks:
            self._wrap_attention(block.attn)
            self.attn_modules.append(block.attn)

    @staticmethod
    def _wrap_attention(module):
        def wrapped_forward(attn_self, x, attn_mask=None, is_causal=False):
            B, N, C = x.shape
            qkv = attn_self.qkv(x).reshape(
                B, N, 3, attn_self.num_heads, attn_self.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn_self.q_norm(q), attn_self.k_norm(k)
            q = q * attn_self.scale

            attn = q @ k.transpose(-2, -1)
            attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
            attn = maybe_add_mask(attn, attn_bias)
            attn = attn.softmax(dim=-1)
            attn = attn_self.attn_drop(attn)
            attn = attn.requires_grad_()
            attn.retain_grad()
            attn_self.attn_weights = attn

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, attn_self.attn_dim)
            x = attn_self.norm(x)
            x = attn_self.proj(x)
            x = attn_self.proj_drop(x)
            return x

        module.forward = types.MethodType(wrapped_forward, module)

    @torch.enable_grad()
    def __call__(self, input_tensor: torch.Tensor):
        """Returns (logit, cam) for a single image batch of shape (1, 3, 224, 224)."""
        for module in self.attn_modules:
            module.attn_weights = None
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logit = self.model(input_tensor)
        self.model.zero_grad()
        logit.backward()
        return logit.detach(), self._compute_gmar()

    def _compute_gmar(self) -> np.ndarray:
        device = self.attn_modules[0].attn_weights.device
        N = self.attn_modules[0].attn_weights.shape[-1]
        joint_attention = torch.eye(N, device=device)
        identity = torch.eye(N, device=device)

        for module in self.attn_modules:
            attn = module.attn_weights[0]
            grad = module.attn_weights.grad[0]

            head_importance = F.relu(grad.mean(dim=(1, 2)))
            if head_importance.sum() < 1e-8:
                head_importance = torch.ones_like(head_importance)
            head_importance = head_importance / head_importance.sum()

            attn_weighted = torch.einsum("hij,h->ij", attn, head_importance)
            attn_weighted = attn_weighted / attn_weighted.sum(dim=-1, keepdim=True)
            attn_residual = attn_weighted + identity
            attn_residual = attn_residual / attn_residual.sum(dim=-1, keepdim=True)

            joint_attention = attn_residual @ joint_attention

        cam = joint_attention[0, 1:].reshape(14, 14).detach().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(cam)


def overlay_cam_on_image(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] saliency map (any size) onto an RGB uint8 image as a JET heatmap."""
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * img_np.astype(float) + alpha * heatmap.astype(float)
    return np.clip(overlay, 0, 255).astype(np.uint8)
