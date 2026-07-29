import argparse
import sys
import types
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.layers.attention import maybe_add_mask, resolve_self_attn_mask
from torchvision import transforms

from cancer_dataset import TestDataset, build_val_transform, IMAGENET_MEAN, IMAGENET_STD
from finetune import FineTuneModel, MLPHead

DEFAULTS = dict(
    test_dir="test_dataset",
    classifier="outputs/best_classifier.pt",
    output="outputs/gmar",
)


class ViTGMAR:
    def __init__(self, model: FineTuneModel):
        self.model = model
        self.attn_modules = []
        self._patch_attention_modules()

    def _patch_attention_modules(self):
        for block in self.model.backbone.blocks:
            attn_module = block.attn
            self._wrap_attention(attn_module)
            self.attn_modules.append(attn_module)

    def _wrap_attention(self, module):
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

    def _reset_attentions(self):
        for module in self.attn_modules:
            module.attn_weights = None
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                module.attn_weights.grad = None

    @torch.enable_grad()
    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self._reset_attentions()
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logit = self.model(input_tensor)
        self.model.zero_grad()
        logit.backward()

        cam = self._compute_gmar()
        return cam

    def _compute_gmar(self) -> np.ndarray:
        if not self.attn_modules:
            raise RuntimeError("No attention modules registered for GMAR.")

        device = self.attn_modules[0].attn_weights.device
        N = self.attn_modules[0].attn_weights.shape[-1]
        joint_attention = torch.eye(N, device=device)
        identity = torch.eye(N, device=device)

        for module in self.attn_modules:
            attn = module.attn_weights[0]
            grad = module.attn_weights.grad[0]

            head_importance = F.relu(grad.mean(dim=(1, 2)))
            if head_importance.sum() < 1e-8:
                head_importance = torch.ones_like(head_importance, device=device)
            head_importance = head_importance / head_importance.sum()

            attn_weighted = torch.einsum('hij,h->ij', attn, head_importance)
            attn_weighted = attn_weighted / attn_weighted.sum(dim=-1, keepdim=True)
            attn_residual = attn_weighted + identity
            attn_residual = attn_residual / attn_residual.sum(dim=-1, keepdim=True)

            joint_attention = attn_residual @ joint_attention

        cam = joint_attention[0, 1:]
        cam = cam.reshape(14, 14).detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def overlay_cam_on_image(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * img_np.astype(float) + alpha * heatmap.astype(float)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="GMAR for ViT")
    parser.add_argument("--test_dir", default=DEFAULTS["test_dir"])
    parser.add_argument("--classifier", default=DEFAULTS["classifier"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GMAR Visualisation")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    if not Path(args.classifier).exists():
        print(f"ERROR: Classifier not found: {args.classifier}", file=sys.stderr)
        sys.exit(1)

    backbone = timm.create_model(
        "vit_base_patch16_224", pretrained=False,
        num_classes=0, global_pool="token",
    )
    head = MLPHead(in_dim=backbone.embed_dim)
    model = FineTuneModel(backbone, head).to(device)
    state_dict = torch.load(args.classifier, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded classifier: {args.classifier}")

    gmar = ViTGMAR(model)

    test_ds = TestDataset(root=args.test_dir)
    filenames = test_ds.get_filenames()

    print(f"\nGenerating GMAR for {len(test_ds)} test images...\n")

    for idx in range(len(test_ds)):
        img_tensor, label = test_ds[idx]
        fname = filenames[idx]
        label_str = "malignant" if label.item() == 1 else "benign"

        input_tensor = img_tensor.unsqueeze(0).to(device)
        cam = gmar(input_tensor)

        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
        pred_str = "malignant" if prob >= 0.5 else "benign"

        img_np = denormalize_image(img_tensor)
        overlay = overlay_cam_on_image(img_np, cam, alpha=0.45)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\nTrue: {label_str}", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet", interpolation="bilinear")
        axes[1].set_title("GMAR Map (14×14)", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"GMAR Overlay\nPred: {pred_str} ({prob:.3f})", fontsize=12)
        axes[2].axis("off")

        fig.suptitle(fname, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        save_path = output_dir / f"gmar_{Path(fname).stem}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx+1}/{len(test_ds)}] {fname}  ->  {save_path.name}")

    print(f"\nAll GMAR outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
