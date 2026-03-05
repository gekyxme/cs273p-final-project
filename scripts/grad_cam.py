"""
Grad-CAM Visualisation for PawpularityModel

Implements Grad-CAM from scratch using PyTorch forward/backward hooks.
Highlights the image regions the CNN focused on when predicting Pawpularity.

Usage:
    python scripts/grad_cam.py                         # uses best.pt, 8 random images
    python scripts/grad_cam.py --n_images 16           # visualise 16 images
    python scripts/grad_cam.py --checkpoint checkpoints/image_only.pt

Output: runs/gradcam/gradcam_grid.png
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import PawpularityModel
from src.dataset import TABULAR_COLS, IMAGENET_MEAN, IMAGENET_STD

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM using hooks.

    Registers:
        forward hook  → captures feature map activations at target layer
        backward hook → captures gradients flowing back through target layer
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, tabular: torch.Tensor) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            image   : (1, 3, H, W) tensor on model device
            tabular : (1, 12) tensor on model device

        Returns:
            heatmap : (H, W) numpy array in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(image, tabular)     # (1, 1)
        output.backward()                       # scalar → backprop

        # Global average pool the gradients over spatial dims → (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input image size and normalise to [0, 1]
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    original_img: np.ndarray,  # (H, W, 3) uint8
    heatmap: np.ndarray,       # (H, W) float [0, 1]
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap onto the original image. Returns (H, W, 3) uint8."""
    cmap   = plt.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * colored + (1 - alpha) * original_img).astype(np.uint8)
    return blended


def load_model(checkpoint_path: Path, device: torch.device) -> PawpularityModel:
    ck  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PawpularityModel(
        fusion_mode=cfg.get("fusion_mode", "fusion_concat"),
        freeze_backbone=cfg["freeze_backbone"],
        tabular_input_dim=cfg["tabular_input_dim"],
        tabular_hidden_dim=cfg["tabular_hidden_dim"],
        tabular_output_dim=cfg["tabular_output_dim"],
        fusion_hidden_dim=cfg["fusion_hidden_dim"],
        dropout=cfg["dropout"],
    )
    # Remap old checkpoints that used 'fusion_head' → new name 'head'
    state = {
        k.replace("fusion_head.", "head."): v
        for k, v in ck["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg


def get_val_samples(cfg: dict, n: int, seed: int = 42):
    """Return n random samples from the validation split."""
    data_dir = REPO_ROOT / cfg["data_dir"]
    if cfg.get("debug_mode", False):
        csv_path = data_dir / "debug" / "train.csv"
        img_dir  = data_dir / "debug" / "train"
    else:
        csv_path = data_dir / "raw" / "train.csv"
        img_dir  = data_dir / "resized" / "train"

    df = pd.read_csv(csv_path)
    random.seed(seed)
    sample_df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return sample_df, img_dir


def preprocess_image(img_path: Path) -> tuple:
    """Return (tensor (1,3,224,224), original numpy array (224,224,3))."""
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.LANCZOS)
    original = np.array(img)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = tfm(img).unsqueeze(0)   # (1, 3, 224, 224)
    return tensor, original


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--n_images",   default=8, type=int)
    parser.add_argument("--seed",       default=42, type=int)
    args = parser.parse_args()

    device = torch.device("cpu")   # Grad-CAM hooks work more reliably on CPU
    ckpt_path = REPO_ROOT / args.checkpoint

    print(f"[Grad-CAM] Loading checkpoint: {ckpt_path}")
    model, cfg = load_model(ckpt_path, device)
    print(f"[Grad-CAM] fusion_mode: {cfg.get('fusion_mode', 'fusion_concat')}")

    # Target layer: last MBConv block of EfficientNet-B0 features
    target_layer = model.image_encoder.features[-1]
    grad_cam     = GradCAM(model, target_layer)

    # Load sample images
    sample_df, img_dir = get_val_samples(cfg, n=args.n_images, seed=args.seed)

    # ── Generate visualisations ───────────────────────────────────────────────
    n_cols  = 4
    n_rows  = (args.n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 3, n_rows * 6))
    axes = axes.flatten()

    print(f"[Grad-CAM] Generating {args.n_images} heatmaps ...")
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = img_dir / (row["Id"] + ".jpg")
        if not img_path.exists():
            print(f"  [SKIP] {img_path.name} not found")
            continue

        img_tensor, original = preprocess_image(img_path)
        tab_tensor = torch.tensor(
            row[TABULAR_COLS].values.astype(np.float32)
        ).unsqueeze(0)

        img_tensor  = img_tensor.to(device).requires_grad_(False)
        tab_tensor  = tab_tensor.to(device)

        # Need gradients for the image path through the model
        img_tensor = img_tensor.detach()

        heatmap   = grad_cam(img_tensor, tab_tensor)
        overlayed = overlay_heatmap(original, heatmap)

        true_score = int(row["Pawpularity"])
        with torch.no_grad():
            pred_score = model(img_tensor, tab_tensor).item()

        # Original image row
        ax_orig = axes[i]
        ax_orig.imshow(original)
        ax_orig.set_title(f"True: {true_score}", fontsize=9)
        ax_orig.axis("off")

        # Grad-CAM overlay row
        ax_cam = axes[args.n_images + i]
        ax_cam.imshow(overlayed)
        ax_cam.set_title(f"Pred: {pred_score:.1f}", fontsize=9)
        ax_cam.axis("off")

    # Hide any unused axes
    for j in range(args.n_images, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Grad-CAM — Top row: original | Bottom row: heatmap overlay\n"
        "Warm colours = regions CNN weighted most heavily",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    out_dir  = REPO_ROOT / "runs" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradcam_grid.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    grad_cam.remove_hooks()
    print(f"[Grad-CAM] Saved → {out_path}")


if __name__ == "__main__":
    main()
