"""
Step 8: Training Loop

Usage:
    # Debug run (100 images, quick sanity check):
    python -m src.train

    # Full training:
    python -m src.train --config configs/default.yaml --debug_mode false

    # Override any config value on the CLI:
    python -m src.train --epochs 50 --batch_size 16 --debug_mode false
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import PawpularityModel
from src.utils import count_parameters, get_device, rmse, set_seed

# ── TensorBoard (optional — graceful fallback if not installed) ───────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False


def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            # cast booleans passed as strings from argparse
            if isinstance(cfg.get(k), bool):
                cfg[k] = str(v).lower() in ("true", "1", "yes")
            else:
                cfg[k] = type(cfg[k])(v) if k in cfg else v
    return cfg


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> float:
    """Run one training epoch. Returns epoch RMSE."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for batch_idx, (images, tabular, targets) in enumerate(loader):
        images  = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images, tabular)          # (B, 1)
        loss  = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} | batch {batch_idx+1}/{len(loader)} "
                  f"| loss {avg_loss:.4f}")

    epoch_preds   = torch.cat(all_preds)
    epoch_targets = torch.cat(all_targets)
    return rmse(epoch_preds, epoch_targets)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run validation. Returns val RMSE."""
    model.eval()
    all_preds, all_targets = [], []

    for images, tabular, targets in loader:
        images  = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        preds = model(images, tabular)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    return rmse(torch.cat(all_preds), torch.cat(all_targets))


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train Pawpularity model")
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--debug_mode",  default=None)
    parser.add_argument("--epochs",      default=None, type=int)
    parser.add_argument("--batch_size",  default=None, type=int)
    parser.add_argument("--lr",          default=None, type=float)
    parser.add_argument("--freeze_backbone", default=None)
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {
            "debug_mode":      args.debug_mode,
            "epochs":          args.epochs,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "freeze_backbone": args.freeze_backbone,
        },
    )

    # ── Setup ─────────────────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent
    set_seed(cfg["seed"])
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PawpularityModel(
        freeze_backbone=cfg["freeze_backbone"],
        tabular_input_dim=cfg["tabular_input_dim"],
        tabular_hidden_dim=cfg["tabular_hidden_dim"],
        tabular_output_dim=cfg["tabular_output_dim"],
        fusion_hidden_dim=cfg["fusion_hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    print(f"[Model] Trainable params: {count_parameters(model):,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    criterion = nn.MSELoss()

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if TENSORBOARD:
        log_dir = repo_root / cfg["log_dir"]
        writer  = SummaryWriter(log_dir=str(log_dir))
        print(f"[TensorBoard] Logging to {log_dir}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    ckpt_dir  = repo_root / cfg["checkpoint_dir"]
    best_rmse = math.inf
    best_epoch = 0

    print(f"\n{'='*55}")
    print(f" Training for {cfg['epochs']} epochs | "
          f"debug_mode={cfg['debug_mode']} | device={device}")
    print(f"{'='*55}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_rmse = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, cfg["log_interval"],
        )
        val_rmse = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train RMSE {train_rmse:.4f} | "
              f"val RMSE {val_rmse:.4f} | "
              f"lr {current_lr:.2e} | "
              f"{elapsed:.1f}s")

        # MPS sync for accurate timing
        if device.type == "mps":
            torch.mps.synchronize()

        # TensorBoard
        if writer:
            writer.add_scalar("RMSE/train", train_rmse, epoch)
            writer.add_scalar("RMSE/val",   val_rmse,   epoch)
            writer.add_scalar("LR",         current_lr, epoch)

        # Checkpoint
        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_epoch = epoch
            save_checkpoint(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse":         val_rmse,
                    "cfg":              cfg,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  ✓ New best val RMSE: {best_rmse:.4f} — checkpoint saved.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f" Training complete.")
    print(f" Best val RMSE: {best_rmse:.4f} at epoch {best_epoch}")
    print(f" Checkpoint   : {ckpt_dir / 'best.pt'}")
    print(f"{'='*55}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
