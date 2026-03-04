"""
PawpularityDataset and DataLoader factory.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# The 12 binary metadata features (column order matches train.csv)
TABULAR_COLS = [
    "Subject Focus", "Eyes", "Face", "Near", "Action",
    "Accessory", "Group", "Collage", "Human", "Occlusion",
    "Info", "Blur",
]

# ImageNet normalisation stats (EfficientNet was pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool, image_size: int = 224):
    """Return torchvision transform pipeline for train or val/test."""
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class PawpularityDataset(Dataset):
    """
    Loads pre-resized 224×224 images + tabular metadata from the PetFinder
    Pawpularity dataset.

    Returns per item:
        image    : FloatTensor (3, H, W)
        tabular  : FloatTensor (12,)
        target   : FloatTensor (1,)  — Pawpularity score in [1, 100]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform=None,
        is_test: bool = False,
    ):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.is_test   = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── Image ──────────────────────────────────────────────────────────
        img_path = self.img_dir / (row["Id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ── Tabular ────────────────────────────────────────────────────────
        tabular = torch.tensor(
            row[TABULAR_COLS].values.astype(np.float32), dtype=torch.float32
        )

        # ── Target ─────────────────────────────────────────────────────────
        if self.is_test:
            target = torch.tensor([0.0], dtype=torch.float32)  # placeholder
        else:
            target = torch.tensor([float(row["Pawpularity"])], dtype=torch.float32)

        return image, tabular, target


def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from cfg.

    Expects cfg keys:
        data_dir, debug_mode, val_split, batch_size, num_workers, seed, image_size
    """
    repo_root = Path(__file__).resolve().parent.parent
    data_dir  = repo_root / cfg["data_dir"]

    if cfg.get("debug_mode", False):
        csv_path = data_dir / "debug" / "train.csv"
        img_dir  = data_dir / "debug" / "train"
    else:
        csv_path = data_dir / "raw" / "train.csv"
        img_dir  = data_dir / "resized" / "train"

    df = pd.read_csv(csv_path)

    # Stratified split: bin Pawpularity into 4 quantile groups for stratification
    df["_strat_bin"] = pd.qcut(df["Pawpularity"], q=4, labels=False, duplicates="drop")

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.get("val_split", 0.2),
        random_state=cfg.get("seed", 42),
        stratify=df["_strat_bin"],
    )

    train_df = train_df.drop(columns=["_strat_bin"])
    val_df   = val_df.drop(columns=["_strat_bin"])

    image_size = cfg.get("image_size", 224)

    train_dataset = PawpularityDataset(
        df=train_df,
        img_dir=img_dir,
        transform=get_transforms(is_train=True, image_size=image_size),
    )
    val_dataset = PawpularityDataset(
        df=val_df,
        img_dir=img_dir,
        transform=get_transforms(is_train=False, image_size=image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # pin_memory=True causes issues with MPS
        drop_last=True,    # avoid single-sample batches that break BatchNorm
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,
    )

    print(f"[Dataset] train={len(train_dataset)}  val={len(val_dataset)}  "
          f"(debug_mode={cfg.get('debug_mode', False)})")

    return train_loader, val_loader
