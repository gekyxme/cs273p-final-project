"""
Step 2: Offline Image Resizing + Debug Subset Creation

Resizes all train/test images to 224x224 and saves them to data/resized/.
Also creates a 100-image debug subset in data/debug/ for rapid loop testing.

Usage:
    python scripts/resize_images.py
"""

import shutil
import random
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW    = REPO_ROOT / "data" / "raw"
DATA_RESIZED = REPO_ROOT / "data" / "resized"
DATA_DEBUG  = REPO_ROOT / "data" / "debug"
TARGET_SIZE = (224, 224)
DEBUG_N     = 100
SEED        = 42


def resize_split(split: str):
    """Resize all .jpg images for a given split (train or test)."""
    src_dir = DATA_RAW / split
    dst_dir = DATA_RESIZED / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(src_dir.glob("*.jpg"))
    if not images:
        print(f"[WARNING] No .jpg files found in {src_dir}")
        return

    print(f"\n[INFO] Resizing {len(images)} {split} images → {TARGET_SIZE} ...")
    skipped = 0
    for img_path in tqdm(images, desc=split, unit="img"):
        dst_path = dst_dir / img_path.name
        if dst_path.exists():
            skipped += 1
            continue
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                img.save(dst_path, format="JPEG", quality=95)
        except Exception as e:
            print(f"  [ERROR] Could not process {img_path.name}: {e}")

    done = len(images) - skipped
    print(f"  [OK] {done} resized, {skipped} skipped (already existed).")


def create_debug_subset():
    """Copy a random 100-image subset + matching CSV rows to data/debug/."""
    debug_img_dir = DATA_DEBUG / "train"
    debug_img_dir.mkdir(parents=True, exist_ok=True)

    train_csv = DATA_RAW / "train.csv"
    if not train_csv.exists():
        print("[WARNING] train.csv not found, skipping debug subset.")
        return

    df = pd.read_csv(train_csv)

    # Sample DEBUG_N rows
    random.seed(SEED)
    sample_df = df.sample(n=min(DEBUG_N, len(df)), random_state=SEED).reset_index(drop=True)

    print(f"\n[INFO] Creating debug subset ({len(sample_df)} images) ...")
    copied = 0
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="debug", unit="img"):
        img_name = row["Id"] + ".jpg"
        src = DATA_RESIZED / "train" / img_name
        dst = debug_img_dir / img_name
        if not src.exists():
            print(f"  [WARNING] Resized image not found: {src}. Run resize first.")
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    # Save the matching CSV
    debug_csv = DATA_DEBUG / "train.csv"
    sample_df.to_csv(debug_csv, index=False)
    print(f"  [OK] {copied} images + CSV saved to {DATA_DEBUG}")
    print(f"  Pawpularity stats → mean={sample_df['Pawpularity'].mean():.2f}, "
          f"std={sample_df['Pawpularity'].std():.2f}, "
          f"min={sample_df['Pawpularity'].min()}, "
          f"max={sample_df['Pawpularity'].max()}")


def main():
    print("=== Step 2: Offline Image Resizing ===")

    for split in ["train", "test"]:
        resize_split(split)

    create_debug_subset()

    # Final summary
    train_count = len(list((DATA_RESIZED / "train").glob("*.jpg")))
    test_count  = len(list((DATA_RESIZED / "test").glob("*.jpg")))
    debug_count = len(list((DATA_DEBUG / "train").glob("*.jpg")))
    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  data/resized/train/ : {train_count} images")
    print(f"  data/resized/test/  : {test_count} images")
    print(f"  data/debug/train/   : {debug_count} images")
    print(f"  All images are 224×224 JPEG, ready for PyTorch DataLoader.")
    print(f"─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
