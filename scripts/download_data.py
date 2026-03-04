import os
import subprocess
import zipfile
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_RAW   = REPO_ROOT / "data" / "raw"
DATASET    = "schulta/petfinder-pawpularity-score-clean"
DATASET_ZIP = "petfinder-pawpularity-score-clean.zip"

def check_kaggle_credentials():
    """Verify ~/.kaggle/kaggle.json exists and has correct permissions."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "\n[ERROR] ~/.kaggle/kaggle.json not found.\n"
            "  1. Go to https://www.kaggle.com/settings → API → 'Create New Token'\n"
            "  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. Run: chmod 600 ~/.kaggle/kaggle.json\n"
        )
    # Kaggle API requires 600 permissions
    current_mode = oct(kaggle_json.stat().st_mode)[-3:]
    if current_mode != "600":
        print(f"[WARNING] kaggle.json permissions are {current_mode}, fixing to 600...")
        kaggle_json.chmod(0o600)
    print("[OK] Kaggle credentials found.")

def download_competition_data():
    """Download dataset zip into data/raw/ using the Kaggle CLI."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_RAW / DATASET_ZIP
    if zip_path.exists():
        print(f"[SKIP] Zip already exists at {zip_path}")
        return zip_path

    print(f"[INFO] Downloading dataset '{DATASET}' ...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-p", str(DATA_RAW)
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[ERROR] Kaggle download failed:\n{result.stderr}\n"
            "Make sure you are logged in and the dataset is accessible at:\n"
            f"  https://www.kaggle.com/datasets/{DATASET}"
        )

    print("[OK] Download complete.")
    return zip_path

def unzip_data(zip_path: Path):
    """Unzip the competition archive into data/raw/."""
    print(f"[INFO] Unzipping {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_RAW)

    # Some nested zips (train.zip, test.zip) — unzip those too
    for nested_zip in DATA_RAW.glob("*.zip"):
        if nested_zip == zip_path:
            continue
        print(f"  [INFO] Unzipping nested: {nested_zip.name}")
        with zipfile.ZipFile(nested_zip, "r") as z:
            z.extractall(DATA_RAW)
        nested_zip.unlink()  # remove nested zip after extraction

    print("[OK] Unzip complete.")

def print_summary():
    """Print a quick summary of what was downloaded."""
    print("\n── Dataset Summary ──────────────────────────────────────")

    # CSVs
    for csv_name in ["train.csv", "test.csv"]:
        csv_path = DATA_RAW / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"\n{csv_name}: {df.shape[0]} rows × {df.shape[1]} cols")
            if "Pawpularity" in df.columns:
                print(f"  Pawpularity  →  mean={df['Pawpularity'].mean():.2f}, "
                      f"std={df['Pawpularity'].std():.2f}, "
                      f"min={df['Pawpularity'].min()}, "
                      f"max={df['Pawpularity'].max()}")
        else:
            print(f"[WARNING] {csv_name} not found at {csv_path}")

    # Images
    for split in ["train", "test"]:
        img_dir = DATA_RAW / split
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg"))
            print(f"\n{split}/ images: {len(images)}")
        else:
            print(f"[WARNING] {split}/ directory not found at {img_dir}")

    print("\n─────────────────────────────────────────────────────────")
    print(f"[OK] Raw data lives in: {DATA_RAW}")

def main():
    print("=== Step 1: Kaggle Data Download ===\n")
    check_kaggle_credentials()
    zip_path = download_competition_data()
    unzip_data(zip_path)
    print_summary()

if __name__ == "__main__":
    main()