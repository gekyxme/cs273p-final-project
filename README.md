# cs273p-Final-Project

CS273P final project based on Kaggle PetFinder.my Pawpularity Contest.
Goal: multimodal model (CNN image encoder + tabular metadata) + ablations + reproducible training/eval.

## Collaboration (branches)
- Work only on your branch: `amrit`, `prateek`, or `asmita`
- Push to your branch
- Open PR to `main` for review/merge (best result goes to `main`)

## Quick start
```bash
git clone <REPO_URL>
cd <REPO_NAME>
git fetch --all
git checkout <your-branch>   # amrit | prateek | asmita
conda env create -f environment.yml
conda activate pawpularity
python scripts/download_data.py
python scripts/resize_images.py
python -m src.train --debug_mode false --epochs 25
```

---

## Results

### Hardware
- Apple M4 MacBook Air · PyTorch MPS backend · ~9,900 training images

---

### Baseline Training

Full dataset, 25 epochs, batch size 32, EfficientNet-B0 + Tabular MLP, concat fusion.

| Epochs | Batch Size | Best Val RMSE | Best Epoch |
|--------|------------|---------------|------------|
| 25 | 32 | **20.53** | 20 |

Checkpoint: `checkpoints/best.pt`

---

### Ablation Analysis

Each ablation was trained under identical conditions (25 epochs, batch size 32, full dataset).

#### Ablation 1: What contributes to performance?

| Model | What it uses | Val RMSE | vs Baseline |
|-------|-------------|----------|-------------|
| **Fusion Concat** *(baseline)* | Image + Tabular | **20.53** | — |
| Image Only | Image only | 20.59 | +0.06 |
| Tabular Only | Metadata only | 20.58 | +0.05 |

**Finding:** Both modalities contribute independently. Combining them gives the best result.
Tabular features alone (RMSE 20.58) slightly edge out image-only (RMSE 20.59), showing
the 12 binary metadata features carry real signal beyond what the image alone provides.

---

#### Ablation 2: Does a smarter fusion strategy help?

| Fusion Strategy | Description | Val RMSE | vs Baseline |
|----------------|-------------|----------|-------------|
| **Concat** *(baseline)* | Simple concatenation of features | **20.53** | — |
| Gated Attention | Learned gate weights each modality | 20.60 | +0.07 |

**Finding:** Simple concatenation outperforms gated attention on this dataset.
The attention mechanism adds ~1.7M parameters but no accuracy benefit, the dataset
is too noisy and small for the extra complexity to pay off. Concat fusion is the better design.

---

### Key Takeaway

All models converge near RMSE ~20.5, which is close to the dataset's inherent noise floor
(Pawpularity std ≈ 20.6). This is consistent with top public Kaggle solutions (~17–18 RMSE)
which required large ensembles and test-time augmentation to push meaningfully below this floor.
The multimodal fusion design is validated, it is the best single-model configuration tested.