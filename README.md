# cs273p-Final-Project

CS273P final project based on Kaggle PetFinder.my Pawpularity Contest. [web:43]  
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

## Training Log

### Baseline: Multimodal Fusion (EfficientNet-B0 + Tabular MLP)
| Run | Epochs | Batch Size | Train RMSE | Val RMSE | Notes |
|-----|--------|------------|------------|----------|-------|
| Baseline | 25 | 32 | 20.63 | **20.53** | Full dataset (~9,900 images), MPS (M4 MacBook Air) |

> Best checkpoint saved at epoch 20. Target range for competitive Kaggle submissions: 17–18.