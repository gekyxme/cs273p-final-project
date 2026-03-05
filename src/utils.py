"""
Utility helpers: seeding, device selection, metrics.
"""

import os
import random
import math
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make training fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no CUDA, harmless
    # MPS does not expose a manual_seed, but CPU seed covers it
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return the best available device: MPS → CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Squared Error (scalar, Python float)."""
    with torch.no_grad():
        mse = torch.mean((preds.squeeze() - targets.squeeze()) ** 2)
        return math.sqrt(mse.item())


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
