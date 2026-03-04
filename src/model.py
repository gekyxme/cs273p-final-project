"""
Multimodal Pawpularity Model

Architecture:
    ImageEncoder   : EfficientNet-B0 (pretrained) → 1280-dim features
    TabularEncoder : MLP → 32-dim features
    FusionHead     : Concat(1280+32) → FC layers → scalar score
"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ImageEncoder(nn.Module):
    """
    EfficientNet-B0 with the classification head replaced by Identity.
    Output: (B, 1280)
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Keep everything except the final classifier
        self.features   = base.features    # convolutional backbone
        self.avgpool    = base.avgpool     # AdaptiveAvgPool2d → (B, 1280, 1, 1)
        self.output_dim = base.classifier[1].in_features  # 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 1280, 7, 7)
        x = self.avgpool(x)           # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)       # (B, 1280)
        return x


class TabularEncoder(nn.Module):
    """
    MLP for 12 binary metadata features.
    Output: (B, tabular_output_dim)
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, output_dim)


class PawpularityModel(nn.Module):
    """
    Multimodal fusion model.

    Forward signature:
        model(image, tabular) → (B, 1)  raw Pawpularity score in [1, 100] range
    """

    def __init__(
        self,
        freeze_backbone: bool = False,
        tabular_input_dim: int = 12,
        tabular_hidden_dim: int = 64,
        tabular_output_dim: int = 32,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.image_encoder   = ImageEncoder(freeze_backbone=freeze_backbone)
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            output_dim=tabular_output_dim,
            dropout=dropout,
        )

        fused_dim = self.image_encoder.output_dim + self.tabular_encoder.output_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),  # raw score output
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)       # (B, 1280)
        tab_feat = self.tabular_encoder(tabular)   # (B, 32)
        fused    = torch.cat([img_feat, tab_feat], dim=1)  # (B, 1312)
        return self.fusion_head(fused)             # (B, 1)
