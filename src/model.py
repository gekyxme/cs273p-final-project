"""
Multimodal Pawpularity Model

Supports four modes (set via config `fusion_mode`):
    fusion_concat    : EfficientNet-B0 + Tabular MLP → Concat → FC head  (baseline)
    fusion_attention : EfficientNet-B0 + Tabular MLP → Gated Attention → FC head
    image_only       : EfficientNet-B0 → FC head  (ablation)
    tabular_only     : Tabular MLP → FC head  (ablation)
"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

VALID_MODES = ("fusion_concat", "fusion_attention", "image_only", "tabular_only")


class ImageEncoder(nn.Module):
    """
    EfficientNet-B0 with the classification head replaced by Identity.
    Output: (B, 1280)
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        self.features   = base.features
        self.avgpool    = base.avgpool
        self.output_dim = base.classifier[1].in_features  # 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (B, 1280)


class TabularEncoder(nn.Module):
    """
    MLP for 12 binary metadata features.
    Output: (B, output_dim)
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
        return self.net(x)


class GatedAttentionFusion(nn.Module):
    """
    Gated attention fusion: learns a gate vector that weights how much
    of each modality contributes to the fused representation.

    Given img_feat (B, D_img) and tab_feat (B, D_tab):
        combined = Concat[img, tab]           (B, D_img + D_tab)
        gate     = Sigmoid(Linear(combined))  (B, D_img + D_tab)
        fused    = gate ⊙ combined            (B, D_img + D_tab)
    """

    def __init__(self, img_dim: int, tab_dim: int):
        super().__init__()
        combined_dim = img_dim + tab_dim
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid(),
        )
        self.output_dim = combined_dim

    def forward(self, img_feat: torch.Tensor, tab_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.gate(combined) * combined


def _make_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Shared FC regression head: Linear → BN → GELU → Dropout → Linear(1)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class PawpularityModel(nn.Module):
    """
    Unified model supporting concat fusion, attention fusion, image-only,
    and tabular-only modes.

    Forward: model(image, tabular) → (B, 1)
    """

    def __init__(
        self,
        fusion_mode: str = "fusion_concat",
        freeze_backbone: bool = False,
        tabular_input_dim: int = 12,
        tabular_hidden_dim: int = 64,
        tabular_output_dim: int = 32,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert fusion_mode in VALID_MODES, \
            f"fusion_mode must be one of {VALID_MODES}, got '{fusion_mode}'"
        self.fusion_mode = fusion_mode

        # ── Encoders (only instantiate what is needed) ────────────────────────
        if fusion_mode != "tabular_only":
            self.image_encoder = ImageEncoder(freeze_backbone=freeze_backbone)

        if fusion_mode != "image_only":
            self.tabular_encoder = TabularEncoder(
                input_dim=tabular_input_dim,
                hidden_dim=tabular_hidden_dim,
                output_dim=tabular_output_dim,
                dropout=dropout,
            )

        # ── Fusion + head ─────────────────────────────────────────────────────
        if fusion_mode == "fusion_concat":
            in_dim = self.image_encoder.output_dim + self.tabular_encoder.output_dim
            self.head = _make_head(in_dim, fusion_hidden_dim, dropout)

        elif fusion_mode == "fusion_attention":
            self.attention = GatedAttentionFusion(
                img_dim=self.image_encoder.output_dim,
                tab_dim=self.tabular_encoder.output_dim,
            )
            self.head = _make_head(self.attention.output_dim, fusion_hidden_dim, dropout)

        elif fusion_mode == "image_only":
            self.head = _make_head(self.image_encoder.output_dim, fusion_hidden_dim, dropout)

        elif fusion_mode == "tabular_only":
            self.head = _make_head(self.tabular_encoder.output_dim, fusion_hidden_dim, dropout)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "fusion_concat":
            img_feat = self.image_encoder(image)
            tab_feat = self.tabular_encoder(tabular)
            fused    = torch.cat([img_feat, tab_feat], dim=1)
            return self.head(fused)

        elif self.fusion_mode == "fusion_attention":
            img_feat = self.image_encoder(image)
            tab_feat = self.tabular_encoder(tabular)
            fused    = self.attention(img_feat, tab_feat)
            return self.head(fused)

        elif self.fusion_mode == "image_only":
            return self.head(self.image_encoder(image))

        elif self.fusion_mode == "tabular_only":
            return self.head(self.tabular_encoder(tabular))
