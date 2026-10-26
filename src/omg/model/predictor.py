from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalPredictor(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.modalities = tuple(feature_dims)
        self.projections = nn.ModuleDict(
            {
                modality: nn.Linear(int(dimension), hidden_dim)
                for modality, dimension in feature_dims.items()
            }
        )
        self.modality_embeddings = nn.Parameter(
            torch.empty(len(self.modalities), hidden_dim)
        )
        nn.init.normal_(self.modality_embeddings, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.projections[modality](features[modality])
                + self.modality_embeddings[index]
                for index, modality in enumerate(self.modalities)
            ],
            dim=1,
        )
        encoded = self.transformer(tokens)
        return self.head(encoded.mean(dim=1)).squeeze(-1)
