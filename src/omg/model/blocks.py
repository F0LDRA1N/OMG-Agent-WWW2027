from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    value = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(value, (0, dimension - value.shape[1]))


class FeatureContextEncoder(nn.Module):
    """Projects pooled observed features and explicit availability bits to d_c."""

    def __init__(self, feature_dims: dict[str, int], shared_dim: int) -> None:
        super().__init__()
        self.modalities = tuple(feature_dims)
        self.feature_dims = feature_dims
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(int(dimension), shared_dim),
                    nn.LayerNorm(shared_dim),
                    nn.GELU(),
                )
                for modality, dimension in feature_dims.items()
            }
        )
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * len(self.modalities) + len(self.modalities), shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

    def pool(
        self,
        features: dict[str, torch.Tensor],
        feature_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            modality: masked_mean(features[modality], feature_masks[modality])
            for modality in self.modalities
        }

    def from_pooled(
        self,
        pooled: dict[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        encoded = []
        for column, modality in enumerate(self.modalities):
            value = self.projections[modality](pooled[modality])
            encoded.append(value * availability[:, column : column + 1].to(value.dtype))
        bits = availability.to(encoded[0].dtype)
        return self.fusion(torch.cat([*encoded, bits], dim=-1))

    def forward(
        self,
        features: dict[str, torch.Tensor],
        feature_masks: dict[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled = self.pool(features, feature_masks)
        return self.from_pooled(pooled, availability), pooled


class LatentAutoencoder(nn.Module):
    """Target-specific feature-to-latent maps A_t/B_t from the paper."""

    def __init__(self, feature_dim: int, latent_dim: int) -> None:
        super().__init__()
        width = max(128, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, feature_dim),
        )

    def encode(self, feature: torch.Tensor) -> torch.Tensor:
        return self.encoder(feature)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class CrossAttention(nn.Module):
    def __init__(self, width: int, condition_dim: int, heads: int = 8) -> None:
        super().__init__()
        heads = min(heads, width)
        while width % heads:
            heads -= 1
        self.query_norm = nn.LayerNorm(width)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query_norm(value).unsqueeze(1)
        key_value = self.condition_projection(self.condition_norm(condition))
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=None if condition_mask is None else ~condition_mask,
            need_weights=False,
        )
        return value + attended.squeeze(1)


class SelfAttention(nn.Module):
    """Self-attention used by each vector U-Net residual stage."""

    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__()
        heads = min(heads, width)
        while width % heads:
            heads -= 1
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value).unsqueeze(1)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        return value + attended.squeeze(1)


class ZeroEvidenceAdapter(nn.Module):
    """Vector analogue of a zero-initialized 1x1 evidence convolution."""

    def __init__(self, evidence_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(evidence_dim, output_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.projection(evidence.mean(dim=1))


class DiffusionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        time_dim: int,
        semantic_dim: int,
        evidence_dim: int | None,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, output_dim)
        self.time = nn.Linear(time_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.self_attention = SelfAttention(output_dim)
        self.semantic = CrossAttention(output_dim, semantic_dim)
        self.evidence = (
            ZeroEvidenceAdapter(evidence_dim, output_dim)
            if evidence_dim is not None
            else None
        )
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
        evidence: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = self.input(value) + self.time(time)
        hidden = self.self_attention(hidden)
        hidden = self.semantic(hidden, semantic, semantic_mask)
        if self.evidence is not None and evidence is not None:
            hidden = hidden + self.evidence(evidence)
        return self.skip(value) + self.output(self.norm(hidden))


class ConditionedUNet(nn.Module):
    """Eleven-stage vector U-Net with semantic and early evidence attention."""

    def __init__(
        self,
        latent_dim: int,
        widths: list[int],
        semantic_dim: int,
        context_dim: int,
        evidence_dim: int,
        evidence_encoder_layers: int,
    ) -> None:
        super().__init__()
        if len(widths) != 4:
            raise ValueError("The paper specifies four U-Net widths")
        self.time_embedding_dim = 128
        self.time_dim = 512
        self.time_projection = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_dim),
            nn.SiLU(),
        )
        self.input = DiffusionBlock(
            latent_dim + context_dim,
            widths[0],
            self.time_dim,
            semantic_dim,
            evidence_dim if evidence_encoder_layers > 0 else None,
        )
        encoder_dims = list(zip(widths[:-1], widths[1:]))
        self.encoders = nn.ModuleList(
            [
                DiffusionBlock(
                    input_dim,
                    output_dim,
                    self.time_dim,
                    semantic_dim,
                    evidence_dim if index < evidence_encoder_layers - 1 else None,
                )
                for index, (input_dim, output_dim) in enumerate(encoder_dims)
            ]
        )
        self.middle = nn.ModuleList(
            [
                DiffusionBlock(widths[-1], widths[-1], self.time_dim, semantic_dim, None),
                DiffusionBlock(widths[-1], widths[-1], self.time_dim, semantic_dim, None),
            ]
        )
        decoder_dims = [
            (widths[-1] * 2, widths[-2]),
            (widths[-2] * 2, widths[-3]),
            (widths[-3] * 2, widths[-4]),
        ]
        self.decoders = nn.ModuleList(
            [
                DiffusionBlock(input_dim, output_dim, self.time_dim, semantic_dim, None)
                for input_dim, output_dim in decoder_dims
            ]
        )
        self.final = DiffusionBlock(
            widths[0] * 2, widths[0], self.time_dim, semantic_dim, None
        )
        self.output_semantic = CrossAttention(widths[0], semantic_dim)
        self.output = nn.Linear(widths[0], latent_dim)

    def forward(
        self,
        noisy: torch.Tensor,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
        evidence: torch.Tensor | None,
    ) -> torch.Tensor:
        time = self.time_projection(
            sinusoidal_embedding(timesteps, self.time_embedding_dim).to(
                noisy.dtype
            )
        )
        skips = []
        hidden = self.input(
            torch.cat([noisy, context], dim=-1),
            time,
            semantic,
            semantic_mask,
            evidence,
        )
        skips.append(hidden)
        for block in self.encoders:
            hidden = block(hidden, time, semantic, semantic_mask, evidence)
            skips.append(hidden)
        for block in self.middle:
            hidden = block(hidden, time, semantic, semantic_mask, None)
        for block in self.decoders:
            hidden = block(
                torch.cat([hidden, skips.pop()], dim=-1),
                time,
                semantic,
                semantic_mask,
                None,
            )
        hidden = self.final(
            torch.cat([hidden, skips.pop()], dim=-1),
            time,
            semantic,
            semantic_mask,
            None,
        )
        hidden = self.output_semantic(hidden, semantic, semantic_mask)
        return self.output(hidden)
