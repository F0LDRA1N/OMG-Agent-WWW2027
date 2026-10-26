from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConditionedUNet, LatentAutoencoder


def cosine_alpha_bar(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    values = torch.cos(((points / steps + offset) / (1 + offset)) * math.pi / 2) ** 2
    values = values / values[0]
    betas = (1 - values[1:] / values[:-1]).clamp(0.0001, 0.9999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class TargetExecutor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        widths: list[int],
        shared_dim: int,
        align_dim: int,
        evidence_encoder_layers: int,
        diffusion_steps: int,
        condition_dropout: float,
    ) -> None:
        super().__init__()
        self.autoencoder = LatentAutoencoder(feature_dim, latent_dim)
        self.unet = ConditionedUNet(
            latent_dim,
            widths,
            shared_dim,
            latent_dim,
            latent_dim,
            evidence_encoder_layers,
        )
        self.context_projection = nn.Linear(shared_dim, latent_dim)
        self.semantic_head = self._alignment_mlp(feature_dim, align_dim)
        self.plan_semantic_head = self._alignment_mlp(shared_dim, align_dim)
        self.evidence_head = self._alignment_mlp(feature_dim, align_dim)
        self.memory_evidence_head = self._alignment_mlp(latent_dim, align_dim)
        self.condition_dropout = condition_dropout
        self.register_buffer("alpha_bar", cosine_alpha_bar(diffusion_steps))

    @staticmethod
    def _alignment_mlp(input_dim: int, align_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, align_dim),
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )

    def _conditions(
        self,
        plan_tokens: torch.Tensor,
        plan_mask: torch.Tensor,
        evidence: torch.Tensor,
        drop: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not drop:
            return plan_tokens, plan_mask, evidence
        semantic_keep = (
            torch.rand(plan_tokens.shape[0], device=plan_tokens.device)
            >= self.condition_dropout
        )
        evidence_keep = (
            torch.rand(plan_tokens.shape[0], device=plan_tokens.device)
            >= self.condition_dropout
        )
        semantic = plan_tokens * semantic_keep[:, None, None].to(plan_tokens.dtype)
        semantic_mask = plan_mask & semantic_keep[:, None]
        semantic_mask[:, 0] = True
        memory = evidence * evidence_keep[:, None, None].to(evidence.dtype)
        return semantic, semantic_mask, memory

    def training_loss(
        self,
        target_feature: torch.Tensor,
        observed_context: torch.Tensor,
        plan_tokens: torch.Tensor,
        plan_mask: torch.Tensor,
        plan_pooled: torch.Tensor,
        evidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        clean = self.autoencoder.encode(target_feature)
        timestep = torch.randint(
            0, self.alpha_bar.shape[0], (clean.shape[0],), device=clean.device
        )
        alpha = self.alpha_bar[timestep].to(clean.dtype).unsqueeze(1)
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        semantic, semantic_mask, memory = self._conditions(
            plan_tokens, plan_mask, evidence, drop=self.training
        )
        context = self.context_projection(observed_context)
        predicted_noise = self.unet(
            noisy, context, timestep, semantic, semantic_mask, memory
        )
        clean_estimate = (
            noisy - (1 - alpha).sqrt() * predicted_noise
        ) / alpha.sqrt().clamp_min(1e-6)
        reconstructed = self.autoencoder.decode(clean_estimate)
        semantic_loss = 1 - F.cosine_similarity(
            self.semantic_head(reconstructed),
            self.plan_semantic_head(plan_pooled),
            dim=-1,
        ).mean()
        evidence_loss = F.l1_loss(
            self.evidence_head(reconstructed),
            self.memory_evidence_head(evidence.mean(dim=1)),
        )
        return {
            "noise": F.mse_loss(predicted_noise, noise),
            "semantic": semantic_loss,
            "evidence": evidence_loss,
            "reconstructed": reconstructed,
        }

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        observed_context: torch.Tensor,
        plan_tokens: torch.Tensor,
        plan_mask: torch.Tensor,
        evidence: torch.Tensor,
        ddim_steps: int,
        guidance_scale: float,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if eta != 0.0:
            raise ValueError("The reported deterministic DDIM protocol requires eta=0")
        latent_dim = self.autoencoder.encoder[-1].out_features
        latent = torch.randn(
            batch_size,
            latent_dim,
            device=plan_tokens.device,
            dtype=plan_tokens.dtype,
        )
        total = self.alpha_bar.shape[0]
        schedule = torch.linspace(total - 1, 0, ddim_steps, device=latent.device).long()
        null_tokens = torch.zeros_like(plan_tokens)
        null_mask = torch.zeros_like(plan_mask)
        null_mask[:, 0] = True
        null_evidence = torch.zeros_like(evidence)
        context = self.context_projection(observed_context)
        for position, timestep_value in enumerate(schedule):
            timestep = torch.full(
                (batch_size,), int(timestep_value), device=latent.device, dtype=torch.long
            )
            conditional = self.unet(
                latent, context, timestep, plan_tokens, plan_mask, evidence
            )
            unconditional = self.unet(
                latent,
                context,
                timestep,
                null_tokens,
                null_mask,
                null_evidence,
            )
            predicted_noise = unconditional + guidance_scale * (
                conditional - unconditional
            )
            alpha = self.alpha_bar[timestep_value].to(latent.dtype)
            clean = (latent - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if position + 1 == len(schedule):
                latent = clean
            else:
                next_alpha = self.alpha_bar[schedule[position + 1]].to(latent.dtype)
                latent = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * predicted_noise
        return self.autoencoder.decode(latent)
