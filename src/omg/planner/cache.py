from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class PlanProjection(nn.Module):
    def __init__(self, hidden_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, shared_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.projection(hidden.detach())
        if token_mask is None:
            pooled = tokens.mean(dim=-2)
        else:
            weights = token_mask.to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        return tokens, pooled


class ConsistencyVerifier(nn.Module):
    def __init__(self, shared_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(shared_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def logits(self, plan: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([plan, observation], dim=-1)).squeeze(-1)

    def forward(self, plan: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return self.logits(plan, observation).sigmoid()


@dataclass
class SelectedPlans:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    pooled: torch.Tensor
    consistency: torch.Tensor
    candidate_index: torch.Tensor
    round_index: torch.Tensor
    verifier_loss: torch.Tensor


@dataclass
class ReflectionDiagnostic:
    candidate_index: int
    consistency: float
    field: str


def _candidate_tensors(
    entries: list[dict[str, Any]],
    hidden_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_counts = [len(entry["candidates"]) for entry in entries]
    if min(candidate_counts, default=0) < 1:
        raise ValueError("Every plan-cache entry must contain at least one candidate")
    max_candidates = max(candidate_counts)
    max_tokens = max(
        int(candidate["hidden"].shape[0])
        for entry in entries
        for candidate in entry["candidates"]
    )
    hidden = torch.zeros(
        len(entries), max_candidates, max_tokens, hidden_dim, device=device
    )
    token_mask = torch.zeros(
        len(entries), max_candidates, max_tokens, dtype=torch.bool, device=device
    )
    scores = torch.full((len(entries), max_candidates), float("-inf"), device=device)
    schema = torch.ones((len(entries), max_candidates), device=device)
    rounds = torch.zeros(
        (len(entries), max_candidates), dtype=torch.long, device=device
    )
    for row, entry in enumerate(entries):
        for column, candidate in enumerate(entry["candidates"]):
            value = candidate["hidden"].to(device=device, dtype=hidden.dtype)
            if value.ndim != 2 or value.shape[-1] != hidden_dim:
                raise ValueError(f"Invalid plan hidden shape: {tuple(value.shape)}")
            hidden[row, column, : value.shape[0]] = value
            token_mask[row, column, : value.shape[0]] = True
            scores[row, column] = float(candidate["logprob"])
            schema[row, column] = float(candidate.get("schema_penalty", 0.0))
            rounds[row, column] = int(candidate.get("round", 0))
    return hidden, token_mask, scores, schema, rounds


class CachedPlanSelector(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        shared_dim: int,
        verifier_weight: float,
        schema_penalty: float,
        verifier_threshold: float,
        max_rounds: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = PlanProjection(hidden_dim, shared_dim)
        self.verifier = ConsistencyVerifier(shared_dim)
        self.verifier_weight = verifier_weight
        self.schema_penalty = schema_penalty
        self.verifier_threshold = verifier_threshold
        self.max_rounds = max_rounds

    def _auxiliary_verifier_loss(
        self,
        entries: list[dict[str, Any]],
        observations: torch.Tensor,
    ) -> torch.Tensor:
        positive_hidden: list[torch.Tensor] = []
        negative_hidden: list[torch.Tensor] = []
        observation_rows: list[torch.Tensor] = []
        for row, entry in enumerate(entries):
            positives = entry.get("positive_hidden", [])
            negatives = entry.get("negative_hidden", [])
            for positive, negative in zip(positives, negatives):
                positive_hidden.append(
                    positive.to(
                        device=observations.device, dtype=observations.dtype
                    )
                )
                negative_hidden.append(
                    negative.to(
                        device=observations.device, dtype=observations.dtype
                    )
                )
                observation_rows.append(observations[row])
        if not positive_hidden:
            return observations.new_zeros(())
        positive_pad = pad_sequence(positive_hidden, batch_first=True)
        negative_pad = pad_sequence(negative_hidden, batch_first=True)
        positive_mask = (
            torch.arange(positive_pad.shape[1], device=observations.device)[None, :]
            < torch.tensor(
                [value.shape[0] for value in positive_hidden], device=observations.device
            )[:, None]
        )
        negative_mask = (
            torch.arange(negative_pad.shape[1], device=observations.device)[None, :]
            < torch.tensor(
                [value.shape[0] for value in negative_hidden], device=observations.device
            )[:, None]
        )
        _, positive = self.projection(positive_pad, positive_mask)
        _, negative = self.projection(negative_pad, negative_mask)
        obs = torch.stack(observation_rows)
        logits = torch.cat(
            [self.verifier.logits(positive, obs), self.verifier.logits(negative, obs)]
        )
        labels = torch.cat(
            [
                torch.ones_like(logits[: len(obs)]),
                torch.zeros_like(logits[: len(obs)]),
            ]
        )
        return F.binary_cross_entropy_with_logits(logits, labels)

    def reflection_diagnostics(
        self,
        entries: list[dict[str, Any]],
        observations: torch.Tensor,
    ) -> list[ReflectionDiagnostic | None]:
        """Attribute a rejected first-round plan to one schema field."""
        with torch.enable_grad():
            hidden, token_mask, logprob, schema, rounds = _candidate_tensors(
                entries, self.hidden_dim, observations.device
            )
            shape = hidden.shape
            tokens, pooled = self.projection(
                hidden.flatten(0, 1), token_mask.flatten(0, 1)
            )
            tokens = tokens.view(shape[0], shape[1], shape[2], -1)
            pooled = pooled.view(shape[0], shape[1], -1)
            repeated = observations.detach()[:, None, :].expand_as(pooled)
            consistency = self.verifier(
                pooled.flatten(0, 1), repeated.flatten(0, 1)
            ).view(shape[0], shape[1])
            score = (
                logprob
                + self.verifier_weight * consistency
                - self.schema_penalty * schema
            ).masked_fill(rounds != 0, float("-inf"))
            selected = score.argmax(dim=1)
            rows = torch.arange(shape[0], device=observations.device)
            selected_tokens = tokens[rows, selected]
            selected_mask = token_mask[rows, selected]
            weights = selected_mask.to(selected_tokens.dtype).unsqueeze(-1)
            selected_pooled = (
                selected_tokens * weights
            ).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            selected_consistency = self.verifier(
                selected_pooled, observations.detach()
            )
            gradients = torch.autograd.grad(
                (1 - selected_consistency).sum(),
                selected_tokens,
                retain_graph=False,
                create_graph=False,
            )[0]
        fields = ("entity", "action", "attribute")
        diagnostics: list[ReflectionDiagnostic | None] = []
        for row in range(shape[0]):
            value = float(selected_consistency[row].detach().cpu())
            if value >= self.verifier_threshold:
                diagnostics.append(None)
                continue
            field_count = min(len(fields), int(selected_mask[row].sum()))
            norms = gradients[row, :field_count].norm(dim=-1)
            field_index = int(norms.argmax().detach().cpu())
            diagnostics.append(
                ReflectionDiagnostic(
                    candidate_index=int(selected[row].detach().cpu()),
                    consistency=value,
                    field=fields[field_index],
                )
            )
        return diagnostics

    def forward(
        self,
        entries: list[dict[str, Any]],
        observations: torch.Tensor,
        compute_verifier_loss: bool = True,
        supervision_observations: torch.Tensor | None = None,
    ) -> SelectedPlans:
        hidden, token_mask, logprob, schema, rounds = _candidate_tensors(
            entries, self.hidden_dim, observations.device
        )
        shape = hidden.shape
        tokens, pooled = self.projection(
            hidden.flatten(0, 1), token_mask.flatten(0, 1)
        )
        pooled = pooled.view(shape[0], shape[1], -1)
        repeated_observation = observations[:, None, :].expand_as(pooled)
        consistency = self.verifier(
            pooled.flatten(0, 1), repeated_observation.flatten(0, 1)
        ).view(shape[0], shape[1])
        score = logprob + self.verifier_weight * consistency - self.schema_penalty * schema
        first_score = score.masked_fill(rounds != 0, float("-inf"))
        selected = first_score.argmax(dim=1)
        rows = torch.arange(shape[0], device=observations.device)
        for round_index in range(1, self.max_rounds):
            needs_reflection = consistency[rows, selected] < self.verifier_threshold
            round_score = score.masked_fill(rounds != round_index, float("-inf"))
            has_round = round_score.isfinite().any(dim=1)
            replacement = round_score.argmax(dim=1)
            selected = torch.where(
                needs_reflection & has_round, replacement, selected
            )
        selected_tokens = tokens.view(shape[0], shape[1], shape[2], -1)[rows, selected]
        selected_mask = token_mask[rows, selected]
        verifier_loss = observations.new_zeros(())
        if compute_verifier_loss:
            verifier_loss = self._auxiliary_verifier_loss(
                entries,
                (
                    observations
                    if supervision_observations is None
                    else supervision_observations
                ),
            )
        return SelectedPlans(
            tokens=selected_tokens,
            token_mask=selected_mask,
            pooled=pooled[rows, selected],
            consistency=consistency[rows, selected],
            candidate_index=selected,
            round_index=rounds[rows, selected],
            verifier_loss=verifier_loss,
        )
