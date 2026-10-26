from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.manifest import MODALITIES
from ..planner.cache import CachedPlanSelector
from ..retrieval.memory import TrainingMemory
from .blocks import FeatureContextEncoder
from .diffusion import TargetExecutor
from .predictor import MultimodalPredictor


class OMGAgent(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        feature_dims = {key: int(value) for key, value in cfg.data.feature_dims.items()}
        shared_dim = int(cfg.model.planner.shared_dim)
        self.modalities = tuple(MODALITIES)
        self.context_encoder = FeatureContextEncoder(feature_dims, shared_dim)
        self.plan_selector = CachedPlanSelector(
            hidden_dim=int(cfg.model.planner.hidden_dim),
            shared_dim=shared_dim,
            verifier_weight=float(cfg.model.planner.verifier_weight),
            schema_penalty=float(cfg.model.planner.schema_penalty),
            verifier_threshold=float(cfg.model.planner.verifier_threshold),
            max_rounds=int(cfg.model.planner.max_rounds),
        )
        self.executors = nn.ModuleDict(
            {
                modality: TargetExecutor(
                    feature_dim=feature_dims[modality],
                    latent_dim=int(cfg.model.executor.latent_dims[modality]),
                    widths=[int(value) for value in cfg.model.executor.widths],
                    shared_dim=shared_dim,
                    align_dim=int(cfg.model.executor.align_dim),
                    evidence_encoder_layers=int(
                        cfg.model.executor.evidence_encoder_layers
                    ),
                    diffusion_steps=int(cfg.model.executor.diffusion_steps),
                    condition_dropout=float(cfg.model.executor.condition_dropout),
                )
                for modality in self.modalities
            }
        )
        self.predictor = MultimodalPredictor(
            feature_dims,
            hidden_dim=int(cfg.model.predictor.hidden_dim),
            layers=int(cfg.model.predictor.layers),
            heads=int(cfg.model.predictor.heads),
        )
        self.ddim_steps = int(cfg.model.executor.ddim_steps)
        self.ddim_eta = float(cfg.model.executor.ddim_eta)
        self.guidance_scale = float(cfg.model.executor.guidance_scale)
        self.loss_weights = {
            key: float(value) for key, value in cfg.training.loss_weights.items()
        }
        self.memory: TrainingMemory | None = None

    def attach_memory(self, memory: TrainingMemory) -> None:
        self.memory = memory

    def refresh_memory(self) -> None:
        if self.memory is None:
            raise RuntimeError("Training memory is not attached")
        self.memory.refresh(
            self.plan_selector,
            self.context_encoder,
        )

    def _require_memory(self) -> TrainingMemory:
        if self.memory is None:
            raise RuntimeError("Training memory is not attached")
        return self.memory

    def forward(
        self,
        batch: dict[str, Any],
        reconstruct_with_ddim: bool = False,
    ) -> dict[str, Any]:
        memory = self._require_memory()
        availability = batch["availability"]
        context, pooled = self.context_encoder(
            batch["features"], batch["feature_masks"], availability
        )
        supervision_context = None
        if self.training:
            full_availability = torch.ones_like(availability)
            supervision_context = self.context_encoder.from_pooled(
                pooled, full_availability
            )
        plans = self.plan_selector(
            batch["plans"],
            context,
            compute_verifier_loss=self.training,
            supervision_observations=supervision_context,
        )
        completed = {
            modality: pooled[modality] for modality in self.modalities
        }
        component_losses: dict[str, list[torch.Tensor]] = {
            "noise": [],
            "semantic": [],
            "evidence": [],
        }
        retrieval: dict[str, Any] = {}

        for target_index, target in enumerate(self.modalities):
            missing = ~availability[:, target_index]
            if not missing.any():
                continue
            rows = missing.nonzero(as_tuple=False).squeeze(1)
            row_ids = [batch["ids"][int(row)] for row in rows]
            evidence = memory.retrieve(
                target=target,
                query_plan=plans.pooled.index_select(0, rows),
                query_context=context.index_select(0, rows),
                query_availability=availability.index_select(0, rows),
                query_ids=row_ids,
                plan_selector=self.plan_selector,
                context_encoder=self.context_encoder,
                target_encoder=self.executors[target].autoencoder.encode,
            )
            retrieval[target] = {
                "rows": rows,
                "indices": evidence.indices,
                "similarities": evidence.similarities,
            }
            if reconstruct_with_ddim:
                reconstructed = self.executors[target].sample(
                    batch_size=len(rows),
                    observed_context=context.index_select(0, rows),
                    plan_tokens=plans.tokens.index_select(0, rows),
                    plan_mask=plans.token_mask.index_select(0, rows),
                    evidence=evidence.values,
                    ddim_steps=self.ddim_steps,
                    guidance_scale=self.guidance_scale,
                    eta=self.ddim_eta,
                )
            else:
                output = self.executors[target].training_loss(
                    target_feature=pooled[target].index_select(0, rows),
                    observed_context=context.index_select(0, rows),
                    plan_tokens=plans.tokens.index_select(0, rows),
                    plan_mask=plans.token_mask.index_select(0, rows),
                    plan_pooled=plans.pooled.index_select(0, rows),
                    evidence=evidence.values,
                )
                reconstructed = output["reconstructed"]
                for loss_name in component_losses:
                    component_losses[loss_name].append(output[loss_name])
            replacement = completed[target].clone()
            replacement.index_copy_(0, rows, reconstructed)
            completed[target] = replacement

        predictions = self.predictor(completed)
        task_loss = F.mse_loss(predictions, batch["labels"])
        zeros = task_loss.new_zeros(())
        reduced = {
            name: torch.stack(values).mean() if values else zeros
            for name, values in component_losses.items()
        }
        total = (
            reduced["noise"]
            + self.loss_weights["semantic"] * reduced["semantic"]
            + self.loss_weights["evidence"] * reduced["evidence"]
            + self.loss_weights["task"] * task_loss
            + self.loss_weights["verifier"] * plans.verifier_loss
        )
        return {
            "loss": total,
            "losses": {
                **reduced,
                "task": task_loss,
                "verifier": plans.verifier_loss,
            },
            "predictions": predictions,
            "completed_features": completed,
            "selected_candidate": plans.candidate_index,
            "planning_round": plans.round_index,
            "consistency": plans.consistency,
            "retrieval": retrieval,
        }
