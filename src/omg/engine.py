from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .data.manifest import ManifestRecord
from .metrics import sentiment_metrics
from .model.agent import OMGAgent
from .retrieval.memory import TrainingMemory


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "labels": batch["labels"].to(device),
        "availability": batch["availability"].to(device),
        "features": {
            key: value.to(device, non_blocking=True)
            for key, value in batch["features"].items()
        },
        "feature_masks": {
            key: value.to(device, non_blocking=True)
            for key, value in batch["feature_masks"].items()
        },
    }


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"Unsupported precision: {precision}")


def build_agent(
    cfg: Any,
    train_records: list[ManifestRecord],
    plan_cache: dict[str, Any],
    device: torch.device,
) -> OMGAgent:
    model = OMGAgent(cfg)
    retriever = cfg.model.retriever
    memory = TrainingMemory(
        records=train_records,
        plan_cache=plan_cache,
        feature_dims=dict(cfg.data.feature_dims),
        plan_hidden_dim=int(cfg.model.planner.hidden_dim),
        shared_dim=int(cfg.model.planner.shared_dim),
        key_dim=int(retriever.key_dim),
        top_k=int(retriever.top_k),
        temperature=float(retriever.temperature),
        nlist=int(retriever.nlist),
        nprobe=int(retriever.nprobe),
        index_seed=int(retriever.index_seed),
    )
    model.attach_memory(memory)
    model.to(device)
    return model


def optimizer_for(model: OMGAgent, cfg: Any) -> torch.optim.Optimizer:
    rates = cfg.training.learning_rates
    if model.memory is None:
        raise RuntimeError("Attach memory before creating the optimizer")
    groups = [
        {
            "name": "plan_projection",
            "params": list(model.plan_selector.projection.parameters()),
            "lr": float(rates.plan_projection),
        },
        {
            "name": "verifier",
            "params": list(model.plan_selector.verifier.parameters()),
            "lr": float(rates.verifier),
        },
        {
            "name": "retriever",
            "params": [
                *model.context_encoder.parameters(),
                *model.memory.query_projection.parameters(),
                *model.memory.key_projection.parameters(),
            ],
            "lr": float(rates.retriever),
        },
        {
            "name": "vae_and_alignment",
            "params": [
                parameter
                for executor in model.executors.values()
                for module in (
                    executor.autoencoder,
                    executor.context_projection,
                    executor.semantic_head,
                    executor.plan_semantic_head,
                    executor.evidence_head,
                    executor.memory_evidence_head,
                )
                for parameter in module.parameters()
            ],
            "lr": float(rates.vae),
        },
        {
            "name": "unet",
            "params": [
                parameter
                for executor in model.executors.values()
                for parameter in executor.unet.parameters()
            ],
            "lr": float(rates.unet),
        },
        {
            "name": "predictor",
            "params": list(model.predictor.parameters()),
            "lr": float(rates.predictor),
        },
    ]
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len({id(parameter) for parameter in grouped}) != len(grouped):
        raise RuntimeError("A parameter occurs in multiple optimizer groups")
    missing = {id(parameter) for parameter in trainable} - {
        id(parameter) for parameter in grouped
    }
    if missing:
        raise RuntimeError(f"{len(missing)} trainable parameters lack an optimizer group")
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.no_grad()
def predict_loader(
    model: OMGAgent,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    precision: str,
    batch_preprocessor: Callable[[dict[str, Any], OMGAgent], None] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels: list[float] = []
    predictions: list[float] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        if batch_preprocessor is not None:
            batch_preprocessor(batch, model)
        with autocast_context(device, precision):
            output = model(batch, reconstruct_with_ddim=True)
        batch_predictions = output["predictions"].float().cpu().tolist()
        batch_labels = batch["labels"].float().cpu().tolist()
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)
        retrieval_by_row: dict[int, dict[str, list[str]]] = {}
        if model.memory is None:
            raise RuntimeError("Evaluation model has no training memory")
        for target, retrieval in output["retrieval"].items():
            global_rows = retrieval["rows"].detach().cpu().tolist()
            neighbor_rows = retrieval["indices"].detach().cpu().tolist()
            for global_row, neighbors in zip(global_rows, neighbor_rows):
                retrieval_by_row.setdefault(int(global_row), {})[target] = [
                    model.memory.sample_ids[int(neighbor)] for neighbor in neighbors
                ]
        for index, sample_id in enumerate(batch["ids"]):
            rows.append(
                {
                    "id": sample_id,
                    "label": batch_labels[index],
                    "prediction": batch_predictions[index],
                    "available": "".join(
                        "1" if value else "0"
                        for value in batch["availability"][index].cpu().tolist()
                    ),
                    "selected_candidate": int(
                        output["selected_candidate"][index].cpu()
                    ),
                    "planning_round": int(output["planning_round"][index].cpu()),
                    "consistency": float(output["consistency"][index].cpu()),
                    "retrieved_train_ids": retrieval_by_row.get(index, {}),
                }
            )
    return sentiment_metrics(labels, predictions), rows


def save_checkpoint(
    path: str | Path,
    model: OMGAgent,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_mae: float,
    run_metadata: dict[str, Any],
) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mae": best_mae,
            "run_metadata": run_metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_model_checkpoint(path: str | Path, model: OMGAgent) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return payload
