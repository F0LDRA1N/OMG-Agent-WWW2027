from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access and no implicit defaults."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_node(item) for item in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    cfg = _node(raw)
    project_root = path.parent.parent
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(project_root)
    for key in ("manifest", "mask_root", "plan_cache", "output_root"):
        value = Path(cfg.experiment[key])
        if not value.is_absolute():
            cfg.experiment[key] = str((project_root / value).resolve())
    validate_paper_contract(cfg)
    return cfg


def clone_config(cfg: ConfigNode) -> ConfigNode:
    return _node(deepcopy(dict(cfg)))


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def dump_config(cfg: ConfigNode, path: str | Path) -> None:
    payload = _plain(cfg)
    payload.pop("_config_path", None)
    payload.pop("_project_root", None)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def require_choice(value: Any, choices: set[Any], name: str) -> None:
    if value not in choices:
        options = ", ".join(map(str, sorted(choices)))
        raise ValueError(f"{name}={value!r}; expected one of: {options}")


def _expect(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name}={actual!r}; paper contract requires {expected!r}")


def validate_paper_contract(cfg: ConfigNode) -> None:
    """Reject silent drift from the dimensions and protocols reported in OMGFinal."""
    _expect(list(cfg.experiment.seeds), [1234, 1235, 1236, 1237, 1238], "seeds")
    _expect(int(cfg.experiment.plan_generation_seed), 2027, "plan_generation_seed")
    _expect(
        [float(value) for value in cfg.experiment.missing_rates],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "missing_rates",
    )
    _expect(list(cfg.data.modalities), ["text", "vision", "audio"], "modalities")
    _expect(dict(cfg.data.feature_dims), {"text": 768, "vision": 35, "audio": 74}, "feature_dims")
    expected_splits = {
        "mosi": {"train": 1284, "valid": 229, "test": 686},
        "mosei": {"train": 16326, "valid": 1871, "test": 4659},
    }
    dataset = str(cfg.experiment.dataset)
    require_choice(dataset, set(expected_splits), "dataset")
    _expect(dict(cfg.data.expected_split_sizes), expected_splits[dataset], "split_sizes")

    planner = cfg.model.planner
    _expect(str(planner.model_id), "Qwen/Qwen2.5-Omni-3B", "planner.model_id")
    for name, expected in (
        ("hidden_dim", 2048),
        ("shared_dim", 512),
        ("candidates", 5),
        ("max_rounds", 2),
    ):
        _expect(int(planner[name]), expected, f"planner.{name}")
    for name, expected in (
        ("temperature", 1.0),
        ("verifier_threshold", 0.6),
        ("verifier_weight", 0.3),
        ("schema_penalty", 0.1),
    ):
        _expect(float(planner[name]), expected, f"planner.{name}")

    retriever = cfg.model.retriever
    _expect(int(retriever.key_dim), 512, "retriever.key_dim")
    _expect(int(retriever.top_k), 10, "retriever.top_k")
    _expect(float(retriever.temperature), 0.07, "retriever.temperature")
    _expect(str(retriever.index_type), "ivfflat", "retriever.index_type")
    _expect(int(retriever.nlist), 256, "retriever.nlist")
    _expect(int(retriever.nprobe), 32, "retriever.nprobe")
    _expect(int(retriever.index_seed), 2027, "retriever.index_seed")

    executor = cfg.model.executor
    _expect(
        dict(executor.latent_dims),
        {"text": 128, "vision": 16, "audio": 32},
        "executor.latent_dims",
    )
    _expect(list(executor.widths), [128, 256, 512, 512], "executor.widths")
    _expect(int(executor.evidence_encoder_layers), 3, "executor.evidence_encoder_layers")
    _expect(int(executor.diffusion_steps), 100, "executor.diffusion_steps")
    _expect(int(executor.ddim_steps), 50, "executor.ddim_steps")
    _expect(float(executor.ddim_eta), 0.0, "executor.ddim_eta")
    _expect(float(executor.guidance_scale), 1.5, "executor.guidance_scale")
    _expect(float(executor.condition_dropout), 0.1, "executor.condition_dropout")
    _expect(int(executor.align_dim), 256, "executor.align_dim")
    _expect(str(executor.semantic_layers), "all", "executor.semantic_layers")
    _expect(int(cfg.model.predictor.hidden_dim), 256, "predictor.hidden_dim")
    _expect(int(cfg.model.predictor.layers), 3, "predictor.layers")
    _expect(int(cfg.model.predictor.heads), 8, "predictor.heads")

    _expect(str(cfg.training.optimizer), "adamw", "training.optimizer")
    _expect(int(cfg.training.epochs), 100, "training.epochs")
    _expect(int(cfg.training.batch_size), 32 if dataset == "mosi" else 128, "training.batch_size")
    _expect([float(value) for value in cfg.training.betas], [0.9, 0.999], "training.betas")
    _expect(float(cfg.training.eps), 1.0e-8, "training.eps")
    _expect(float(cfg.training.weight_decay), 0.01, "training.weight_decay")
    _expect(int(cfg.training.warmup_steps), 500, "training.warmup_steps")
    _expect(str(cfg.training.scheduler), "cosine", "training.scheduler")
    _expect(float(cfg.training.gradient_clip_norm), 1.0, "training.gradient_clip_norm")
    _expect(str(cfg.training.precision), "bf16", "training.precision")
    _expect(int(cfg.training.memory_refresh_epochs), 1, "training.memory_refresh_epochs")
    _expect(
        {key: float(value) for key, value in cfg.training.learning_rates.items()},
        {
            "plan_projection": 1.0e-4,
            "verifier": 5.0e-5,
            "retriever": 1.0e-4,
            "vae": 5.0e-5,
            "unet": 1.0e-4,
            "predictor": 5.0e-4,
        },
        "training.learning_rates",
    )
    _expect(
        dict(cfg.training.loss_weights),
        {"semantic": 0.1, "evidence": 0.1, "task": 0.1, "verifier": 1.0},
        "training.loss_weights",
    )
