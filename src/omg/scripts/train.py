from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ..config import dump_config, load_config
from ..data.dataset import OMGDataset, build_dataloader, load_plan_cache
from ..data.manifest import load_manifest, records_for_split, validate_manifest
from ..data.masks import load_mask_manifest, mask_path
from ..engine import (
    autocast_context,
    build_agent,
    cosine_scheduler,
    device_from_arg,
    move_batch,
    optimizer_for,
    predict_loader,
    save_checkpoint,
)
from ..reproducibility import (
    append_jsonl,
    environment_metadata,
    seed_everything,
    sha256_file,
    write_json,
)


def _run_directory(cfg, mr: float, seed: int) -> Path:
    return Path(cfg.experiment.output_root) / f"mr_{mr:.1f}" / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mr", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mr not in [float(value) for value in cfg.experiment.missing_rates]:
        raise ValueError(f"MR {args.mr} is not configured")
    if args.seed not in [int(value) for value in cfg.experiment.seeds]:
        raise ValueError(f"Seed {args.seed} is not configured")
    run_directory = _run_directory(cfg, args.mr, args.seed)
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"{run_directory} is not empty; move the partial run or choose "
            "a new output root"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = device_from_arg(args.device)

    records = load_manifest(cfg.experiment.manifest)
    validate_manifest(
        records,
        dict(cfg.data.feature_dims),
        dict(cfg.data.expected_split_sizes),
        inspect_arrays=False,
    )
    plans = load_plan_cache(
        cfg.experiment.plan_cache,
        manifest_path=cfg.experiment.manifest,
        expected_model_id=str(cfg.model.planner.model_id),
        expected_hidden_dim=int(cfg.model.planner.hidden_dim),
        expected_ids=[record.sample_id for record in records],
        expected_generation_seed=int(cfg.experiment.plan_generation_seed),
    )
    manifest_sha256 = sha256_file(cfg.experiment.manifest)
    datasets = {}
    mask_files = {}
    mask_metadata = {}
    for split in ("train", "valid"):
        split_records = records_for_split(records, split)
        path = mask_path(cfg.experiment.mask_root, args.mr, args.seed, split)
        masks, metadata = load_mask_manifest(
            path, [record.sample_id for record in split_records]
        )
        if (
            float(metadata.get("mr", -1)) != args.mr
            or int(metadata.get("seed", -1)) != args.seed
        ):
            raise ValueError(f"Mask metadata mismatch in {path}")
        datasets[split] = OMGDataset(records, split, masks, plans)
        mask_files[split] = path
        mask_metadata[split] = metadata
    loaders = {
        "train": build_dataloader(
            datasets["train"],
            int(cfg.training.batch_size),
            int(cfg.data.num_workers),
            args.seed,
            shuffle=True,
        ),
        "valid": build_dataloader(
            datasets["valid"],
            int(cfg.training.batch_size),
            int(cfg.data.num_workers),
            args.seed,
            shuffle=False,
        ),
    }
    model = build_agent(
        cfg, records_for_split(records, "train"), plans, device
    )
    optimizer = optimizer_for(model, cfg)
    total_steps = int(cfg.training.epochs) * len(loaders["train"])
    scheduler = cosine_scheduler(
        optimizer, int(cfg.training.warmup_steps), total_steps
    )
    dump_config(cfg, run_directory / "config.resolved.yaml")
    write_json(run_directory / "environment.json", environment_metadata())
    write_json(
        run_directory / "mask.sha256",
        {
            split: {
                "sha256": sha256_file(path),
                "metadata": mask_metadata[split],
            }
            for split, path in mask_files.items()
        },
    )
    run_metadata = {
        "format_version": 1,
        "dataset": str(cfg.experiment.dataset),
        "mr": float(args.mr),
        "seed": int(args.seed),
        "manifest_sha256": manifest_sha256,
        "planner_model_id": str(cfg.model.planner.model_id),
        "plan_cache_format": 7,
        "plan_generation_seed": int(cfg.experiment.plan_generation_seed),
        "train_mask_sha256": sha256_file(mask_files["train"]),
        "valid_mask_sha256": sha256_file(mask_files["valid"]),
    }
    write_json(run_directory / "run.json", run_metadata)

    best_mae = float("inf")
    precision = str(cfg.training.precision)
    for epoch in range(1, int(cfg.training.epochs) + 1):
        if (epoch - 1) % int(cfg.training.memory_refresh_epochs) == 0:
            model.refresh_memory()
        model.train()
        epoch_totals: dict[str, float] = {}
        samples = 0
        progress = tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                output = model(batch, reconstruct_with_ddim=False)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.training.gradient_clip_norm)
            )
            optimizer.step()
            scheduler.step()
            batch_size = len(batch["ids"])
            samples += batch_size
            values = {"total": output["loss"], **output["losses"]}
            for name, value in values.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + float(
                    value.detach().float().cpu()
                ) * batch_size
            progress.set_postfix(loss=f"{float(output['loss'].detach()):.4f}")

        model.refresh_memory()
        cuda_devices = (
            list(range(torch.cuda.device_count()))
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(args.seed)
            validation_metrics, _ = predict_loader(
                model, loaders["valid"], device, precision
            )
        history = {
            "epoch": epoch,
            "train": {
                name: value / max(samples, 1)
                for name, value in epoch_totals.items()
            },
            "valid": validation_metrics,
            "learning_rates": {
                group.get("name", str(index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
        }
        append_jsonl(run_directory / "history.jsonl", history)
        if validation_metrics["mae"] < best_mae:
            best_mae = validation_metrics["mae"]
            save_checkpoint(
                run_directory / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_mae,
                run_metadata,
            )
        save_checkpoint(
            run_directory / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_mae,
            run_metadata,
        )
        print(json.dumps(history, sort_keys=True))


if __name__ == "__main__":
    main()
