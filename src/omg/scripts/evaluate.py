from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..data.dataset import OMGDataset, build_dataloader, load_plan_cache
from ..data.manifest import (
    MODALITIES,
    load_manifest,
    records_for_split,
    validate_manifest,
)
from ..data.masks import (
    INCOMPLETE_PATTERNS,
    ALL_OBSERVED,
    load_mask_manifest,
    mask_path,
    pattern_key,
)
from ..engine import (
    build_agent,
    device_from_arg,
    load_model_checkpoint,
    predict_loader,
)
from ..planner.prompts import diagnostic_feedback
from ..planner.qwen import QwenPlannerBackend
from ..reproducibility import seed_everything, sha256_file, write_json


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _loader(cfg, records, plans, masks, seed):
    dataset = OMGDataset(records, "test", masks, plans)
    return build_dataloader(
        dataset,
        int(cfg.training.batch_size),
        int(cfg.data.num_workers),
        seed,
        shuffle=False,
    )


def _field_content(text: str, field: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0)).get(field, "")
            return json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)
        except json.JSONDecodeError:
            pass
    return field


def _live_reflector(
    backend: QwenPlannerBackend,
    cfg: Any,
    seed: int,
    regenerate_first_round: bool,
):
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else []
    )

    def preprocess(batch: dict[str, Any], model) -> None:
        if regenerate_first_round:
            for row, sample_id in enumerate(batch["ids"]):
                availability = (
                    batch["availability"][row].detach().cpu().tolist()
                )
                pattern = "".join(
                    "1" if value else "0" for value in availability
                )
                digest = hashlib.sha256(
                    f"{seed}|{sample_id}|{pattern}|online-first".encode()
                ).digest()
                generation_seed = int.from_bytes(digest[:8], "big") % (2**31)
                available = [
                    modality
                    for modality, value in zip(MODALITIES, availability)
                    if value
                ]
                missing = [
                    modality
                    for modality, value in zip(MODALITIES, availability)
                    if not value
                ]
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(generation_seed)
                    if cuda_devices:
                        torch.cuda.manual_seed_all(generation_seed)
                    generated = backend.generate_candidates(
                        batch["raw"][row],
                        available,
                        missing,
                        count=int(cfg.model.planner.candidates),
                        temperature=float(cfg.model.planner.temperature),
                    )
                entry = batch["plans"][row]
                batch["plans"][row] = {
                    **entry,
                    "candidates": [
                        backend.cache_candidate(candidate, 0)
                        for candidate in generated
                    ],
                }
        with torch.no_grad():
            observations, _ = model.context_encoder(
                batch["features"],
                batch["feature_masks"],
                batch["availability"],
            )
        diagnostics = model.plan_selector.reflection_diagnostics(
            batch["plans"], observations
        )
        for row, diagnostic in enumerate(diagnostics):
            if diagnostic is None:
                continue
            availability = batch["availability"][row].detach().cpu().tolist()
            pattern = "".join("1" if value else "0" for value in availability)
            digest = hashlib.sha256(
                f"{seed}|{batch['ids'][row]}|{pattern}|live-reflection".encode()
            ).digest()
            generation_seed = int.from_bytes(digest[:8], "big") % (2**31)
            entry = batch["plans"][row]
            initial = entry["candidates"][diagnostic.candidate_index]
            feedback = diagnostic_feedback(
                diagnostic.field,
                _field_content(initial["text"], diagnostic.field),
                diagnostic.consistency,
            )
            available = [
                modality
                for modality, value in zip(MODALITIES, availability)
                if value
            ]
            missing = [
                modality
                for modality, value in zip(MODALITIES, availability)
                if not value
            ]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(generation_seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(generation_seed)
                reflected = backend.generate_candidates(
                    batch["raw"][row],
                    available,
                    missing,
                    count=int(cfg.model.planner.candidates),
                    temperature=float(cfg.model.planner.temperature),
                    feedback=feedback,
                )
            first_round = [
                candidate
                for candidate in entry["candidates"]
                if int(candidate.get("round", 0)) == 0
            ]
            batch["plans"][row] = {
                **entry,
                "candidates": [
                    *first_round,
                    *[
                        backend.cache_candidate(candidate, 1)
                        for candidate in reflected
                    ],
                ],
            }

    return preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mr", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--protocol", choices=("random", "fixed"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--live-reflection", action="store_true")
    parser.add_argument(
        "--online-planner",
        action="store_true",
        help="Generate both planning rounds online; includes --live-reflection",
    )
    parser.add_argument("--planner-device-map", default="auto")
    parser.add_argument("--planner-dtype", default="bfloat16")
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mr not in [float(value) for value in cfg.experiment.missing_rates]:
        raise ValueError(f"MR {args.mr} is not configured")
    if args.seed not in [int(value) for value in cfg.experiment.seeds]:
        raise ValueError(f"Seed {args.seed} is not configured")
    if args.protocol == "fixed" and args.mr != 0.7:
        raise ValueError("The paper's fixed-availability protocol uses the MR=0.7 checkpoint")
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    records = load_manifest(cfg.experiment.manifest)
    validate_manifest(
        records,
        dict(cfg.data.feature_dims),
        dict(cfg.data.expected_split_sizes),
        inspect_arrays=False,
    )
    train_records = records_for_split(records, "train")
    test_records = records_for_split(records, "test")
    plans = load_plan_cache(
        cfg.experiment.plan_cache,
        manifest_path=cfg.experiment.manifest,
        expected_model_id=str(cfg.model.planner.model_id),
        expected_hidden_dim=int(cfg.model.planner.hidden_dim),
        expected_ids=[record.sample_id for record in records],
        expected_generation_seed=int(cfg.experiment.plan_generation_seed),
    )
    model = build_agent(cfg, train_records, plans, device)
    run_directory = (
        Path(cfg.experiment.output_root)
        / f"mr_{args.mr:.1f}"
        / f"seed_{args.seed}"
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_directory / "best.pt"
    checkpoint_payload = load_model_checkpoint(checkpoint, model)
    checkpoint_metadata = checkpoint_payload.get("run_metadata")
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError(
            f"{checkpoint} has no v5 run metadata; refusing an untraceable evaluation"
        )
    expected_checkpoint_metadata = {
        "dataset": str(cfg.experiment.dataset),
        "mr": float(args.mr),
        "seed": int(args.seed),
        "manifest_sha256": sha256_file(cfg.experiment.manifest),
        "planner_model_id": str(cfg.model.planner.model_id),
        "plan_cache_format": 7,
        "plan_generation_seed": int(cfg.experiment.plan_generation_seed),
    }
    mismatches = {
        key: (checkpoint_metadata.get(key), expected)
        for key, expected in expected_checkpoint_metadata.items()
        if checkpoint_metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint/config/protocol metadata mismatch: {mismatches}")
    model.refresh_memory()
    precision = str(cfg.training.precision)
    batch_preprocessor = None
    if args.live_reflection or args.online_planner:
        backend = QwenPlannerBackend(
            str(cfg.model.planner.model_id),
            device_map=args.planner_device_map,
            dtype=args.planner_dtype,
            flash_attention=args.flash_attention,
        )
        batch_preprocessor = _live_reflector(
            backend,
            cfg,
            args.seed,
            regenerate_first_round=args.online_planner,
        )

    if args.protocol == "random":
        path = mask_path(cfg.experiment.mask_root, args.mr, args.seed, "test")
        masks, metadata = load_mask_manifest(
            path, [record.sample_id for record in test_records]
        )
        if (
            float(metadata.get("mr", -1)) != args.mr
            or int(metadata.get("seed", -1)) != args.seed
        ):
            raise ValueError(f"Mask metadata mismatch in {path}")
        metrics, rows = predict_loader(
            model,
            _loader(cfg, records, plans, masks, args.seed),
            device,
            precision,
            batch_preprocessor=batch_preprocessor,
        )
        metrics["mask_sha256"] = sha256_file(path)
        metrics["mask_metadata"] = metadata
        metrics["live_reflection"] = args.live_reflection or args.online_planner
        metrics["online_planner"] = args.online_planner
        metrics["checkpoint_epoch"] = int(checkpoint_payload["epoch"])
        metrics["checkpoint_best_valid_mae"] = float(
            checkpoint_payload["best_mae"]
        )
        metrics["checkpoint"] = str(checkpoint.resolve())
        suffix = (
            "random.online"
            if args.online_planner
            else "random.live"
            if args.live_reflection
            else "random"
        )
        write_json(run_directory / f"metrics.{suffix}.json", metrics)
        _write_jsonl(run_directory / f"predictions.{suffix}.jsonl", rows)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    fixed_patterns = (*INCOMPLETE_PATTERNS, ALL_OBSERVED)
    all_metrics: dict[str, Any] = {}
    all_rows = []
    for pattern in fixed_patterns:
        key = pattern_key(pattern)
        masks = {record.sample_id: pattern for record in test_records}
        metrics, rows = predict_loader(
            model,
            _loader(cfg, records, plans, masks, args.seed),
            device,
            precision,
            batch_preprocessor=batch_preprocessor,
        )
        metrics["live_reflection"] = args.live_reflection or args.online_planner
        metrics["online_planner"] = args.online_planner
        metrics["checkpoint_epoch"] = int(checkpoint_payload["epoch"])
        metrics["checkpoint_best_valid_mae"] = float(
            checkpoint_payload["best_mae"]
        )
        metrics["checkpoint"] = str(checkpoint.resolve())
        all_metrics[key] = metrics
        all_rows.extend({**row, "fixed_pattern": key} for row in rows)
    suffix = (
        "fixed.online"
        if args.online_planner
        else "fixed.live"
        if args.live_reflection
        else "fixed"
    )
    write_json(run_directory / f"metrics.{suffix}.json", all_metrics)
    _write_jsonl(run_directory / f"predictions.{suffix}.jsonl", all_rows)
    print(json.dumps(all_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
