from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..config import load_config
from ..data.manifest import (
    MODALITIES,
    load_manifest,
    records_for_split,
    validate_manifest,
)
from ..data.masks import (
    ALL_OBSERVED,
    INCOMPLETE_PATTERNS,
    load_mask_manifest,
    mask_path,
    pattern_key,
)
from ..reproducibility import sha256_file, write_json

METRICS = ("acc2", "f1", "acc7", "mae", "corr")
VARIANT_SUFFIX = {"cached": "", "live": ".live", "online": ".online"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _require_metrics(payload: dict[str, Any], location: str) -> None:
    for metric in METRICS:
        if metric not in payload or not math.isfinite(float(payload[metric])):
            raise ValueError(f"{location}: missing or non-finite metric {metric}")


def _audit_retrieval(
    rows: list[dict[str, Any]],
    train_ids: set[str],
    expected_targets: dict[str, set[str]],
    location: str,
) -> None:
    for row in rows:
        sample_id = str(row["id"])
        retrieved = row.get("retrieved_train_ids")
        if not isinstance(retrieved, dict):
            raise ValueError(f"{location}/{sample_id}: invalid retrieval provenance")
        if set(retrieved) != expected_targets[str(row["available"])]:
            raise ValueError(
                f"{location}/{sample_id}: retrieved targets {sorted(retrieved)} "
                f"do not match availability {row['available']}"
            )
        for target, neighbors in retrieved.items():
            if len(neighbors) != 10:
                raise ValueError(
                    f"{location}/{sample_id}/{target}: expected 10 neighbors"
                )
            if len(set(neighbors)) != len(neighbors):
                raise ValueError(
                    f"{location}/{sample_id}/{target}: duplicate neighbors"
                )
            foreign = set(map(str, neighbors)) - train_ids
            if foreign:
                raise ValueError(
                    f"{location}/{sample_id}/{target}: non-training neighbors "
                    f"{sorted(foreign)[:5]}"
                )


def _run_directory(root: Path, mr: float, seed: int) -> Path:
    return root / f"mr_{mr:.1f}" / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit run completeness, mask identity, and train-only retrieval"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_SUFFIX),
        default="cached",
        help="Evaluation artifact variant to audit",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.experiment.output_root)
    records = load_manifest(cfg.experiment.manifest)
    validate_manifest(
        records,
        dict(cfg.data.feature_dims),
        dict(cfg.data.expected_split_sizes),
        inspect_arrays=False,
    )
    train_ids = {
        record.sample_id for record in records_for_split(records, "train")
    }
    test_records = records_for_split(records, "test")
    test_ids = {record.sample_id for record in test_records}
    expected_targets = {}
    for pattern in (*INCOMPLETE_PATTERNS, ALL_OBSERVED):
        key = pattern_key(pattern)
        expected_targets[key] = {
            modality
            for modality, available in zip(MODALITIES, pattern)
            if not available
        }

    missing_artifacts: list[str] = []
    audited_runs = 0
    suffix = VARIANT_SUFFIX[args.variant]
    for mr in [float(value) for value in cfg.experiment.missing_rates]:
        for seed in [int(value) for value in cfg.experiment.seeds]:
            run = _run_directory(root, mr, seed)
            required = [
                run / "best.pt",
                run / "last.pt",
                run / "run.json",
                run / "environment.json",
                run / "mask.sha256",
                run / "history.jsonl",
                run / f"metrics.random{suffix}.json",
                run / f"predictions.random{suffix}.jsonl",
            ]
            if mr == 0.7:
                required.extend(
                    [
                        run / f"metrics.fixed{suffix}.json",
                        run / f"predictions.fixed{suffix}.jsonl",
                    ]
                )
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                if args.allow_incomplete:
                    missing_artifacts.extend(absent)
                    continue
                raise FileNotFoundError(
                    f"MR={mr:.1f}/seed={seed}: missing {absent}"
                )

            run_metadata = _read_json(run / "run.json")
            expected_run_metadata = {
                "dataset": str(cfg.experiment.dataset),
                "mr": mr,
                "seed": seed,
                "manifest_sha256": sha256_file(cfg.experiment.manifest),
                "planner_model_id": str(cfg.model.planner.model_id),
                "plan_cache_format": 7,
                "plan_generation_seed": int(
                    cfg.experiment.plan_generation_seed
                ),
            }
            mismatches = {
                key: (run_metadata.get(key), expected)
                for key, expected in expected_run_metadata.items()
                if run_metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"{run}/run.json mismatch: {mismatches}")
            stored_mask_hashes = _read_json(run / "mask.sha256")
            for split in ("train", "valid"):
                split_records = records_for_split(records, split)
                split_mask_path = mask_path(
                    cfg.experiment.mask_root, mr, seed, split
                )
                load_mask_manifest(
                    split_mask_path,
                    [record.sample_id for record in split_records],
                )
                actual_hash = sha256_file(split_mask_path)
                if (
                    stored_mask_hashes.get(split, {}).get("sha256")
                    != actual_hash
                    or run_metadata.get(f"{split}_mask_sha256")
                    != actual_hash
                ):
                    raise ValueError(f"{run}: {split} mask hash mismatch")

            history = _read_jsonl(run / "history.jsonl")
            expected_epochs = int(cfg.training.epochs)
            if [int(row["epoch"]) for row in history] != list(
                range(1, expected_epochs + 1)
            ):
                raise ValueError(f"{run}: incomplete or non-sequential history")

            random_metrics = _read_json(
                run / f"metrics.random{suffix}.json"
            )
            _require_metrics(random_metrics, f"{run}/random")
            if random_metrics.get("checkpoint") != str(
                (run / "best.pt").resolve()
            ):
                raise ValueError(f"{run}/random: evaluation did not use best.pt")
            if bool(random_metrics.get("online_planner")) != (
                args.variant == "online"
            ) or bool(random_metrics.get("live_reflection")) != (
                args.variant in {"live", "online"}
            ):
                raise ValueError(f"{run}/random: planner variant flags mismatch")
            random_rows = _read_jsonl(
                run / f"predictions.random{suffix}.jsonl"
            )
            counts = Counter(str(row["id"]) for row in random_rows)
            if set(counts) != test_ids or set(counts.values()) != {1}:
                raise ValueError(f"{run}/random: test IDs are not exactly once")
            masks, mask_metadata = load_mask_manifest(
                mask_path(cfg.experiment.mask_root, mr, seed, "test"),
                test_ids,
            )
            expected_available = {
                sample_id: pattern_key(pattern)
                for sample_id, pattern in masks.items()
            }
            for row in random_rows:
                if str(row["available"]) != expected_available[str(row["id"])]:
                    raise ValueError(
                        f"{run}/random/{row['id']}: prediction mask mismatch"
                    )
            test_mask_path = mask_path(
                cfg.experiment.mask_root, mr, seed, "test"
            )
            if random_metrics.get("mask_sha256") != sha256_file(test_mask_path):
                raise ValueError(f"{run}/random: test-mask hash mismatch")
            if random_metrics.get("mask_metadata") != mask_metadata:
                raise ValueError(f"{run}/random: mask metadata mismatch")
            _audit_retrieval(
                random_rows, train_ids, expected_targets, f"{run}/random"
            )

            if mr == 0.7:
                fixed_metrics = _read_json(
                    run / f"metrics.fixed{suffix}.json"
                )
                expected_patterns = set(expected_targets)
                if set(fixed_metrics) != expected_patterns:
                    raise ValueError(f"{run}/fixed: missing availability metrics")
                for pattern, metrics in fixed_metrics.items():
                    _require_metrics(metrics, f"{run}/fixed/{pattern}")
                    if metrics.get("checkpoint") != str(
                        (run / "best.pt").resolve()
                    ):
                        raise ValueError(
                            f"{run}/fixed/{pattern}: evaluation did not use best.pt"
                        )
                    if bool(metrics.get("online_planner")) != (
                        args.variant == "online"
                    ) or bool(metrics.get("live_reflection")) != (
                        args.variant in {"live", "online"}
                    ):
                        raise ValueError(
                            f"{run}/fixed/{pattern}: planner variant flags mismatch"
                        )
                fixed_rows = _read_jsonl(
                    run / f"predictions.fixed{suffix}.jsonl"
                )
                by_id: dict[str, set[str]] = defaultdict(set)
                for row in fixed_rows:
                    pattern = str(row.get("fixed_pattern"))
                    if str(row["available"]) != pattern:
                        raise ValueError(
                            f"{run}/fixed/{row['id']}: pattern mismatch"
                        )
                    by_id[str(row["id"])].add(pattern)
                if set(by_id) != test_ids or any(
                    patterns != expected_patterns for patterns in by_id.values()
                ):
                    raise ValueError(
                        f"{run}/fixed: each test ID must cover all seven patterns"
                    )
                if len(fixed_rows) != len(test_ids) * len(expected_patterns):
                    raise ValueError(f"{run}/fixed: duplicate prediction rows")
                _audit_retrieval(
                    fixed_rows, train_ids, expected_targets, f"{run}/fixed"
                )
            audited_runs += 1

    report = {
        "audit_version": 1,
        "dataset": str(cfg.experiment.dataset),
        "variant": args.variant,
        "audited_runs": audited_runs,
        "expected_runs": len(cfg.experiment.missing_rates)
        * len(cfg.experiment.seeds),
        "missing_artifacts": missing_artifacts,
        "status": "complete" if not missing_artifacts else "incomplete",
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / f"audit.{args.variant}.json"
    )
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
