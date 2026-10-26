from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..data.dataset import load_plan_cache
from ..data.manifest import load_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--plans", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_manifest(cfg.experiment.manifest)
    report = validate_manifest(
        records,
        dict(cfg.data.feature_dims),
        dict(cfg.data.expected_split_sizes),
        inspect_arrays=not args.metadata_only,
    )
    if args.plans:
        plan_path = Path(cfg.experiment.plan_cache)
        load_plan_cache(
            plan_path,
            manifest_path=cfg.experiment.manifest,
            expected_model_id=str(cfg.model.planner.model_id),
            expected_hidden_dim=int(cfg.model.planner.hidden_dim),
            expected_ids=[record.sample_id for record in records],
            expected_generation_seed=int(cfg.experiment.plan_generation_seed),
        )
        report["plan_cache"] = str(plan_path)
        report["plan_cache_valid"] = True
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
