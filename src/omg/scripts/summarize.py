from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("acc2", "f1", "acc7", "mae", "corr")
VARIANT_SUFFIX = {"cached": "", "live": ".live", "online": ".online"}


def _coordinates(path: Path, root: Path) -> tuple[float, int]:
    relative = path.relative_to(root)
    mr = float(relative.parts[0].removeprefix("mr_"))
    seed = int(relative.parts[1].removeprefix("seed_"))
    return mr, seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[1234, 1235, 1236, 1237, 1238],
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--variant", choices=tuple(VARIANT_SUFFIX), default="cached"
    )
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    groups: dict[tuple[str, float, str], list[tuple[int, dict[str, float]]]] = defaultdict(list)
    suffix = VARIANT_SUFFIX[args.variant]
    for path in sorted(
        root.glob(f"mr_*/seed_*/metrics.random{suffix}.json")
    ):
        mr, seed = _coordinates(path, root)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        groups[("random", mr, "mixed")].append((seed, payload))
    for path in sorted(
        root.glob(f"mr_*/seed_*/metrics.fixed{suffix}.json")
    ):
        mr, seed = _coordinates(path, root)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for pattern, metrics in payload.items():
            groups[("fixed", mr, pattern)].append((seed, metrics))
    if not groups:
        raise FileNotFoundError(f"No metric files below {root}")
    if not args.allow_incomplete:
        expected_groups = {
            ("random", round(index / 10, 1), "mixed")
            for index in range(8)
        } | {
            ("fixed", 0.7, pattern)
            for pattern in ("100", "010", "001", "110", "101", "011", "111")
        }
        if set(groups) != expected_groups:
            missing = sorted(expected_groups - set(groups))
            extra = sorted(set(groups) - expected_groups)
            raise ValueError(
                f"Result grid mismatch: missing={missing}, extra={extra}"
            )

    summary: list[dict[str, Any]] = []
    for (protocol, mr, pattern), runs in sorted(groups.items()):
        seeds = sorted(seed for seed, _ in runs)
        expected_seeds = sorted(args.expected_seeds)
        if not args.allow_incomplete and seeds != expected_seeds:
            raise ValueError(
                f"{protocol}/MR={mr}/{pattern}: found seeds {seeds}; "
                f"expected seeds {expected_seeds}"
            )
        row: dict[str, Any] = {
            "protocol": protocol,
            "mr": mr,
            "pattern": pattern,
            "seeds": seeds,
            "n": len(runs),
        }
        for metric in METRICS:
            values = np.asarray([float(run[metric]) for _, run in runs])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summary.append(row)

    output_json = root / f"summary.{args.variant}.json"
    temporary_json = output_json.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_json.replace(output_json)
    output_csv = root / f"summary.{args.variant}.csv"
    temporary_csv = output_csv.with_suffix(".csv.tmp")
    fieldnames = ["protocol", "mr", "pattern", "n", "seeds"] + [
        f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")
    ]
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({**row, "seeds": ",".join(map(str, row["seeds"]))})
    temporary_csv.replace(output_csv)
    print(f"Wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
