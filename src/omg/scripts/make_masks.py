from __future__ import annotations

import argparse

from ..config import load_config
from ..data.manifest import SPLITS, load_manifest, validate_manifest
from ..data.masks import mask_path, write_mask_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_manifest(cfg.experiment.manifest)
    validate_manifest(
        records,
        dict(cfg.data.feature_dims),
        dict(cfg.data.expected_split_sizes),
        inspect_arrays=False,
    )
    for mr in cfg.experiment.missing_rates:
        for seed in cfg.experiment.seeds:
            for split in SPLITS:
                write_mask_manifest(
                    mask_path(cfg.experiment.mask_root, float(mr), int(seed), split),
                    records,
                    str(cfg.experiment.dataset),
                    split,
                    float(mr),
                    int(seed),
                )


if __name__ == "__main__":
    main()
