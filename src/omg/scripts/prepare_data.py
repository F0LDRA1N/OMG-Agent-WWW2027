from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import load_config
from ..data.manifest import MODALITIES
from ..reproducibility import sha256_file, write_json


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"Expected [time, feature] array, got {result.shape}")
    return result


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).numpy()
    return float(np.asarray(value).reshape(-1)[0])


def _filename(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")[:80]
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'sample'}-{digest}"


def _load_raw_index(path: str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    index_path = Path(path).expanduser().resolve()
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Raw index must map sample IDs to raw modality fields")
    result = {}
    for key, value in payload.items():
        fields = dict(value)
        for modality in ("image", "audio", "video"):
            if fields.get(modality):
                raw_path = Path(str(fields[modality])).expanduser()
                if not raw_path.is_absolute():
                    raw_path = index_path.parent / raw_path
                fields[modality] = str(raw_path.resolve())
        result[str(key)] = fields
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the legacy unified .pt feature bundle to the v5 manifest"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="mosi_unified.pt or mosei_unified.pt")
    parser.add_argument(
        "--raw-index",
        help="JSON mapping sample IDs to text/image/audio/video fields",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = Path(args.input).expanduser().resolve()
    manifest = Path(cfg.experiment.manifest)
    if manifest.exists() and not args.force:
        raise FileExistsError(f"{manifest} exists; pass --force to replace it")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Unified feature input must be a sample-ID mapping")
    raw_index = _load_raw_index(args.raw_index)
    feature_root = manifest.parent / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    split_alias = {"val": "valid", "validation": "valid"}
    expected_dims = {
        key: int(value) for key, value in cfg.data.feature_dims.items()
    }
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(payload):
            item = payload[sample_id]
            split = split_alias.get(str(item["split"]).lower(), str(item["split"]).lower())
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"{sample_id}: unsupported split {split!r}")
            stem = _filename(str(sample_id))
            source_keys = {"text": "text", "vision": "visual", "audio": "audio"}
            paths = {}
            for modality in MODALITIES:
                value = _array(item[source_keys[modality]]["feat"])
                if value.shape[1] != expected_dims[modality]:
                    raise ValueError(
                        f"{sample_id}/{modality}: dimension {value.shape[1]} "
                        f"!= configured {expected_dims[modality]}"
                    )
                destination = feature_root / f"{stem}.{modality}.npy"
                np.save(destination, value, allow_pickle=False)
                paths[modality] = str(destination.relative_to(manifest.parent))
            raw = dict(raw_index.get(str(sample_id), {}))
            raw.setdefault("text", str(item.get("annotation", "")))
            record = {
                "id": str(sample_id),
                "split": split,
                "label": _scalar(item["label_reg"]),
                "features": paths,
                "raw": raw,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(manifest)
    write_json(
        manifest.parent / "provenance.json",
        {
            "dataset": str(cfg.experiment.dataset),
            "source": str(source),
            "source_sha256": sha256_file(source),
            "raw_index": (
                str(Path(args.raw_index).expanduser().resolve())
                if args.raw_index
                else None
            ),
            "raw_index_sha256": sha256_file(args.raw_index) if args.raw_index else None,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "converter": "omg-prepare-data/v5",
        },
    )


if __name__ == "__main__":
    main()
