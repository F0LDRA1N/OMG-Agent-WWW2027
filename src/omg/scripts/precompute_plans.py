from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..data.manifest import MODALITIES, load_manifest, validate_manifest
from ..data.masks import parse_pattern
from ..planner.prompts import diagnostic_feedback
from ..planner.qwen import QwenPlannerBackend, controlled_corruptions
from ..reproducibility import sha256_file

PATTERNS = ("111", "100", "010", "001", "110", "101", "011")


def _compact_candidate(
    candidate: Any,
    backend: QwenPlannerBackend,
    round_index: int,
) -> dict[str, Any]:
    return backend.cache_candidate(candidate, round_index)


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _seed_generation(base_seed: int, sample_id: str, pattern: str, round_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{sample_id}|{pattern}|{round_index}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _raw_requirements(raw: dict[str, Any]) -> None:
    if not raw.get("text"):
        raise ValueError("Every record needs raw.text for exact-pattern planning")
    if not (raw.get("image") or raw.get("video")):
        raise ValueError("Every record needs raw.image or raw.video")
    if not raw.get("audio"):
        raise ValueError(
            "Every record needs a separate raw.audio file. Reusing raw.video for "
            "an audio-only pattern could expose the missing visual modality."
        )
    for field in ("image", "video", "audio"):
        if raw.get(field) and not Path(str(raw[field])).is_file():
            raise FileNotFoundError(f"Missing raw.{field} file: {raw[field]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    generation_seed = (
        int(cfg.experiment.plan_generation_seed)
        if args.seed is None
        else args.seed
    )
    if generation_seed != int(cfg.experiment.plan_generation_seed):
        raise ValueError(
            "--seed must match experiment.plan_generation_seed so caches are traceable"
        )
    records = load_manifest(cfg.experiment.manifest)
    validate_manifest(
        records,
        dict(cfg.data.feature_dims),
        dict(cfg.data.expected_split_sizes),
        inspect_arrays=False,
    )
    if args.limit is not None:
        records = records[: args.limit]
    output_path = Path(cfg.experiment.plan_cache)
    if output_path.exists():
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "plans" not in payload:
            raise ValueError(f"Invalid existing plan cache: {output_path}")
        plans = payload["plans"]
    else:
        payload = {
            "metadata": {
                "version": 7,
                "model_id": str(cfg.model.planner.model_id),
                "manifest_sha256": sha256_file(cfg.experiment.manifest),
                "patterns": list(PATTERNS),
                "candidates_per_round": int(cfg.model.planner.candidates),
                "rounds": int(cfg.model.planner.max_rounds),
                "generation_seed": generation_seed,
                "cache_hidden_dtype": "float16",
                "cache_rule": (
                    "Entity/Action/Attribute-pooled Thinker states; "
                    "exact sample and availability pattern"
                ),
            },
            "plans": {},
        }
        plans = payload["plans"]
    if (
        int(payload.get("metadata", {}).get("generation_seed", generation_seed))
        != generation_seed
    ):
        raise ValueError(
            "Existing plan cache was generated with a different --seed"
        )
    if int(payload.get("metadata", {}).get("version", -1)) != 7:
        raise ValueError(
            "Existing plan cache uses an incompatible format; regenerate it"
        )
    expected_metadata = {
        "model_id": str(cfg.model.planner.model_id),
        "manifest_sha256": sha256_file(cfg.experiment.manifest),
        "patterns": list(PATTERNS),
        "candidates_per_round": int(cfg.model.planner.candidates),
        "rounds": int(cfg.model.planner.max_rounds),
    }
    metadata_mismatches = {
        key: (payload["metadata"].get(key), expected)
        for key, expected in expected_metadata.items()
        if payload["metadata"].get(key) != expected
    }
    if metadata_mismatches:
        raise ValueError(
            f"Existing plan cache metadata mismatch: {metadata_mismatches}"
        )
    backend = QwenPlannerBackend(
        str(cfg.model.planner.model_id),
        device_map=args.device_map,
        dtype=args.dtype,
        flash_attention=args.flash_attention,
    )
    completed_since_save = 0
    for record in records:
        if record.sample_id in plans and all(
            pattern in plans[record.sample_id] for pattern in PATTERNS
        ):
            continue
        _raw_requirements(record.raw)
        sample_plans = plans.setdefault(record.sample_id, {})
        supervision: dict[str, Any] | None = None
        for pattern_value in PATTERNS:
            if pattern_value in sample_plans:
                if pattern_value == "111":
                    supervision = {
                        "positive_hidden": sample_plans[pattern_value].get(
                            "positive_hidden", []
                        ),
                        "negative_hidden": sample_plans[pattern_value].get(
                            "negative_hidden", []
                        ),
                    }
                continue
            pattern = parse_pattern(pattern_value)
            available = [
                modality
                for modality, is_available in zip(MODALITIES, pattern)
                if is_available
            ]
            missing = [
                modality
                for modality, is_available in zip(MODALITIES, pattern)
                if not is_available
            ]
            _seed_generation(generation_seed, record.sample_id, pattern_value, 0)
            first = backend.generate_candidates(
                record.raw,
                available,
                missing,
                count=int(cfg.model.planner.candidates),
                temperature=float(cfg.model.planner.temperature),
            )
            first_best = max(
                first,
                key=lambda candidate: candidate.logprob
                - float(cfg.model.planner.schema_penalty) * candidate.schema_penalty,
            )
            candidates = [
                _compact_candidate(candidate, backend, 0) for candidate in first
            ]
            if int(cfg.model.planner.max_rounds) > 1:
                for candidate_index, initial in enumerate(first):
                    feedback = diagnostic_feedback(
                        "plan", initial.text, score=0.0
                    )
                    _seed_generation(
                        generation_seed,
                        record.sample_id,
                        pattern_value,
                        100 + candidate_index,
                    )
                    reflected = backend.generate_candidates(
                        record.raw,
                        available,
                        missing,
                        count=1,
                        temperature=float(cfg.model.planner.temperature),
                        feedback=feedback,
                    )
                    candidates.append(
                        _compact_candidate(reflected[0], backend, 1)
                    )
            if pattern_value == "111" and record.split == "train":
                positive = (
                    backend.encode_plan_text(first_best.text)
                    .mean(dim=0, keepdim=True)
                    .to(torch.float16)
                )
                negatives = [
                    backend.encode_plan_text(text).mean(dim=0, keepdim=True).to(torch.float16)
                    for text in controlled_corruptions(first_best.text)
                ]
                supervision = {
                    "positive_hidden": [positive for _ in negatives],
                    "negative_hidden": negatives,
                }
            elif pattern_value == "111":
                supervision = {
                    "positive_hidden": [],
                    "negative_hidden": [],
                }
            if supervision is None:
                raise RuntimeError("Pattern 111 must be generated before incomplete patterns")
            sample_plans[pattern_value] = {
                "candidates": candidates,
                **supervision,
            }
        completed_since_save += 1
        if completed_since_save >= args.save_every:
            _atomic_save(payload, output_path)
            completed_since_save = 0
    _atomic_save(payload, output_path)


if __name__ == "__main__":
    main()
