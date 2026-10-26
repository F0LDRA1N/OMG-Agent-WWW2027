# OMG-Agent

Source code for the supplied OMG-Agent v5 implementation, organized around
the Planner--Retriever--Executor graph. Dataset files, pretrained model weights,
trained checkpoints, and experiment outputs are not included in this archive.

## What is fixed by the reproducibility contract

- Official CMU-MOSI/MOSEI train, validation, and test partitions are read from
  a JSONL manifest and checked against the reported split sizes.
- With probability `MR`, a sample receives one of the six nonempty incomplete
  availability patterns uniformly; otherwise it remains fully observed.
  `MR=0.0` therefore always means all modalities are observed.
- Masks are deterministic functions of dataset, split, MR, seed, and sample
  ID. Five runs use seeds `1234..1238`.
- Random-MR results use a separately trained checkpoint at every MR. The seven
  fixed-availability columns all use the checkpoint trained at `MR=0.7`.
- Every target memory has exactly `N_train` members. Its membership is
  training-only; validation and test samples cannot become neighbors.
- Observed modalities bypass reconstruction. Multiple missing targets are
  reconstructed independently, so one generated target never conditions
  another target.
- Missing-target ground truth is available only to training reconstruction
  losses. Evaluation zeroes missing features before collation.
- Validation MAE selects the checkpoint. No test metric participates in model
  selection.

The YAML loader rejects changes to paper-critical dimensions and
hyperparameters. See `IMPLEMENTATION_NOTES.md` for how the implementation resolves
contradictory statements that remain in the manuscript sources.

## Environment

Use a CUDA environment compatible with PyTorch 2.6:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m nltk.downloader wordnet
```

Qwen2.5-Omni also requires `ffmpeg`. FlashAttention-2 is optional.

## Data

Each line of `manifest.jsonl` describes one utterance:

```json
{
  "id": "video_segment_id",
  "split": "train",
  "label": 1.5,
  "features": {
    "text": "features/video_segment_id.text.npy",
    "vision": "features/video_segment_id.vision.npy",
    "audio": "features/video_segment_id.audio.npy"
  },
  "raw": {
    "text": "transcript",
    "image": "raw/video_segment_id.jpg",
    "audio": "raw/video_segment_id.wav",
    "video": "raw/video_segment_id.mp4"
  }
}
```

Feature arrays are `[sequence_length, feature_dim]`, with dimensions
768/35/74 for text/vision/audio. Paths are relative to the manifest. A separate
audio file is required because using audio from a video in an audio-only
availability pattern could expose the missing visual stream.

To convert the legacy unified feature bundle:

```bash
omg-prepare-data \
  --config configs/mosi.yaml \
  --input /path/to/mosi_unified.pt \
  --raw-index /path/to/mosi_raw_index.json
```

The raw index maps sample IDs to any of `text`, `image`, `audio`, and `video`.
The converter writes feature arrays, the manifest, and `provenance.json` with
source hashes.

Validate the converted dataset before generating any artifact:

```bash
omg-validate-data --config configs/mosi.yaml
```

## Plans and masks

Generate all masks and the deterministic training plan cache:

```bash
omg-make-masks --config configs/mosi.yaml
omg-precompute-plans --config configs/mosi.yaml
omg-validate-data --config configs/mosi.yaml --plans
```

The cache is keyed by exact sample ID and availability pattern. It is never
keyed by speaker. Cache metadata binds it to the manifest hash, Qwen model ID,
format version, and generation seed.

The cache stores five initial candidates and a deterministic second-round
candidate bank needed during joint training. Evaluation offers three planner
modes:

- Default: fully cached candidates; measures the paper's
  retrieval-and-execution path with planning excluded.
- `--live-reflection`: reuse cached first-round candidates, then use the
  trained verifier's gradient attribution to generate the focused second round.
- `--online-planner`: generate both rounds from raw observed modalities. This
  is the faithful end-to-end verify--reflect--replan path and includes Planner
  runtime.

## Train and evaluate

Run the random-missing sweep:

```bash
for mr in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  for seed in 1234 1235 1236 1237 1238; do
    omg-train --config configs/mosi.yaml --mr "$mr" --seed "$seed"
    omg-evaluate --config configs/mosi.yaml --mr "$mr" --seed "$seed" \
      --protocol random --online-planner
  done
done
```

Evaluate all seven fixed-availability patterns only from the `MR=0.7`
checkpoints:

```bash
for seed in 1234 1235 1236 1237 1238; do
  omg-evaluate --config configs/mosi.yaml --mr 0.7 --seed "$seed" \
    --protocol fixed --online-planner
done
```

Omit `--online-planner` for cached-plan evaluation, or replace it with
`--live-reflection`. Planner placement and precision can be controlled with
`--planner-device-map`, `--planner-dtype`, and `--flash-attention`.

The evaluator refuses a checkpoint whose dataset, MR, seed, manifest hash, or
Planner model ID differs from the requested run.

## Summarize and audit

For online Planner artifacts:

```bash
omg-summarize --run-root runs/mosi --variant online
omg-audit-run --config configs/mosi.yaml --variant online
```

Use `--variant cached` or `--variant live` for the other modes. Summaries report
the mean and population standard deviation over exactly five seeds. The audit
checks all 40 random-MR runs, all five fixed-pattern runs, 100-epoch histories,
exact test membership, mask hashes, all seven fixed patterns, finite metrics,
and every retrieved neighbor's membership in the training split.

## Model correspondence

- Frozen `Qwen/Qwen2.5-Omni-3B` Thinker with 2048-dimensional hidden states.
- Structured Entity/Action/Attribute plan interface projected to 512
  dimensions; five candidates, threshold 0.6, and at most two total rounds.
- Verifier MLP `1024 -> 256 -> 1`, trained with complete-sample positives and
  controlled WordNet/action/attribute corruptions.
- One target-specific training memory per modality; 512-dimensional cosine
  keys, FAISS IVFFlat, Top-K 10, and temperature 0.07.
- Target latent dimensions 128/16/32 for text/vision/audio.
- Four encoder blocks, two bottleneck blocks, and four decoder blocks.
  Semantic cross-attention spans all 11 stages; zero-initialized evidence
  adapters occur in the first three encoder stages.
- Cosine diffusion horizon 100; deterministic 50-step DDIM (`eta=0`) and CFG
  scale 1.5. Semantic and evidence conditions are independently dropped with
  probability 0.1 during training.
- A three-layer Transformer performs sentiment regression over observed and
  reconstructed modality features.
- AdamW, component learning rates, 500-step warm-up, cosine decay, gradient
  clipping at 1.0, and 100 epochs follow the detailed appendix profile.

## Run artifacts

Each `runs/<dataset>/mr_<MR>/seed_<seed>/` directory contains:

- `run.json`, `config.resolved.yaml`, and `environment.json`
- `mask.sha256`
- `best.pt`, `last.pt`, and `history.jsonl`
- `metrics.random[.live|.online].json`
- `predictions.random[.live|.online].jsonl`
- for `MR=0.7`, the corresponding fixed-protocol files

Prediction rows record the selected candidate, planning round, verifier score,
availability pattern, and exact retrieved training IDs. Summary and audit
files are written under `runs/<dataset>/`.
