# Implementation notes for the supplied v5 snapshot

These notes record the implementation choices made for the original v5
snapshot and its original manuscript specification. The executable settings
are defined by the source code and configuration files included here.

## Objective

The main method equation labels the downstream task objective as
cross-entropy, while the task is continuous sentiment regression and the
detailed appendix defines task loss as MSE on the one-step clean estimate.
The code uses MSE.

The executor uses L1 for evidence alignment, matching the main method equation.
An older appendix sentence calls this cosine distance; that sentence is not
used as the executable definition.

## Dimensions

The revised method and parameter accounting specify Qwen hidden width 2048,
shared plan/context width 512, and retrieval-key width 512. An older retriever
paragraph still says 4096/256/256. The code uses 2048/512/512.

Target latent widths are text 128, vision 16, and audio 32. Alignment heads use
width 256.

The text occasionally calls `(A_t, B_t)` a VAE, but neither a posterior
parameterization nor a KL term is defined in the total objective. v5 implements
the explicitly defined deterministic encoder/decoder maps and does not invent
an unreported KL weight.

## Optimizer

The experiment prose mentions a global Adam learning rate of `2e-3`. The
detailed appendix gives AdamW, weight decay `0.01`, and component-specific
rates. The code follows the detailed AdamW table, including betas, epsilon,
500-step warm-up, cosine decay, gradient clipping, and 100 epochs.

## Random missingness

`MR` is implemented as the probability that an example is assigned an
incomplete pattern. Conditional on being incomplete, one of the six nonempty
patterns is sampled uniformly. It is not an independent per-modality drop
probability. This makes `MR=0.7` well-defined while always retaining at least
one observed modality.

Every random-MR point is separately trained and evaluated. Consequently, the
random `MR=0.0` checkpoint is not the same training condition as the main
fixed-availability `T+V+A` control, which comes from the `MR=0.7` checkpoint.

An older preprocessing subsection additionally proposes modality dropout
`p=0.2` and text augmentation. Applying that augmentation would change the
explicit per-sample masks and make each nominal MR point describe a different
effective missingness process. v5 therefore uses only the fixed, hashed mask
protocol stated in the revised experiment section.

## Planner execution modes

Joint training requires repeatable LLM states, so the training cache contains
deterministically generated first-round candidates and a bootstrap second-round
bank. The learned verifier still scores and selects these candidates.

For the exact field-attributed loop described in the method, evaluation must
use `--live-reflection` or `--online-planner`. The latter regenerates both
rounds from the exact observed raw inputs. Cached evaluation is reserved for
the paper's offline-plan retrieval/execution latency path and must not be
reported as cold end-to-end Planner latency.

## Retrieval isolation

Each target modality has one memory with exactly one member per training
sample. Source context uses the two non-target modalities for that target.
Memory membership never changes, while projected keys and target values are
refreshed during optimization. The final FAISS index is refreshed once and
then held fixed during evaluation.

The revised main equation mentions a straight-through estimator for Top-K
indices, whereas the detailed gradient-flow appendix states that FAISS indices
are non-differentiable constants and gradients flow through the selected
queries, keys, and aggregation weights. v5 follows the detailed appendix.

The implementation makes no extra claim that the official MOSI/MOSEI
partitions are speaker-independent. It relies on the manifest partitions,
indexes training IDs only, and does not use speaker identity for cache keys,
retrieval fallback, or reuse.

The main verifier paragraph calls its observation projection frozen, while the
joint-optimization description trains the feature/retrieval interface. v5 uses
one jointly trained feature-context encoder for verifier and retriever so that
the two consume the same `d_c=512` observation representation.

## Zero-initialized evidence adapter

The vector-feature implementation uses a zero-initialized linear analogue of a
1x1 ZeroConv. An older appendix sentence additionally initializes a
multiplicative scalar to zero. Initializing both the projection and its
multiplier to zero would make both gradients zero at initialization, so v5
uses the zero projection alone; its contribution is exactly zero initially and
its weights can learn on the first update.
