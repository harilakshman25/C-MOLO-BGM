# Dog LoRA pilot — 26 October 2026

The user reviewed all 40 clips and accepted them, reporting no birds/other anomalies,
some reflections and a few angry barks. This is recorded in DOG_DATA_REVIEW.json; no
repeated CSV review is required. All 24 train / 8 validation / 8 test assignments remain.
The test split is excluded from latent encoding, optimization and checkpoint selection.
Because the user reports listening to all clips, it is not an unseen listening set.

## Run the smoke test first

Push the local code changes, then on gpu73:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
```

On cluster100, using the existing GPU environment (no additional dependencies):

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-gpu
sbatch --partition=gpupart_v100 jobs/train_dog_lora.sbatch "$CONDA_PREFIX/bin/python" smoke
```

Read logs/cmolo-lora-JOBID.out and .err. Expected final marker: LORA SMOKE PASSED.
This job tests adapter math, caches frozen features from train/validation only, verifies
zero-adapter/base equivalence, takes two optimizer steps, and saves/reloads an adapter.
It uses only two validation clips for the smoke check. This is not a quality result.

The first job includes codec feature caching, which may take longer than generation.
Use only one training job at a time while the shared cache is being built. Subsequent
jobs reuse data/esc50-dog/cache-v1.pt only if model/data/review hashes match. The cache
uses trusted local torch files with weights_only loading. No test latents are cached.

## After the smoke test passes

Run a separate, fresh pilot from the base model (not from the two-step smoke weights):

```bash
sbatch --partition=gpupart_v100 jobs/train_dog_lora.sbatch "$CONDA_PREFIX/bin/python" pilot
```

Do not run this until the smoke log has been checked. The pilot runs 200 optimizer steps,
validates at step zero and every 50 steps, and saves checkpoints. Batch size 1 with four
gradient-accumulation steps means 800 training-clip draws, sampled with replacement.
This is intentionally small but can still overfit two minutes of audio. Stop/adjust based
on validation and listening, not test-set tuning. The one-hour job is a bound, not a runtime
prediction. Checkpoint files permit inference but this version does not resume optimizer state.

## What is trained

- One frozen Stable Audio Open 1.0 base revision, same as the working baseline.
- Rank 8, alpha 8 low-rank matrices on self/cross-attention query and value projections.
- Base weights frozen in FP16; trainable adapters/optimizer state FP32; gradient scaling
  and activation checkpointing enabled. No PEFT or full-model training dependencies added.
- Learning rate 1e-4, AdamW, zero weight decay, norm clipping 1.0, 10% conditioning dropout.
- All captions remain generic "A dog barking." Reflections/aggressive tone are not given
  invented per-recording captions. Category variety may remain audible after adaptation.

## Training objective and deliberate pilot assumptions

Inspection used the exact Diffusers 0.32.2 source distribution. Its cosine scheduler uses
sigma_data=1, t=2*atan(sigma)/pi, input scaling 1/sqrt(1+sigma^2), and a negative v-prediction
output coefficient. Therefore with alpha=cos(pi*t/2), beta=sin(pi*t/2):

    noisy = alpha * clean_latent + beta * noise
    target_velocity = alpha * noise - beta * clean_latent

The trainer samples t uniformly inside (0,1); it does not train by stepping the inference
scheduler or treating model outputs as epsilon. The tests reconstruct the clean latent
using the scheduler's actual output preconditioning formula.

This is a compatible adaptation objective, not a reproduction of upstream pretraining:
we cache the FP32 codec posterior mean (not repeated stochastic posterior samples), pad
the waveform to the actual full model length before encoding, and keep the five-second
duration conditioning. Mono recordings are duplicated to two channels; this does not add
real stereo detail. The loss weights active frames by 1 and padded frames by 0.1, normalized
by the summed weights, so long silence does not dominate the clip while still being trained.
These choices must be disclosed and reconsidered if the pilot fails.

Text and duration embeddings and rotary positions follow the verified pipeline. During
caption dropout the full cross-attention condition is zeroed, but global duration is kept,
matching the pipeline's unconditional branch when no negative prompt is used.

Validation uses the same fixed noise draws and t values (0.2, 0.5, 0.8) every evaluation,
without caption dropout. Only the eight validation clips choose the best checkpoint.
If no trained checkpoint beats step zero, best-adapter.pt deliberately remains the
untrained baseline; report that honestly. A loss decrease does not establish audio improvement.

## Artifacts and inference

Each output folder has status.json, config.json, metrics.jsonl, report.json, packages.txt,
best-adapter.pt, last-adapter.pt, and validation-step checkpoints. Checkpoints contain
only adapter weights plus base/data/config metadata, not the entire audio backbone.
No training data or checkpoint is committed to Git.

The prompt-comparison script now accepts --adapter PATH. It loads the same base revision,
validates adapter targets and records the adapter hash. Smoke-test and step-zero adapters
are rejected as quality-evaluation candidates. Use an actual trained pilot checkpoint;
select based on the validation report before producing the same six prompt/seed samples.
The adapter format is project-native, not a standard PEFT format. Real-model training and
inference with these adapters remain unverified until the server smoke and pilot complete.

The local adapter tests verify zero initialization, gradients, frozen base weights,
target selection, serialization/reload and scheduler-compatible velocity mathematics.
They do not establish full-model VRAM fit, stability or acoustic quality.

Local verification on 26 October: 22 tests passed, including a miniature actual
StableAudioDiTModel forward/backward with gradient checkpointing. Executed with project-local
CPU-only torch 2.7.1 and Diffusers 0.32.2 under Python 3.13.5; the server continues using
its existing Python 3.11 CUDA environment. Test libraries were installed only under ignored
outputs/training-runtime; no existing Python environment or model weights were changed.
