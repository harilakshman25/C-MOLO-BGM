# Regional conditioning experiment — 28 October 2026

The previous adapter-only masks did not establish bark timing control. This follow-up
changes the event branch's text conditioning, while keeping the background branch
rain-only. Both branches see the same latent, diffusion timestep, rotary positions and
full ten-second duration. Their v predictions are blended before CFG; the pipeline
still performs a single scheduler update. This remains an experimental composition
method, not a guarantee of sound presence, timing or background preservation.

The background branch always disables the adapter. The event branch is run either with
the adapter disabled (regional-base control) or enabled (regional-lora). Event conditioning
includes the pinned pipeline's text and start/end duration tokens and zero unconditional
CFG cross-attention branch. Global duration conditioning is unchanged. No negative prompt
is used. No waveform cuts, gates or repositioning are applied.

Two equal-length intervals, early 1–4 seconds and late 6–9 seconds, use 0.25-second soft
edges. Masks use actual codec hop length / sample rate and are zero across padding.
For each seed (42, 123, 2026) there are seven outputs:

| Prefix | Condition |
|---|---|
| background | Rain-only base reference |
| global-base | Rain + dog prompt throughout, adapter off |
| global-lora | Rain + dog prompt throughout, adapter on |
| early-base | Two-prompt blend, 1–4 seconds, adapter off |
| early-lora | Same blend with dog adapter in event branch |
| late-base | Two-prompt blend, 6–9 seconds, adapter off |
| late-lora | Same blend with dog adapter in event branch |

All seven outputs for a seed share one attenuation factor for sample-peak headroom.
They are not individually loudness-normalized. Report whether a dog is audible, when
it barks, rain continuity and artifacts; do not treat higher volume as timing success.
Compare early/late for the same seed first, and base/lora for the same window second.
Do not select only the best seed when reporting results.

## Server command

After committing and pushing local changes:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda activate cmolo-gpu
sbatch --partition=gpupart_v100 jobs/compare_regional.sbatch "$CONDA_PREFIX/bin/python" outputs/lora-pilot-29967/best-adapter.pt
```

The job loads the existing model and adapter once, then checks zero-mask/background
and full-mask/event endpoints with adapter off and on (two inference steps each,
maximum waveform error <= 1e-4). Failure stops generation. Successful checks are saved
to smoke-checks.json; these are not listening-quality examples. Then it generates
21 ten-second clips at 100 steps. No additional training or dependencies are needed.

Logs: logs/cmolo-regional-JOBID.out and .err. Markers: REGIONAL SMOKE PASSED followed
by REGIONAL COMPARISON COMPLETE. Copy outputs/regional-JOBID for review. It contains
WAVs, both masks, status.json, smoke-checks.json and experiment.json with hashes,
conditions and pass counts. Local numerical tests do not replace server endpoint checks
or listening assessment. The model instance must not serve concurrent generations
while its temporary forward wrapper is installed.

## Server verification — job 30101, 28 October

The copied run completed all 21 samples in 461.48 seconds including model loading
and endpoint checks. All four endpoint waveform errors were exactly 0.0 (zero/full
mask, event adapter off/on). Local verification confirmed all 21 WAV hashes, ten-second
44.1 kHz stereo PCM16 format, both masks against the expected codec-frame geometry,
and 100 background plus 100 event passes for each regional sample. Each seed's seven
outputs share one gain factor; no full-scale PCM samples were found. These signal
checks do not establish absence of audible distortion or correct bark placement.
Details are in outputs/regional-30101/verification.json. Listening assessment is pending;
the assistant inspected metadata and signal data, not acoustic events by listening.

## User listening result — 28 October

The user listened to all samples and reported audible dog barking only in
global-base-2026.wav and global-lora-2026.wav, around seconds 2, 3 and 6. All other
files contained only rain. Seed 123 was very quiet. These are listener observations;
no new distortion or background-quality ratings were supplied for this run.

Thus all 12 regional outputs (two windows x two adapter settings x three seeds)
lacked audible barking. Neither regional prompting alone nor regional prompting with
the dog adapter demonstrated event presence or timing control in this pilot. The two
global seed-2026 files retained the same approximate bark times. Inaudible barking is
not evidence of successful exclusion outside the requested window: the required event
was also absent inside it. Quiet outputs may affect audibility, but do not justify
claiming hidden successful events or a proven cause for failure.

The implementation checks passed, but the acoustic hypothesis was unsupported in this
small experiment. No general impossibility claim follows. Regional prediction blending
may suppress the desired event; its cause has not been established. Do not claim that
LoRA guarantees higher fidelity or that latent fusion guarantees temporal placement.

Recommended next deliverable: a clearly labeled separate-stem baseline, generating
rain and an isolated dog clip independently and placing the dog clip within the target
interval using the existing timeline mixer. This offers deterministic waveform placement,
but event presence still requires checking the generated dog clip. It is a baseline/demo,
not evidence that latent C-MoLo works or a novel contribution over the senior's mixing
pipeline. Preserve these failed composition runs for honest comparison and future work.
