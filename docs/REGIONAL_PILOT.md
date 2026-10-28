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
