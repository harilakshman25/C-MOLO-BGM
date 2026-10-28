# Temporal adapter pilot — 27 October 2026

This first experiment isolates temporal adapter influence. It does not yet compose
multiple experts or use different prompts for different time regions.

For each shared noisy latent and timestep, compute base and dog-adapter v predictions.
Blend v = base + mask * (expert - base), before classifier-free guidance. Both CFG
branches use the same mask, so linear CFG commutes with this blend. The original
Diffusers 0.32.2 pipeline performs exactly one scheduler step on the blended prediction.
No audio is cut, muted, repositioned or crossfaded after generation.

The mask covers seconds 3–6 with 0.25-second soft edges in a ten-second scene. Frame
spacing is the actual VAE hop length divided by sample rate. Padded latent frames have
zero mask. Attention and codec decoding can spread effects beyond the selected region.

All conditions share the same prompt: steady rain with a dog barking. Therefore the
base can bark outside the interval even with a zero adapter mask. This experiment
cannot establish guaranteed event placement. A subsequent regional-conditioning
experiment is needed if adapter-only masking provides little temporal control.

## Run

Commit/push local changes, then on the server:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda activate cmolo-gpu
sbatch --partition=gpupart_v100 jobs/compare_temporal.sbatch "$CONDA_PREFIX/bin/python" outputs/lora-pilot-29967/best-adapter.pt
```

The job first runs two-step real-model smoke generations to compare zero-mask against
base and one-mask against full adapter, with maximum waveform error <= 1e-4. These are
implementation checks, not quality samples. Failure stops the job before the pilot.
The same job then generates nine 100-step samples: three modes by seeds 42, 123, 2026.
It loads the model again for the pilot. It requires no new dependencies or training.

Logs: logs/cmolo-temporal-JOBID.out and .err. Completion markers are TEMPORAL SMOKE
PASSED and TEMPORAL COMPARISON COMPLETE. Outputs: outputs/temporal-JOBID. Compare
base-SEED.wav, full-SEED.wav and temporal-SEED.wav for each seed. Listen for barking
before 3 seconds, during 3–6 seconds and afterward; note rain continuity and artifacts.
Temporal masks are saved as mask.npy, and experiment.json records settings, adapter
identity, prediction-pass counts and WAV hashes.

All three outputs for a seed receive one shared attenuation if necessary for headroom.
There is no independent peak normalization or loudness matching. Differences in
loudness are possible and are not themselves evidence of better quality.

The forward wrapper is temporary and restores adapter strengths even after failure.
Do not run concurrent generations on the same model instance. This is a single-expert
CLI research prototype; it is not yet a multi-expert engine or a server implementation.

Local tests cover shared inputs, mask interpolation, exact endpoints, CFG equivalence
and cleanup after failures. Full-size model parity and acoustic outcomes must be
verified by the server job; no local CUDA result is claimed.

## Server result — 28 October, job 30033

The real-model smoke check passed: both base/zero-mask and full/one-mask waveform
maximum absolute errors were 0.0 on the two-step check. All nine 100-step comparison
clips completed. The comparison run took 160.52 seconds including loading. The logs
show approximately 10–11 seconds per base/full generation and 22 seconds per temporal
generation, consistent with two model passes per denoising step.

Local verification of the copied files confirmed all 13 WAV hashes (four smoke, nine
pilot), ten-second 44.1 kHz stereo PCM16 format, and no full-scale PCM samples. Saved
masks match the expected codec-frame mask and are zero outside seconds 3–6, including
padding. Recorded temporal calls are 100 base plus 100 expert passes per sample.
Each seed's modes share the same output gain. verification.json in the copied pilot
folder contains the file checks. Different peaks/loudness remain possible.

These establish implementation execution and endpoint parity, not acoustic event
placement or preserved rain. Listening assessment is pending. Use the nine pilot
files, not the two-step smoke audio, to judge barking before/during/after seconds 3–6
and background continuity. The assistant inspected metadata and PCM data, not sound
events by listening. The logged warnings did not prevent completion.

## User listening assessment — 28 October

| Seed | Base | Full adapter | Temporal adapter |
|---|---|---|---|
| 42 | Rain only | Continuous background dog with rain | Rain only |
| 123 | Very quiet rain only | Similar to base | Similar to base |
| 2026 | Quiet rain and barks around 2, 3, 6 seconds | Better overall sound; possibly quieter barks at the same times | Better overall sound than full; same bark times |

The user reported good rain and no distortion. Times are approximate listening
observations, not annotated event boundaries. No numerical quality ratings were inferred.

Conclusion: this three-seed pilot does not demonstrate requested temporal bark placement.
For 42, full adaptation makes barking audible but temporal adaptation does not. For 123,
none of the conditions produces audible barking. For 2026, temporal adaptation retains
approximately the base bark times, including a bark before the target interval. Background
quality was acceptable to the listener. Differences in level should not be interpreted
as a general perceptual-quality gain.

The next hypothesis is regional semantic conditioning: use rain-only conditioning for
the background prediction and rain-plus-dog conditioning for the event prediction,
then blend the predictions using the same interval mask. Keep the full scene duration
for both passes. Include a masked two-prompt base-model control with adapter disabled,
alongside the same composition with the dog adapter enabled only in the event branch,
to separate prompt composition effects from LoRA effects. Compare shifted intervals
with paired seeds to test whether bark timing follows the requested window. This is
a proposed follow-up, not an implemented or validated result. Attention and decoding
can still spread sound across mask boundaries; exact timing is not guaranteed.
