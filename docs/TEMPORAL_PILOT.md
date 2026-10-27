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
