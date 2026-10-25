# Start here — 25 October

We have a working audio-pipeline scaffold, not a trained C-MoLo model. Today's concrete
milestone is to run it on gpu73, pass the tests, and produce a 10-second diagnostic WAV.
Then we establish an allocated GPU environment and generate real baseline audio.

## What exists

The current flow is: story -> draft event list -> validation -> sound inputs -> timed mix
-> WAV and report. The separate fusion module tests how expert predictions would combine;
it is not yet connected to a neural audio model.

| File/directory | Purpose |
|---|---|
| `README.md` | General project instructions and commands. |
| `pyproject.toml` | Package name, dependencies and `bgm` command installation. |
| `environment.yml` | New CPU Conda environment; no PyTorch or model downloads. |
| `.gitignore` | Keeps generated audio, datasets, weights and logs out of Git. |
| `.gitattributes` | Keeps Linux scripts compatible when edited on Windows. |
| `bgm/` | Python implementation, detailed below. |
| `examples/story.txt` | Small story to test the workflow. |
| `examples/timeline.json` | Reviewed example: rain, dog and piano with timing/volume settings. |
| `tests/test_core.py` | 16 automated checks for timing, mixing, fusion and data checks. |
| `docs/ARCHITECTURE.md` | Full technical proposal; read later. |
| `docs/WORKFLOW.md` | Original schedule and experiment design; early dates are now outdated. |
| `docs/VALIDATION.md` | What was actually tested locally on 19 October. |
| `docs/START_HERE.md` | This short onboarding guide and updated priorities. |
| `scripts/smoke_cpu.py` | Runs tests and creates a fresh diagnostic demo on CPU. |
| `jobs/gpu_probe.sbatch` | Small Slurm allocation/inventory check; no model training. |
| `outputs/` | Generated timelines, WAVs, masks and reports; created by runs, ignored by Git. |
| `data/`, `checkpoints/`, `logs/` | Future local datasets, model weights and job logs; ignored by Git. |

Inside `bgm/`:

| File | Purpose |
|---|---|
| `__init__.py` | Package version. |
| `__main__.py` | Allows `python -m bgm`. |
| `cli.py` | Reads commands such as `plan`, `validate`, `render`, `masks`, `dataset-check`. |
| `timeline.py` | Validates cue timings; drafts simple keyword-based cues. No LLM yet. |
| `backends.py` | Test-tone generator and optional, locally untested Stable Audio inference adapter. |
| `audio.py` | Reads/writes WAVs; places sounds; fades; lowers backgrounds during narration. |
| `fusion.py` | Numerical temporal masks and expert-prediction fusion. No trained LoRAs yet. |
| `service.py` | Runs generation/mixing and saves artifacts; later reused by the server. |
| `dataset.py` | Checks dataset manifests for missing files and train/test source leakage. |

A completed demo folder contains `mix.wav` (final audio), `stems/` (individual inputs),
`timeline.json` (settings), `report.json` (statistics/hashes), and `status.json` (run state).
Test tones/noise only verify plumbing. They do not demonstrate realistic dog/rain/piano.

## Your laptop -> GitHub -> server workflow

Keep editing and committing on Windows, then pull on gpu73. Because you report a shared
NFS home, the same checkout may already be visible on cluster100: verify with `pwd -P`
and `git rev-parse HEAD`, rather than cloning copies unnecessarily. NFS shares disk files;
CPU RAM and GPU VRAM still belong to the machine running each process.
Avoid pulling code or modifying a shared environment while a job is running from it.

After committing/pushing these new setup files, run the following **inside your VS Code
remote terminal on gpu73**, not in Windows PowerShell:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda env create -f environment.yml
conda activate cmolo-cpu
python scripts/smoke_cpu.py
```

The script hides GPUs only for its child processes and caps numerical-library CPU threads.
Expected result: 16 tests pass and a new `outputs/cpu-.../demo/mix.wav` is printed.
For listening, download that WAV from VS Code's remote Explorer and play it locally.
No GPU allocation or model download is needed for this milestone.

If `cmolo-cpu` already exists, activate it and run `python -m pip install -e .` instead
of recreating it. The environment is installed in your Conda environment location,
not inside the Git repository. Environment creation needs package download access.
Do not run `sudo` or install dependencies into `(base)`.

For future logins:

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-cpu
python scripts/smoke_cpu.py
```

For a separate CPU-only command, the exact syntax is:

```bash
CUDA_VISIBLE_DEVICES="" python -m bgm validate examples/timeline.json
```

`export CUDA_VISIBLE_DEVICES=""` also works, with **no spaces around `=`**, but persists in
that shell. It hides GPUs; it does not reserve CPU RAM or move a CUDA-only backend to CPU.
Our optional Stable Audio backend currently requires CUDA and will reject a CPU-only run.
The CPU environment deliberately has no inference stack yet.

## Cluster allocation check (later, separate terminal)

From Windows, connect using your working alias `ssh cse-cluster-100`. On the cluster head:

```bash
cd ~/C-MOLO-BGM
pwd -P
git rev-parse HEAD
sinfo -o '%P %a %l %D %G'
mkdir -p logs
sbatch --partition=gpupart_v100 jobs/gpu_probe.sbatch
squeue -u "$USER"
```

The partition name comes from your supplied cluster100 output. This assumes the site
accepts `--gres=gpu:1`; site-specific account/QOS requirements may need adjustment.
Use the returned job number to inspect `logs/cmolo-probe-JOBNUMBER.out` and `.err`.
The job checks allocated-node access, shared checkout visibility, GPU model and driver.
It does not consume a long training slot or prove PyTorch/model compatibility.

Do not carry `CUDA_VISIBLE_DEVICES=""` into GPU jobs, and do not manually select GPU 0
inside Slurm scripts. Slurm supplies device visibility. Use a fresh cluster login shell
for submission; a CPU-only prefix command on gpu73 does not affect that shell.

Before choosing cluster115/116, obtain their actual `sinfo` output and allocation rules.
We only have the screenshot's hardware summary, not their partition names or availability.
gpu73's two L40S cards were busy in your snapshot; do not assume a nominal 48 GB card
means 48 GB is currently available. A GPU stack must also match the allocated architecture:
do not assume one newest PyTorch build supports V100, P100, P4000 and H100 alike.

## Revised target for the 28 October review

- **25 October:** pass CPU checks; verify GPU allocation and environment access.
- **26 October:** generate actual baseline audio on an allocated GPU; curate one category's
  licensed training/validation data and verify a compatible LoRA training path.
- **27 October:** if feasible, train/evaluate one expert on held-out examples. Otherwise
  present real baseline results, working pipeline and measured integration limitations.
- **28 October:** demonstrate only verified results. Two experts and full fused generation
  are stretch goals, not a promise for the remaining three days.

Next information needed after CPU success: allocated GPU model, driver, and job output.
Then create a separate `cmolo-gpu` environment with a verified PyTorch build and pinned
model dependencies. Sharing a filesystem does not guarantee identical drivers/libraries
across nodes. Keep the working CPU environment intact during GPU setup.

References: [Conda environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html),
[Slurm GPU allocation](https://slurm.schedmd.com/gres.html),
[sbatch options](https://slurm.schedmd.com/sbatch.html).
