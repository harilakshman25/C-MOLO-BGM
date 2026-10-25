# First real audio — 26 October

Server job 29953 passed the FP16 GPU compute test on a V100 32 GB with torch 2.7.1+cu118,
Diffusers 0.32.2 and Transformers 4.49.0. The next milestone is one five-second dog-bark
sample from the unmodified base model. This is not LoRA training or C-MoLo fusion.

## 1. Model access

Sign in to https://huggingface.co/stabilityai/stable-audio-open-1.0 and review/accept its
access conditions if you agree. The model repository is gated. Use the same account for
server login, with a read token permitted to access this gated model. Do not paste the
token into chat, scripts, Git, or shell command arguments.

## 2. Download on gpu73

Push these local changes, then run in the gpu73 VS Code terminal:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda activate cmolo-gpu
huggingface-cli login
python scripts/download_model.py
```

Paste your token only into the login prompt; adding it as a Git credential is not needed.
Downloads require network access and several GB of disk space. This script does not load
the model or use a GPU. Wait for `Download complete` before submitting the job. A 401/403
usually means account/token/model-access permission needs correction, not a CUDA issue.

The script downloads the Diffusers folders and records the resolved model commit SHA.
Reruns reuse the saved revision and cached files. Checkpoints stay outside Git. It does
not auto-upgrade the verified Python environment or download the separate native checkpoint.

## 3. Submit from cluster100

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-gpu
mkdir -p logs
sbatch --partition=gpupart_v100 jobs/baseline.sbatch "$CONDA_PREFIX/bin/python"
squeue -u "$USER"
```

Read the returned job ID's logs:

```bash
cat logs/cmolo-baseline-JOBID.out
tail -n 40 logs/cmolo-baseline-JOBID.err
```

Progress bars may appear in `.err`; that alone does not indicate failure. The success
marker is `BASELINE COMPLETE`. A one-hour allocation is a limit, not a runtime estimate.
Inference is offline in the job; it reads the already-downloaded files from shared NFS.

## 4. Listen and report

The final audio is `outputs/baseline-JOBID/mix.wav`. Download it through VS Code's remote
Explorer and listen locally. Also inspect `report.json` for peak GPU memory, generation
settings, model revision and runtime; `packages.txt` records installed versions.

Tell us whether a dog is recognizable, whether speech/music intrudes, and whether there
are clicks, distortion or silence. Keep an unsuccessful sample too: a fixed-seed baseline
is evidence, not a guaranteed good demo. The next comparisons need multiple prompts/seeds.

This job requests one GPU and 32 GB system RAM; these are separate resources. A short
requested output does not guarantee low model memory use because internal latent padding
can be longer. If CUDA runs out of memory, preserve the logs; the wrapper supports
`--offload` as a subsequent experiment. Do not change the working package versions blindly.

The scripts have been locally syntax-checked; real inference still requires this server run.
