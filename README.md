# BackgroundMellow extension: C-MoLo workbench

**Getting started on gpu73 / the cluster:** read [START_HERE](docs/START_HERE.md)
for the short file guide, CPU Conda setup, and revised 25-28 October priorities.

**Current stage (28 October):** base generation and dog LoRA training/inference passed.
Temporal and regional composition ran correctly but did not establish bark timing control.
The next deliverable is the [narrated forest-cabin demo](docs/STORY_DEMO.md), combining
local CPU TTS, ten sound categories and an inspectable separate-stem mix. Its first
server run is pending. Progress report and presentation are in `deliverables/`.

CLI-first research scaffold for category audio LoRAs and temporal diffusion composition.
Start with [architecture](docs/ARCHITECTURE.md) and the [19-28 October workflow](docs/WORKFLOW.md).

## Implemented now

- Validated JSON timelines with overlapping events; a small keyword draft planner.
- Codec-frame-aware soft masks and numerical normalized/residual fusion functions.
- A tested reference denoising loop with one scheduler update per iteration.
- Stereo WAV mixing, fades, narration-driven background ducking and sample-peak headroom.
- Deterministic diagnostic audio, external WAV stems, and an optional Stable Audio baseline adapter.
- Run artifacts, failure status, seeds and WAV hashes; dataset split-leakage checks.

**New, awaiting full-model GPU verification:** single-expert temporal prediction blending,
with endpoint parity checks before the nine-sample listening experiment. All 24 local
tests pass; server parity and acoustic timing remain unverified.

**Not implemented yet:** local LLM planner, TTS/forced aligner, multi-expert model integration,
formal perceptual evaluation, server or frontend. The base Stable Audio inference path
has passed on the V100. Diagnostic tones are engineering fixtures, not dog,
rain or piano synthesis and not evidence of C-MoLo quality.

## Quick start

Use Python 3.10+ for core functionality; use a separate Python 3.11 environment for the
candidate GPU stack. Run from this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m bgm validate examples/timeline.json
.\.venv\Scripts\python -m bgm plan examples/story.txt --duration 10 --out outputs/draft.json
.\.venv\Scripts\python -m bgm render examples/timeline.json --backend diagnostic --out outputs/demo
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Output folders must be new; use a different run name when rerunning. The checked-in example
uses manually authored timings. Keyword drafts use estimated word-rate timing and require
review. They cannot understand negation or arbitrary sound categories.

This machine also has NumPy in the Codex bundled Python; the initial verification used:

```powershell
$bgmPython = 'C:\Users\harilakshman\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $bgmPython -m bgm render examples/timeline.json --backend diagnostic --out outputs/my-demo
```

### Use actual audio files

Provide one 16-bit PCM WAV per cue named with its ID (e.g. dog_01.wav). All WAVs must match
the timeline sample rate and have one or two channels. Each stem must span at least the cue
duration. Short files fail rather than being silently looped; long files are truncated by
the mixer. Narration must fit the timeline. Start with pre-aligned, reviewed timing.

```powershell
python -m bgm render examples/timeline.json --backend files --stems data/stems --narration data/narration.wav --out outputs/real-stems-01
python -m bgm dataset-check data/manifest.jsonl
```

### Export latent masks

Below, 0.02 seconds is an illustrative frame interval, **not a verified model stride**.
Use the actual codec stride/sample-rate and padded frame count for model integration.

```powershell
python -m bgm masks examples/timeline.json --frames 500 --frame-seconds 0.02 --out outputs/masks.npz
```

### Optional real model baseline (GPU stage)

Install a CUDA-compatible PyTorch build for the server, then `python -m pip install -e ".[inference]"`.
Accept the model's license as appropriate and download a pinned Stable Audio Open 1.0
Diffusers checkpoint separately. The adapter only loads local files. Change the timeline
sample_rate to 44100 and keep each cue <=47 seconds. For example:

```powershell
python -m bgm render data/timeline-44100.json --backend stable-audio --model checkpoints/stable-audio-open-1.0 --steps 100 --out outputs/base-01
```

`--offload` uses model CPU offload. Memory fit is unverified on the local 6 GB GPU; the
V100 32 GB is the verified baseline environment. Training is exposed separately through
scripts/train_dog_lora.py; real-model fusion is not implemented. Use the pinned
requirements-gpu-v100.txt for the working server environment.

## Files and reuse

`bgm/service.py` is the transport-independent run entry point. Keep GPU/model work behind
backends; a future server worker calls this service. CLI argument handling is in cli.py,
timeline validation in timeline.py, numerical experiments in fusion.py, and mixing in audio.py.

Each completed run contains mix.wav, stems/, timeline.json, report.json, status.json and
optional narration.wav. report.json's backend label distinguishes diagnostic output from
model output. Current metrics are signal diagnostics, not research quality scores.
