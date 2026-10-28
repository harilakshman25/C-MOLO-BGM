# Story demo: narration, music and effects

This separate-stem CLI baseline is distinct from the unsuccessful latent timing
experiments. It uses a manually authored eight-sentence forest-cabin story, not an
automatic LLM planner. Measured TTS sentence boundaries anchor authored within-sentence
cue offsets; these are not word-level forced alignment. All requested categories are
included: rain, footsteps, piano, wood, door creak, dog, running, forest, trees and wind.

## 1. CPU preparation on gpu73

Commit and push local changes, then run on gpu73 (not the Slurm login node):

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda activate cmolo-gpu
export CUDA_VISIBLE_DEVICES=""
python -m pip install -r requirements-narration.txt
python scripts/download_tts.py
python scripts/prepare_narration.py --out outputs/forest-narration
```

Do not upgrade torch or transformers. MMS English TTS uses the existing Transformers
stack on CPU. The download script pins the resolved snapshot and saves its identity.
MMS TTS has a CC-BY-NC-4.0 model license: https://huggingface.co/facebook/mms-tts-eng.
No hosted inference API or voice cloning is used. The new inference path still needs
its first server check. If the output directory already exists, use a fresh name and
pass it as the third argument to the job below; do not overwrite earlier runs.

The narration folder contains narration.wav, segments.json, timeline.json, tts-report.json
and status.json. Check that narration is intelligible before final presentation. Speech
is resampled to 44.1 kHz with polyphase filtering; 2.5-second gaps leave room for effects.

## 2. Submit sound generation and first mix on master

Use a separate master terminal so the CPU-only CUDA variable is not carried over:

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-gpu
sbatch --partition=gpupart_v100 jobs/story_demo.sbatch "$CONDA_PREFIX/bin/python" outputs/lora-pilot-29967/best-adapter.pt outputs/forest-narration
```

Slurm controls GPU visibility. The job generates 20 candidate WAVs (two per category),
loads the base and adapter once, and enables the adapter only for dog generation.
Candidate 0 is provisionally selected for every category, then the first mix is rendered.
New candidates, including piano, are not assumed successful until heard. Stable Audio
is an initial short piano-bed option; it is not a guarantee of coherent long-form music.

```bash
cat logs/cmolo-story-JOBID.out
tail -60 logs/cmolo-story-JOBID.err
```

Expected markers: SOUND LIBRARY COMPLETE and STORY DEMO COMPLETE. Copy
outputs/forest-narration, outputs/sound-library-JOBID and outputs/story-demo-JOBID locally.

## 3. Listen and remix without a GPU

The main output is outputs/story-demo-JOBID/mix.wav. Listen for narration intelligibility,
all intended sound events, appropriate levels, and loop seams. Candidate presence is
not automatically recognized. To replace a poor candidate, edit the matching filename
in outputs/sound-library-JOBID/selection.json, for example dog-0.wav to dog-1.wav.
Adjust cue gain_db/start_s/end_s in the narration folder's timeline.json if needed.
Then on gpu73:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/render_story.py --narration outputs/forest-narration --library outputs/sound-library-JOBID --out outputs/story-demo-remix-01
```

Only ambience/music loop using crossfades. Effects play once, with zero padding if needed.
This can guarantee waveform placement within the cue, but cannot guarantee that the
model generated the intended sound or its onset at the cue start. Quiet non-silent
sources get capped RMS gain adjustment; near-silent sources fail visibly. Existing
mixing ducks ambience/music during narration and controls sample peaks; it is not
LUFS mastering or a true-peak limiter. All current settings are demo defaults to review.

## Presentation preparation through 3 November

- 28–29 October: get narration and first story mix running; choose candidate sounds.
- 30 October: render a second story by editing the scene JSON; record limitations.
- 31 October: freeze examples and compare base/dog-adapter and modular/latent results.
- 1 November: update report and slides with verified story results; retain negative runs.
- 2 November: rehearse, export copies and keep local audio backups.
- 3 November: present implementation, evidence and remaining research questions.

Server/frontend work follows a stable CLI demo; no web app is needed to demonstrate
the architecture. Report the separate-stem baseline honestly alongside the research
branch, rather than presenting it as a successful latent-fusion model.
