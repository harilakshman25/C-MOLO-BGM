# Prompt comparison and first dataset

The user reports that baseline job 29955 contains a clear dog bark with possible ongoing
bird chirping. This is a subjective observation, not an independently measured result.
The original prompt requests an outdoor field recording, so birds may be contextual.

## Six-sample prompt study

Run the original outdoor prompt and an isolated-bark prompt with seeds 42, 123 and 2026.
Keep cue ID dog_01, five-second duration, model revision, 100 steps, scheduler and all mix
settings identical. The actual generator seed is derived from the seed and cue ID, as in
the original run. This is a comparison of two prompt formulations; it does not isolate
the causal effect of one word, and negation in positive prompts is not guaranteed to work.
No separate negative prompt is introduced in this comparison.

After pushing changes, pull on gpu73. Then on cluster100:

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-gpu
mkdir -p logs
sbatch --partition=gpupart_v100 jobs/compare_prompts.sbatch "$CONDA_PREFIX/bin/python"
```

Look for COMPARISON COMPLETE in logs/cmolo-prompts-JOBID.out. Download the six WAVs in
outputs/prompts-JOBID/listen/ and rate them using ratings.csv. Use a comfortable fixed
playback volume. Rate bark clarity 1 (unclear) through 5 (very clear); mark birds yes/no/unsure,
other unwanted sounds, and distortion. Filenames conceal the prompt label; experiment.json
contains the key. Listen before opening the key. This is informal pilot listening, not a
formal blinded study or a loudness-normalized perceptual benchmark. One listener and three
seeds are not enough for a general superiority claim. Preserve failures and all seeds.

The model loads once. Each sample retains its timeline, WAVs and report; experiment.json
records common model/configuration details and peak GPU memory. An optional --diagnostic
flag exists for CPU plumbing tests; its output is not part of the research comparison.

## Public dataset: ESC-50 dog category

[Official ESC-50 repository](https://github.com/karolpiczak/ESC-50) provides 40 five-second
clips per class. Its five folds keep segments from the same original recording together.
Our pilot uses folds 1-3 for training, fold 4 for validation and fold 5 for test: expected
24/8/8 clips, only two minutes of training audio before exclusions. This is a small adapter
feasibility dataset, not a substitute for the proposed 200-video collection.

On gpu73, no GPU needed:

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-cpu
python scripts/prepare_esc50_dog.py
```

The script downloads only dog WAVs plus metadata, README and the full LICENSE. It resolves
and saves a source commit SHA, hashes recordings, checks source-level split leakage, and
creates data/esc50-dog/review.csv. No new Python dependencies are required. Reruns retain
the dataset revision and existing review sheet. Network access to GitHub is required.

Review the training and validation clips for intelligible barking, birds, speech, music,
noise and clipping. Mark keep/reject and correct captions; do not assume a class label
means a clean isolated recording. Do not copy excluded validation/test clips into training.
Reserve test audio for final assessment; no training decisions based on test performance.
The manifest remains marked pending until review is incorporated into a curated manifest.

ESC-50 includes dataset-level and individual source licensing/attribution details in its
LICENSE; these are retained rather than replaced with a guessed per-clip license. Consult
the bundled terms when sharing recordings or derivative datasets. No audio is committed to Git.

Also check potential overlap with the base model's Freesound training corpus before
claiming unseen-audio generalization. Source-separated local folds prevent our own split
leakage but cannot prove the base model never saw a recording.

Next gate: inspect the prompt ratings and data review, then implement one category LoRA
using the verified v-prediction scheduler. If cleaner prompting already fixes birds, report
that result; do not force a LoRA-improvement claim. Training is not yet implemented here.
