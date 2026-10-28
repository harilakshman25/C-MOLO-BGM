# Word-timing refinement and 2 November deadline

Preserve outputs/story-demo-30111 as the first successful 62.33-second baseline.
The new variant reuses its narration and selected sound library. Local Whisper base.en
estimates word endpoints within each known narration segment. An explicit anchor file
maps six cue IDs to spoken triggers. A cue begins 80 ms after the trigger word ends.
This is ASR-based timing estimation, not forced alignment against a fixed transcript.

Only discrete footsteps, barking, running, wood/door creaks and the first piano entrance
are moved. Rain and other ambience stay in place, including the one-second opening
rain. Stem-internal silence and fades can still delay the audible event beyond cue onset.
We do not claim perfect word synchronization without listening and annotations.

After committing/pushing, run on gpu73 in the existing environment:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda activate cmolo-gpu
export CUDA_VISIBLE_DEVICES=""
python scripts/download_aligner.py
python scripts/align_narration.py --narration outputs/forest-narration --out outputs/forest-aligned
```

Only after ALIGNMENT COMPLETE:

```bash
python scripts/render_story.py --narration outputs/forest-aligned --library outputs/sound-library-30111 --out outputs/story-demo-aligned-01
```

No Slurm job, retraining, audio regeneration or dependency upgrade is needed. Download
needs network access; subsequent inference uses local files. The aligner model identity
is pinned in checkpoints/whisper-base-en/download_manifest.json. If a destination exists,
use a fresh output name rather than overwriting earlier results.

Missing/duplicate triggers, invalid timestamps and substantially mismatched transcripts
cause a visible failure. recognized-words.json remains available for diagnosis, while
the original mix is unchanged. Send the error rather than bypassing the check.

Review alignment-report.json and compare the original and revised mixes for early effects,
narration masking and whether each expected sound is still audible. Changes are provisional
until reviewed. Source: https://huggingface.co/openai/whisper-base.en and Transformers
ASR pipeline word-timestamp support. Existing Transformers 4.49.0 is retained; CPU FP32
and eager attention are selected for this integration. First server run remains pending.

## Revised schedule

- 28–29 October: run the timing refinement and select the better story mix.
- 30 October: second short story and a small, clearly labeled listening comparison.
- 31 October: freeze demonstrations and update the report/slides with measured evidence.
- 1 November: rehearse and save offline WAV, PDF and PowerPoint backups.
- 2 November: presentation/submission deadline. Do not plan required work beyond this date.
