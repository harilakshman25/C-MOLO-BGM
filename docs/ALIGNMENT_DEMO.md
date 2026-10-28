# Word-timing refinement and 2 November deadline

## Current action: manual annotation after two failed ASR runs

The small.en retry also stopped safely because barked was recognized as Bart; creak
and piano were also misrecognized. No aligned remix was produced. Stop automatic
recognition retries for this demo. Keep the original narration; recognition errors
alone do not prove the speech is unintelligible to a human.

Run `python scripts/make_timing_page.py` locally, then open
outputs/forest-narration/manual-timing.html in a browser. Listen and mark the end of
each of the six specified words. Export the JSON or paste it into the chat. No times
are prefilled or inferred. The page references the adjacent local narration.wav.

After saving the user's exported JSON as outputs/forest-manual-times.json:

```bash
python scripts/apply_manual_timing.py --marks outputs/forest-manual-times.json --out outputs/forest-manual
python scripts/render_story.py --narration outputs/forest-manual --library outputs/sound-library-30111 --out outputs/story-demo-manual-01
```

These commands can run on CPU with the existing numpy environment. The marks must
match the narration hash, contain all six cue IDs and lie inside their sentence bounds.
Timing is explicitly recorded as human annotation, not model-derived alignment. Input
marks and original outputs are preserved. First listen to the remixed output before
selecting it for presentation. The earlier automatic attempts below are retained as
experiment history, not instructions to retry them again.

## First attempt and retry

The first base.en run failed safely: the footsteps scene had transcript similarity
0.70, below the unchanged 0.75 threshold. Other recognized triggers included Bart for
barked, creek for creak and pea in a for piano. No aligned remix was produced.
Do not weaken matching or substitute these words silently. The cause of recognition
errors is not established; missing end timestamps were also reported.

Retry on gpu73 after pulling the fix:

```bash
conda activate cmolo-gpu
export CUDA_VISIBLE_DEVICES=""
python scripts/download_aligner.py --size small
python scripts/align_narration.py --model checkpoints/whisper-small-en --narration outputs/forest-narration --out outputs/forest-aligned-small-01
```

The larger English checkpoint is a retry, not a guaranteed fix. One second of trailing
silence is added to each recognition input without changing the original narration or
word-time origin. All matching/timestamp checks remain. Only after ALIGNMENT COMPLETE:

```bash
python scripts/render_story.py --narration outputs/forest-aligned-small-01 --library outputs/sound-library-30111 --out outputs/story-demo-aligned-small-01
```

If this retry still fails, preserve the successful original demo and manually annotate
the six trigger times rather than spending repeated runs on uncertain ASR matching.

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
