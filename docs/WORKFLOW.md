# Execution plan: 19-28 October 2026

Dates use Asia/Kolkata. Target review: Wednesday 28 October. GPU available from Thursday
22 October, about 48 GB VRAM. The plan assumes access through 26 October; confirm hours
and whether storage persists before starting long training jobs.

| Date | Work | Exit evidence |
|---|---|---|
| 19 Oct | Architecture, CLI contracts, mask/fusion math, baseline mixer | Runnable diagnostic CLI and invariant tests |
| 20 Oct | Curate dog/rain, provenance, source-level splits | Audited manifests; data counts and usable minutes |
| 21 Oct | Prepare short scenes, annotations and evaluation forms | Fixed held-out prompts; expected event intervals; frozen experiment list |
| 22 Oct | Set up GPU; record model revision/environment; generate base audio | 5-10s baseline WAVs, peak VRAM, runtime; decide adapter target modules |
| 23 Oct | Implement verified LoRA trainer; pilot first expert | Reloadable adapter; held-out comparison against base; train/validation curves |
| 24 Oct | Train second expert; compare single-expert performance | Dog/rain checkpoints, data provenance and validation report |
| 25 Oct | Integrate torch masked prediction composition | Shared-latent loop; same scheduler; identity and overlap tests on real model |
| 26 Oct | Run fixed-seed ablations and listening evaluation | Complete result table, audio, failures, timing and memory measurements |
| 27 Oct | Freeze code/results; prepare diagrams and review narrative | Reproducible CLI commands and representative blind-selected samples |
| 28 Oct | Review with professor | Honest evidence, limitations and next steps; no last-minute model changes |

## Scope and decision gates

Minimum credible result: runnable baseline, one category adapter evaluated on held-out
clips, and a clearly documented composition experiment. Target result: two experts with
overlap ablations. Stretch: piano, local LLM planning and local narration integration.
Server and frontend follow CLI validation; they are not required for the 28 October review.

- If no real backbone inference succeeds on 22 October, fix environment/model access
  before training. Capture the error, versions and resource use.
- If no adapter can be trained/reloaded by 23 October, present the baseline and integration
  findings; do not label diagnostic tone output as a LoRA result.
- If two experts are ready but fusion suppresses one event, retain that as a measured
  negative result and compare residual fusion, guidance and mask width on validation only.
- If fusion underperforms stem mixing, the extension can still establish when specialization
  helps and when waveform composition is preferable. Do not claim it wins regardless.

## Controlled experiments

Use the same reviewed timeline, narration, final loudness protocol and prompt set.
Suggested pilot: 12 short scenes x 3 fixed seeds. Include dog alone, rain alone,
sequential dog/rain, partial overlap, full overlap, repeated barks, empty intervals,
and category combinations absent from training. Keep these separate from validation.

| ID | Generation | Isolates |
|---|---|---|
| A | Base model, one combined prompt | Monolithic general-model baseline |
| B | Base model, separate cues, waveform mix | Benefit of explicit timeline and mixing |
| C | Category LoRAs, separate cues, waveform mix | Benefit of specialization over B |
| D | Masked conditional base predictions, no LoRA | Benefit of temporal composition alone |
| E | Masked category LoRAs, normalized fusion | Proposed composition compared with C and D |
| F | Masked category LoRAs, residual fusion | Prediction-composition rule |

Also compare hard versus soft masks and mask fade widths on validation. Keep prompts,
duration and generation budget explicit; equal seeds across different algorithms do not
make their trajectories equivalent. Report compute as well as quality. Start sequential
expert passes to fit memory; measure before adding batching.

## Evaluation without overstating results

1. **Event presence and overlap recall:** blind listeners mark which requested events are
   audible, including whether both are audible during overlaps. A global text/audio score
   is secondary and does not prove simultaneous event presence.
2. **Timing:** annotate actual generated-audio onset/end, with uncertainty. Use one-to-one
   class-aware matching to expected events, report onset error, temporal IoU, misses and
   false positives. Unmatched targets receive zero in aggregate recall/IoU. Do not evaluate
   the planner's own timestamps as if they were measured audio timing.
3. **Fidelity and artifacts:** randomize labels for pairwise listening and report number of
   listeners, scenes and ratings. Include pumping, clicks, leakage and category suppression.
4. **Narration:** measure ASR word error against the known transcript on voice-only and
   mixed audio, with consistent recognizer settings; also rate intelligibility by ear.
5. **Signal checks:** report clipping/sample peak now; add LUFS and true peak before
   loudness-matched listening. More spectral flatness, more onsets or larger dynamic range
   do not automatically mean better audio. Silence can inflate dynamic-range measurements.
6. **Efficiency:** peak allocated/reserved GPU memory, wall time, real-time factor,
   adapter size, steps, active experts, training GPU-hours and dataset minutes.

Save per-scene/per-seed raw results; summarize uncertainty by resampling scenes rather
than pretending all ratings from the same scene are independent. Small pilot results are
feasibility evidence. Dataset-level distribution metrics require enough independent samples;
avoid presenting a tiny-sample FAD as conclusive. Never force the base model to fail for a demo.

## Dataset JSONL contract

Each line is an object like:

```json
{"id":"dog_0001","path":"clips/dog_0001.wav","source_id":"recording_004","category":"dog","caption":"Two short dog barks outdoors","license":"CC0-1.0","split":"train"}
```

Paths are relative to the manifest directory. Keep a provenance table with original URL,
download/permission details and segment boundaries. Dataset-check validates basic WAV
readability, unique IDs, source-level split isolation and exact file duplicates. It does
not establish legal clearance, signal quality, native-format consistency or near-duplicate
absence; inspect these before training. Training data is not bundled with the project.

## Next implementation work after this milestone

1. Pick the local TTS and aligner after checking language needs and hardware compatibility.
   Persist word tokens, timestamps, confidence, transcript offsets and audio hash.
2. Replace keyword drafts with constrained local LLM JSON output; schema-validate and allow
   manual edits. Reject unknown categories or explicitly fall back to the base generator.
3. On GPU, pin the candidate backbone and inspect its actual modules; implement LoRA train,
   save/load, isolated-expert inference, and the torch scheduler composition hook.
4. Verify that zero-strength/disabled adapters reproduce the base model within tolerance;
   verify one-expert/full-mask equivalence and no accidental repeated scheduler stepping.
5. Add measured audio-event evaluation, then run the experiment matrix.
6. Wrap the tested service in a queued server, then build the timeline/audio comparison UI.
