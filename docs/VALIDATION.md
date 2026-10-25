# Initial validation: 19 October 2026

Executed locally with bundled Python 3.12.14 and NumPy 2.3.5. The ordinary system Python
is 3.13.5 and does not currently have NumPy/PyTorch installed. No environment packages,
model weights, training data or GPU dependencies were downloaded during this milestone.

## Results

- `python -m unittest discover -s tests -v`: **16 tests passed**.
- `python -m bgm validate examples/timeline.json`: valid, 3 cues / 10 seconds.
- `python -m bgm plan examples/story.txt --duration 10 --out outputs/draft.json`: completed.
- `python -m bgm render examples/timeline.json --backend diagnostic --out outputs/demo`: completed.
- `python -m bgm masks examples/timeline.json --frames 500 --frame-seconds 0.02 --out outputs/masks.npz`: completed.

Commands above used the full bundled Python path documented in README.md.

Diagnostic mix: 240,000 stereo samples, 24 kHz, 16-bit PCM, exactly 10 seconds.
SHA-256: f440490c59b545f3a31691b8b2b2f6abfbe8f5480edd1dc45455ed7c09ca4c73.
The render contains no narration. Dog/piano cues are tones and the rain cue is noise;
none are neural synthesis. Signal checks are not perceptual quality evidence.

Tests cover valid overlaps, invalid intervals and unsafe IDs, explicit estimated planner
timing, uncovered-region fallback, normalized overlap, identical-expert invariance,
empty expert bank, padded latent masks, soft boundaries, shape rejection, shared-latent
evaluation and one scheduler step per iteration, sample placement, peak headroom,
narration-driven ducking, short-stem/long-narration rejection, WAV roundtrip, deterministic
hashes, refusal to overwrite runs, failure status, and source/hash dataset leakage.

## Still unverified

- Optional Stable Audio adapter, dependency compatibility, GPU memory and runtime.
- Real model mask semantics, scheduler prediction conversions, LoRA target modules and training.
- Local TTS/LLM/alignment integrations (not implemented).
- Audio quality, actual event timing, overlap coverage and narration intelligibility.
- Any performance improvement over the senior's framework or a properly mixed baseline.

The local hardware query reports an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM.
The planned 48 GB GPU is user-provided future access from 22 October; it was not inspected.
