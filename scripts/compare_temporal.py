"""Paired base/full/temporal dog-adapter pilot; no waveform timing edits."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from bgm.audio import write_wav
from bgm.backends import StableAudioBackend
from bgm.fusion import temporal_masks
from bgm.lora import set_strength
from bgm.temporal import masked_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/stable-audio-open-1.0"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.json"
    status.write_text(json.dumps({"state": "running"}))
    started = time.perf_counter()
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Use one allocated GPU")
        import diffusers
        if diffusers.__version__ != "0.32.2":
            raise RuntimeError("Temporal integration verified against diffusers 0.32.2 only")
        steps = 2 if args.smoke else 100
        backend = StableAudioBackend(args.model, steps=steps, adapter_path=args.adapter)
        pipe = backend.pipe
        if type(pipe.scheduler).__name__ != "CosineDPMSolverMultistepScheduler" or pipe.scheduler.config.prediction_type != "v_prediction":
            raise RuntimeError("Expected verified v-prediction scheduler")
        sr = pipe.vae.sampling_rate
        frames = pipe.transformer.config.sample_size
        spacing = pipe.vae.hop_length / sr
        cue = {"start_s": 3.0, "end_s": 6.0, "fade_s": 0.25}
        mask = torch.tensor(temporal_masks([cue], frames, spacing)[0], device="cuda", dtype=torch.float32)
        np.save(args.out / "mask.npy", mask.cpu().numpy())
        prompt = "Steady rain falling with a dog barking, natural field recording, no speech, no music."
        metadata = {"model": backend.adapter_metadata["model"], "adapter": backend.adapter_metadata,
                    "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                    "prompt": prompt, "duration_s": 10, "interval": cue,
                    "frame_seconds": spacing, "latent_frames": frames,
                    "steps": steps, "guidance_scale": 7.0, "smoke_test_only": args.smoke,
                    "torch": torch.__version__, "diffusers": diffusers.__version__,
                    "scheduler_config": dict(pipe.scheduler.config), "samples": [],
                    "limitation": "Mask controls adapter influence, not guaranteed bark boundaries. Base prompt includes barking in every mode."}
        seeds = (42,) if args.smoke else (42, 123, 2026)
        for seed in seeds:
            results = {}
            modes = ("base", "full", "zero-mask", "one-mask") if args.smoke else ("base", "full", "temporal")
            for mode in modes:
                print(f"Generating {mode}-{seed}", flush=True)
                set_strength(pipe.transformer, 0 if mode == "base" else 1)
                active_mask = torch.zeros_like(mask) if mode == "zero-mask" else torch.ones_like(mask) if mode == "one-mask" else mask
                context = masked_adapter(pipe.transformer, active_mask) if mode not in ("base", "full") else nullcontext({})
                torch.manual_seed(seed)
                with torch.inference_mode(), context as stats:
                    result = pipe(prompt=prompt, audio_end_in_s=10, num_inference_steps=steps,
                                  guidance_scale=7.0, generator=torch.Generator("cuda").manual_seed(seed),
                                  output_type="pt").audios[0].float().cpu().numpy().T
                if not np.isfinite(result).all() or result.shape != (10*sr, 2):
                    raise ValueError("Invalid generated audio")
                results[mode] = result
                metadata["samples"].append({"seed": seed, "mode": mode, "prediction_calls": dict(stats)})
            if args.smoke:
                for direct, wrapped in (("base", "zero-mask"), ("full", "one-mask")):
                    error = float(np.max(np.abs(results[direct]-results[wrapped])))
                    metadata[f"{direct}_endpoint_max_error"] = error
                    if error > 1e-4:
                        raise RuntimeError(f"{direct} endpoint parity failed: {error}")
            # One common attenuation for this seed; do not independently boost files.
            peak = max(float(np.max(np.abs(x))) for x in results.values())
            gain = min(1.0, 10**(-1/20)/max(peak, 1e-12))
            for mode, result in results.items():
                path = args.out / f"{mode}-{seed}.wav"
                write_wav(path, result*gain, sr)
                entry = next(x for x in metadata["samples"] if x["seed"] == seed and x["mode"] == mode)
                entry.update(file=path.name, common_gain=gain, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        metadata["elapsed_s"] = time.perf_counter()-started
        (args.out / "experiment.json").write_text(json.dumps(metadata, indent=2, default=str))
        status.write_text(json.dumps({"state": "completed"}))
        print("TEMPORAL SMOKE PASSED" if args.smoke else "TEMPORAL COMPARISON COMPLETE", flush=True)
    except Exception as exc:
        status.write_text(json.dumps({"state": "failed", "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
