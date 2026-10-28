"""Regional prompt composition, with and without dog LoRA, and shifted windows."""
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
from bgm.temporal import masked_adapter, event_conditioning

BACKGROUND = "Steady rain falling, natural field recording, no speech, no music."
EVENT = "Steady rain falling with a dog barking, natural field recording, no speech, no music."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("checkpoints/stable-audio-open-1.0"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.json"
    status.write_text(json.dumps({"state": "running"}))
    started = time.perf_counter()
    try:
        import diffusers
        if diffusers.__version__ != "0.32.2" or torch.cuda.device_count() != 1:
            raise RuntimeError("Use pinned diffusers 0.32.2 and one allocated GPU")
        backend = StableAudioBackend(args.model, adapter_path=args.adapter)
        pipe = backend.pipe
        if type(pipe.scheduler).__name__ != "CosineDPMSolverMultistepScheduler" or pipe.scheduler.config.prediction_type != "v_prediction":
            raise RuntimeError("Expected cosine v-prediction scheduler")
        condition = event_conditioning(pipe, EVENT)
        sr, frames = pipe.vae.sampling_rate, pipe.transformer.config.sample_size
        spacing = pipe.vae.hop_length/sr
        masks = {}
        windows = {"early": (1., 4.), "late": (6., 9.)}
        for name, (start, end) in windows.items():
            values = temporal_masks([dict(start_s=start, end_s=end, fade_s=0.25)], frames, spacing)[0]
            masks[name] = torch.tensor(values, device="cuda", dtype=torch.float32)
            np.save(args.out/f"mask-{name}.npy", values)
        report = dict(model=backend.adapter_metadata["model"], adapter=backend.adapter_metadata,
                      adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                      background_prompt=BACKGROUND, event_prompt=EVENT, duration_s=10,
                      windows=windows, fade_s=0.25, frame_seconds=spacing, latent_frames=frames,
                      steps=100, guidance_scale=7.0, negative_prompt=None,
                      torch=torch.__version__, diffusers=diffusers.__version__,
                      scheduler_config=dict(pipe.scheduler.config), samples=[], smoke_checks={},
                      limitation="Regional predictions do not guarantee event timing. All modes use full ten-second duration conditioning.")

        def generate(seed, steps, prompt, strength, mask=None):
            set_strength(pipe.transformer, strength)
            context = nullcontext({}) if mask is None else masked_adapter(
                pipe.transformer, mask, event_condition=condition, event_strength=strength)
            torch.manual_seed(seed)
            with torch.inference_mode(), context as stats:
                sound = pipe(prompt=prompt, audio_end_in_s=10, guidance_scale=7.0,
                             num_inference_steps=steps, output_type="pt",
                             generator=torch.Generator("cuda").manual_seed(seed)).audios[0].float().cpu().numpy().T
            if sound.shape != (10*sr, 2) or not np.isfinite(sound).all():
                raise ValueError("Invalid generated audio")
            if mask is not None and stats != dict(steps=steps, base_passes=steps, expert_passes=steps):
                raise RuntimeError("Unexpected denoiser call counts")
            return sound, dict(stats)

        # Endpoint parity validates hand-built event embeddings against the pipeline.
        background, _ = generate(42, 2, BACKGROUND, 0)
        for strength in (0, 1):
            for endpoint in (0, 1):
                print(f"Checking regional endpoint {endpoint}, adapter {strength}", flush=True)
                expected = background if endpoint == 0 else generate(42, 2, EVENT, strength)[0]
                actual, _ = generate(42, 2, BACKGROUND, strength,
                                     torch.full_like(masks["early"], endpoint))
                error = float(np.max(np.abs(expected-actual)))
                report["smoke_checks"][f"endpoint-{endpoint}-adapter-{strength}"] = error
                if error > 1e-4:
                    raise RuntimeError(f"Regional endpoint parity failed: {error}")
        (args.out/"smoke-checks.json").write_text(json.dumps(report["smoke_checks"], indent=2))
        print("REGIONAL SMOKE PASSED", flush=True)
        for seed in (42, 123, 2026):
            results = {}
            cases = [("background", BACKGROUND, 0, None), ("global-base", EVENT, 0, None),
                     ("global-lora", EVENT, 1, None)]
            cases += [(f"{name}-{label}", BACKGROUND, strength, mask)
                      for name, mask in masks.items() for label, strength in (("base", 0), ("lora", 1))]
            for name, prompt, strength, mask in cases:
                print(f"Generating {name}-{seed}", flush=True)
                results[name], stats = generate(seed, 100, prompt, strength, mask)
                report["samples"].append(dict(seed=seed, mode=name, prediction_calls=stats))
            peak = max(float(np.max(np.abs(x))) for x in results.values())
            gain = min(1., 10**(-1/20)/max(peak, 1e-12))
            for name, sound in results.items():
                path = args.out/f"{name}-{seed}.wav"
                write_wav(path, sound*gain, sr)
                entry = next(x for x in report["samples"] if x["seed"] == seed and x["mode"] == name)
                entry.update(file=path.name, common_gain=gain, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (args.out/"experiment.json").write_text(json.dumps(report, indent=2, default=str))
        report["elapsed_s"] = time.perf_counter()-started
        (args.out/"experiment.json").write_text(json.dumps(report, indent=2, default=str))
        status.write_text(json.dumps({"state": "completed"}))
        print("REGIONAL COMPARISON COMPLETE", flush=True)
    except Exception as exc:
        status.write_text(json.dumps({"state": "failed", "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
