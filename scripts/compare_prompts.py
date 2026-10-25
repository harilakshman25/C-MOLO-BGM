"""Generate a small paired-seed prompt study; diagnostic mode is for CPU validation."""
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from bgm.backends import DiagnosticBackend, StableAudioBackend, cue_seed
from bgm.service import render

SEEDS = (42, 123, 2026)
ISOLATED = "An isolated dog barking, no other animals, no birds, no speech, no music."


def cases(template):
    for seed in SEEDS:
        for condition in ("outdoor", "isolated"):
            plan = copy.deepcopy(template)
            plan["seed"] = seed
            if condition == "isolated":
                plan["cues"][0]["prompt"] = ISOLATED
            yield condition, seed, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/stable-audio-open-1.0"))
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--adapter", type=Path, help="Trained project-native LoRA checkpoint")
    args = parser.parse_args()
    if args.diagnostic and args.adapter:
        raise ValueError("Cannot evaluate an adapter with diagnostic tones")
    template = json.loads(Path("examples/dog-baseline.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.json"
    status.write_text(json.dumps({"state": "running"}))
    try:
        started = time.perf_counter()
        metadata = {"purpose": "prompt-only pilot, not a trained-LoRA evaluation", "steps": 100,
                    "guidance_scale": 7.0, "negative_prompt": None, "seeds": SEEDS,
                    "diagnostic": args.diagnostic}
        if args.diagnostic:
            backend = DiagnosticBackend()
        else:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Use a single allocated GPU")
            torch.cuda.reset_peak_memory_stats()
            metadata["model"] = json.loads((args.model / "download_manifest.json").read_text())
            print("Loading model once for six samples...", flush=True)
            backend = StableAudioBackend(args.model, steps=100, adapter_path=args.adapter)
            if args.adapter:
                import hashlib
                metadata["purpose"] = "dog-LoRA pilot; compare to unchanged base prompt study"
                metadata["adapter"] = backend.adapter_metadata
                metadata["adapter_sha256"] = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
            metadata.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                            gpu=torch.cuda.get_device_name(0), dtype="float16",
                            scheduler=type(backend.pipe.scheduler).__name__,
                            scheduler_config=dict(backend.pipe.scheduler.config))
        mapping = []
        for condition, seed, plan in cases(template):
            name = f"{condition}-{seed}"
            print(f"Generating {name}", flush=True)
            report = render(plan, args.out / name, backend=backend)
            mapping.append({"condition": condition, "seed": seed,
                            "generator_seed": cue_seed(seed, "dog_01"),
                            "prompt": plan["cues"][0]["prompt"], "run": name,
                            "mix_sha256": report["sha256"]["mix.wav"]})
        # Neutral filenames reduce expectation bias; key remains available for later analysis.
        random.Random(7026).shuffle(mapping)
        listen = args.out / "listen"
        listen.mkdir()
        with (args.out / "ratings.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", "bark_clarity_1_to_5", "birds_yes_no_unsure",
                             "other_unwanted_sounds", "distortion_yes_no_unsure", "notes"])
            for index, item in enumerate(mapping, 1):
                sample = f"sample_{index:02d}.wav"
                shutil.copyfile(args.out / item["run"] / "mix.wav", listen / sample)
                item["sample"] = sample
                writer.writerow([sample, "", "", "", "", ""])
        metadata["samples"] = mapping
        metadata["elapsed_including_load_s"] = time.perf_counter() - started
        if not args.diagnostic:
            metadata["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            metadata["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
        (args.out / "experiment.json").write_text(json.dumps(metadata, indent=2, default=str))
        (args.out / "packages.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        status.write_text(json.dumps({"state": "completed"}))
        print(f"COMPARISON COMPLETE: {listen}\nRatings: {args.out / 'ratings.csv'}", flush=True)
    except Exception as exc:
        status.write_text(json.dumps({"state": "failed", "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
