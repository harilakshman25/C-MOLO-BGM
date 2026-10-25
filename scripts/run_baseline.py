"""One real-model generation with runtime and peak GPU-memory metadata."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from bgm.backends import StableAudioBackend
from bgm.service import render
from bgm.timeline import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/stable-audio-open-1.0"))
    parser.add_argument("--timeline", type=Path, default=Path("examples/dog-baseline.json"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()
    plan = validate(json.loads(args.timeline.read_text()))
    if not 1 <= args.steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    manifest = json.loads((args.model / "download_manifest.json").read_text())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run inside a single-GPU allocation")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print("Loading local Stable Audio checkpoint...", flush=True)
    backend = StableAudioBackend(args.model, steps=args.steps, offload=args.offload)
    print("Generating five-second baseline (no LoRA)...", flush=True)
    report = render(plan, args.out, backend=backend)
    torch.cuda.synchronize()
    report["inference"] = {
        "model": manifest, "steps": args.steps, "guidance_scale": 7.0,
        "dtype": "float16", "offload": args.offload,
        "scheduler": type(backend.pipe.scheduler).__name__,
        "scheduler_config": dict(backend.pipe.scheduler.config),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_including_model_load_s": time.perf_counter()-started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        "note": "Allocator peaks exclude some driver and non-PyTorch memory."
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.out / "packages.txt").write_text(packages)
    print(json.dumps(report["inference"], indent=2, default=str), flush=True)
    print(f"BASELINE COMPLETE: {args.out / 'mix.wav'}", flush=True)


if __name__ == "__main__":
    main()
