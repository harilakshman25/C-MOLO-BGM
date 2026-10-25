import argparse
import json
from pathlib import Path
import numpy as np
from .timeline import draft, validate
from .fusion import temporal_masks
from .backends import DiagnosticBackend, StableAudioBackend
from .service import render
from .dataset import audit


def main():
    parser = argparse.ArgumentParser(description="CLI-first C-MoLo research workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Draft keyword cues; timing is an estimate")
    p.add_argument("story", type=Path)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    for name in ("validate", "masks", "render"):
        p = sub.add_parser(name)
        p.add_argument("timeline", type=Path)
        if name == "masks":
            p.add_argument("--frames", type=int, required=True)
            p.add_argument("--frame-seconds", type=float, required=True)
            p.add_argument("--out", type=Path, required=True)
        if name == "render":
            p.add_argument("--backend", choices=["diagnostic", "files", "stable-audio"], required=True)
            p.add_argument("--stems", type=Path)
            p.add_argument("--narration", type=Path)
            p.add_argument("--model", type=Path)
            p.add_argument("--steps", type=int, default=100)
            p.add_argument("--offload", action="store_true")
            p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("dataset-check")
    p.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = draft(args.story.read_text(encoding="utf-8"), args.duration, args.seed)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Draft saved to {args.out}; review estimated timing and missing events.")
        elif args.command == "dataset-check":
            print(json.dumps(audit(args.manifest), indent=2))
        else:
            plan = validate(json.loads(args.timeline.read_text(encoding="utf-8")))
            if args.command == "validate":
                print(f"Valid timeline: {len(plan['cues'])} cues, {plan['duration_s']} seconds")
            elif args.command == "masks":
                masks = temporal_masks(plan["cues"], args.frames, args.frame_seconds)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.out, masks=masks, cue_ids=[c["id"] for c in plan["cues"]],
                         frame_seconds=args.frame_seconds)
                print(f"Saved {len(masks)} masks; verify frame spacing against your codec")
            else:
                if args.backend == "files" and not args.stems:
                    raise ValueError("--stems is required for files backend")
                if args.backend == "stable-audio" and (not args.model or not args.model.is_dir()):
                    raise ValueError("--model must point to a downloaded Stable Audio model directory")
                if not 1 <= args.steps <= 1000:
                    raise ValueError("--steps must be between 1 and 1000")
                if args.backend == "stable-audio":
                    if plan["sample_rate"] != 44100 or any(c["end_s"]-c["start_s"] > 47 for c in plan["cues"]):
                        raise ValueError("Stable Audio requires 44100 Hz and cues <=47 seconds")
                backend = (DiagnosticBackend() if args.backend == "diagnostic" else
                           StableAudioBackend(args.model, args.steps, args.offload) if args.backend == "stable-audio" else None)
                report = render(plan, args.out, backend=backend,
                                stems_dir=args.stems if args.backend == "files" else None,
                                narration_path=args.narration)
                print(json.dumps(report, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
