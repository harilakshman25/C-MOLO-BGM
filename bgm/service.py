"""Transport-independent run service: a future worker can call render directly."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from . import __version__
from .audio import mix, read_wav, write_wav
from .backends import cue_seed
from .timeline import validate


def render(plan, output_dir, backend=None, stems_dir=None, narration_path=None):
    validate(plan)
    if (backend is None) == (stems_dir is None):
        raise ValueError("Choose exactly one generated backend or input stem directory")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = output_dir / "status.json"
    status.write_text(json.dumps({"state": "running"}), encoding="utf-8")
    try:
        stems = {}
        for cue in plan["cues"]:
            sound = (backend.generate(cue, plan["sample_rate"], cue_seed(plan["seed"], cue["id"]))
                     if backend else read_wav(Path(stems_dir)/f"{cue['id']}.wav", plan["sample_rate"]))
            stems[cue["id"]] = sound
            write_wav(output_dir/"stems"/f"{cue['id']}.wav", sound, plan["sample_rate"])
        narration = read_wav(narration_path, plan["sample_rate"]) if narration_path else None
        if narration is not None:
            write_wav(output_dir/"narration.wav", narration, plan["sample_rate"])
        output, metrics = mix(plan, stems, narration)
        write_wav(output_dir/"mix.wav", output, plan["sample_rate"])
        (output_dir/"timeline.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        hashes = {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in output_dir.rglob("*.wav")}
        report = {"state": "completed", "version": __version__, "python": platform.python_version(),
                  "numpy": np.__version__, "backend": backend.name if backend else "external-stems",
                  "elapsed_s": time.perf_counter()-started, "seed": plan["seed"],
                  "metrics": metrics, "sha256": hashes}
        (output_dir/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        status.write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        return report
    except Exception as exc:
        status.write_text(json.dumps({"state": "failed", "error": str(exc)}), encoding="utf-8")
        raise
