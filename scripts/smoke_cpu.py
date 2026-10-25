"""Run from any directory with the project's Python; never touches a GPU."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    run = root / "outputs" / ("cpu-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    commands = [
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
        ["-m", "bgm", "validate", "examples/timeline.json"],
        ["-m", "bgm", "plan", "examples/story.txt", "--duration", "10", "--out", str(run / "draft.json")],
        ["-m", "bgm", "render", "examples/timeline.json", "--backend", "diagnostic", "--out", str(run / "demo")],
        ["-m", "bgm", "masks", "examples/timeline.json", "--frames", "500", "--frame-seconds", "0.02", "--out", str(run / "masks.npz")],
    ]
    for args in commands:
        print("Running: python " + " ".join(args), flush=True)
        subprocess.run([sys.executable, *args], cwd=root, env=env, check=True)
    print(f"\nCPU checks passed. Diagnostic audio (not AI-generated sounds): {run / 'demo' / 'mix.wav'}")


if __name__ == "__main__":
    main()
