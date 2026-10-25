"""Download a revision-pinned Diffusers snapshot. Does not load/use a GPU."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/stable-audio-open-1.0"))
    args = parser.parse_args()
    repo = "stabilityai/stable-audio-open-1.0"
    marker = args.out / "download_manifest.json"
    # On reruns retain the first successful revision instead of silently updating weights.
    revision = json.loads(marker.read_text())["revision"] if marker.exists() else HfApi().model_info(repo, revision=args.revision).sha
    print(f"Downloading {repo} at {revision}. No GPU required.", flush=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(args.out),
                      allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                      "text_encoder/*", "projection_model/*", "transformer/*", "vae/*",
                                      "README.md", "LICENSE*"], max_workers=2)
    if not (args.out / "model_index.json").exists():
        raise RuntimeError("Incomplete Diffusers checkpoint: missing model_index.json")
    marker.write_text(json.dumps({"repo_id": repo, "revision": revision}, indent=2))
    print(f"Download complete: {args.out.resolve()}")


if __name__ == "__main__":
    main()
