"""Download just ESC-50 dog clips; preserve source folds and provenance for review."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import wave
from bgm.dataset import audit


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "cmolo-research-pilot"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/esc50-dog"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clips").mkdir(exist_ok=True)
    marker = args.out / "source_revision.txt"
    revision = marker.read_text().strip() if marker.exists() else json.loads(fetch("https://api.github.com/repos/karolpiczak/ESC-50/commits/master"))["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Invalid source commit SHA")
    marker.write_text(revision + "\n")
    base = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{revision}"
    for name in ("LICENSE", "README.md", "meta/esc50.csv"):
        target = args.out / Path(name).name
        if not target.exists():
            target.write_bytes(fetch(f"{base}/{name}"))
    rows = list(csv.DictReader(io.StringIO((args.out / "esc50.csv").read_text())))
    dogs = [row for row in rows if row["category"] == "dog"]
    if len(dogs) != 40:
        raise ValueError(f"Expected 40 dog clips; found {len(dogs)}. Review upstream changes.")
    records = []
    for row in dogs:
        filename = row["filename"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.wav", filename):
            raise ValueError("Unsafe source filename")
        audio = args.out / "clips" / filename
        if not audio.exists():
            payload = fetch(f"{base}/audio/{filename}")
            # Validate before saving so interrupted downloads cannot masquerade as clips.
            with wave.open(io.BytesIO(payload), "rb") as handle:
                if handle.getnframes() <= 0:
                    raise ValueError(f"Empty audio: {filename}")
            temporary = audio.with_suffix(".part")
            temporary.write_bytes(payload)
            temporary.replace(audio)
        with wave.open(str(audio), "rb") as handle:
            sr = handle.getframerate()
            duration = handle.getnframes()/sr
            channels = handle.getnchannels()
        fold = int(row["fold"])
        if fold not in range(1, 6):
            raise ValueError("Unexpected fold")
        records.append({"id": audio.stem, "path": f"clips/{filename}",
                        "source_id": "freesound-" + row["src_file"], "category": "dog",
                        "caption": "A dog barking.", "caption_source": "category-derived; needs listening review",
                        "license": "See bundled LICENSE for dataset and individual recording terms",
                        "split": "train" if fold <= 3 else "validation" if fold == 4 else "test",
                        "fold": fold, "source_url": f"https://freesound.org/s/{row['src_file']}/",
                        "download_url": f"{base}/audio/{filename}", "dataset_revision": revision,
                        "sample_rate": sr, "channels": channels, "duration_s": duration,
                        "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "review_status": "pending"})
        print(f"Ready: {filename}", flush=True)
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    review = args.out / "review.csv"
    if not review.exists():
        with review.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "split", "keep_yes_no", "birds_yes_no_unsure", "other_sounds", "caption_after_listening", "notes"])
            for row in records:
                writer.writerow([row["id"], row["split"], "", "", "", "", ""])
    report = audit(manifest)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("PREPARATION COMPLETE. Clips are pending listening review; not approved training data yet.")


if __name__ == "__main__":
    main()
