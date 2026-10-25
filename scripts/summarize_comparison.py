"""Join listener ratings to experiment labels and verify exported PCM WAVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import wave
import numpy as np


def summarize(root):
    experiment = json.loads((root / "experiment.json").read_text())
    if experiment.get("diagnostic"):
        raise ValueError("Diagnostic tones are not research results")
    ratings = {}
    warnings = []
    with (root / "ratings.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = ["sample", "bark_clarity_1_to_5", "birds_yes_no_unsure", "other_unwanted_sounds", "distortion_yes_no_unsure", "notes"]
        if reader.fieldnames != expected:
            raise ValueError("Unexpected ratings columns")
        for row in reader:
            # Preserve user text where an unquoted comma appeared in the final notes column.
            extra = row.pop(None, None)
            if extra:
                row["notes"] = ",".join([row["notes"], *extra])
                warnings.append(f"{row['sample']}: unquoted note commas recovered; original CSV unchanged")
            if row["sample"] in ratings:
                raise ValueError("Duplicate rating")
            clarity = int(row["bark_clarity_1_to_5"])
            if not 1 <= clarity <= 5:
                raise ValueError("Clarity must be 1 through 5")
            for key in ("birds_yes_no_unsure", "distortion_yes_no_unsure"):
                if row[key].strip().lower() not in ("yes", "no", "unsure"):
                    raise ValueError(f"Invalid {key}")
            row["bark_clarity_1_to_5"] = clarity
            ratings[row["sample"]] = row
    if set(ratings) != {item["sample"] for item in experiment["samples"]}:
        raise ValueError("Ratings must cover exactly the generated sample set")
    results = []
    for item in experiment["samples"]:
        name = item["sample"]
        if Path(name).name != name:
            raise ValueError("Unsafe sample path")
        path = root / "listen" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["mix_sha256"]:
            raise ValueError(f"Hash mismatch: {name}")
        with wave.open(str(path), "rb") as handle:
            sr, channels, width, frames = handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes()
            if (sr, channels, width, frames) != (44100, 2, 2, 220500):
                raise ValueError(f"Unexpected WAV format/duration: {name}")
            samples = np.frombuffer(handle.readframes(frames), dtype="<i2").astype(float)/32768
        results.append({**item, "rating": ratings[name], "wav": {
            "hash_matches": True, "duration_s": frames/sr, "sample_rate": sr,
            "channels": channels, "sample_peak": float(np.max(np.abs(samples))),
            "rms": float(np.sqrt(np.mean(samples**2))),
            "pcm_full_scale_samples": int(np.sum(np.abs(samples) >= 32767/32768))}})
    groups = {}
    for condition in ("outdoor", "isolated"):
        rows = [row["rating"] for row in results if row["condition"] == condition]
        groups[condition] = {"samples": len(rows),
            "mean_listener_bark_clarity": sum(row["bark_clarity_1_to_5"] for row in rows)/len(rows),
            "birds_reported": sum(row["birds_yes_no_unsure"] == "yes" for row in rows),
            "distortion_reported": sum(row["distortion_yes_no_unsure"] == "yes" for row in rows),
            "distortion_uncertain": sum(row["distortion_yes_no_unsure"] == "unsure" for row in rows)}
    return {"groups": groups, "samples": results, "parsing_notes": warnings,
            "interpretation": "Single-listener, three-seed pilot. Ratings are user observations, not automated sound recognition. PCM checks do not prove absence of upstream distortion.",
            "elapsed_including_load_s": experiment["elapsed_including_load_s"],
            "peak_allocated_gib": experiment["peak_allocated_gib"],
            "peak_reserved_gib": experiment["peak_reserved_gib"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = summarize(args.run)
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"groups": report["groups"], "parsing_notes": report["parsing_notes"],
                      "wav_checks": "All six hashes and formats match; inspect analysis.json for signal statistics"}, indent=2))


if __name__ == "__main__":
    main()
