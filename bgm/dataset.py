"""Audit category manifests for missing audio and source/duplicate split leakage."""
import hashlib
import json
import wave
from pathlib import Path


def audit(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Dataset manifest is empty")
    sources, hashes, ids, counts = {}, {}, set(), {}
    for row in records:
        for key in ("id", "path", "source_id", "category", "caption", "license", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Required dataset field: {key}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate clip ID: {row['id']}")
        ids.add(row["id"])
        split = row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        audio = path.parent/row["path"]
        with wave.open(str(audio), "rb") as handle:
            if handle.getnframes() == 0:
                raise ValueError(f"Empty WAV: {audio}")
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        for store, key in ((sources, row["source_id"]), (hashes, digest)):
            if key in store and store[key] != split:
                raise ValueError(f"Cross-split leakage for clip {row['id']}")
            store[key] = split
        group = f"{row['category']}:{split}"
        counts[group] = counts.get(group, 0)+1
    return {"clips": len(records), "counts": counts,
            "note": "Exact file hashes and source IDs only; near-duplicate and license verification remain manual"}
