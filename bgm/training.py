"""Data checks and frozen feature caching for the initial dog adaptation pilot."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .audio import read_wav
from .dataset import audit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_records(manifest, review):
    decision = json.loads(Path(review).read_text())
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if decision.get("decision") != "accept_all_for_pilot" or len(rows) != decision["expected_clips"]:
        raise ValueError("Dataset does not match the accepted 40-clip review")
    audit(manifest)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    if counts != {"train": 24, "validation": 8, "test": 8}:
        raise ValueError(f"Unexpected split counts: {counts}")
    for row in rows:
        if row["category"] != "dog":
            raise ValueError("This pilot accepts only dog clips")
        expected_split = "train" if int(row["fold"]) <= 3 else "validation" if int(row["fold"]) == 4 else "test"
        if row["split"] != expected_split:
            raise ValueError("Official source folds must remain unchanged")
        if digest(Path(manifest).parent/row["path"]) != row["sha256"]:
            raise ValueError(f"Audio changed since preparation: {row['id']}")
    # Test waveforms are not encoded or used to optimize/select the model.
    return [row for row in rows if row["split"] != "test"]


def cache_features(pipe, rows, manifest, cache_path, identity):
    cache_path = Path(cache_path)
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Cache metadata mismatch; choose a new cache path")
        return saved
    frames = int(pipe.transformer.config.sample_size)
    sr, hop = pipe.vae.config.sampling_rate, pipe.vae.hop_length
    full_samples = frames * hop
    pipe.text_encoder.to(device="cuda", dtype=torch.float32).eval()
    pipe.projection_model.to(device="cuda", dtype=torch.float32).eval()
    conditions = {}
    with torch.no_grad():
        for caption in sorted({row["caption"] for row in rows}):
            prompt = pipe.encode_prompt(caption, "cuda", False)
            start, end = pipe.encode_duration(0., 5., "cuda", False, 1)
            conditions[caption] = {"cross": torch.cat([prompt, start, end], dim=1).cpu(),
                                   "global": torch.cat([start, end], dim=2).cpu()}
    pipe.text_encoder.to("cpu")
    pipe.projection_model.to("cpu")
    torch.cuda.empty_cache()
    pipe.vae.to(device="cuda", dtype=torch.float32).eval()
    clips = []
    with torch.no_grad():
        for row in rows:
            audio = read_wav(Path(manifest).parent/row["path"], sr)
            if len(audio) != 5*sr or full_samples < len(audio):
                raise ValueError("Pilot requires five-second native-rate clips that fit the model")
            # Duplicate mono to stereo, preserve amplitude; no fake spatial information.
            waveform = torch.zeros(1, 2, full_samples, device="cuda", dtype=torch.float32)
            waveform[0, :, :len(audio)] = torch.from_numpy(audio.T.astype(np.float32)).to("cuda")
            latent = pipe.vae.encode(waveform).latent_dist.mode().cpu()
            if latent.shape[-1] != frames or not torch.isfinite(latent).all():
                raise ValueError("Unexpected/non-finite codec latent")
            clips.append({"id": row["id"], "split": row["split"], "caption": row["caption"],
                          "latent": latent, "active_frames": min(frames, math.ceil(len(audio)/hop))})
            print(f"Cached {row['split']} {row['id']}", flush=True)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    saved = {"identity": identity, "clips": clips, "conditions": conditions,
             "frames": frames, "hop_length": hop, "sample_rate": sr,
             "encoder_policy": "FP32 posterior mode, waveform padded with silence to full model length"}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(cache_path)
    return saved
