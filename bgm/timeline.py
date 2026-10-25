"""Validated, versioned timeline shared by CLI and future server."""
import math
import re


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def validate(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Expected timeline schema_version 1")
    number(plan.get("duration_s"), "duration_s", 0.1, 600)
    sr = plan.get("sample_rate")
    if type(sr) is not int or sr not in (16000, 22050, 24000, 44100, 48000):
        raise ValueError("Unsupported integer sample_rate")
    if type(plan.get("seed")) is not int or not 0 <= plan["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(plan.get("story"), str):
        raise ValueError("story must be text")
    if not isinstance(plan.get("cues"), list) or len(plan["cues"]) > 256:
        raise ValueError("cues must be a list of at most 256 entries")
    seen = set()
    for cue in plan["cues"]:
        if not isinstance(cue, dict):
            raise ValueError("Each cue must be an object")
        cid = cue.get("id")
        if not isinstance(cid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid) or cid in seen:
            raise ValueError("Cue IDs must be unique safe filenames")
        seen.add(cid)
        if cue.get("kind") not in ("sfx", "ambience", "music"):
            raise ValueError(f"{cid}: kind must be sfx, ambience or music")
        for key in ("category", "prompt"):
            if not isinstance(cue.get(key), str) or not cue[key].strip():
                raise ValueError(f"{cid}: {key} is required")
        for key in ("start_s", "end_s"):
            number(cue.get(key), key, 0, plan["duration_s"])
        if cue["end_s"] <= cue["start_s"]:
            raise ValueError(f"{cid}: end_s must exceed start_s")
        number(cue.get("gain_db", -12), "gain_db", -80, 12)
        number(cue.get("fade_s", 0.05), "fade_s", 0, (cue["end_s"] - cue["start_s"]) / 2)
    return plan


VOCABULARY = {
    "dog": (r"\b(dog|dogs|bark|barks|barking)\b", "sfx", "A dog barking clearly outdoors", 1.5),
    "rain": (r"\b(rain|raining|rainfall)\b", "ambience", "Steady rain falling, no speech", None),
    "piano": (r"\bpiano\b", "music", "Gentle solo piano instrumental, no vocals", None),
    "footsteps": (r"\b(footsteps|walking|walked)\b", "sfx", "Footsteps on a wooden floor", 2.0),
}


def draft(story, duration_s, seed=42):
    """Small deterministic keyword fixture; NOT an LLM or forced aligner."""
    if not story.strip():
        raise ValueError("Story is empty")
    number(duration_s, "duration_s", 0.1, 600)
    words = list(re.finditer(r"\S+", story))
    cues = []
    for category, (pattern, kind, prompt, length) in VOCABULARY.items():
        matches = list(re.finditer(pattern, story, re.I))
        if not matches:
            continue
        # Continuous beds use the first mention; discrete events retain repeated mentions.
        for match in (matches[:1] if length is None else matches):
            index = sum(w.start() < match.start() for w in words)
            start = min(duration_s - 0.05, duration_s * index / len(words))
            end = duration_s if length is None else min(duration_s, start + length)
            cues.append({"id": f"{category}_{len(cues):03d}", "category": category,
                         "kind": kind, "prompt": prompt, "start_s": start, "end_s": end,
                         "gain_db": -18 if kind != "sfx" else -8,
                         "fade_s": min(0.05, (end-start)/2), "trigger_word_index": index})
    return validate({"schema_version": 1, "story": story, "duration_s": duration_s,
                     "sample_rate": 24000, "seed": seed,
                     "timing_source": "heuristic-estimate-review-required", "cues": cues})
