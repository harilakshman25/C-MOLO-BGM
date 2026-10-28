"""Measured sentence timing and reusable stem composition; no latent fusion."""
import numpy as np
from .timeline import validate


def level_audio(audio, target_db=-22, max_boost_db=18):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite stereo audio")
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-7:
        raise ValueError("Near-silent source; regenerate or replace this candidate")
    gain = min(10**(max_boost_db/20), 10**(target_db/20)/rms, 0.85/peak)
    return audio*gain, float(20*np.log10(gain))


def fit_stem(sound, samples, sample_rate, loop=False):
    if samples < 1 or len(sound) < 2:
        raise ValueError("Invalid stem length")
    if len(sound) >= samples:
        return sound[:samples].copy()
    if not loop:
        # Isolated effects are played once; the remaining cue is silent.
        return np.pad(sound, ((0, samples-len(sound)), (0, 0)))
    overlap = max(1, min(round(.3*sample_rate), len(sound)//4))
    result = sound.copy()
    while len(result) < samples:
        ramp = np.linspace(0, 1, overlap)[:, None]
        bridge = result[-overlap:]*(1-ramp)+sound[:overlap]*ramp
        result = np.concatenate([result[:-overlap], bridge, sound[overlap:]])
    return result[:samples]


def build_plan(story, segments, library, sample_rate=44100):
    if len(story["segments"]) != len(segments) or not segments:
        raise ValueError("Narration segments do not match story")
    duration = segments[-1]["end_s"]+3
    cues = []
    for i, (scene, timing) in enumerate(zip(story["segments"], segments)):
        start, end = timing["start_s"], timing["end_s"]
        next_start = segments[i+1]["start_s"] if i+1<len(segments) else duration
        for j, category in enumerate(scene["effects"]):
            spec = library[category]
            kind = spec["kind"]
            if kind == "sfx":
                begin = start + (end-start)*(0.65 if category == "dog" else 0.35)
                finish = min(next_start, begin+spec["seconds"])
            else:
                begin, finish = start, next_start
            cues.append(dict(id=f"s{i:02d}_{category}_{j}", category=category, kind=kind,
                             prompt=spec["prompt"], start_s=begin, end_s=finish,
                             gain_db=-3 if kind=="sfx" else -5,
                             fade_s=min(.3, (finish-begin)/2)))
    # Continuous quiet rain under the outdoor scenes; scene-specific rain adds texture.
    cues.append(dict(id="rain_bed",category="rain",kind="ambience",prompt=library["rain"]["prompt"],
                     start_s=0, end_s=duration, gain_db=-12, fade_s=.8))
    return validate(dict(schema_version=1, story=" ".join(x["text"] for x in story["segments"]),
                         duration_s=duration, sample_rate=sample_rate, seed=story["seed"], cues=cues,
                         timing_source="Measured TTS sentence boundaries; authored within-sentence cue offsets, not forced alignment"))
