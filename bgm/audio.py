"""PCM I/O and deterministic baseline mixer. No generative-quality claims."""
import wave
from pathlib import Path
import numpy as np


def read_wav(path, sample_rate):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() not in (1, 2):
            raise ValueError("Use mono/stereo 16-bit PCM WAV")
        if handle.getframerate() != sample_rate:
            raise ValueError(f"WAV must be {sample_rate} Hz; resample explicitly before use")
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(float)/32768
        audio = audio.reshape(-1, handle.getnchannels())
    return np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio


def write_wav(path, audio, sample_rate):
    audio = np.asarray(audio)
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError("Expected finite stereo audio [samples, 2]")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(audio, -1, 1)*32767).astype("<i2").tobytes())


def duck_envelope(narration, sample_rate, reduction_db=-10):
    """10ms block RMS detector with attack/release smoothing, not a speech VAD."""
    block = max(1, round(sample_rate*0.01))
    envelope = np.ones(len(narration))
    state = 1.0
    for offset in range(0, len(narration), block):
        end = min(len(narration), offset+block)
        active = np.sqrt(np.mean(narration[offset:end]**2)) > 0.01
        target = 10**(reduction_db/20) if active else 1.0
        tau = 0.02 if target < state else 0.2
        t = np.arange(1, end-offset+1)/sample_rate
        envelope[offset:end] = target+(state-target)*np.exp(-t/tau)
        state = envelope[end-1]
    return envelope


def mix(plan, stems, narration=None):
    sr, duration = plan["sample_rate"], plan["duration_s"]
    samples = round(sr*duration)
    voice = np.zeros((samples, 2))
    if narration is not None:
        if len(narration) > samples:
            raise ValueError("Narration exceeds timeline; align/revise duration before rendering")
        voice[:len(narration)] = narration
    envelope = duck_envelope(voice, sr)
    output = voice.copy()
    for cue in plan["cues"]:
        begin, end = round(cue["start_s"]*sr), round(cue["end_s"]*sr)
        sound = np.asarray(stems[cue["id"]], dtype=float)
        if sound.ndim != 2 or sound.shape[1] != 2 or not np.isfinite(sound).all():
            raise ValueError(f"Invalid stereo stem {cue['id']}")
        if len(sound) < end-begin:
            raise ValueError(f"Stem {cue['id']} is shorter than its cue")
        sound = sound[:end-begin].copy()*10**(cue.get("gain_db", -12)/20)
        fade = min(round(cue.get("fade_s", 0.05)*sr), len(sound)//2)
        if fade:
            ramp = 0.5-0.5*np.cos(np.linspace(0, np.pi, fade))
            sound[:fade] *= ramp[:, None]
            sound[-fade:] *= ramp[::-1, None]
        if cue["kind"] in ("music", "ambience"):
            sound *= envelope[begin:end, None]
        output[begin:end] += sound
    peak = float(np.max(np.abs(output)))
    gain = min(1, 10**(-1/20)/max(peak, 1e-12))
    output *= gain
    return output, {"peak_before_gain": peak, "master_gain_db": float(20*np.log10(gain)),
                    "sample_peak_after_gain": float(np.max(np.abs(output))),
                    "rms": float(np.sqrt(np.mean(output**2))),
                    "narration_present": narration is not None,
                    "note": "Sample-peak gain only; not true-peak limiting or LUFS normalization"}
