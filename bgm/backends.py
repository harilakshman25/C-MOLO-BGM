"""Explicit diagnostic and optional real inference backends."""
import hashlib
import numpy as np


def cue_seed(seed, cue_id):
    return (seed + int.from_bytes(hashlib.sha256(cue_id.encode()).digest()[:4], "little")) % 2**32


class DiagnosticBackend:
    name = "diagnostic-tones-NOT-generated-sounds"

    def generate(self, cue, sample_rate, seed):
        n = round(cue["end_s"]*sample_rate)-round(cue["start_s"]*sample_rate)
        t = np.arange(n)/sample_rate
        frequency = 180 + cue_seed(0, cue["category"]) % 600
        rng = np.random.default_rng(seed)
        signal = 0.15*np.sin(2*np.pi*frequency*t)
        if cue["kind"] == "ambience":
            signal = rng.normal(0, 0.04, n)
        return np.repeat(signal[:, None], 2, axis=1)


class StableAudioBackend:
    """Baseline inference only. Requires locally downloaded weights and torch stack.

    No LoRA loading/fusion is implied. Sequential CPU offload reduces residency,
    but does not establish that inference fits a particular GPU.
    """
    name = "stable-audio-open-baseline"

    def __init__(self, model_path, steps=100, offload=False):
        try:
            import torch
            from diffusers import StableAudioPipeline
        except ImportError as exc:
            raise RuntimeError("Install the inference extra in the GPU environment first") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("This backend requires a CUDA GPU")
        self.torch, self.steps = torch, steps
        self.pipe = StableAudioPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.float16, local_files_only=True)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

    def generate(self, cue, sample_rate, seed):
        if sample_rate != self.pipe.vae.sampling_rate:
            raise ValueError(f"Timeline sample_rate must equal {self.pipe.vae.sampling_rate}")
        duration = cue["end_s"]-cue["start_s"]
        if duration > 47:
            raise ValueError("Stable Audio baseline limits cues to 47 seconds; split the scene")
        with self.torch.inference_mode():
            audio = self.pipe(prompt=cue["prompt"], audio_end_in_s=duration,
                              num_inference_steps=self.steps,
                              generator=self.torch.Generator("cuda").manual_seed(seed),
                              output_type="pt").audios[0]
        return audio.float().cpu().numpy().T
