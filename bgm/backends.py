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

    Optional project-native LoRA loading is supported; temporal fusion is not.
    Model CPU offload reduces residency,
    but does not establish that inference fits a particular GPU.
    """
    name = "stable-audio-open-baseline"

    def __init__(self, model_path, steps=100, offload=False, adapter_path=None):
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
        self.adapter_metadata = None
        if adapter_path is not None:
            if offload:
                raise ValueError("Adapter loading with offload is not yet validated")
            import json
            from pathlib import Path
            from .lora import inject_lora, load_adapter_state
            checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
            meta = checkpoint["metadata"]
            if meta.get("format") != "cmolo-linear-lora-v1":
                raise ValueError("Unknown adapter format")
            identity = json.loads((Path(model_path)/"download_manifest.json").read_text())
            if meta["model"] != identity:
                raise ValueError("Adapter trained on a different base revision")
            if meta.get("smoke_test_only") or meta["step"] == 0:
                raise ValueError("Use a trained pilot checkpoint, not a smoke/step-zero adapter")
            targets = inject_lora(self.pipe.transformer, meta["rank"], meta["alpha"])
            if targets != meta["targets"]:
                raise ValueError("Adapter target modules mismatch")
            load_adapter_state(self.pipe.transformer, checkpoint["adapter"])
            self.pipe.transformer.requires_grad_(False).eval()
            self.adapter_metadata = meta
            self.name = "stable-audio-open-dog-lora"

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
