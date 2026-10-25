"""Check the allocated GPU and inference imports, without downloading a model."""
import json
import os
import platform
import torch
import diffusers
import transformers
from diffusers import StableAudioPipeline, CosineDPMSolverMultistepScheduler
from importlib.metadata import version


def main():
    # Pipeline import alone does not exercise optional scheduler dependencies.
    scheduler = CosineDPMSolverMultistepScheduler()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run inside the allocated GPU job, not on the head node.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one allocated visible GPU; found {torch.cuda.device_count()}")
    torch.manual_seed(42)
    # Device 0 here is the first GPU exposed to this process, not physical node GPU 0.
    a = torch.randn(256, 256, device="cuda:0", dtype=torch.float16)
    result = a @ a.T
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("GPU matrix test produced non-finite values")
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "status": "GPU CHECK PASSED",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "pipeline_import": StableAudioPipeline.__name__,
        "scheduler_constructed": type(scheduler).__name__,
        "torchsde": version("torchsde"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_gib": round(props.total_memory / 2**30, 2),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "fp16_matrix_test": "passed",
        "note": "No model loaded; model inference and memory requirements remain untested."
    }, indent=2))


if __name__ == "__main__":
    main()
