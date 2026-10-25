"""Minimal project-native LoRA for Stable Audio q/v Linear projections.

This explicit adapter format is not a PEFT checkpoint. Base weights are never saved.
"""
import math
import re
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Expected Linear, positive rank and alpha")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.strength = rank, float(alpha), 1.0
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        if self.strength == 0:
            return original
        # FP32 trainable parameters/branch are important for FP16 frozen-weight training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = F.linear(F.linear(x.float(), self.lora_a.float()), self.lora_b.float())
        return original + (residual * (self.alpha/self.rank) * self.strength).to(original.dtype)


def inject_lora(model, rank=8, alpha=8.0):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Adapters already installed")
    model.requires_grad_(False)
    pattern = re.compile(r"transformer_blocks\.\d+\.attn[12]\.to_[qv]$")
    targets = [name for name, module in model.named_modules() if pattern.fullmatch(name) and isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("No expected Stable Audio attention projections found")
    for name in targets:
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha))
    return targets


def adapter_parameters(model):
    return [p for name, p in model.named_parameters() if name.endswith((".lora_a", ".lora_b"))]


def adapter_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if name.endswith((".lora_a", ".lora_b"))}


def load_adapter_state(model, state):
    expected = adapter_state(model)
    if set(expected) != set(state):
        raise ValueError("Adapter keys do not match target modules")
    for key, value in state.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor: {key}")
    model.load_state_dict(state, strict=False)


def set_strength(model, strength):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Invalid adapter strength")
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.strength = strength


def velocity_example(clean, noise, time):
    """Cosine VP noising consistent with sigma_data=1 and t=2*atan(sigma)/pi.

    The scheduler operates on unscaled x+sigma*noise and scales it by
    1/sqrt(1+sigma^2). The network therefore sees alpha*x+beta*noise.
    Its target is alpha*noise-beta*x, not epsilon.
    """
    if clean.shape != noise.shape or time.shape != (clean.shape[0],):
        raise ValueError("Invalid velocity batch shapes")
    if not ((time > 0) & (time < 1)).all():
        raise ValueError("Training times must be strictly inside (0,1)")
    shape = (-1,) + (1,)*(clean.ndim-1)
    angle = time.reshape(shape) * (math.pi/2)
    alpha, beta = angle.cos(), angle.sin()
    return alpha*clean + beta*noise, alpha*noise-beta*clean
