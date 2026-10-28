"""Single-expert prediction blending for the pinned Stable Audio pipeline."""
from contextlib import contextmanager
import torch
from .lora import LoRALinear, set_strength


@contextmanager
def masked_adapter(model, mask, event_condition=None, event_strength=1.0):
    """Blend v predictions before pipeline CFG and its single scheduler step.

    Both passes see the same latent and timestep. Optional event_condition replaces
    only the expert cross-attention input; global duration remains unchanged.
    event_strength=0 is the regional-prompt control without adapter influence.
    This temporarily
    wraps forward; callers must use one inference at a time on this model.
    """
    if mask.ndim != 1 or not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
        raise ValueError("Expected finite one-dimensional mask in [0,1]")
    if event_strength not in (0, 1):
        raise ValueError("Pilot event strength must be zero or one")
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if not modules:
        raise ValueError("No installed adapter")
    strengths = [m.strength for m in modules]
    original = model.forward
    had_override = "forward" in model.__dict__
    previous_override = model.__dict__.get("forward")
    stats = {"steps": 0, "base_passes": 0, "expert_passes": 0}

    def forward(*args, **kwargs):
        if kwargs.get("return_dict", True):
            raise ValueError("Temporal wrapper requires return_dict=False")
        latent = args[0] if args else kwargs["hidden_states"]
        if latent.shape[-1] != mask.numel():
            raise ValueError("Mask must cover the full padded latent time axis")
        set_strength(model, 0)
        base = original(*args, **kwargs)[0]
        stats["base_passes"] += 1
        event_kwargs = dict(kwargs)
        if event_condition is not None:
            background_condition = kwargs["encoder_hidden_states"]
            if event_condition.shape != background_condition.shape or not torch.isfinite(event_condition).all():
                raise ValueError("Event/background condition shapes must match and be finite")
            event_kwargs["encoder_hidden_states"] = event_condition.to(background_condition)
        set_strength(model, event_strength)
        expert = original(*args, **event_kwargs)[0]
        stats["expert_passes"] += 1
        if base.shape != expert.shape or base.shape[-1] != mask.numel():
            raise ValueError("Unexpected prediction shape")
        weights = mask.to(device=base.device, dtype=torch.float32)
        mixed = (base.float() + weights * (expert.float() - base.float())).to(base.dtype)
        # Exact endpoints also support strict zero/full-mask regression checks.
        mixed = torch.where(weights == 0, base, torch.where(weights == 1, expert, mixed))
        if not torch.isfinite(mixed).all():
            raise ValueError("Nonfinite fused prediction")
        stats["steps"] += 1
        return (mixed,)

    model.forward = forward
    try:
        yield stats
    finally:
        if had_override:
            model.forward = previous_override
        else:
            del model.forward
        for module, strength in zip(modules, strengths):
            module.strength = strength


@torch.inference_mode()
def event_conditioning(pipe, prompt, duration=10):
    """Pinned pipeline's batch-one CFG conditioning, no negative prompt."""
    device = pipe._execution_device
    text = pipe.encode_prompt(prompt, device, False)
    start, end = pipe.encode_duration(0.0, float(duration), device, False, 1)
    conditional = torch.cat([text, start, end], dim=1)
    return torch.cat([torch.zeros_like(conditional), conditional], dim=0)
