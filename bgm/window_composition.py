"""Experimental audio adaptation of local-view diffusion consensus.

One shared latent and scheduler; each local view changes BOTH text and duration
conditioning. This is not a trained temporal model and provides no timing guarantee.
"""
from contextlib import contextmanager
import math
import torch
from .lora import LoRALinear, set_strength


@torch.inference_mode()
def conditioning(pipe, prompt, duration):
    device = pipe._execution_device
    text = pipe.encode_prompt(prompt, device, False)
    start, end = pipe.encode_duration(0., float(duration), device, False, 1)
    cross = torch.cat([text, start, end], dim=1)
    glob = torch.cat([start, end], dim=2)
    return torch.cat([torch.zeros_like(cross), cross]), torch.cat([glob, glob])


@contextmanager
def window_consensus(model, views, local_weight=4.):
    """v = (v_global + sum(weight * scatter(v_local))) / (1 + sum(weight)).

    Each view contains begin/end latent indices, cross/global embeddings, and
    adapter strength. Local rotary positions restart at zero, including the
    prepended duration token. Overlaps are normalized, not added as waveforms.
    CFG and exactly one scheduler update are performed by the original pipeline.
    """
    if not math.isfinite(local_weight) or local_weight < 0:
        raise ValueError("Invalid local weight")
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    strengths = [m.strength for m in modules]
    original = model.forward
    had_override = 'forward' in model.__dict__
    previous = model.__dict__.get('forward')
    stats = dict(steps=0, global_passes=0, local_passes=0)

    def forward(hidden_states, timestep, **kwargs):
        if kwargs.get('return_dict', True) or kwargs.get('attention_mask') is not None:
            raise ValueError("Requires tuple output and no attention mask")
        set_strength(model, 0)
        base = original(hidden_states, timestep, **kwargs)[0]
        stats['global_passes'] += 1
        numerator = base.float().clone()
        denominator = torch.ones(base.shape[-1], device=base.device, dtype=torch.float32)
        for view in views:
            begin, end = view['begin'], view['end']
            if not 0 <= begin < end <= hidden_states.shape[-1]:
                raise ValueError("Invalid local view bounds")
            strength = view.get('strength', 0)
            if strength not in (0, 1) or (strength and not modules):
                raise ValueError("Requested adapter is not installed")
            if local_weight == 0:
                continue
            local = dict(kwargs, encoder_hidden_states=view['cross'], global_hidden_states=view['global'])
            # Prefix positions correspond to a short independent local coordinate system.
            rotary = kwargs.get('rotary_embedding')
            if rotary is not None:
                local['rotary_embedding'] = tuple(x[:end-begin+1] for x in rotary)
            set_strength(model, strength)
            pred = original(hidden_states[..., begin:end], timestep, **local)[0]
            if pred.shape != base[..., begin:end].shape or not torch.isfinite(pred).all():
                raise ValueError("Invalid local prediction")
            weight = torch.ones(end-begin, device=base.device, dtype=torch.float32)*local_weight
            fade = min(view.get('fade_frames', 5), (end-begin)//2)
            if fade:
                ramp = torch.linspace(0, 1, fade+2, device=base.device)[1:-1]
                weight[:fade] *= ramp
                weight[-fade:] *= ramp.flip(0)
            numerator[..., begin:end] += pred.float()*weight
            denominator[begin:end] += weight
            stats['local_passes'] += 1
        result = (numerator/denominator).to(base.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite consensus")
        stats['steps'] += 1
        return (result,)

    model.forward = forward
    try:
        yield stats
    finally:
        if had_override:
            model.forward = previous
        else:
            del model.forward
        for module, strength in zip(modules, strengths):
            module.strength = strength
