"""Scheduler-independent numerical fusion primitives, not a trained audio model."""
import numpy as np


def temporal_masks(cues, frames, frame_duration_s):
    """Sample soft masks at latent frame centres. Include padded frames in frames.

    frame_duration_s must come from the codec stride / sample rate, not clip duration
    divided by the model's padded latent length.
    """
    if type(frames) is not int or frames <= 0 or not np.isfinite(frame_duration_s) or frame_duration_s <= 0:
        raise ValueError("Positive frames and frame_duration_s required")
    time = (np.arange(frames) + 0.5) * frame_duration_s
    masks = np.zeros((len(cues), frames), dtype=np.float64)
    for i, cue in enumerate(cues):
        start, end, fade = cue["start_s"], cue["end_s"], cue.get("fade_s", 0.05)
        if not all(np.isfinite(x) for x in (start, end, fade)) or start < 0 or end <= start or not 0 <= fade <= (end-start)/2:
            raise ValueError("Invalid mask interval")
        mask = ((time >= start) & (time < end)).astype(float)
        if fade > 0:
            ramp = np.minimum(np.clip((time-start)/fade, 0, 1), np.clip((end-time)/fade, 0, 1))
            mask *= 0.5 - 0.5*np.cos(np.pi*ramp)
        masks[i] = mask
    return masks


def fuse(base, experts, masks, mode="normalized", strength=1.0):
    """Fuse predictions in ONE shared scheduler parameterization.

    base: [..., T]; experts: [K, ..., T]; masks: [K, T]. No CFG is applied here.
    normalized: base fills uncovered weight; overlapping experts form a convex blend.
    residual: base + strength * sum(mask * (expert - base)); experimental ablation.
    """
    base, experts, masks = [np.asarray(x, dtype=np.float64) for x in (base, experts, masks)]
    if base.ndim < 1 or experts.ndim != base.ndim+1 or experts.shape[1:] != base.shape:
        raise ValueError("Expert and base prediction shapes disagree")
    if masks.shape != (experts.shape[0], base.shape[-1]):
        raise ValueError("Masks must have shape [experts, latent_time]")
    if not all(np.isfinite(x).all() for x in (base, experts, masks)) or (masks < 0).any() or (masks > 1).any():
        raise ValueError("Finite predictions and masks in [0,1] required")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    weights = masks.reshape((masks.shape[0],) + (1,)*(base.ndim-1) + (base.shape[-1],))
    if mode == "residual":
        return base + strength * (weights * (experts-base)).sum(axis=0)
    if mode != "normalized":
        raise ValueError("Unknown fusion mode")
    total = masks.sum(axis=0)
    background = np.maximum(0, 1-total)
    return (background*base + (weights*experts).sum(axis=0)) / np.maximum(1, total)


def sample_fused(latents, timesteps, predict_base, predict_expert, step, masks, mode="normalized"):
    """Reference loop for synthetic tests. Real torch adapter integration is pending.

    Predict callbacks receive the same z and timestep. step(prediction,t,z) executes
    exactly once per timestep. Callbacks own scheduler input scaling and guidance.
    """
    z = np.array(latents, dtype=float, copy=True)
    for t in timesteps:
        base = predict_base(z, t)
        predictions = [predict_expert(i, z, t) for i in range(len(masks))]
        experts = np.stack(predictions) if predictions else np.empty((0,)+z.shape)
        prediction = fuse(base, experts, masks, mode=mode)
        z = step(prediction, t, z)
    return z
