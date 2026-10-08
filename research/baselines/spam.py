"""Baseline: SPAM -- Spike-Aware Adam with Momentum reset (Huang et al., arXiv 2501.06842).

Written from scratch from the paper. The authors' repo (TianjinYellow/SPAM-Optimizer) has NO license, so
no code is taken from it; it was used only to cross-check behaviour (schedule timing, defaults). SPAM is
AdamW plus three mechanisms, implemented here on the fork's stock AdamW through `pre_opt_step` (the
final, accumulated + clipped gradient, right before `opt.step()`):

  1. Momentum reset: every `delta_t` steps, zero m and v. Adam's per-param `step` is KEPT, so bias
     correction does not restart -- which is why 3 exists.
  2. Spike-aware clipping: entries with g^2 > theta * v are set to sign(g) * sqrt(theta * v), where v is
     the second moment BEFORE this step's update. Off for the first `grace` steps after each reset
     (v is ~0 there, so every entry would read as a spike).
  3. Cosine warmup: for `warmup` steps after each reset the update is scaled from 0.01 up to 1.

All three schedules are keyed to the GLOBAL step (training step 0 = SPAM's first step), so a fork at
step T is in the same reset/clip/warmup phase as if SPAM had run from the start of training.

Known departures from the authors' release (both shared by every branch, so Δ comparisons are unaffected):
  * The optimizer is torch AdamW, not an HF-style AdamW: eps is added to sqrt(v / bc2) rather than
    sqrt(v), and weight decay is applied before the update rather than after.
  * Warmup scales each group's lr by s for one step and its weight_decay by 1/s, so the decoupled
    decay lr * wd is unchanged (up to fp rounding); the authors scale only the update. Both are
    restored right after `opt.step()`.
  * Dense only: no sparse-momentum masks.

Defaults: theta=5000, delta_t=500, warmup=150 (the paper's settings, also the authors' documented usage);
grace=20 matches the authors' released configuration. (Their launch scripts pass the 1000-step LR warmup
as SPAM's warmup instead; we follow the paper value.)
"""

from __future__ import annotations

import math

from research.baselines.reset import zero_moments

THETA = 5000.0
DELTA_T = 500
WARMUP = 150
GRACE = 20


def warmup_scale(c: int, warmup: int = WARMUP) -> float:
    """Update scale `c` steps after a reset: a half-cosine ramp from 0.01 to 1 over `warmup` steps
    (1 - 0.99 * (1 + cos(pi * c / (warmup + 1))) / 2), 1.0 from c >= warmup."""
    if c >= warmup:
        return 1.0
    return 1.0 - 0.99 * (1.0 + math.cos(math.pi * c / (warmup + 1))) / 2.0


def phase(step: int, *, theta: float = THETA, delta_t: int = DELTA_T, warmup: int = WARMUP, grace: int = GRACE):
    """What SPAM does at global `step`: (reset_now, clip_on, update_scale). Pure. `delta_t=0` disables
    resets (and so warmup)."""
    cur = step + 1  # 1-based count of optimizer steps taken, including this one
    reset_now = delta_t > 0 and cur % delta_t == 0
    clip_on = theta != 0 and cur >= grace and (delta_t == 0 or cur % delta_t >= grace)
    # No warmup before the first reset; after a reset the warmup clock is 0 on the reset step itself.
    scale = warmup_scale(cur % delta_t, warmup) if delta_t > 0 and cur >= delta_t else 1.0
    return reset_now, clip_on, scale


def spike_clip_(opt, theta: float = THETA) -> int:
    """Clip every grad entry with g^2 > theta * v to sign(g) * sqrt(theta * v), in place, using AdamW's
    current `exp_avg_sq`. Params with no grad or no state yet are skipped. Returns the entries clipped."""
    import torch

    n = 0
    with torch.no_grad():
        for group in opt.param_groups:
            for p in group["params"]:
                v = opt.state.get(p, {}).get("exp_avg_sq")
                if p.grad is None or v is None:
                    continue
                g = p.grad
                mask = g.square() > theta * v
                # torch.where, not boolean index_put: same values, no deterministic-mode caveats.
                g.copy_(torch.where(mask, g.sign() * (v * theta).sqrt(), g))
                n += int(mask.sum())
    return n


def as_pre_opt_step(*, theta: float = THETA, delta_t: int = DELTA_T, warmup: int = WARMUP, grace: int = GRACE):
    """A `pre_opt_step(step, ctx)` running SPAM's reset -> clip -> warmup for this step on ctx['opt'].
    Records {reset, clipped, scale} in ctx['spam'] for logging; returns an undo() when it scaled the LR."""

    def pre_opt_step(step, ctx):
        opt = ctx["opt"]
        reset_now, clip_on, scale = phase(step, theta=theta, delta_t=delta_t, warmup=warmup, grace=grace)
        if reset_now:
            zero_moments(opt)
        clipped = spike_clip_(opt, theta) if clip_on else 0
        ctx["spam"] = {"reset": reset_now, "clipped": clipped, "scale": scale}
        if scale == 1.0:
            return None

        saved = [(g["lr"], g["weight_decay"]) for g in opt.param_groups]
        for g in opt.param_groups:
            g["lr"] = g["lr"] * scale
            g["weight_decay"] = g["weight_decay"] / scale

        def undo():
            for g, (lr, wd) in zip(opt.param_groups, saved):
                g["lr"], g["weight_decay"] = lr, wd

        return undo

    return pre_opt_step
