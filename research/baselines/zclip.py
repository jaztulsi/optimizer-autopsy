"""Baseline: ZClip -- adaptive z-score gradient-norm clipping (Kumar et al., arXiv 2504.02507).

Reimplemented following bluorion-com/ZClip (Apache-2.0), `zclip/zclip.py` (`ZClip.step`, non-FSDP path).
ZClip keeps an EMA of the total gradient norm's mean and variance; when a step's norm has z-score >
z_thresh it clips to
    mean + z_thresh * std / (z / z_thresh)          ("adaptive_scaling", the default clip_option)
capped at `max_grad_norm`, and feeds the clipped value (not the spike) back into the EMA. The first
`warmup` calls only buffer norms (and apply plain max_grad_norm clipping), then seed mean/var from them.

ZClip REPLACES global clipping, so its branch runs with `optim.grad_clip = 0` (see `apply_to_cfg`) and
reads the raw total norm from ctx["grad_norm"]; `max_norm` takes over the trunk's clip value.

Fork protocol -- shadow priming. A kill-test fork starts at the detector trigger t0, a few steps before
the spike: far short of ZClip's 25-step warmup, so a cold ZClip would just be global clipping. Instead
attach `observer()` to the trunk's `on_step`: every trunk step runs ZClip's exact statistics update on
that step's raw norm WITHOUT touching gradients, and the fork starts from `state_dict()` at t0. The one
approximation: the trunk itself ran without ZClip, which only matters on steps ZClip would have clipped
(rare on a clean pre-spike trunk).

Defaults are the reference's (alpha=0.97, z_thresh=2.5, max_grad_norm=1.0, eps=1e-6, warmup_steps=25,
mode="zscore", clip_option="adaptive_scaling", clip_factor=1.0, skip_update_on_spike=False). Only the
default mode is implemented; "percentile" / clip_option="mean" are not.
"""

from __future__ import annotations

import copy

ALPHA = 0.97
Z_THRESH = 2.5
MAX_NORM = 1.0
EPS = 1e-6
WARMUP = 25


def apply_to_cfg(cfg: dict) -> dict:
    """Return a cfg with the global clip off (shallow copy, like `clip.apply_to_cfg`): ZClip replaces it."""
    return {**cfg, "optim": {**cfg.get("optim", {}), "grad_clip": 0.0}}


class ZClip:
    """Stateful ZClip. Call as a `pre_opt_step(step, ctx)` in a fork; `observe(norm)` / `observer()` to
    prime it on the trunk. `state_dict()` / `load_state_dict()` carry the primed statistics into a fork."""

    def __init__(
        self,
        *,
        alpha: float = ALPHA,
        z_thresh: float = Z_THRESH,
        max_norm: float | None = MAX_NORM,
        eps: float = EPS,
        warmup: int = WARMUP,
        clip_factor: float = 1.0,
        skip_update_on_spike: bool = False,
    ):
        self.alpha, self.z_thresh, self.max_norm, self.eps = alpha, z_thresh, max_norm, eps
        self.warmup, self.clip_factor, self.skip_update_on_spike = warmup, clip_factor, skip_update_on_spike
        self.buffer: list[float] = []
        self.initialized = False
        self.mean: float | None = None
        self.var: float | None = None

    # -- the reference's statistics, same order of operations ---------------------------------------

    def _clip_val(self, norm: float) -> float | None:
        std = self.var**0.5
        z = (norm - self.mean) / (std + self.eps)
        if z > self.z_thresh:
            eta = z / self.z_thresh  # larger outliers get a tighter threshold
            return (self.mean + (self.z_thresh * std) / eta) * self.clip_factor
        return None

    def _update(self, norm: float) -> float | None:
        """Advance the statistics by one step on raw total norm `norm`; return the norm to clip to
        (None = leave the gradient alone). Shared by the fork hook and trunk-side priming."""
        if not self.initialized:
            self.buffer.append(norm)
            if len(self.buffer) >= self.warmup:
                self.mean = sum(self.buffer) / len(self.buffer)
                self.var = sum((x - self.mean) ** 2 for x in self.buffer) / len(self.buffer)
                self.initialized = True
                self.buffer = []
            return self.max_norm  # warmup: plain max-norm clipping

        clip_val = self._clip_val(norm)
        effective = clip_val if clip_val is not None else norm
        if self.max_norm is not None:
            effective = min(effective, self.max_norm)
        if not (clip_val is not None and self.skip_update_on_spike):
            ema_in = clip_val if clip_val is not None else norm
            self.mean = self.alpha * self.mean + (1 - self.alpha) * ema_in
            self.var = self.alpha * self.var + (1 - self.alpha) * (ema_in - self.mean) ** 2  # NEW mean, as ref
        return effective

    # -- fork / trunk entry points ---------------------------------------------------------------

    def __call__(self, step, ctx):
        """`pre_opt_step`: clip ctx['model']'s grads in place to ZClip's threshold for this step."""
        norm = float(ctx["grad_norm"])
        target = self._update(norm)
        clipped = target is not None and norm > target
        if clipped:
            coef = target / (norm + 1e-6)  # the reference's apply_in_place_clipping
            for p in ctx["model"].parameters():
                if p.grad is not None:
                    p.grad.mul_(coef)
        ctx["zclip"] = {"norm": norm, "target": target, "clipped": clipped, "mean": self.mean}

    def observe(self, norm: float) -> None:
        """Shadow step: advance the statistics on a trunk step's raw norm, touching no gradient."""
        self._update(float(norm))

    def observer(self):
        """An `on_step(step, info)` that primes this ZClip on the trunk (info['grad_norm'] is pre-clip)."""
        return lambda step, info: self.observe(info["grad_norm"])

    def state_dict(self) -> dict:
        return {k: copy.copy(getattr(self, k)) for k in ("buffer", "initialized", "mean", "var")}

    def load_state_dict(self, sd: dict) -> None:
        for k, v in sd.items():
            setattr(self, k, copy.copy(v))


def as_pre_opt_step(state: dict | None = None, **params) -> ZClip:
    """A fresh ZClip `pre_opt_step`, optionally starting from a primed `state_dict()` (copied, so one
    primed state can seed many branches). One instance per fork: it is stateful."""
    z = ZClip(**params)
    if state is not None:
        z.load_state_dict(state)
    return z
