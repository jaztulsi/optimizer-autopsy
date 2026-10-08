"""Baseline: AdaGC -- tensor-wise adaptive gradient clipping (Wang et al., arXiv 2502.11034, ICML 2026).

Reimplemented following lishuai-97/AdaGC (Apache-2.0),
`megatron-llama.patch` -> `megatron/optimizer/clip_grads.py::AdaGC`
(clip_mode="AdaGC", the unsharded tensor-wise variant). NOTE: AdaGC clips each tensor against an EMA of
ITS OWN past gradient norm -- not against the weight norm, as this file's earlier stub said (that is
AGC, Brock et al. 2021). Per parameter tensor i:
    h_i = min(1, ratio * gamma_i / ||g_i||);   g_i <- h_i * g_i;   gamma_i <- beta * gamma_i + (1-beta) * ||h_i g_i||
For the first `start_clip_steps` calls it instead applies plain global-norm clipping (`max_norm`) and
seeds each gamma_i with the MINIMUM post-clip tensor norm seen in that window.

AdaGC REPLACES global clipping, so its branch runs with `optim.grad_clip = 0` (see `apply_to_cfg`);
`max_norm` (warm-up phase only) takes over the trunk's clip value.

Fork protocol -- shadow priming, as for ZClip: a fork starts a few steps before the spike, far short of
the 100-step warm-up, so attach `observer()` to the trunk's `on_step` and start the fork from
`state_dict()`. The trunk's grads at on_step are AFTER its own global clip (coef c); priming rescales
them by 1/c where AdaGC would have seen raw norms (the adaptive phase) and keeps them as-is where it
would have seen globally clipped norms (the warm-up phase). The one approximation: the trunk itself ran
without AdaGC, which only matters on steps AdaGC would have clipped.

Defaults are the reference's (clip_ratio=1.04, beta=0.99, eps=1e-6, start_clip_steps=100), matching
its example launch (`--clip-ratio 1.04 --clip-beta 0.99 --clip-grad 1.0`). State is keyed by parameter
NAME (forks build fresh params; tied weights appear once via `named_parameters()`).
"""

from __future__ import annotations

RATIO = 1.04
BETA = 0.99
EPS = 1e-6
START = 100
MAX_NORM = 1.0


def apply_to_cfg(cfg: dict) -> dict:
    """Return a cfg with the global clip off (shallow copy, like `clip.apply_to_cfg`): AdaGC replaces it."""
    return {**cfg, "optim": {**cfg.get("optim", {}), "grad_clip": 0.0}}


class AdaGC:
    """Stateful AdaGC. Call as a `pre_opt_step(step, ctx)` in a fork; `observer()` primes it on the trunk.
    `state_dict()` / `load_state_dict()` carry {steps, gamma by param name} into a fork."""

    def __init__(
        self,
        *,
        ratio: float = RATIO,
        beta: float = BETA,
        eps: float = EPS,
        start_clip_steps: int = START,
        max_norm: float = MAX_NORM,
    ):
        self.ratio, self.beta, self.eps, self.start, self.max_norm = ratio, beta, eps, start_clip_steps, max_norm
        self.steps = 0
        self.gamma: dict = {}  # param name -> 0-dim fp32 tensor (on the param's device)

    def _step(self, model, total_norm: float, *, apply: bool, seen_coef: float = 1.0) -> int:
        """One AdaGC call. `apply=False` (priming) updates gamma without touching grads; `seen_coef` is
        the global-clip coefficient already applied to the grads we see (trunk on_step), 1.0 in a fork.
        Returns how many tensors were adaptively clipped."""
        import torch

        self.steps += 1
        n_clipped = 0
        with torch.no_grad():
            if self.steps <= self.start:  # warm-up: global clip, seed gamma with the min post-clip norm
                coef = self.max_norm / (total_norm + self.eps)
                if apply and coef < 1.0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(coef)
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    n = torch.linalg.vector_norm(p.grad.float())
                    if not apply:  # priming: raw norm, then AdaGC's own warm-up clip
                        n = n / seen_coef * min(1.0, coef)
                    g = self.gamma.get(name)
                    g = None if g is None else g.to(n.device)
                    self.gamma[name] = n if g is None else torch.minimum(g, n)
                return 0

            for name, p in model.named_parameters():  # adaptive phase: per-tensor, against own EMA
                if p.grad is None:
                    continue
                n = torch.linalg.vector_norm(p.grad.float()) / seen_coef  # raw norm
                g = self.gamma.setdefault(name, n).to(n.device)
                thresh = self.ratio * g
                over = n > thresh
                if apply:
                    # x1.0 on tensors under threshold is a bitwise no-op.
                    p.grad.mul_(torch.where(over, thresh / (n + self.eps), torch.ones_like(n)).to(p.grad.dtype))
                    n_clipped += int(over)
                self.gamma[name] = g * self.beta + torch.where(over, thresh, n) * (1.0 - self.beta)
        return n_clipped

    def __call__(self, step, ctx):
        """`pre_opt_step`: clip ctx['model']'s grads in place (global in warm-up, tensor-wise after)."""
        n = self._step(ctx["model"], float(ctx["grad_norm"]), apply=True)
        ctx["adagc"] = {"steps": self.steps, "clipped_tensors": n, "warmup": self.steps <= self.start}

    def observer(self, trunk_clip: float):
        """An `on_step(step, info)` that primes this AdaGC on a trunk run with global clip `trunk_clip`
        (the trunk cfg's optim.grad_clip; 0 = none). Grads at on_step are post-clip; this undoes that."""

        def on_step(step, info):
            raw = float(info["grad_norm"])
            seen = min(1.0, trunk_clip / (raw + 1e-6)) if trunk_clip and trunk_clip > 0 else 1.0  # torch's coef
            self._step(info["model"], raw, apply=False, seen_coef=seen)

        return on_step

    def state_dict(self) -> dict:
        return {"steps": self.steps, "gamma": {k: v.detach().cpu().clone() for k, v in self.gamma.items()}}

    def load_state_dict(self, sd: dict, device=None) -> None:
        self.steps = int(sd["steps"])
        self.gamma = {k: v.clone() if device is None else v.to(device) for k, v in sd["gamma"].items()}


def as_pre_opt_step(state: dict | None = None, device=None, **params) -> AdaGC:
    """A fresh AdaGC `pre_opt_step`, optionally from a primed `state_dict()` (copied). One per fork."""
    a = AdaGC(**params)
    if state is not None:
        a.load_state_dict(state, device=device)
    return a
