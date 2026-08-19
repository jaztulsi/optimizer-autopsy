"""Baseline Bs (magnitude half): global gradient-norm clipping during the fork.

`train_forward` reads `cfg['optim']['grad_clip']` ONCE and clips every step, so 'clip' is a cfg-level
setting, not a `pre_step` (grads don't exist yet when `pre_step` fires at the top of the step). This
returns a cfg copy with the clip norm set -- the branch runs under it. For `lr_bump` this is the
load-bearing half of Bs (skip is a no-op there).
"""

from __future__ import annotations


def apply_to_cfg(cfg: dict, max_norm: float) -> dict:
    """Return a cfg with `optim.grad_clip = max_norm` (shallow copy; only `optim` is replaced, so the
    caller's cfg is untouched and every other optim key is preserved)."""
    return {**cfg, "optim": {**cfg.get("optim", {}), "grad_clip": float(max_norm)}}
