"""LR schedule, applied by `trunk.train_forward` itself at the top of every step (before `pre_step`).

It lives in the training loop, not in a hook, so every path that trains honours it: trunk, resume,
`prime_trunk`, `fork_determinism_gate`, B* (which has no pre_step) and every fork. An LR recipe in a
`pre_step` (spikes.induce `lr_bump`, the cooldown arm) then scales `lr_at(step, cfg)`, the scheduled LR.

    optim:
      lr: 3.0e-4                 # peak LR
      schedule: {kind: cosine, warmup: 100}
      # kind: cosine | wsd. Optional: total (default train.max_steps), min_lr (default 0.1*lr),
      # decay (wsd only: length of the final linear decay, default 0.2*total).

No `schedule` key (the proxy pilot) = constant LR, and the loop is byte-for-byte unchanged.
Torch-free on purpose, so spikes.induce stays torch-free.
"""

from __future__ import annotations

import math

_KEYS = {"kind", "warmup", "total", "min_lr", "decay"}


def lr_at(step: int, cfg: dict) -> float:
    """The scheduled LR at optimizer step `step`. Exactly cfg["optim"]["lr"] when there is no schedule."""
    o = cfg["optim"]
    lr, s = o["lr"], o.get("schedule")
    if not s:
        return lr
    if set(s) - _KEYS or s.get("kind") not in ("cosine", "wsd"):
        raise ValueError(f"bad optim.schedule {s!r}: kind must be cosine|wsd, keys from {sorted(_KEYS)}")
    warmup = s.get("warmup", 0)
    total = s.get("total", cfg["train"]["max_steps"])
    min_lr = s.get("min_lr", 0.1 * lr)
    if step < warmup:
        return lr * (step + 1) / warmup
    if s["kind"] == "cosine":
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * p)) * (lr - min_lr)
    decay = s.get("decay", round(0.2 * total))  # wsd: warmup, stable at lr, then linear to min_lr
    start = total - decay
    if step < start:
        return lr
    return lr - min(1.0, (step - start) / max(1, decay)) * (lr - min_lr)
