"""Held-out evaluation for forks: Δ = val_loss(branch) - val_loss(B*), measured on val slots that never
overlap the localizer's probe slots (PLAN_V8 §6.2 / §6.10, fixes improvement.md A1: the battery's `final` is
the last TRAINING loss).

Slots are `data.get_batch("val", slot, ...)` indices. get_batch wraps modulo the val length, so "slot >= 1000"
is not held out: the proxy val split is 0.005 * 100M ~ 500k tokens, i.e. ~30 slots of 64 x 256 before the
first wrap. `heldout_steps` picks the slots that are both pre-wrap and token-disjoint from the probe.

`horizon_probe` is an `on_step` observer for `run_branch` / `run_fork`: no grad, no RNG, train mode restored,
so it never changes the trajectory (tests/test_autopsy.py checks this bitwise on CPU). Every branch that is
compared must carry the same probe.
"""

from __future__ import annotations

PROBE_STEPS = range(4)  # val slots reserved for the localizer probe (PLAN_V8 §6.10)


def starts(n_tokens: int, steps, batch: int, block: int) -> list[int]:
    """Row start offsets that `get_batch` reads for `steps` (same formula, prepare.py:153-154). Each row
    touches tokens [s, s + block + 1)."""
    span = n_tokens - block - 1
    if span <= 0:
        raise ValueError(f"val shard too small: {n_tokens} tokens for block {block}")
    return [((st * batch * block) % span + i * block) % span for st in steps for i in range(batch)]


def overlaps(a: list[int], b: list[int], block: int) -> bool:
    """True if any row window in `a` shares a token with any row window in `b` (windows are block+1 long)."""
    import bisect

    b = sorted(b)
    for s in a:
        i = bisect.bisect_left(b, s)
        if (i < len(b) and b[i] - s < block + 1) or (i > 0 and s - b[i - 1] < block + 1):
            return True
    return False


def _n_val(data_dir: str) -> int:
    from research.data.prepare import _memmap

    return len(_memmap(data_dir, "val"))


def heldout_steps(data_dir: str, batch: int, block: int, probe_steps=PROBE_STEPS, probe_batch=None) -> list[int]:
    """Val slots before the first modulo wrap whose windows are token-disjoint from the probe slots'."""
    n = _n_val(data_dir)
    probe = starts(n, probe_steps, probe_batch or batch, block)
    per_slot = batch * block
    return [s for s in range((n - block - 1) // per_slot) if not overlaps(starts(n, [s], batch, block), probe, block)]


def assert_disjoint(data_dir: str, steps, batch: int, block: int, probe_steps=PROBE_STEPS, probe_batch=None) -> None:
    """Raise if the eval slots `steps` share a token with the probe slots (run before trusting any Δ)."""
    n = _n_val(data_dir)
    if overlaps(starts(n, steps, batch, block), starts(n, probe_steps, probe_batch or batch, block), block):
        raise ValueError(f"eval slots overlap the probe slots {list(probe_steps)} on a {n}-token val split")


def val_loss(model, data_dir: str, steps, batch: int) -> float:
    """Mean held-out loss over val slots `steps` (equal tokens per slot, so this is the token mean)."""
    import torch

    from research.data.prepare import get_batch

    steps = list(steps)
    if not steps:
        raise ValueError("val_loss needs at least one slot")
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    total = 0.0
    try:
        with torch.no_grad():
            for s in steps:
                x, y = get_batch("val", s, batch, model.cfg.block_size, data_dir, dev)
                total += model(x, y)[1].item()
    finally:
        model.train(was_training)
    return total / len(steps)


def horizon_probe(data_dir: str, steps, batch: int, horizons, out: dict):
    """`on_step(step, info)` that records `out[step] = val_loss(...)` after the update at each step in
    `horizons` (absolute trunk steps; end of schedule = max_steps - 1)."""
    steps, horizons = list(steps), set(horizons)

    def on_step(step, info):
        if step in horizons:
            out[step] = val_loss(info["model"], data_dir, steps, batch)

    return on_step
