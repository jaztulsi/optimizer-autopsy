"""Baselines Bg / Bvreset: naive global optimizer-state reset -- the blunt upper bounds.

Bg zeroes both moments (m, v -> 0); Bvreset zeroes only v and keeps m. They are the blunt versions of
our repair operator (and of its v-only arm, Bv); the localizer is what we claim beats them. In the
Task-9 kill-test the spike is REPLAYED forward inside the fork, so the reset fires ONCE right after the
spike window (zeroing the freshly poisoned moments) -- not at fork start, where the state is still
clean and zeroing it would prove nothing.

Bvreset caveat, by design: with v=0 and m kept, the next step's v is just (1-beta2) * g^2 while Adam's
bias correction (step count kept) no longer divides it back up, so the first post-reset updates are
~1/sqrt(1-beta2) larger than usual (~4.5x at beta2=0.95). That is the naive baseline as specified;
SPAM's post-reset warmup exists to tame exactly this.
"""

from __future__ import annotations

BOTH = ("exp_avg", "exp_avg_sq")
V_ONLY = ("exp_avg_sq",)


def zero_moments(opt, keys=BOTH) -> int:
    """Zero the named AdamW moments (m=`exp_avg`, v=`exp_avg_sq`) of every param in place; Adam's per-param
    `step` is kept. Returns how many tensors were zeroed."""
    n = 0
    for st in opt.state.values():
        for key in keys:
            if key in st:
                st[key].zero_()
                n += 1
    return n


def as_pre_step(at_step: int, keys=BOTH):
    """A `pre_step(step, ctx)` that performs the global reset of `keys` once, when `step == at_step`."""

    def pre_step(step, ctx):
        if step == at_step:
            zero_moments(ctx["opt"], keys)

    return pre_step
