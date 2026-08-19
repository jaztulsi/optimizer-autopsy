"""Baseline Bg: naive full optimizer-state reset (m, v -> 0) -- the blunt upper bound.

The blunt version of our repair operator; the localizer is what we claim beats this. In the Task-9
kill-test the spike is REPLAYED forward inside the fork, so the reset fires ONCE right after the
spike window (zeroing the freshly poisoned moments) -- not at fork start, where the state is still
clean and zeroing it would prove nothing.
"""

from __future__ import annotations


def zero_moments(opt) -> int:
    """Zero every AdamW moment (m=`exp_avg`, v=`exp_avg_sq`) in place. Returns how many were zeroed."""
    n = 0
    for st in opt.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key].zero_()
                n += 1
    return n


def as_pre_step(at_step: int):
    """A `pre_step(step, ctx)` that performs the global reset once, when `step == at_step`."""

    def pre_step(step, ctx):
        if step == at_step:
            zero_moments(ctx["opt"])

    return pre_step
