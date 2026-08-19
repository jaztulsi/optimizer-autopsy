"""Baseline Bs (data-side half): skip the offending batch around the spike.

In the Task-9 fork replay, the `corrupt_batch` spike hook sets `ctx['batch']=garbage` inside its
window; skip runs AFTER it (see `fork.compose` ordering) and resets `ctx['batch']` back to the CLEAN,
ALIGNED slot -- `None` makes `train_forward` fetch `get_batch` for that same `gstep`, so the data
stream is never shifted ('skip to the next batch' is forbidden). For non-batch spikes (`lr_bump`,
etc.) there is no bad batch, so skip is a no-op and the clip half of Bs does the work.
"""

from __future__ import annotations


def as_pre_step(inject_step: int, width: int):
    """Within `[inject_step, inject_step+width)`, force the clean aligned batch (undo a corruption)."""

    def pre_step(step, ctx):
        if inject_step <= step < inject_step + width:
            ctx["batch"] = None  # -> train_forward fetches the aligned clean slot for this gstep

    return pre_step
