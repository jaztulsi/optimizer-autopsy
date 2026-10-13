"""Baseline: skip-step -- skip the OPTIMIZER UPDATE itself over the spike window.

The practice of dropping an update when a step looks bad (e.g. PaLM, Chowdhery et al. 2022, restarted
before a spike and skipped ~200-500 batches; many trainers skip any step with a non-finite or outsized
grad). Distinct from `skip.py`, which keeps the update but swaps the corrupted batch for the clean one:
here the batch is consumed (forward + backward run, the data cursor advances as usual, so the stream
stays aligned) but `opt.step()` never fires -- w, m, v and Adam's step count are left exactly as they
were. For non-batch spikes (`lr_bump`, `tiny_eps`, `precision`) this is the only "skip" that does
anything, since there is no bad batch to swap.

Two timings, same hook (`as_pre_opt_step(start, width)` skips [start, start + width)):
  * Bskipstep, oracle: start = the known `inject_step`. The most favourable version (a real trainer only
    sees the symptom), which is the right side to err on for a comparison we want to beat.
  * Bskipstep_det, detector-timed: start = the detector trigger t0, same `width` budget. What a real
    trainer could actually do. The detector is causal and the fork replays the trunk bitwise, so t0 from
    `first_trigger` on the trunk's logs is exactly when it would fire live inside the fork.
"""

from __future__ import annotations


def as_pre_opt_step(inject_step: int, width: int):
    """A `pre_opt_step(step, ctx)` that skips the optimizer update for every step in the window."""

    def pre_opt_step(step, ctx):
        if inject_step <= step < inject_step + width:
            ctx["skip_update"] = True

    return pre_opt_step
