"""Baseline Brollback: rollback-and-skip at matched compute -- the production default (PaLM / OPT).

PaLM (Chowdhery et al. 2022, arXiv 2204.02311) "re-started training from a checkpoint roughly 100 steps
before the spike started, and skipped roughly 200-500 data batches"; OPT-175B (Zhang et al. 2022, arXiv
2205.01068) likewise restarted from earlier checkpoints when loss diverged. It is the answer a reviewer
expects us to beat, since the fork harness already keeps snapshots.

What the branch models, on a fork from the battery snapshot at `inject_step`:
  1. The trainer runs the spike: steps inject_step .. t0, where the detector fires at t0 = inject_step + L.
     Those L + 1 steps are wasted compute. They are identical to B0's first L + 1 steps (same snapshot,
     same hooks, bitwise replay), so the branch does not re-run them -- it only charges them.
  2. It rolls back to the newest clean checkpoint, the fork snapshot itself (w, m, v, Adam step at
     `inject_step`).
  3. It resumes on data it has not seen: post-rollback step s reads step s + skip's slots. The default
     skip = L + 1 continues the data cursor exactly where it was at detection (no batch read twice);
     pass a larger skip for PaLM-style extra skipping.
  4. Matched compute: it trains N - (L + 1) steps after rollback, so its total forward/backward compute
     equals the N steps every other branch gets. This is prereg-v2's cost model ("rollback cost is
     counted as L + 1 recomputed steps").

The spike does NOT recur after rollback (its recipe hook is not replayed). That is the most favourable
assumption for rollback -- PaLM found replaying from an earlier state without the skipped batches gave no
spike -- and the right side to err on for a baseline we want to beat.

CONFOUND, by design: rollback-and-skip changes the data order, so it is a practitioner baseline, never
`B*` (the clean, data-aligned counterfactual). Report it labelled as data-shifting. Its train losses are
on different batches from the other branches', so compare branches on the held-out metric.
"""

from __future__ import annotations


def plan(inject_step: int, detector_t0: int, skip: int | None = None) -> tuple[int, int]:
    """(wasted_steps, data_skip) for a rollback triggered at `detector_t0` to the snapshot at `inject_step`.
    Raises if the detector fired before the injection (nothing to roll back) or if `skip` would re-read
    batches the trainer already consumed."""
    wasted = detector_t0 - inject_step + 1  # inject_step .. t0 inclusive: the step at t0 ran (its loss fired)
    if wasted < 1:
        raise ValueError(f"detector_t0={detector_t0} precedes inject_step={inject_step}: no spike to roll back")
    skip = wasted if skip is None else skip
    if skip < wasted:
        raise ValueError(f"skip={skip} < {wasted} wasted steps: the trainer would re-read batches it already saw")
    return wasted, skip


def as_pre_step(skip: int):
    """A `pre_step(step, ctx)` that shifts the data cursor by `skip` steps' worth of batches, every step."""

    def pre_step(step, ctx):
        ctx["data_offset"] = skip

    return pre_step
