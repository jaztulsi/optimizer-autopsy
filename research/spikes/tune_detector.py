"""Tune the online spike detector against induced spikes (known ground truth).

The detector thresholds a statistic over the streaming trunk logs -- a rolling **loss z-score** and/or
a **grad-norm EMA** -- to fire a PRE-spike trigger `t0` BEFORE the loss peak, so the snapshot captures
the poisoned-but-not-yet-exploded state (context §15.2). Everything here is a pure function of the
logged (loss, grad_norm) arrays, so it is CPU-only unit-testable with no torch (see `_selfcheck`).

Scoring against ground truth (context §15.1): a recipe records the KNOWN `inject_step` (the cause). The
spike EVENT is the loss `peak_step`, MEASURED from the trajectory (argmax in a post-inject window) --
derived, never eyeballed. A detection is valid iff the trigger fired AFTER the cause and BEFORE the
explosion, with enough lead:  `inject_step <= t0 < peak_step`  and  `lead = peak_step - t0 >= L`.
A false positive is any trigger in the clean prefix `[warmup, inject_step)`.

DoD (context §15.2): median lead >= `L` at false-positive rate <= `f` for >= 3 of the 4 recipes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorParams:
    """One operating point of `detect_spike`. `tune()` sweeps a grid of these."""

    z: float = 4.0  # fire if rolling loss z-score exceeds this
    win: int = 20  # rolling window for the loss mean/std baseline
    ema_beta: float = 0.9  # grad-norm EMA smoothing
    ema_mult: float = 3.0  # fire if grad_norm exceeds ema_mult * (past EMA)
    warmup: int = 10  # no firing before this step (need a baseline first)
    use_gradnorm: bool = True


def trigger_mask(loss_hist, gradnorm_hist, p: DetectorParams) -> list[bool]:
    """Causal per-step firing decision: at each step t it uses ONLY data in [0, t] (loss z-score vs
    the previous `win` steps; grad-norm vs the EMA of PAST grad-norms). A NaN/Inf loss fires
    immediately. Returns a bool per step."""
    loss = np.asarray(loss_hist, dtype=float)
    n = len(loss)
    mask = [False] * n
    ema = None
    for t in range(n):
        gt = float(gradnorm_hist[t]) if gradnorm_hist is not None else None
        fired = False
        if t >= p.warmup:
            if not np.isfinite(loss[t]):
                fired = True  # NaN/Inf loss = definite spike
            else:
                past = loss[max(0, t - p.win) : t]
                if len(past) >= 2:
                    mu, sd = past.mean(), past.std()
                    if sd > 0 and (loss[t] - mu) / sd > p.z:
                        fired = True
                if p.use_gradnorm and ema is not None and gt is not None and np.isfinite(gt):
                    if gt > p.ema_mult * ema:
                        fired = True
        mask[t] = fired
        if gt is not None and np.isfinite(gt):  # update EMA AFTER testing t (keeps the test causal)
            ema = gt if ema is None else p.ema_beta * ema + (1 - p.ema_beta) * gt
    return mask


def first_trigger(loss_hist, gradnorm_hist, p: DetectorParams) -> int | None:
    """The pre-spike trigger `t0`: first step `trigger_mask` fires, else None."""
    for t, fired in enumerate(trigger_mask(loss_hist, gradnorm_hist, p)):
        if fired:
            return t
    return None


def spike_occurred(losses, inject_step: int, *, rise: float = 2.0, window: int = 25) -> dict:
    """MEASURE the spike event from the loss trajectory (machine-checkable ground truth, not eyeball).

    baseline = median loss over the clean prefix `[0, inject_step)`; peak = max loss in
    `[inject_step, inject_step+window)`. `occurred` iff peak/baseline >= `rise` (or a NaN/Inf appears in
    the window -- a definite explosion). Returns {occurred, peak_step, peak_loss, baseline, ratio}.
    """
    L = np.asarray(losses, dtype=float)
    n = len(L)
    pre = L[:inject_step]
    base = float(np.nanmedian(pre)) if len(pre) else float(L[0])
    lo, hi = inject_step, min(n, inject_step + window)
    seg = L[lo:hi]
    if len(seg) == 0:
        return {"occurred": False, "peak_step": inject_step, "peak_loss": base, "baseline": base, "ratio": 0.0}
    if not np.all(np.isfinite(seg)):  # NaN/Inf = explosion; peak = first non-finite step
        pk = lo + int(np.argmax(~np.isfinite(seg)))
        return {"occurred": True, "peak_step": pk, "peak_loss": float("inf"), "baseline": base, "ratio": float("inf")}
    pk = lo + int(np.argmax(seg))
    peak = float(L[pk])
    ratio = peak / base if base > 0 else float("inf")
    return {"occurred": ratio >= rise, "peak_step": pk, "peak_loss": peak, "baseline": base, "ratio": ratio}


def score_spike(losses, gradnorms, inject_step: int, p: DetectorParams, *, min_lead: int) -> dict:
    """Score one labeled run: did the detector fire in the valid pre-spike window with enough lead,
    and how many clean-prefix steps false-fired. See the module docstring for the validity rule."""
    occ = spike_occurred(losses, inject_step)
    peak = occ["peak_step"]
    mask = trigger_mask(losses, gradnorms, p)

    clean = [mask[t] for t in range(p.warmup, min(inject_step, len(mask)))]
    fp_steps, clean_steps = int(sum(clean)), len(clean)

    t0 = next((t for t, fired in enumerate(mask) if fired), None)
    detected = bool(occ["occurred"] and t0 is not None and inject_step <= t0 < peak and (peak - t0) >= min_lead)
    return {
        "occurred": occ["occurred"],
        "peak_step": peak,
        "t0": t0,
        "detected": detected,
        "lead": (peak - t0) if detected else None,
        "fp_steps": fp_steps,
        "clean_steps": clean_steps,
    }


def _aggregate(labeled_runs, params: DetectorParams, min_lead: int) -> dict:
    leads, detected, fp_steps, fp_total = [], 0, 0, 0
    for r in labeled_runs:
        s = score_spike(r["losses"], r.get("gradnorms"), r["inject_step"], params, min_lead=min_lead)
        if s["detected"]:
            detected += 1
            leads.append(s["lead"])
        fp_steps += s["fp_steps"]
        fp_total += s["clean_steps"]
    return {
        "n_detected": detected,
        "median_lead": float(np.median(leads)) if leads else 0.0,
        "fp_rate": (fp_steps / fp_total) if fp_total else 0.0,
    }


def default_grid() -> list[DetectorParams]:
    return [DetectorParams(z=z, ema_mult=em) for z in (3.0, 4.0, 5.0, 6.0) for em in (2.5, 3.0, 4.0, 5.0)]


def tune(labeled_runs, *, L: int, f: float, grid=None) -> dict | None:
    """Sweep the threshold grid; keep operating points that pass the DoD (>=3/4 detected, median
    lead >= L, FP rate <= f); return the DoD-passing point with the LARGEST median lead, or None."""
    best = None
    for p in grid or default_grid():
        agg = _aggregate(labeled_runs, p, L)
        need = math.ceil(0.75 * len(labeled_runs))
        passed = agg["n_detected"] >= need and agg["median_lead"] >= L and agg["fp_rate"] <= f
        if passed and (best is None or agg["median_lead"] > best["median_lead"]):
            best = {"params": p, **agg, "need": need, "passed": True}
    return best


def dod_check(labeled_runs, params: DetectorParams, *, L: int, f: float) -> dict:
    """Evaluate a fixed operating point against the Task 8 DoD. Returns the metrics + pass/fail."""
    agg = _aggregate(labeled_runs, params, L)
    need = math.ceil(0.75 * len(labeled_runs))
    passed = agg["n_detected"] >= need and agg["median_lead"] >= L and agg["fp_rate"] <= f
    return {"passed": passed, "need": need, **agg}


_PEAK_LAG = 5  # loss peaks this many steps AFTER injection (the 1/(1-beta2) poison-propagation lag)


def _synth_run(inject_step: int, *, n: int = 70, seed: int = 0, spike: bool = True):
    """A synthetic trunk log: smooth decay + noise; if `spike`, the grad-norm bursts AT injection
    (early t0) while the loss ramps to a peak `_PEAK_LAG` steps later -- the lag that creates the
    lead the detector must exploit. For the CPU-only self-check (no torch needed)."""
    rng = np.random.default_rng(seed)
    steps = np.arange(n)
    loss = 3.0 + 4.0 * np.exp(-steps / 25.0) + rng.normal(0, 0.03, n)
    grad = 1.0 + rng.normal(0, 0.05, n).clip(-0.2, 0.2)
    if spike:
        for k in range(0, 13):  # loss bump peaking at inject + _PEAK_LAG
            t = inject_step + k
            if t < n:
                loss[t] += 9.0 * math.exp(-((k - _PEAK_LAG) ** 2) / (2 * 2.0**2))
        for k in range(0, 3):  # grad-norm burst right at the bad step (the early detectable signal)
            t = inject_step + k
            if t < n:
                grad[t] += 15.0
    return {"losses": loss.tolist(), "gradnorms": grad.tolist(), "inject_step": inject_step}


def _selfcheck() -> None:
    """CPU-only (numpy) verification of the scoring + tuning machinery -- exercises the real code
    paths, not a read-through: a broken z-score, EMA, lead rule, or FP count fails an assert here."""
    L, f = 2, 0.05
    runs = [_synth_run(inj, seed=i) for i, inj in enumerate((40, 45, 38, 50))]

    # Ground truth is measured, not assumed: each spike is detected and peaks _PEAK_LAG after inject.
    for r in runs:
        occ = spike_occurred(r["losses"], r["inject_step"])
        assert occ["occurred"], f"synthetic spike at {r['inject_step']} not measured as a spike"
        assert occ["peak_step"] == r["inject_step"] + _PEAK_LAG, (r["inject_step"], occ["peak_step"])

    # A clean run must NOT false-fire in its prefix (guards the FP path).
    clean = _synth_run(40, seed=99, spike=False)
    p0 = DetectorParams()
    assert first_trigger(clean["losses"], clean["gradnorms"], p0) is None, "detector false-fired on a clean run"

    # Tuning finds a DoD-passing operating point; re-checking it agrees.
    best = tune(runs, L=L, f=f)
    assert best is not None and best["passed"], "tune found no DoD-passing operating point"
    assert best["n_detected"] >= 3, best
    verify = dod_check(runs, best["params"], L=L, f=f)
    assert verify["passed"] and verify["median_lead"] >= L and verify["fp_rate"] <= f, verify
    print(
        f"tune_detector selfcheck OK: {best['n_detected']}/4 detected, "
        f"median lead={best['median_lead']:.0f} (>= {L}), FP={best['fp_rate']:.3f} (<= {f})"
    )


if __name__ == "__main__":
    _selfcheck()
