"""Natural spikes: pre-registered stress presets, an offline spike finder, and the search driver.

The reviewer's objection to induced spikes is answered with spikes nothing injects: proxy trunks run under
hyperparameters the literature ties to real instabilities (RESEARCH.md §2) -- a hot constant LR (Wortsman
et al. 2023: small models reproduce large-scale instabilities at high LR), a slow second moment beta2=0.999
(Bai et al. 2026: v decouples from g^2), and no gradient clipping. Every preset x seed is one capture run
(capture.py, one process each); whatever spikes appear are found offline by `find_spikes` with rules fixed
here BEFORE any run (PLAN V7 §4.2): the first qualifying spike of each run, in seed order, enters K2.

Seeds change the initialization only -- `get_batch` is a pure function of the step, so every seed reads
the same data in the same order. `alignment_test` uses that: spikes that line up across seeds point at the
data (OLMo 2's repeated n-grams); spikes that don't point at the optimizer dynamics.

Run on the AMD GPU:  python -m research.spikes.natural --search      (never on a laptop)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

# Pre-registered (PLAN V7 §4.2). 8 presets x 5 seeds x 3000 steps = 120k proxy steps (~2 P100-h).
LR_MULTS = (5, 10)  # x the proxy LR 3e-4, held constant (the proxy config has no warmup/decay)
BETA2S = (0.95, 0.999)
CLIPS = (1.0, 0.0)
STRESS = {
    f"lr{m}x_b2{b2}_clip{c:g}": {"optim.lr": 3.0e-4 * m, "optim.betas": [0.9, b2], "optim.grad_clip": c}
    for m, b2, c in itertools.product(LR_MULTS, BETA2S, CLIPS)
}
SEEDS = (0, 1, 2, 3, 4)
STEPS = 3000
BATCH = 16

# find_spikes rule (pre-registered): after `warmup`, a step is spiky if its loss is >= `rise` x the trailing
# median AND >= `z` robust (MAD) z-scores above it, or non-finite. Spiky steps <= `merge` apart form one event.
WARMUP, WIN, RISE, Z, MERGE, RECOVER = 100, 50, 1.3, 6.0, 3, 100


def find_spikes(
    losses,
    *,
    warmup: int = WARMUP,
    win: int = WIN,
    rise: float = RISE,
    z: float = Z,
    merge: int = MERGE,
    recover: int = RECOVER,
) -> list[dict]:
    """Find natural spike events in a loss trace (pure numpy, causal baseline).

    Each event: {start, peak_step, peak_loss, baseline, ratio, malignant}. `baseline` is the trailing
    median before `start`; `malignant` (K2-V2's restart tier) = the loss never returns below 1.1 x
    baseline within `recover` steps of the peak, or the run went non-finite.
    """
    L = np.asarray(losses, dtype=float)
    flagged = []
    for t in range(max(warmup, win), len(L)):
        past = L[t - win : t]
        past = past[np.isfinite(past)]
        if len(past) < 2:
            continue
        base = float(np.median(past))
        mad = 1.4826 * float(np.median(np.abs(past - base)))
        if not np.isfinite(L[t]):
            flagged.append((t, base))
        elif L[t] >= rise * base and (L[t] - base) > z * max(mad, 1e-12):
            flagged.append((t, base))

    events: list[list] = []
    for t, base in flagged:
        if events and t - events[-1][-1][0] <= merge:
            events[-1].append((t, base))
        else:
            events.append([(t, base)])

    out = []
    for ev in events:
        start, base = ev[0]
        end = ev[-1][0]
        seg = L[start : end + 1]
        if not np.all(np.isfinite(seg)):
            pk = start + int(np.argmax(~np.isfinite(seg)))
            out.append(
                {
                    "start": start,
                    "peak_step": pk,
                    "peak_loss": float("inf"),
                    "baseline": base,
                    "ratio": float("inf"),
                    "malignant": True,
                }
            )
            continue
        pk = start + int(np.argmax(seg))
        after = L[pk + 1 : pk + 1 + recover]
        recovered = bool(np.any(np.isfinite(after) & (after < 1.1 * base)))
        out.append(
            {
                "start": start,
                "peak_step": pk,
                "peak_loss": float(L[pk]),
                "baseline": base,
                "ratio": float(L[pk] / base),
                "malignant": not recovered,
            }
        )
    return out


def alignment_test(
    starts_by_seed: list, n_steps: int, *, tol: int = 5, warmup: int = WARMUP, n_perm: int = 2000, seed: int = 0
) -> dict:
    """Do natural spikes hit the SAME steps across seeds more than chance? Statistic: cross-seed pairs of
    spike starts within `tol` steps. Null: each seed keeps its spike count, times redrawn uniformly in
    [warmup, n_steps). Returns {stat, p_value} (one-sided, add-one corrected)."""

    def stat(sets):
        return sum(
            int(np.sum(np.abs(np.subtract.outer(a, b)) <= tol))
            for a, b in itertools.combinations([np.asarray(s, dtype=int) for s in sets], 2)
        )

    observed = stat(starts_by_seed)
    rng = np.random.default_rng(seed)
    hits = sum(
        stat([rng.integers(warmup, n_steps, size=len(s)) for s in starts_by_seed]) >= observed for _ in range(n_perm)
    )
    return {"stat": observed, "p_value": (1 + hits) / (1 + n_perm)}


def summarize(runs: list[dict]) -> dict:
    """Per preset: the events of every seed, how many seeds spiked, and the cross-seed alignment test."""
    out = {}
    for name in sorted({r["stress"] for r in runs}):
        rs = sorted((r for r in runs if r["stress"] == name), key=lambda r: r["seed"])
        events = {r["seed"]: find_spikes(r["losses"]) for r in rs}
        starts = [[e["start"] for e in events[s]] for s in events]
        out[name] = {
            "events": events,
            "seeds_with_spike": sum(1 for s in events if events[s]),
            "alignment": alignment_test(starts, len(rs[0]["losses"])) if sum(map(len, starts)) >= 2 else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Natural-spike stress search (GPU-only).")
    ap.add_argument("--search", action="store_true", help="run every preset x seed, then summarize")
    ap.add_argument("--out", default=os.environ.get("EVIDENCE_OUT", "results"))
    a = ap.parse_args()
    if not a.search:
        ap.print_help()
        return

    run_dir = os.path.join(a.out, "natural_runs")
    os.makedirs(run_dir, exist_ok=True)
    paths = []
    for name, overrides in STRESS.items():
        for seed in SEEDS:
            spec = {
                "run_id": f"{name}-s{seed}",
                "stress": name,
                "overrides": {**overrides, "train.batch_size": BATCH},
                "seed": seed,
                "steps": STEPS,
                "recipe": None,
                "out_dir": os.path.join(run_dir, f"{name}-s{seed}"),
            }
            subprocess.run([sys.executable, "-m", "research.spikes.capture", "--one", json.dumps(spec)], check=True)
            paths.append(os.path.join(spec["out_dir"], "run.json"))

    runs = []
    for p in paths:
        with open(p) as fp:
            runs.append(json.load(fp))
    summary = summarize(runs)
    with open(os.path.join(a.out, "natural_search.json"), "w") as fp:
        json.dump({"presets": STRESS, "seeds": SEEDS, "steps": STEPS, "summary": summary, "runs": paths}, fp, indent=2)
    for name, s in summary.items():
        print(f"[{name}] seeds with a spike: {s['seeds_with_spike']}/{len(SEEDS)} alignment={s['alignment']}")

    from research.spikes.k2 import build_k2, save_k2

    manifest = build_k2(paths)  # only triggers that land on a find_spikes event enter K2
    save_k2(manifest, os.path.join(a.out, "k2_manifest_natural.json"))
    print(f"natural K2 entries: {len(manifest.entries)} -> k2_manifest_natural.json")


if __name__ == "__main__":
    main()
