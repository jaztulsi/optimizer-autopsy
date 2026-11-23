"""Does the localizer separate spikes from normal steps, and does it agree with itself? (Weeks 5 and 7.)

Inputs are `pipeline.summary_stats` rows, each with a `label` ("spike" or "normal") and a `seed`:

  * `separation` -- for a statistic (psi_k, max_score, max_drift_ratio), the AUC of spikes vs
    normal steps: the chance a random spike scores higher than a random normal step (Mann-Whitney,
    ties count 1/2). 0.5 = no separation, 1.0 = perfect. Reported per seed and pooled.
  * `selection_agreement` -- mean pairwise Jaccard overlap of the selected groups across spikes of
    the same kind (e.g. all lr_bump spikes over seeds). High = the localizer points at the same
    place each time the same kind of spike happens.

Plain python + numpy, so it is tested without torch. Nothing here is a pass/fail gate: PLAN_V8 §7.2
is where thresholds for claims are pre-registered, and this module only reports.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def auc(pos, neg) -> float | None:
    """P(random pos > random neg) + 0.5 * P(tie). None if either side is empty."""
    p = np.asarray([x for x in pos if x is not None and np.isfinite(x)], dtype=np.float64)
    n = np.asarray([x for x in neg if x is not None and np.isfinite(x)], dtype=np.float64)
    if p.size == 0 or n.size == 0:
        return None
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (p.size * n.size))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def separation(rows, stat) -> dict:
    """AUC of `stat` for spikes vs normal steps, pooled and per seed."""

    def split(rs):
        return [r[stat] for r in rs if r["label"] == "spike"], [r[stat] for r in rs if r["label"] == "normal"]

    pos, neg = split(rows)
    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        sp, nm = split([r for r in rows if r["seed"] == seed])
        per_seed[seed] = {"auc": auc(sp, nm), "n_spike": len(sp), "n_normal": len(nm)}
    return {"stat": stat, "auc": auc(pos, neg), "n_spike": len(pos), "n_normal": len(neg), "per_seed": per_seed}


def selection_agreement(rows) -> dict:
    """Mean pairwise Jaccard of `selected` across spikes, grouped by spike `kind`."""
    out = {}
    for kind in sorted({r.get("kind") for r in rows if r["label"] == "spike"}, key=str):
        sel = [r["selected"] for r in rows if r["label"] == "spike" and r.get("kind") == kind]
        pairs = [jaccard(a, b) for a, b in combinations(sel, 2)]
        out[str(kind)] = {
            "n_spikes": len(sel),
            "mean_jaccard": float(np.mean(pairs)) if pairs else None,
            "empty_selections": sum(1 for s in sel if not s),
        }
    return out


def report(rows, stats=("psi_k", "max_score", "max_drift_ratio")) -> dict:
    return {
        "n_rows": len(rows),
        "separation": {s: separation(rows, s) for s in stats},
        "selection_agreement": selection_agreement(rows),
    }
