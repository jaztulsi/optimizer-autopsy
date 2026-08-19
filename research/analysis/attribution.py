"""Attribution science (C2): tie the localized poison to the counterfactual fork outcome.

Does repairing exactly the localized groups recover the run, and does repairing random/other
groups not? This is the causal claim the paper makes.

Also home to the **Gate B verdict** (Task 9, the cheap-fix kill-test): given the per-branch Δ's
from the fork battery, decide PROCEED (build the localizer/repair) vs PIVOT (cheap fix suffices ->
C1+C2 paper), and render the one-page report. Kept stdlib-only so it runs on the CPU CI runner and
the GPU image alike (no scipy/statsmodels pin needed here).
"""

from __future__ import annotations

import random
import statistics

# TODO: attribute(snapshot, localization, forks) -> effect of repairing localized vs control groups.
# TODO: necessity_sufficiency(forks) -> is the localized set necessary AND sufficient for recovery.

_CHEAP = "Bs"  # the cheap fix under test (skip + clip); Δ = val_loss(branch) - val_loss(B*)


def _paired_ci(deltas, *, confidence: float = 0.95, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap CI of the mean of paired Δ's -- stdlib only, deterministic via `seed`.
    # ponytail: scipy.stats.bootstrap (BCa, bias-corrected) is the drop-in if we ever need it."""
    a = [float(x) for x in deltas]
    if not a:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(a)
    if len(a) < 2:  # no spread to resample -> point "interval"
        return mean, mean, mean
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(a, k=len(a))) for _ in range(n_boot))
    lo = boots[int((1 - confidence) / 2 * n_boot)]
    hi = boots[min(n_boot - 1, int((1 + confidence) / 2 * n_boot))]
    return mean, lo, hi


def gate_b_verdict(
    deltas: dict, *, tau: float, recipes_total: int = 4, confidence: float = 0.95, seed: int = 0
) -> dict:
    """Evaluate the Gate-B cheap-fix kill-test.

    `deltas`: `{recipe: {branch: [Δ per matched seed]}}`, Δ = val_loss(branch) - val_loss(B*).
    A branch **recovers** on a recipe iff its paired-bootstrap CI upper bound <= `tau` (reaches clean).
    Decision:
      * PROCEED       -- Bs fails to recover on >=1 recipe (CI *lower* bound > tau): a real gap a cheap
                         fix can't close -> build the localizer/repair.
      * PIVOT         -- Bs recovers on EVERY recipe present: the cheap fix suffices -> C1+C2 paper.
      * INCONCLUSIVE  -- neither holds (a Bs CI straddles tau): need more seeds / recipes.
    The verdict is **PROVISIONAL** unless all `recipes_total` recipes are present.
    """
    recipes = list(deltas)
    per: dict = {}
    for r, branch_deltas in deltas.items():
        per[r] = {}
        for b, ds in branch_deltas.items():
            mean, lo, hi = _paired_ci(ds, confidence=confidence, seed=seed)
            per[r][b] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(list(ds)), "recovers": hi <= tau}

    def _bs(r, key):
        return per[r].get(_CHEAP, {}).get(key)

    bs_recovers = {r: bool(_bs(r, "recovers")) for r in recipes}
    bs_clear_gap = {r: (_bs(r, "ci_lo") is not None and _bs(r, "ci_lo") > tau) for r in recipes}

    if recipes and any(bs_clear_gap[r] for r in recipes):
        decision = "PROCEED"
    elif recipes and all(bs_recovers[r] for r in recipes):
        decision = "PIVOT"
    else:
        decision = "INCONCLUSIVE"

    return {
        "decision": decision,
        "provisional": len(recipes) < recipes_total,
        "tau": tau,
        "recipes_present": recipes,
        "recipes_total": recipes_total,
        "per_recipe": per,
        "bs_recovers": bs_recovers,
    }


def render_verdict(v: dict) -> str:
    """Render the Gate-B verdict as a one-page markdown report.

    Any run with fewer than all recipes is HARD-LABELED provisional in the OUTPUT itself -- a banner
    at the top plus a tag on the decision line -- so a clean-looking partial signal (e.g. an
    lr_bump-only PIVOT) can never be misread later as the binding gate. This is a format guarantee,
    not a caveat paragraph.
    """
    k, total = len(v["recipes_present"]), v["recipes_total"]
    tag = " (PROVISIONAL)" if v["provisional"] else ""
    lines = ["# Gate B — Cheap-Fix Kill-Test Verdict", ""]
    if v["provisional"]:
        present = ", ".join(v["recipes_present"]) or "none"
        lines += [
            f"> ⚠️ **PROVISIONAL — NOT the binding Gate B verdict.** Ran {k}/{total} spike recipes "
            f"({present}). The binding PROCEED/PIVOT decision requires all {total}; treat the call "
            "below as a preliminary signal only.",
            "",
        ]
    lines += [f"**Decision: {v['decision']}{tag}**  ·  τ (recovery margin) = {v['tau']:g}", ""]
    lines += ["| recipe | branch | mean Δ | 95% CI | n | recovers |", "|---|---|---|---|---|---|"]
    for r in v["recipes_present"]:
        for b, s in v["per_recipe"][r].items():
            mean = f"{s['mean']:.4g}" if s["n"] else "—"
            ci = f"[{s['ci_lo']:.4g}, {s['ci_hi']:.4g}]" if s["n"] else "—"
            lines.append(f"| {r} | {b} | {mean} | {ci} | {s['n']} | {'yes' if s['recovers'] else 'no'} |")
    lines += [
        "",
        r"_Rule: a branch recovers iff its Δ-vs-B\* CI upper bound ≤ τ. PROCEED if Bs leaves a gap on "
        "any recipe; PIVOT if Bs recovers on all; else INCONCLUSIVE._",
    ]
    return "\n".join(lines)
