"""Damage triage (PLAN_V8 Appendix A R-F6; proposed, exploratory, not preregistered).

At a detected spike, predict from spike-time optimizer features whether it will leave lasting damage, so only those
spikes get repaired. Labels are exact fork differences against the clean twin B* (Y(none) = val(Px<c>) - val(B*) > 0.01
nats). The fit is CPU numpy; the GPU side is one read-only `pre_opt_step` hook (`features_hook`).
"""

from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
from scipy.stats import rankdata

FEATURES = ("log_x_mean", "log_x_max_tensor", "frac_knee", "cos_gm", "log_lr", "log_mem", "log_step")
BASELINE = ("log_x_mean",)  # spike size alone; the model must beat it
THRESHOLD = 0.01  # nats, prereg-v3 P1's practical size
_TINY = 1e-300


def _log(x: float) -> float:
    return math.log(max(x, _TINY))


def features_hook(steps, out: dict):
    """Read-only `pre_opt_step` (chain it AFTER the plant, so it sees the spiked gradient). At each step in `steps`
    writes out["triage"][str(step)] = {feature: float}, from g, the pre-step m and v = exp_avg_sq,
    x = (1-b2)(g-m)^2 / v. Never touches params, grads or state."""
    from research.autopsy.probation import x_knee

    steps = set(steps)

    def hook(step, ctx):
        if step not in steps:
            return None
        opt = ctx["opt"]
        group = opt.param_groups[0]
        b1, b2 = group["betas"]
        try:
            knee = x_knee(b1, b2)
        except ValueError:
            knee = float("nan")
        n = x_sum = x_max = n_knee = gm = gg = mm = 0.0
        for _, p in ctx["model"].named_parameters():
            st = opt.state.get(p, {})
            if p.grad is None or "exp_avg_sq" not in st:
                continue
            g = p.grad.detach().reshape(-1).double()
            m = st["exp_avg"].detach().reshape(-1).double()
            v = st["exp_avg_sq"].detach().reshape(-1).double()
            x = (1 - b2) * (g - m).square() / v.clamp_min(1e-30)
            n += x.numel()
            x_sum += float(x.sum())
            x_max = max(x_max, float(x.mean()))
            n_knee += float((x >= knee).sum())  # knee nan: no coordinate counts; overwritten with nan below
            gm += float((g * m).sum())
            gg += float(g.square().sum())
            mm += float(m.square().sum())
        denom = math.sqrt(gg) * math.sqrt(mm)
        row = {
            "log_x_mean": _log(x_sum / max(n, 1)),
            "log_x_max_tensor": _log(x_max),
            "frac_knee": n_knee / max(n, 1) if not math.isnan(knee) else float("nan"),
            "cos_gm": gm / denom if denom > 0 else 0.0,
            "log_lr": _log(group["lr"]),
            "log_mem": -_log(1 - b2),
            "log_step": _log(step),
        }
        out.setdefault("triage", {})[str(step)] = {k: float(row[k]) for k in FEATURES}
        return None

    return hook


def dataset(paths, threshold: float = THRESHOLD):
    """Rows from mve result JSONs: one per dose arm `Px<c>` (exactly; not `Px10|U_v@0` etc.) that has a "triage" entry.
    Features = the entry at the smallest step (the onset); Y(none) = val(Px) - val(B*) at the largest step in both;
    y = Y(none) > threshold; group = the run's seed; stratum = dose arm + betas + onset step (one regime, one dose).
    Returns (X, y, groups, rows, skipped): rows are dicts (path, branch, c, stratum, y_none, seed); skipped counts dose
    arms dropped for no "triage" entry (files from before the hook) or a non-finite feature."""
    X, y, groups, rows, skipped = [], [], [], [], {"no_triage": 0, "nonfinite": 0}
    for path in paths:
        with open(path) as f:
            res = json.load(f)
        seed, (b1, b2) = res["stamp"]["cfg"]["seed"], res["stamp"]["cfg"]["optim"]["betas"]
        br = res["branches"]
        for name, b in br.items():
            if not re.fullmatch(r"Px[0-9.e+-]+", name) or "B*" not in br:
                continue
            tri = b.get("metrics", {}).get("triage")
            if not tri:
                skipped["no_triage"] += 1
                continue
            common = set(b["val"]) & set(br["B*"]["val"])
            if not common:
                continue
            h = max(common, key=int)
            y_none = b["val"][h] - br["B*"]["val"][h]
            onset = min(tri, key=int)
            feats = [tri[onset][k] for k in FEATURES]
            if not all(math.isfinite(v) for v in feats):
                skipped["nonfinite"] += 1
                continue
            X.append(feats)
            y.append(int(y_none > threshold))
            groups.append(seed)
            row = {"path": str(path), "branch": name, "c": float(name[2:]), "stratum": f"{name}|b{b1}-{b2}|inj{onset}"}
            rows.append(row | {"y_none": y_none, "seed": seed})
    return np.array(X, float).reshape(-1, len(FEATURES)), np.array(y, int), np.array(groups), rows, skipped


def fit(X, y, l2: float = 1.0, iters: int = 100) -> dict:
    """L2-penalised logistic regression by Newton/IRLS on standardised features. Every weight, the intercept
    included, is penalised, so a one-class fold stays finite."""
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        grad = A.T @ (p - y) + l2 * w
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return {"mu": mu, "sd": sd, "w": w}


def score(model: dict, X):
    """The linear logit."""
    return model["w"][0] + ((X - model["mu"]) / model["sd"]) @ model["w"][1:]


def _pairs(y, k) -> int:
    """Positive-negative pairs inside one stratum mask k."""
    n1 = int((y[k] == 1).sum())
    return n1 * (int(k.sum()) - n1)


def auc(s, y, strata=None) -> float:
    """Mann-Whitney AUC, ties at half. With `strata`, only pairs inside one stratum count (pooled over strata), so
    whatever the strata differ in cannot separate the classes. nan if no stratum holds both classes."""
    s, y = np.asarray(s, float), np.asarray(y)
    strata = np.zeros(len(y)) if strata is None else np.asarray(strata)
    num = den = 0.0
    for k in (strata == c for c in np.unique(strata)):
        if n := _pairs(y, k):
            n1 = int((y[k] == 1).sum())
            num += rankdata(s[k])[y[k] == 1].sum() - n1 * (n1 + 1) / 2
            den += n
    return float(num / den) if den else float("nan")


def loso_scores(X, y, groups, cols, l2: float = 1.0):
    """Leave-one-seed-out logits: each seed is scored by a fit on the other seeds. All nan with fewer than two seeds."""
    cols, s = list(cols), np.full(len(y), np.nan)
    if len(set(groups.tolist())) < 2:
        return s
    for g in set(groups.tolist()):
        te = groups == g
        s[te] = score(fit(X[~te][:, cols], y[~te], l2), X[te][:, cols])
    return s


def report(X, y, groups, strata, l2: float = 1.0) -> dict:
    """All features vs spike size alone, on leave-one-seed-out logits. Primary: the AUC over pairs in the same stratum
    (one dose in one regime: betas and spike step). Pooled, size alone separates c = 1 from c = 30, and across regimes
    the step / LR / beta features could score the regime's base rate instead of the spike. `ceiling`: size alone is
    already >= 0.95 there, so no gain can show. `keep`: auc >= 0.8 and gain >= 0.05 (PLAN_V8 A.3.1). Pooled AUCs are
    reported too."""
    strata = np.asarray(strata)
    s = loso_scores(X, y, groups, range(len(FEATURES)), l2)
    b = loso_scores(X, y, groups, [FEATURES.index(f) for f in BASELINE], l2)
    a, ab = auc(s, y, strata), auc(b, y, strata)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "pairs_within_stratum": sum(_pairs(y, strata == c) for c in np.unique(strata)),
        "auc": a,
        "auc_baseline": ab,
        "gain": a - ab,
        "ceiling": bool(ab >= 0.95),
        "keep": bool(a >= 0.8 and a - ab >= 0.05),
        "auc_pooled": auc(s, y),
        "auc_baseline_pooled": auc(b, y),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args(argv)
    X, y, groups, rows, skipped = dataset(a.paths, a.threshold)
    print(json.dumps(report(X, y, groups, [r["stratum"] for r in rows]) | {"skipped": skipped}, indent=1))


if __name__ == "__main__":
    main()
