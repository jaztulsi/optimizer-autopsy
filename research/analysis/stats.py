"""Statistics for the fork sweep (PLAN_V8 §6.9, §7.0, §7.2).

Convention everywhere: Δ is held-out damage, LOWER IS BETTER. `a` is the method, `b` the comparator, paired by
spike instance (same index = same instance). `clusters` (e.g. the data-order seed) collapses the paired
differences to cluster means first, so the resampling unit is the cluster, not the fork.

scipy and statsmodels are imported lazily: both are pinned (research/requirements.txt, requirements-dev.txt)
and installed in CI, so the tests exercise the real code.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def _diffs(a, b, clusters=None) -> np.ndarray:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.ndim != 1 or d.size < 2:
        raise ValueError(f"need >= 2 paired values, got shape {d.shape}")
    if clusters is None:
        return d
    c = np.asarray(clusters)
    if c.shape != d.shape:
        raise ValueError(f"clusters shape {c.shape} != diffs shape {d.shape}")
    return np.array([d[c == k].mean() for k in np.unique(c)])


def paired_bootstrap(a, b, clusters=None, alpha=0.05, n_resamples=10_000, seed=0) -> dict:
    """BCa (1-alpha) CI for mean(a - b)."""
    from scipy.stats import bootstrap

    d = _diffs(a, b, clusters)
    if np.all(d == d[0]):  # BCa is undefined on a constant sample; the CI is the point
        return {"est": float(d[0]), "lo": float(d[0]), "hi": float(d[0])}
    r = bootstrap((d,), np.mean, confidence_level=1 - alpha, n_resamples=n_resamples, method="BCa", rng=seed)
    return {"est": float(d.mean()), "lo": float(r.confidence_interval.low), "hi": float(r.confidence_interval.high)}


def tost(a, b, margin, clusters=None, alpha=0.05) -> dict:
    """Non-inferiority, superiority and equivalence of `a` vs `b` from one-sided t bounds on mean(a - b).

    noninferior: upper bound < margin (H0: mean >= margin). superior: upper bound < 0. equivalent: both
    one-sided bounds inside (-margin, margin). `p_noninf` / `p_sup` are the one-sided p-values Holm takes."""
    from scipy.stats import t

    if margin <= 0:
        raise ValueError(f"margin must be > 0, got {margin}")
    d = _diffs(a, b, clusters)
    n, est = d.size, float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    if se == 0.0:  # all diffs equal: the bounds collapse onto the estimate
        lo = hi = est
        p_noninf, p_sup = float(est >= margin), float(est >= 0)
    else:
        q = float(t.ppf(1 - alpha, n - 1))
        lo, hi = est - q * se, est + q * se
        p_noninf, p_sup = float(t.cdf((est - margin) / se, n - 1)), float(t.cdf(est / se, n - 1))
    return {
        "est": est,
        "lo": lo,
        "hi": hi,
        "p_noninf": p_noninf,
        "p_sup": p_sup,
        "noninferior": hi < margin,
        "superior": hi < 0,
        "equivalent": lo > -margin and hi < margin,
    }


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER), in the input order."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (FDR), in the input order."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum.accumulate((m / np.arange(m, 0, -1) * p[order[::-1]]))[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def mixed(rows, formula="delta ~ C(branch) + C(site)", group="seed") -> dict:
    """Δ ~ branch + site + (1 | seed) (PLAN_V8 §6.9). `rows`: dicts with the formula's columns plus `group`.
    Returns {term: {est, lo, hi, p}} for the fixed effects (95% Wald CIs, REML)."""
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(list(rows))
    fit = smf.mixedlm(formula, df, groups=df[group]).fit(reml=True)
    ci = fit.conf_int()
    fe = fit.fe_params
    return {k: {"est": fe[k], "lo": ci.loc[k, 0], "hi": ci.loc[k, 1], "p": fit.pvalues[k]} for k in fe.index}


def n_required(sd, delta, alpha=0.05, power=0.9, m=1, n_min=10) -> int:
    """Instances per spike type: max(n_min, ceil((z_{1-alpha/m} + z_power)^2 (sd/delta)^2)), one-sided paired.
    m=1 gives the plan's old 8.6 constant; m=4 (Holm over C-1..C-4, worst case alpha/4) gives 12.4."""
    z = NormalDist().inv_cdf
    k = (z(1 - alpha / m) + z(power)) ** 2
    return max(n_min, math.ceil(k * (sd / delta) ** 2))
