"""Poison score: combine SNR + curvature (+ others) into a single localization map.

Output is a ranked set of parameters/groups the repair operator will act on.

Task 12. Started here: the `v_bar` mechanism (context-ai §4.7) -- the pre-spike EMA of Adam's
second moment that the poison score measures `v_t` against. Nothing in the harness captured a
running average of `v` before this; `VBarAccumulator` is an `on_step(step, info)` hook for
`train_forward`, so it rides the trunk run that is being snapshotted rather than needing a
second pass.

Week 5 (below `VBarAccumulator`): `poison_ratios` (p_i), `psi_k`, `group_drift`, `group_sign`,
`curvature_share`, `null_thresholds` (tau_p, tau_s, tau_d from matched normal steps), `poisoned_directions`
(P), `poison_score`, and `localize`, which returns PLAN_V8 §6.4's `Localization`. `research/localizer/pipeline.py`
turns snapshots into the flat inputs these take.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from research.localizer.snr import as_np


def compose_on_step(*hooks):
    """Chain several `on_step(step, info)` hooks into one.

    `harness.fork.compose` does this for `pre_step`, but `train_forward` takes a single `on_step`
    and nothing composes those -- so a run that wants both the detector and this accumulator needs
    this. Hooks fire in the order given; `info` is the same dict for all of them, so a hook that
    mutates it affects the ones after it.
    """
    hooks = [h for h in hooks if h is not None]

    def on_step(step, info):
        for h in hooks:
            h(step, info)

    return on_step


class VBarAccumulator:
    """Debiased EMA of Adam's second moment `v`, keyed by parameter name, as an `on_step` hook.

    `v_bar <- rho * v_bar + (1 - rho) * v_t`, read back debiased by `1 / (1 - rho**n)`.

    THE DEBIASING IS NOT COSMETIC. An EMA started at zero is badly low for the first ~1/(1-rho)
    steps (100 at rho=0.99). The poison score is
        p_i = |u_i . (v_t - v_bar)| / (|u_i . v_bar| + eps)
    so an undersized `v_bar` inflates the numerator and shrinks the denominator at the same time --
    every direction would look poisoned, and the normal-step null used to set `tau_p` would be
    drawn from the same bias, hiding it. Same correction AdamW applies to its own moments.

    THE LAG IS NOT COSMETIC EITHER. `v_bar` must be PRE-spike (§4.7), but `on_step` fires AFTER
    `opt.step()`, so by the time a detector can see a loss spike at step t, `v_t` already contains
    the spiking gradient. With `lag=1` the hook holds each step's `v` back one step before folding
    it in, so calling `freeze()` the moment a spike fires leaves an EMA built only from steps
    <= t-1. Raise `lag` if the detector needs more steps to make up its mind.

    Memory: one float32 copy of `v` per lag slot, plus one for the EMA. At the proxy shape that is
    ~44 MB each (11.04M params); at 124M it is ~0.5 GB each, so set `store_device="cpu"` there
    (the default) and consider `every>1`.

    Usage:
        vbar = VBarAccumulator(rho=0.99, warmup=50)
        detector = ...                      # calls vbar.freeze() when it fires
        run_trunk(cfg, data_dir, on_step=compose_on_step(detector, vbar))
        ...
        v_bar = vbar.read()                 # {param_name: float32 tensor}, debiased
    """

    def __init__(
        self,
        *,
        rho=0.99,
        warmup=0,
        every=1,
        lag=1,
        store_device="cpu",
        store_dtype=None,
        pause_if=None,
    ):
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if lag < 0:
            raise ValueError("lag must be >= 0")
        self.rho = float(rho)
        self.warmup = int(warmup)
        self.every = int(every)
        self.lag = int(lag)
        self.store_device = store_device
        self.store_dtype = store_dtype
        self.pause_if = pause_if

        self._ema: dict | None = None
        self._pending: deque = deque()
        self.n = 0  # folded updates -- the exponent in the debias term
        self.first_step: int | None = None
        self.last_step: int | None = None
        self.frozen = False
        self.skipped = 0

    # -- the hook -----------------------------------------------------------------------

    def __call__(self, step, info):
        """`on_step(step, info)`; reads `info["opt"]` and `info["model"]`."""
        if self.frozen:
            return
        if step < self.warmup or (step - self.warmup) % self.every:
            return
        if self.pause_if is not None and self.pause_if(step, info):
            self.skipped += 1
            return

        snap = self._read_v(info["model"], info["opt"])
        if snap is None:  # no AdamW state yet (step 0 before the first opt.step())
            return
        self._pending.append((step, snap))
        while len(self._pending) > self.lag:
            self._fold(*self._pending.popleft())

    def _read_v(self, model, opt):
        import torch

        name_by_id = {id(p): n for n, p in model.named_parameters()}
        out = {}
        for group in opt.param_groups:
            for p in group["params"]:
                name = name_by_id.get(id(p))
                if name is None:
                    raise RuntimeError("optimizer param not in model.named_parameters()")
                st = opt.state.get(p, {})
                if "exp_avg_sq" not in st:
                    return None
                dt = self.store_dtype or torch.float32
                out[name] = st["exp_avg_sq"].detach().to(self.store_device, dt).clone()
        return out

    def _fold(self, step, snap):
        if self._ema is None:
            self._ema = {k: t.clone().zero_() for k, t in snap.items()}
            self.first_step = step
        elif self._ema.keys() != snap.keys():
            raise RuntimeError("parameter names changed mid-run")
        for k, t in snap.items():
            self._ema[k].mul_(self.rho).add_(t, alpha=1.0 - self.rho)
        self.n += 1
        self.last_step = step

    # -- control ------------------------------------------------------------------------

    def freeze(self):
        """Stop accumulating. Call this from the spike detector, not after the run: the point of
        `v_bar` is that it never saw the spike. The `lag` slots still in flight are DISCARDED,
        since those are exactly the contaminated steps."""
        self.frozen = True
        self._pending.clear()
        return self

    def flush(self):
        """Fold the lagged slots in anyway -- for a clean run that ended without a spike."""
        while self._pending:
            self._fold(*self._pending.popleft())
        return self

    # -- read ---------------------------------------------------------------------------

    def read(self, *, debias=True) -> dict:
        """{param_name: tensor} -- the debiased EMA. Keys match `harness.snapshot`'s `v`, so this
        drops into the poison score beside a snapshot without renaming."""
        if self._ema is None or self.n == 0:
            raise RuntimeError("v_bar is empty: no steps were folded in (check warmup/lag/freeze)")
        if not debias:
            return {k: t.clone() for k, t in self._ema.items()}
        c = 1.0 - self.rho**self.n
        return {k: (t / c) for k, t in self._ema.items()}

    def read_flat(self, index, *, debias=True):
        """The same thing as one flat vector in `curvature.param_index` order, which is what the
        projections in §4.7/§4.8 consume."""
        import torch

        vb = self.read(debias=debias)
        parts = []
        for name, p in index:
            if name not in vb:
                raise RuntimeError(f"v_bar has no entry for {name!r}")
            parts.append(vb[name].to(p.device, torch.float32).reshape(-1))
        return torch.cat(parts)

    @property
    def effective_window(self) -> float:
        """~1/(1-rho) folded steps, in units of folded updates (multiply by `every` for trunk
        steps). Tells you whether `n` is large enough for the EMA to mean anything."""
        return 1.0 / (1.0 - self.rho)

    def status(self) -> dict:
        return {
            "n": self.n,
            "rho": self.rho,
            "effective_window": self.effective_window,
            "debias": 1.0 - self.rho**self.n if self.n else 0.0,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "pending": len(self._pending),
            "frozen": self.frozen,
            "skipped": self.skipped,
            "warmed_up": self.n >= self.effective_window,
        }

    # -- persistence --------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "ema": self._ema,
            "n": self.n,
            "rho": self.rho,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "frozen": self.frozen,
        }

    def load_state_dict(self, sd) -> None:
        self._ema = sd["ema"]
        self.n = sd["n"]
        self.rho = sd["rho"]
        self.first_step = sd.get("first_step")
        self.last_step = sd.get("last_step")
        self.frozen = sd.get("frozen", False)
        self._pending.clear()


# --------------------------------------------------------------------------------------
# Week 5: poison score, psi_k, null-derived thresholds, localize (context-ai §4.7, PLAN_V8 §6.4)
#
# All numpy: the inputs are flat vectors in `curvature.layout` order (use `snr.flat_from_named`
# on a snapshot's `v` and on a v_bar dict) and a frozen (k, dim) eigenvector block U from
# `curvature.top_k_htilde`. torch is not needed here, so this layer is tested without it.
# --------------------------------------------------------------------------------------

LOCALIZATION_KEYS = ("groups", "selected", "method", "ref", "t", "cost_gpu_s")
GROUP_KEYS = ("tensor", "rows", "score", "sign", "channel")


def _orthonormal_rows(U, tol=1e-3) -> None:
    k = U.shape[0]
    err = float(np.abs(U @ U.T - np.eye(k)).max()) if k else 0.0
    if err > tol:
        raise ValueError(f"eigenvector rows are not orthonormal (max |U U^T - I| = {err:.2e})")


def poison_ratios(U, v, vbar, eps=1e-12) -> np.ndarray:
    """p_i = |u_i . (v - v_bar)| / (|u_i . v_bar| + eps), one value per row of U (context-ai §4.7)."""
    U, v, vbar = as_np(U), as_np(v), as_np(vbar)
    return np.abs(U @ (v - vbar)) / (np.abs(U @ vbar) + eps)


def psi_k(U, v, vbar) -> float:
    """psi_k = ||Pi_k (v - v_bar)|| / ||v - v_bar||: the fraction of the v change inside span(U).

    U must have orthonormal rows (top_k_htilde's eigenvectors do), or Pi_k is not a projection.
    0 when v did not move at all.
    """
    U, v, vbar = as_np(U), as_np(v), as_np(vbar)
    _orthonormal_rows(U)
    d = v - vbar
    nd = float(np.linalg.norm(d))
    if nd == 0.0:
        return 0.0
    return float(np.linalg.norm(U @ d) / nd)


def group_drift(v, vbar, groups, eps=1e-30) -> dict:
    """RMS of log(v / v_bar) per group: how far v moved from its pre-spike average, scale-free.

    This is PLAN_V8 §6.4's magnitude control (||log v - log v_bar||), per group, as an RMS so groups
    of different sizes are comparable. It needs no eigenvectors.
    """
    r = np.log((as_np(v) + eps) / (as_np(vbar) + eps))
    return {gid: float(np.sqrt(np.mean(r[g["start"] : g["stop"]] ** 2))) for gid, g in groups.items()}


def group_sign(v, vbar, groups, eps=1e-30) -> dict:
    """PLAN_V8 §6.1's signed poison per group: +1 if v went UP on balance (over-preconditioned, stall),
    -1 if it went down (under-preconditioned)."""
    r = np.log((as_np(v) + eps) / (as_np(vbar) + eps))
    return {gid: (1 if float(r[g["start"] : g["stop"]].sum()) >= 0 else -1) for gid, g in groups.items()}


def curvature_share(U, directions, groups) -> dict:
    """Share of the selected eigen-directions' energy that falls in each group (sums to 1 over groups).

    For each direction i in `directions`, ||u_i restricted to the group||^2 is that group's share of
    u_i (rows of U are unit vectors). Averaged over the directions. All zeros if `directions` is empty.
    """
    U = as_np(U)
    idx = list(directions)
    if not idx:
        return {gid: 0.0 for gid in groups}
    E = (U[idx] ** 2).sum(axis=0) / len(idx)
    return {gid: float(E[g["start"] : g["stop"]].sum()) for gid, g in groups.items()}


def _quantile(values, q, name, min_n) -> float:
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"no finite null values for {name}: calibration data is missing or empty")
    if a.size < min_n:
        raise ValueError(f"only {a.size} null values for {name} (need >= {min_n}); capture more normal steps")
    return float(np.quantile(a, q))


def null_thresholds(null, *, U=None, groups, q_p=0.95, q_s=0.5, q_d=0.95, min_steps=3, eps=1e-12) -> dict:
    """tau_d, tau_p and tau_s from matched NORMAL steps -- never fixed constants (context-ai §4.7).

    `null` is a list of dicts, one per normal step: {"v", "vbar"} as flat vectors, plus "dir_snr"
    (a length-k array, the directional SNR at that step along the same frozen U) when U is given.

      tau_d = the q_d quantile, over normal steps, of the LARGEST group drift at that step;
      tau_p = the q_p quantile, over normal steps, of the LARGEST p_i at that step (needs U);
      tau_s = the q_s quantile of every directional SNR value at every normal step (needs U).

    WHY THE LARGEST, not every value pooled: a group is tested against tau_d once per group, so with
    G groups a per-value 95th percentile flags about 5% of groups on a perfectly normal step -- with
    the proxy's ~30 tensor groups, nearly every normal step would select something. Taking the
    per-step maximum makes q_d the chance that a normal step selects ANY group. (Found by running
    the planted test over 200 normal draws: pooled gave a 6% per-group false-positive rate.)

    Raises if `null` is empty or has fewer than `min_steps` normal steps. With only a few normal
    steps the quantile is close to the largest value seen, which is coarse; Week 9's calibration
    sweeps are what fix that. The quantiles and sample sizes are returned for the provenance.
    """
    if not null:
        raise ValueError("no normal-step calibration data: thresholds must come from matched normal steps")
    max_drift, max_p, ss = [], [], []
    for row in null:
        if "v" not in row or "vbar" not in row:
            raise ValueError("each null row needs 'v' and 'vbar'")
        max_drift.append(max(group_drift(row["v"], row["vbar"], groups).values()))
        if U is not None:
            if row.get("dir_snr") is None:
                raise ValueError("null row has no 'dir_snr'; it is needed when an eigenbasis is used")
            max_p.append(float(poison_ratios(U, row["v"], row["vbar"], eps).max()))
            ss.extend(as_np(row["dir_snr"]).tolist())
    out = {
        "tau_d": _quantile(max_drift, q_d, "per-step max group drift", min_steps),
        "q_d": q_d,
        "n_normal_steps": len(null),
        "family": "per-step max",
    }
    if U is not None:
        out.update(
            tau_p=_quantile(max_p, q_p, "per-step max poison ratio p_i", min_steps),
            tau_s=_quantile(ss, q_s, "directional SNR", min_steps),
            q_p=q_p,
            q_s=q_s,
            n_null_snr=len(ss),
        )
    return out


def poisoned_directions(p, dir_snr, tau_p, tau_s) -> list[int]:
    """P = { i : p_i > tau_p AND SNR_i < tau_s } -- high poison and low signal (context-ai §4.7)."""
    p, s = as_np(p), as_np(dir_snr)
    if p.shape != s.shape:
        raise ValueError(f"p has shape {p.shape}, dir_snr has {s.shape}")
    return [int(i) for i in range(p.size) if p[i] > tau_p and s[i] < tau_s]


def poison_score(drift, tau_d, share=None) -> dict:
    """Per-group score: drift / tau_d, times (1 + G * curvature share) when an eigenbasis is used.

    drift / tau_d > 1 means the group moved more than the normal-step null allows. G * share is 1
    for a group holding an average share of the poisoned directions, so the curvature factor lifts
    groups where the stiff, poisoned directions live and leaves the rest at x1. Without an
    eigenbasis (the no-HVP localizer) the score is drift / tau_d alone.
    """
    if not tau_d > 0:
        raise ValueError(f"tau_d must be > 0, got {tau_d}")
    G = len(drift)
    out = {}
    for gid, d in drift.items():
        s = d / tau_d
        if share is not None:
            s *= 1.0 + G * share.get(gid, 0.0)
        out[gid] = float(s)
    return out


def localize(
    *,
    v,
    vbar,
    groups,
    null,
    t: int,
    U=None,
    dir_snr=None,
    k_select=5,
    q_p=0.95,
    q_s=0.5,
    q_d=0.95,
    min_steps=3,
    cost_gpu_s=0.0,
    eps=1e-12,
) -> dict:
    """Score every group and select the repair targets; returns PLAN_V8 §6.4's `Localization`.

    `v` is the snapshot's v at the failure step t and `vbar` the pre-spike v_bar, both flat. With
    `U` (frozen top-k eigenvectors of H~ at t) and `dir_snr` (directional SNR along U at t) this is
    the curvature localizer; without them it is the no-HVP localizer (drift only).

    A group is selected if its drift is above tau_d; the selected set is the top `k_select` of
    those by score (it can be empty -- nothing moved more than normal steps do). `rows` is
    (start, stop) or None, so the result is JSON-safe; use `rows_slice` to get a slice back.
    Everything the decision used goes in `provenance`.
    """
    v, vbar = as_np(v), as_np(vbar)
    if v.shape != vbar.shape:
        raise ValueError(f"v has shape {v.shape}, vbar has {vbar.shape}")
    use_eig = U is not None
    if use_eig and dir_snr is None:
        raise ValueError("dir_snr is required with U (P needs both p_i and the directional SNR)")
    if use_eig:
        U = as_np(U)
        if U.shape[1] != v.size:
            raise ValueError(f"U has {U.shape[1]} columns, v has {v.size} entries")

    th = null_thresholds(
        null, U=U if use_eig else None, groups=groups, q_p=q_p, q_s=q_s, q_d=q_d, min_steps=min_steps, eps=eps
    )
    drift = group_drift(v, vbar, groups)
    sign = group_sign(v, vbar, groups)
    prov = {"thresholds": th, "drift": drift, "k_select": int(k_select)}

    share = None
    if use_eig:
        p = poison_ratios(U, v, vbar, eps)
        P = poisoned_directions(p, dir_snr, th["tau_p"], th["tau_s"])
        share = curvature_share(U, P, groups)
        prov.update(
            p=p.tolist(),
            dir_snr=as_np(dir_snr).tolist(),
            poisoned_directions=P,
            psi_k=psi_k(U, v, vbar),
            curvature_share=share,
        )
    score = poison_score(drift, th["tau_d"], share)
    eligible = [gid for gid in groups if drift[gid] > th["tau_d"]]
    selected = sorted(eligible, key=lambda gid: -score[gid])[: int(k_select)]

    loc = {
        "groups": {
            gid: {
                "tensor": g["tensor"],
                "rows": None if g["rows"] is None else [int(g["rows"][0]), int(g["rows"][1])],
                "score": score[gid],
                "sign": sign[gid],
                "channel": "v",
            }
            for gid, g in groups.items()
        },
        "selected": selected,
        "method": "curvature: drift x (1 + G*share of P), P = {p>tau_p, snr<tau_s}" if use_eig else "drift (no HVP)",
        "ref": "vbar",
        "t": int(t),
        "cost_gpu_s": float(cost_gpu_s),
        "provenance": prov,
    }
    validate_localization(loc)
    return loc


def rows_slice(rows):
    """A Localization's `rows` ([start, stop] or None) as the slice PLAN_V8 §6.4 names."""
    return None if rows is None else slice(int(rows[0]), int(rows[1]))


def validate_localization(loc) -> None:
    """Raise if `loc` does not match PLAN_V8 §6.4's Localization contract."""
    for key in LOCALIZATION_KEYS:
        if key not in loc:
            raise ValueError(f"Localization is missing {key!r}")
    if loc["ref"] not in ("B*", "vbar", "fresh_g2"):
        raise ValueError(f"ref must be 'B*', 'vbar' or 'fresh_g2', got {loc['ref']!r}")
    for gid, g in loc["groups"].items():
        for key in GROUP_KEYS:
            if key not in g:
                raise ValueError(f"group {gid!r} is missing {key!r}")
        if g["sign"] not in (1, -1):
            raise ValueError(f"group {gid!r}: sign must be +1 or -1")
        if g["channel"] not in ("m", "v", "m+v"):
            raise ValueError(f"group {gid!r}: channel must be 'm', 'v' or 'm+v'")
        r = g["rows"]
        if r is not None and not isinstance(r, slice) and not (len(r) == 2 and int(r[0]) < int(r[1])):
            raise ValueError(f"group {gid!r}: rows must be None, a slice, or [start, stop)")
        if not np.isfinite(g["score"]):
            raise ValueError(f"group {gid!r}: score is not finite")
    missing = [gid for gid in loc["selected"] if gid not in loc["groups"]]
    if missing:
        raise ValueError(f"selected groups not in groups: {missing}")
    if len(set(loc["selected"])) != len(loc["selected"]):
        raise ValueError("selected has duplicates")


# --------------------------------------------------------------------------------------
# Self-check -- GPU/Kaggle only (CLAUDE.md §1). Not run on the workstation.
# --------------------------------------------------------------------------------------


def _selfcheck_vbar() -> dict:
    """Three properties, each able to fail on its own:
      1. the debiased EMA of a CONSTANT v equals that constant from the very first fold
         (this is what fails if debiasing is dropped),
      2. `lag=1` + `freeze()` excludes the flagged step,
      3. state_dict round-trips.
    Uses a toy model + real AdamW, so the hook runs against the real `info` dict shape.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    lin = nn.Linear(4, 3, bias=False)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)

    def step_once(scale=1.0):
        opt.zero_grad(set_to_none=True)
        x = torch.ones(2, 4) * scale
        lin(x).pow(2).sum().backward()
        opt.step()
        return {"loss": 0.0, "grad_norm": 0.0, "lr": 1e-3, "model": lin, "opt": opt}

    # 1. constant v -> debiased EMA == v after one fold, and after many.
    vbar = VBarAccumulator(rho=0.99, lag=0)
    info = step_once()
    const = {n: torch.full_like(p, 7.0) for n, p in lin.named_parameters()}
    for p in lin.parameters():
        opt.state[p]["exp_avg_sq"] = torch.full_like(p, 7.0)
    for s in range(5):
        vbar(s, info)
    got = vbar.read()
    for n in const:
        err = (got[n] - const[n]).abs().max().item()
        assert err < 1e-5, f"debiased EMA of a constant is off by {err:.3e} (debias broken?)"
    assert abs(vbar.read(debias=False)[next(iter(const))].max().item() - 7.0) > 0.1, (
        "undebiased EMA should still be visibly low after 5 folds -- the test is not testing debias"
    )

    # 2. lag=1 + freeze excludes the flagged step.
    vbar2 = VBarAccumulator(rho=0.9, lag=1)
    for s in range(4):
        info = step_once()
        for p in lin.parameters():
            opt.state[p]["exp_avg_sq"] = torch.full_like(p, float(s + 1))
        vbar2(s, info)
    assert vbar2.last_step == 2, f"lag=1 folded up to step {vbar2.last_step}, expected 2"
    vbar2.freeze()
    assert vbar2.last_step == 2, "freeze() must discard the in-flight (contaminated) slot"
    assert vbar2.n == 3, f"expected 3 folds, got {vbar2.n}"

    # 3. round-trip.
    sd = vbar2.state_dict()
    fresh = VBarAccumulator(rho=0.9, lag=1)
    fresh.load_state_dict(sd)
    a, b = vbar2.read(), fresh.read()
    for n in a:
        assert torch.equal(a[n], b[n]), f"state_dict round-trip changed {n}"

    print(f"v_bar selfcheck OK: {vbar.status()}")
    return {
        "claim": "debiased EMA is exact on a constant v; lag=1 + freeze() excludes the flagged "
        "step; state_dict round-trips",
        "debias_max_abs_err": float(max((got[n] - const[n]).abs().max().item() for n in const)),
        "undebiased_value_after_5_folds": float(vbar.read(debias=False)[next(iter(const))].max().item()),
        "undebiased_expected": 7.0,
        "lag_last_folded_step": int(vbar2.last_step),
        "lag_last_folded_step_expected": 2,
        "lag_n_folds": int(vbar2.n),
        "state_dict_round_trip": True,
        "passed": True,
    }


if __name__ == "__main__":
    _selfcheck_vbar()
