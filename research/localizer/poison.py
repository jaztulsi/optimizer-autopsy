"""Poison score: combine SNR + curvature (+ others) into a single localization map.

Output is a ranked set of parameters/groups the repair operator will act on.

Task 12. Started here: the `v_bar` mechanism (context-ai §4.7) -- the pre-spike EMA of Adam's
second moment that the poison score measures `v_t` against. Nothing in the harness captured a
running average of `v` before this; `VBarAccumulator` is an `on_step(step, info)` hook for
`train_forward`, so it rides the trunk run that is being snapshotted rather than needing a
second pass.

# TODO: poison_score(snapshot, batch, weights) -> combined per-group score from snr + curvature.
# TODO: localize(snapshot, batch, k) -> top-k implicated groups (the repair target set).
"""

from __future__ import annotations

from collections import deque


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
