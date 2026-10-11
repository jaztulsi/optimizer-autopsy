"""Baselines Bg / Bvreset / Bvbar: global optimizer-state resets -- the blunt upper bounds.

Bg zeroes both moments (m, v -> 0); Bvreset zeroes only v and keeps m; Bvbar sets ALL of v back to its
pre-spike average v_bar (m kept) -- the fair global comparison for our selective v repair, which moves
v toward v_bar only in the localized directions. They are the blunt versions of
our repair operator (and of its v-only arm, Bv); the localizer is what we claim beats them. In the
Task-9 kill-test the spike is REPLAYED forward inside the fork, so the reset fires ONCE right after the
spike window (zeroing the freshly poisoned moments) -- not at fork start, where the state is still
clean and zeroing it would prove nothing.

Bvreset caveat, by design: with v=0 and m kept, the next step's v is just (1-beta2) * g^2 while Adam's
bias correction (step count kept) no longer divides it back up, so the first post-reset updates are
~1/sqrt(1-beta2) larger than usual (~4.5x at beta2=0.95). That is the naive baseline as specified;
SPAM's post-reset warmup exists to tame exactly this.
"""

from __future__ import annotations

BOTH = ("exp_avg", "exp_avg_sq")
V_ONLY = ("exp_avg_sq",)


def zero_moments(opt, keys=BOTH) -> int:
    """Zero the named AdamW moments (m=`exp_avg`, v=`exp_avg_sq`) of every param in place; Adam's per-param
    `step` is kept. Returns how many tensors were zeroed."""
    n = 0
    for st in opt.state.values():
        for key in keys:
            if key in st:
                st[key].zero_()
                n += 1
    return n


def set_v_to_vbar(model, opt, v_bar: dict) -> int:
    """Copy `v_bar[name]` into every param's AdamW `exp_avg_sq` in place (m and Adam's `step` kept). Keyed
    by parameter NAME, like snapshots. Strict: a param with optimizer state but no v_bar entry, or a
    shape mismatch, raises (`copy_` would silently broadcast a wrong shape). Returns how many were set."""
    import torch

    n = 0
    for name, p in model.named_parameters():
        st = opt.state.get(p, {})
        if "exp_avg_sq" not in st:
            continue
        if name not in v_bar:
            raise KeyError(f"v_bar has no entry for parameter {name!r}")
        vb = torch.as_tensor(v_bar[name])
        if tuple(vb.shape) != tuple(st["exp_avg_sq"].shape):
            raise ValueError(f"v_bar[{name!r}] shape {tuple(vb.shape)} != v shape {tuple(st['exp_avg_sq'].shape)}")
        st["exp_avg_sq"].copy_(vb.to(st["exp_avg_sq"].device, st["exp_avg_sq"].dtype))
        n += 1
    return n


def as_vbar_pre_step(at_step: int, v_bar: dict):
    """A `pre_step(step, ctx)` that sets v to `v_bar` once, when `step == at_step`."""

    def pre_step(step, ctx):
        if step == at_step:
            set_v_to_vbar(ctx["model"], ctx["opt"], v_bar)

    return pre_step


def as_pre_step(at_step: int, keys=BOTH):
    """A `pre_step(step, ctx)` that performs the global reset of `keys` once, when `step == at_step`."""

    def pre_step(step, ctx):
        if step == at_step:
            zero_moments(ctx["opt"], keys)

    return pre_step
