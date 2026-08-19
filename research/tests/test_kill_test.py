"""Local, torch-free unit tests for the Task-9 cheap-fix kill-test machinery.

Covers the pure control logic that does NOT need a GPU: the four branch factories, the pre_step
composition (ordering + window), NaN-safety, and the PROCEED/PIVOT decision rule + verdict renderer.
The actual number-producing run (fork forward on the real trunk) is GPU-only and lives on Kaggle.

Runnable two ways:  `pytest research/tests/test_kill_test.py`  or  `python -m research.tests.test_kill_test`.
Uses the fake-model/fake-opt pattern from `spikes.induce._selfcheck` -- no torch import anywhere.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from research.analysis.attribution import gate_b_verdict, render_verdict
from research.baselines import clip, reset, skip
from research.harness.fork import Branch, _summarize, compose, make_branches


class _FakeMoment:
    """Stands in for a torch tensor: only `.zero_()` is exercised by the reset baseline."""

    def __init__(self, val: float = 1.0):
        self.val = val
        self.zeroed = False

    def zero_(self):
        self.val = 0.0
        self.zeroed = True


def _fake_opt():
    """An optimizer whose `.state` mirrors AdamW's: {param: {'exp_avg':..., 'exp_avg_sq':...}}."""
    state = {
        "p0": {"exp_avg": _FakeMoment(), "exp_avg_sq": _FakeMoment(), "step": 5},
        "p1": {"exp_avg": _FakeMoment(), "exp_avg_sq": _FakeMoment(), "step": 5},
    }
    return SimpleNamespace(state=state, param_groups=[{"lr": 3e-4}])


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


def test_reset_zero_moments():
    opt = _fake_opt()
    n = reset.zero_moments(opt)
    assert n == 4  # 2 params x (exp_avg, exp_avg_sq)
    assert all(st["exp_avg"].zeroed and st["exp_avg_sq"].zeroed for st in opt.state.values())
    assert all(st["exp_avg"].val == 0.0 and st["exp_avg_sq"].val == 0.0 for st in opt.state.values())


def test_reset_as_pre_step_fires_only_at_step():
    opt = _fake_opt()
    pre = reset.as_pre_step(at_step=12)
    pre(11, {"opt": opt})  # before -> untouched
    assert not any(st["exp_avg"].zeroed for st in opt.state.values())
    pre(12, {"opt": opt})  # at window-end -> zeroed
    assert all(st["exp_avg"].zeroed for st in opt.state.values())


def test_skip_resets_batch_inside_window_only():
    pre = skip.as_pre_step(inject_step=10, width=2)
    # inside [10,12): a garbage override is reset to the aligned clean slot (None)
    for step in (10, 11):
        ctx = {"batch": ("garbage", "garbage")}
        pre(step, ctx)
        assert ctx["batch"] is None, step
    # outside: left untouched
    for step in (9, 12):
        ctx = {"batch": ("garbage", "garbage")}
        pre(step, ctx)
        assert ctx["batch"] == ("garbage", "garbage"), step


def test_clip_apply_to_cfg_is_nonmutating():
    cfg = {"optim": {"lr": 3e-4, "betas": [0.9, 0.95]}, "train": {"batch_size": 16}}
    out = clip.apply_to_cfg(cfg, 1.0)
    assert out["optim"]["grad_clip"] == 1.0
    assert out["optim"]["lr"] == 3e-4 and out["optim"]["betas"] == [0.9, 0.95]  # other keys preserved
    assert "grad_clip" not in cfg["optim"]  # caller's cfg untouched
    assert out["train"] is cfg["train"]  # shallow: unrelated subtrees shared, not copied


# --------------------------------------------------------------------------------------
# Composition + branch factories
# --------------------------------------------------------------------------------------


def test_compose_order_is_spike_then_fix():
    def spike(step, ctx):
        ctx["batch"] = ("garbage", "garbage")
        ctx["lr_bumped"] = True

    fix = skip.as_pre_step(inject_step=0, width=100)  # always in-window
    pre = compose(spike, fix)
    ctx = {"batch": None}
    pre(0, ctx)
    # fix ran AFTER spike -> the corrupted batch was reset to clean, but the LR bump remains
    assert ctx["batch"] is None
    assert ctx["lr_bumped"] is True


def test_compose_drops_none_and_all_none_is_none():
    calls = []
    pre = compose(None, lambda s, c: calls.append(s), None)
    pre(3, {})
    assert calls == [3]
    assert compose(None, None) is None


def _fake_recipe(inject_step=10, width=2):
    """A stand-in for spikes.induce.SpikeRecipe: only pre_step/inject_step/width are read here."""

    def spike(step, ctx):
        if inject_step <= step < inject_step + width:
            ctx["batch"] = ("garbage", "garbage")
            ctx.setdefault("opt", SimpleNamespace()).spiked = True

    return SimpleNamespace(pre_step=spike, inject_step=inject_step, width=width)


def test_make_branches_shapes():
    cfg = {"optim": {"lr": 3e-4}, "train": {"batch_size": 16}, "seed": 0}
    recipe = _fake_recipe(inject_step=10, width=2)
    br = make_branches(cfg, recipe, clip_norm=1.0)
    assert set(br) == {"B0", "B*", "Bg", "Bs"}
    assert all(isinstance(b, Branch) for b in br.values())
    # B* is the clean counterfactual: NO spike replayed.
    assert br["B*"].pre_step is None
    # B0 replays the spike unmodified (exactly the recipe's own hook).
    assert br["B0"].pre_step is recipe.pre_step
    # Bs runs a clip-augmented cfg; the others reuse the trunk cfg.
    assert br["Bs"].cfg["optim"]["grad_clip"] == 1.0
    assert "grad_clip" not in br["B0"].cfg["optim"]


def test_make_branches_bg_resets_after_window():
    cfg = {"optim": {"lr": 3e-4}, "train": {"batch_size": 16}}
    br = make_branches(cfg, _fake_recipe(inject_step=10, width=2), clip_norm=1.0)
    opt = _fake_opt()
    # Before window-end (10+2=12): spike may fire but Bg has NOT reset yet.
    br["Bg"].pre_step(11, {"opt": opt, "batch": None})
    assert not any(st["exp_avg"].zeroed for st in opt.state.values())
    # At window-end: the global reset fires exactly once.
    br["Bg"].pre_step(12, {"opt": opt, "batch": None})
    assert all(st["exp_avg"].zeroed and st["exp_avg_sq"].zeroed for st in opt.state.values())


def test_make_branches_bs_skip_beats_spike_batch_corruption():
    cfg = {"optim": {"lr": 3e-4}, "train": {"batch_size": 16}}
    br = make_branches(cfg, _fake_recipe(inject_step=10, width=2), clip_norm=1.0)
    ctx = {"opt": _fake_opt(), "batch": None}
    br["Bs"].pre_step(10, ctx)  # spike corrupts, skip (runs after) restores the clean aligned slot
    assert ctx["batch"] is None


# --------------------------------------------------------------------------------------
# NaN-safety
# --------------------------------------------------------------------------------------


def test_summarize_nan_safety():
    ok = _summarize("B0", [2.0, 1.5, 1.1])
    assert ok["survival"] == 1 and ok["final"] == 1.1
    for bad in ([2.0, float("nan")], [2.0, float("inf")], []):
        s = _summarize("Bx", bad)
        assert s["survival"] == 0 and math.isnan(s["final"])


# --------------------------------------------------------------------------------------
# Decision rule + renderer
# --------------------------------------------------------------------------------------


def _recover_deltas():  # Bs reaches clean (Δ ~ 0); B* is 0 by definition
    return {"Bs": [0.0, 0.01, -0.005], "B*": [0.0, 0.0, 0.0], "Bg": [0.0, 0.0, 0.0], "B0": [0.6, 0.55, 0.62]}


def _gap_deltas():  # Bs leaves a clear, large gap to clean
    return {"Bs": [0.5, 0.55, 0.52], "B*": [0.0, 0.0, 0.0], "Bg": [0.02, 0.0, 0.03], "B0": [0.6, 0.58, 0.62]}


def test_verdict_proceed_when_gap_remains():
    v = gate_b_verdict({"lr_bump": _gap_deltas()}, tau=0.05)
    assert v["decision"] == "PROCEED"
    assert v["provisional"] is True  # 1 of 4 recipes
    assert v["per_recipe"]["lr_bump"]["Bs"]["recovers"] is False


def test_verdict_pivot_only_when_all_recipes_recover():
    four = {r: _recover_deltas() for r in ("lr_bump", "tiny_eps", "precision", "corrupt_batch")}
    v = gate_b_verdict(four, tau=0.05)
    assert v["decision"] == "PIVOT"
    assert v["provisional"] is False


def test_verdict_inconclusive_when_ci_straddles_tau():
    straddle = {"Bs": [0.0, 0.1, 0.02], "B*": [0.0, 0.0, 0.0]}
    v = gate_b_verdict({"lr_bump": straddle}, tau=0.05)
    assert v["decision"] == "INCONCLUSIVE"


def test_one_clear_gap_forces_proceed_even_if_others_recover():
    mixed = {
        "lr_bump": _gap_deltas(),  # clear gap
        "tiny_eps": _recover_deltas(),  # recovers
    }
    assert gate_b_verdict(mixed, tau=0.05)["decision"] == "PROCEED"


def test_renderer_hard_labels_provisional():
    prov = render_verdict(gate_b_verdict({"lr_bump": _gap_deltas()}, tau=0.05))
    assert "PROVISIONAL — NOT the binding Gate B verdict" in prov
    assert "(PROVISIONAL)" in prov  # tagged on the decision line, not just a caveat
    assert "1/4 spike recipes" in prov

    full = {r: _recover_deltas() for r in ("lr_bump", "tiny_eps", "precision", "corrupt_batch")}
    final = render_verdict(gate_b_verdict(full, tau=0.05))
    assert "PROVISIONAL" not in final  # a full 4-recipe run carries no provisional label
    assert "Decision: PIVOT" in final


def _main() -> None:
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
    print(f"kill-test selfcheck OK: {len(fns)} tests passed (branches, compose, NaN-safety, verdict+renderer)")


if __name__ == "__main__":
    _main()
