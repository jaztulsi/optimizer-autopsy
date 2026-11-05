"""CPU tests for the LR schedule (harness/schedule.py, applied in train_forward) and the fault recipes +
parity residual (autopsy/faults.py). PLAN_V8 §9.2: "lr_bump multiplies the schedule", "bit-flip recipes are
deterministic and reversible", "the parity residual is 0 on honest Adam and nonzero on an injected edit".

Same 1-layer, 16-dim GPT on a synthetic memmap as test_baselines. The sdc_grad checks look at gradients
BEFORE the clip only: at bit 30 the clip's norm overflows fp32 on GPU but maybe not on CPU, so post-clip
behaviour is device-dependent and not tested here.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch, as in test_baselines

import pytest  # noqa: E402
import torch  # noqa: E402

from research.autopsy import faults  # noqa: E402
from research.autopsy.faults import TARGET, flip_bit_  # noqa: E402
from research.harness.fork import run_fork  # noqa: E402
from research.harness.schedule import lr_at  # noqa: E402
from research.harness.snapshot import capture  # noqa: E402
from research.harness.trunk import train_forward  # noqa: E402
from research.spikes.induce import induce  # noqa: E402
from research.tests.test_baselines import _CFG, _data_dir, _fresh, _tmpdir  # noqa: E402

_SCHED = {**_CFG, "optim": {**_CFG["optim"], "schedule": {"kind": "cosine", "warmup": 2}}}  # total = max_steps 6
_NOCLIP = {**_CFG, "optim": {**_CFG["optim"], "grad_clip": 0.0}}


def _diffs(a: dict, b: dict) -> list:
    """(name, flat index) of every element whose bits differ (int32 view: NaN-safe, exact)."""
    out = []
    for n in a:
        out += [(n, int(i)) for i in (a[n].view(torch.int32) != b[n].view(torch.int32)).reshape(-1).nonzero()]
    return out


# --------------------------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------------------------


def test_lr_at_warmup_cosine_wsd_and_validation():
    def cfg(**s):
        return {"optim": {"lr": 1.0, "schedule": s}, "train": {"max_steps": 100}}

    assert lr_at(7, {"optim": {"lr": 3e-4}}) == 3e-4  # no schedule: the cfg lr, exactly (pilot reproducible)

    cos = cfg(kind="cosine", warmup=10)  # min_lr defaults to 0.1 * lr, total to max_steps
    assert [lr_at(s, cos) for s in (0, 4, 9)] == pytest.approx([0.1, 0.5, 1.0])
    assert lr_at(10, cos) == pytest.approx(1.0) and lr_at(55, cos) == pytest.approx(0.55)
    assert lr_at(100, cos) == pytest.approx(0.1) and lr_at(150, cos) == pytest.approx(0.1)
    assert all(lr_at(s, cos) >= lr_at(s + 1, cos) for s in range(10, 100))

    wsd = cfg(kind="wsd", warmup=10, decay=20, min_lr=0.0)
    assert all(lr_at(s, wsd) == 1.0 for s in range(10, 81))  # stable until total - decay
    assert lr_at(90, wsd) == pytest.approx(0.5) and lr_at(100, wsd) == pytest.approx(0.0, abs=1e-12)
    dflt = cfg(kind="wsd")  # decay defaults to 0.2 * total, min_lr to 0.1
    assert lr_at(79, dflt) == 1.0 and lr_at(90, dflt) == pytest.approx(0.55)

    for bad in (cfg(kind="linear"), cfg(kind="cosine", warmpu=10), cfg(warmup=10)):
        with pytest.raises(ValueError):
            lr_at(0, bad)


def _lrs(cfg, data, pre_step=None) -> list:
    """Every step's LR as the optimizer saw it (all param groups must agree)."""
    seen = []

    def rec(step, info):
        seen.append({g["lr"] for g in info["opt"].param_groups})

    model, opt = _fresh(cfg)
    train_forward(model, opt, cfg, data, 0, 6, "cpu", pre_step=pre_step, on_step=rec)
    assert all(len(x) == 1 for x in seen), seen
    return [x.pop() for x in seen]


def test_train_forward_applies_the_schedule_before_pre_step_so_lr_bump_scales_it():
    with _tmpdir() as d:
        data = _data_dir(d)
        want = [lr_at(s, _SCHED) for s in range(6)]
        assert len(set(want)) >= 4  # the schedule moves, so a constant LR can't pass
        assert _lrs(_SCHED, data) == want
        bump = induce("lr_bump", _SCHED, inject_step=3, width=2, factor=10.0)
        assert _lrs(_SCHED, data, bump.pre_step) == [w * (10.0 if s in (3, 4) else 1.0) for s, w in enumerate(want)]


def test_fork_from_a_snapshot_stays_on_the_schedule():
    # A fork restarts train_forward at start_step = 3: the LR must follow the absolute step, not a relative one.
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh(_SCHED)
        train_forward(model, opt, _SCHED, data, 0, 3, "cpu")
        snap = capture(model, opt, step=3)
        seen = []
        run_fork(_SCHED, data, snap, steps=3, device="cpu", seed=False, on_step=lambda s, i: seen.append((s, i["lr"])))
        assert seen == [(s, lr_at(s, _SCHED)) for s in (3, 4, 5)]


# --------------------------------------------------------------------------------------
# Bit flips
# --------------------------------------------------------------------------------------


def test_flip_bit_is_reversible_and_moves_the_exponent():
    torch.manual_seed(0)
    t = torch.randn(64)
    orig = t.clone()
    assert flip_bit_(t, 7, 30)[0] == float(orig[7])
    assert _diffs({"t": t}, {"t": orig}) == [("t", 7)]
    flip_bit_(t, 7, 30)
    assert _diffs({"t": t}, {"t": orig}) == []  # flipping twice restores it bitwise

    before, after = flip_bit_(torch.tensor([1e-8]), 0, 30)
    assert after == before * 2.0**128  # |x| < 2: bit 30 clear -> up-flip
    assert flip_bit_(torch.tensor([4.0]), 0, 30) == (4.0, 4.0 * 2.0**-128)  # |x| >= 2: down-flip
    assert flip_bit_(torch.tensor([3.0]), 0, 31) == (3.0, -3.0)  # sign bit
    for bad in (torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, 4).t()):
        with pytest.raises(TypeError):
            flip_bit_(bad, 0, 30)
    with pytest.raises(ValueError):
        flip_bit_(torch.zeros(4), 0, 32)


def _state_at(data, at, recipe=None, steps=None):
    """Train from scratch; return ({key: {name: tensor}} at the top of step `at`, after the recipe fired), losses."""
    model, opt = _fresh()
    grab = {}

    def pre(step, ctx):
        if recipe is not None:
            recipe.pre_step(step, ctx)
        if step == at:
            for key in ("exp_avg", "exp_avg_sq"):
                grab[key] = {n: opt.state[p][key].clone() for n, p in model.named_parameters()}

    losses = train_forward(model, opt, _CFG, data, 0, steps or at + 1, "cpu", pre_step=pre)
    return grab, losses


@pytest.mark.parametrize("kind,key", [("v_bitflip", "exp_avg_sq"), ("m_bitflip", "exp_avg")])
def test_state_flip_edits_exactly_one_element_deterministically(kind, key):
    other = "exp_avg" if key == "exp_avg_sq" else "exp_avg_sq"
    with _tmpdir() as d:
        data = _data_dir(d)
        clean, _ = _state_at(data, 3)
        r = induce(kind, _CFG, inject_step=3, width=1, index=5, bit=29)
        hit, losses = _state_at(data, 3, r, steps=5)
        assert _diffs(hit[key], clean[key]) == [(TARGET, 5)] and _diffs(hit[other], clean[other]) == []
        was, now = float(clean[key][TARGET].view(-1)[5]), float(hit[key][TARGET].view(-1)[5])
        assert 0 < abs(now) < abs(was) * 1e-18  # bit 29 on |x| < 2 is a down-flip by 2^64
        again, losses2 = _state_at(data, 3, r, steps=5)  # same recipe object, as forks reuse it
        assert _diffs(hit[key], again[key]) == [] and losses == losses2


def test_fault_recipes_reject_bad_use():
    with pytest.raises(ValueError):
        induce("v_bitflip", _CFG, inject_step=3, width=2)  # a flip is a one-step fault
    with pytest.raises(ValueError):
        induce("sdc_grad", _CFG, inject_step=3, width=1, bit=32)
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        with pytest.raises(RuntimeError):  # no Adam state before the first step
            train_forward(model, opt, _CFG, data, 0, 1, "cpu", pre_step=induce("m_bitflip", _CFG, 0).pre_step)
        model, opt = _fresh()
        bad_target = induce("sdc_grad", _CFG, 0, target="nope")
        with pytest.raises(KeyError):
            train_forward(model, opt, _CFG, data, 0, 1, "cpu", pre_step=bad_target.pre_step)


def test_sdc_grad_flips_the_first_microbatch_once_before_the_clip(monkeypatch):
    cfg = {**_NOCLIP, "train": {**_CFG["train"], "grad_accum": 2}}
    calls = []
    real = faults.flip_bit_

    def counting(t, index, bit):
        calls.append((index, bit))
        return real(t, index, bit)

    monkeypatch.setattr(faults, "flip_bit_", counting)

    def run(pre_step=None):
        micro = {}  # (step, micro) -> {name: grad}; grad_hook sees p.grad pre-clip, after the tensor hook
        model, opt = _fresh(cfg)

        def gh(step, m, model):
            micro[step, m] = {n: p.grad.clone() for n, p in model.named_parameters()}

        losses = train_forward(model, opt, cfg, data, 0, 5, "cpu", grad_hook=gh, pre_step=pre_step)
        return micro, losses, dict(model.named_parameters())[TARGET].detach().clone()

    with _tmpdir() as d:
        data = _data_dir(d)
        clean, clean_losses, clean_w = run()
        r = induce("sdc_grad", cfg, inject_step=2, width=1, index=3, bit=26)
        micro, losses, w = run(r.pre_step)
        assert calls == [(3, 26)]  # once: not on microbatch 1, not on steps 3-4
        assert all(_diffs(micro[s, m], clean[s, m]) == [] for s in (0, 1) for m in (0, 1))
        assert _diffs(micro[2, 0], clean[2, 0]) == [(TARGET, 3)]
        got, was = micro[2, 0][TARGET].view(-1).view(torch.int32)[3], clean[2, 0][TARGET].view(-1).view(torch.int32)[3]
        assert int(got) == int(was) ^ (1 << 26)  # exactly bit 26 of that element
        assert _diffs(micro[2, 1], clean[2, 1]) == []  # the second microbatch's gradient is honest
        # The flip reached the weights. Not the loss: here g ~ 2e-5 has bit 26 set (a /256 flip), so the loss moves
        # by ~ g * dw ~ 1e-8, below fp32 resolution at loss ~2.5 (CI 2026-11-05: losses bitwise equal).
        assert losses[:3] == clean_losses[:3] and not torch.equal(w, clean_w)
        _, again, _ = run(r.pre_step)  # reused recipe object: flips again, identically
        assert again == losses and len(calls) == 2


# --------------------------------------------------------------------------------------
# Parity residual
# --------------------------------------------------------------------------------------


def _parity(cfg, data, recipe=None, steps=6):
    out = {}
    model, opt = _fresh(cfg)
    pre = None if recipe is None else recipe.pre_step
    train_forward(model, opt, cfg, data, 0, steps, "cpu", pre_step=pre, on_step=faults.parity_residual(out))
    return out, dict(model.named_parameters())[TARGET].detach().clone()


def test_parity_residual_is_rounding_on_honest_adam_and_flags_state_flips():
    with _tmpdir() as d:
        data = _data_dir(d)
        honest, _ = _parity(_CFG, data)
        assert sorted(honest) == [1, 2, 3, 4, 5]  # step 0 has no previous copy, so it is not recorded
        assert max(max(r.values()) for r in honest.values()) < 1e-5

        for kind, ch, bit in (("v_bitflip", "v", 29), ("m_bitflip", "m", 30)):  # v down-flip; m up-flip
            out, _ = _parity(_CFG, data, induce(kind, _CFG, inject_step=3, width=1, bit=bit), steps=4)
            assert max(max(out[s].values()) for s in (1, 2)) < 1e-5, (kind, out)
            assert out[3][ch] > 1e-2, (kind, out)


def test_parity_residual_cannot_see_a_gradient_fault_by_design():
    # sdc_grad corrupts g, and m and v both integrate the same corrupt g: the recurrence still holds.
    with _tmpdir() as d:
        data = _data_dir(d)
        _, clean_w = _parity(_NOCLIP, data)
        out, w = _parity(_NOCLIP, data, induce("sdc_grad", _NOCLIP, inject_step=3, width=1, bit=26))
        assert not torch.equal(w, clean_w)  # the fault did land (in w; the loss is too coarse, see the sdc test) ...
        assert sorted(out) == [1, 2, 3, 4, 5] and max(max(r.values()) for r in out.values()) < 1e-5  # ... unseen
