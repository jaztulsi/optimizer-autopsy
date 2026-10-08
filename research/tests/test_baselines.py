"""CPU unit tests for the Task-15 gradient-side baselines and the `pre_opt_step` hook they ride on.

Covers: SPAM's schedule (reset / clip / warmup phase vs the reference's bookkeeping), the spike-aware
clip and the LR/WD warmup + undo on a real torch AdamW; skip-step (no update, data still aligned) and the
v-only reset; ZClip's and AdaGC's clipping rules, and that
priming them in shadow mode on a trunk yields EXACTLY the state they'd reach running live (AdaGC
included, whose trunk-side grads arrive already globally clipped); and the hook's wiring in `train_forward` /
`run_fork` -- including that a no-op hook leaves the trajectory BITWISE unchanged (the Δ==0 gate must
not move just because the hook exists). The wiring tests run a 1-layer, 16-dim GPT for a few steps on a
synthetic memmap: unit-test scale, nothing like a trunk run.

Runnable two ways:  `pytest research/tests/test_baselines.py`  or  `python -m research.tests.test_baselines`.
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere in the process; setdefault keeps a launcher's value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math  # noqa: E402
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.baselines import adagc, reset, skip_step, spam, zclip  # noqa: E402
from research.harness.fork import Branch, make_baseline_branches, prime_trunk, run_branch, run_fork  # noqa: E402
from research.harness.snapshot import capture  # noqa: E402
from research.harness.trunk import build_model_opt, run_trunk, train_forward  # noqa: E402

_CFG = {
    "seed": 0,
    "model": {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 8,
        "vocab_size": 16,
        "dropout": 0.0,
        "bias": False,
    },
    "optim": {"lr": 3e-3, "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0},
    "train": {"batch_size": 4, "grad_accum": 1, "max_steps": 6},
}


def _tmpdir():
    # The data memmap keeps train.bin open, which blocks cleanup on Windows; harmless elsewhere.
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _data_dir(d: str) -> str:
    np.tile(np.arange(16, dtype=np.uint16), 500).tofile(f"{d}/train.bin")
    return d


def _weights(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])


def _fresh(cfg=_CFG):
    torch.manual_seed(0)
    return build_model_opt(cfg, "cpu")


# --------------------------------------------------------------------------------------
# SPAM schedule (pure)
# --------------------------------------------------------------------------------------


def test_spam_warmup_scale_matches_reference_cosine():
    # The reference: 1 - CosineAnnealingLR(SGD(lr=0.99), T_max=warmup+1, eta_min=0) at epoch c.
    w = 150
    sgd = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.99)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(sgd, T_max=w + 1, eta_min=0)
    for c in range(w):
        dr = sgd.param_groups[0]["lr"]
        assert math.isclose(spam.warmup_scale(c, w), 1.0 - dr, rel_tol=0, abs_tol=1e-9), c
        sgd.step()
        sched.step()
    assert math.isclose(spam.warmup_scale(0, w), 0.01)
    assert spam.warmup_scale(w, w) == 1.0 and spam.warmup_scale(10 * w, w) == 1.0


def test_spam_phase_matches_reference_bookkeeping():
    kw = {"delta_t": 500, "warmup": 150, "grace": 20}
    # Start of training: no clipping until cur >= grace, never any warmup before the first reset.
    assert spam.phase(0, **kw) == (False, False, 1.0)
    assert spam.phase(18, **kw)[1] is False and spam.phase(19, **kw)[1] is True  # cur = 20
    assert all(spam.phase(t, **kw)[2] == 1.0 for t in range(498))
    # First reset at total_step 499 (cur 500); warmup starts at 0.01 on that same step.
    reset, clip, scale = spam.phase(499, **kw)
    assert reset and not clip and math.isclose(scale, 0.01)
    # Clip is off for `grace` steps after the reset, then back on; warmup ends after `warmup` steps.
    assert spam.phase(499 + 19, **kw)[1] is False and spam.phase(499 + 20, **kw)[1] is True
    assert spam.phase(499 + 149, **kw)[2] < 1.0 and spam.phase(499 + 150, **kw)[2] == 1.0
    assert spam.phase(999, **kw)[0] and not spam.phase(998, **kw)[0]
    # delta_t=0: never reset, never warm up, clip after grace.
    assert spam.phase(10_000, delta_t=0) == (False, True, 1.0)


# --------------------------------------------------------------------------------------
# SPAM on a real AdamW (no model, no data)
# --------------------------------------------------------------------------------------


def _adamw_with_state(v_val: float):
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(2))  # no-decay group, like nanoGPT's biases/norms
    opt = torch.optim.AdamW([{"params": [p], "weight_decay": 0.1}, {"params": [q], "weight_decay": 0.0}], lr=1e-3)
    p.grad, q.grad = torch.ones(4), torch.ones(2)
    opt.step()  # populate exp_avg / exp_avg_sq / step
    opt.state[p]["exp_avg_sq"].fill_(v_val)
    opt.state[q]["exp_avg_sq"].fill_(v_val)
    return p, q, opt


def test_spam_spike_clip_only_touches_spikes():
    p, q, opt = _adamw_with_state(v_val=1e-4)  # theta*v = 0.5 -> bound sqrt(0.5)
    p.grad = torch.tensor([0.1, -0.1, 3.0, -5.0])  # last two are spikes: g^2 > 0.5
    q.grad = None  # params without a grad are skipped
    n = spam.spike_clip_(opt, theta=5000.0)
    bound = math.sqrt(5000.0 * 1e-4)
    assert n == 2
    assert torch.allclose(p.grad, torch.tensor([0.1, -0.1, bound, -bound]))


def test_spam_hook_reset_warmup_and_undo():
    p, q, opt = _adamw_with_state(v_val=1.0)
    step_before = opt.state[p]["step"].clone()
    hook = spam.as_pre_opt_step()  # paper defaults: reset at global step 499
    ctx = {"opt": opt}
    undo = hook(499, ctx)
    assert ctx["spam"] == {"reset": True, "clipped": 0, "scale": spam.warmup_scale(0)}
    assert all(float(opt.state[x]["exp_avg"].abs().sum()) == 0.0 for x in (p, q))
    assert all(float(opt.state[x]["exp_avg_sq"].abs().sum()) == 0.0 for x in (p, q))
    assert torch.equal(opt.state[p]["step"], step_before)  # bias-correction step is kept, as in the reference
    g0, g1 = opt.param_groups
    assert math.isclose(g0["lr"], 1e-3 * 0.01) and math.isclose(g0["lr"] * g0["weight_decay"], 1e-3 * 0.1)
    assert g1["weight_decay"] == 0.0
    undo()
    assert (g0["lr"], g0["weight_decay"], g1["lr"], g1["weight_decay"]) == (1e-3, 0.1, 1e-3, 0.0)
    # Mid-run, no reset and no warmup: nothing to undo.
    assert hook(250, {"opt": opt}) is None


# --------------------------------------------------------------------------------------
# ZClip
# --------------------------------------------------------------------------------------


def _grad_model(grads):
    """A module whose named params p0, p1, ... carry the given grads."""
    m = torch.nn.Module()
    for i, g in enumerate(grads):
        p = torch.nn.Parameter(torch.zeros_like(g))
        p.grad = g.clone()
        m.register_parameter(f"p{i}", p)
    return m


def _total_norm(model) -> float:
    return float(torch.cat([p.grad.reshape(-1) for p in model.parameters()]).norm())


def test_zclip_warmup_then_adaptive_clip():
    z = zclip.ZClip(warmup=3, max_norm=None)
    for n in (1.0, 2.0, 3.0):  # warmup: buffer only, no clip (max_norm=None)
        assert z._update(n) is None
    assert z.initialized and z.mean == 2.0 and math.isclose(z.var, 2 / 3)
    assert z._update(2.1) == 2.1 and math.isclose(z.mean, 0.97 * 2.0 + 0.03 * 2.1)  # z < 2.5: untouched

    z = zclip.ZClip(warmup=3, max_norm=None)
    for n in (1.0, 2.0, 3.0):
        z.observe(n)
    std = math.sqrt(2 / 3)
    zs = (10.0 - 2.0) / (std + 1e-6)
    want = 2.0 + 2.5 * std / (zs / 2.5)  # the reference's adaptive_scaling threshold
    model = _grad_model([torch.full((4,), 5.0)])  # ||g|| = 10
    ctx = {"model": model, "grad_norm": _total_norm(model)}
    z(0, ctx)
    assert ctx["zclip"]["clipped"] and math.isclose(_total_norm(model), want, rel_tol=1e-5)
    # The EMA ingests the CLIPPED value, and the variance uses the updated mean (reference order).
    mean = 0.97 * 2.0 + 0.03 * want
    assert math.isclose(z.mean, mean) and math.isclose(z.var, 0.97 * (2 / 3) + 0.03 * (want - mean) ** 2)


def test_zclip_max_norm_caps_during_warmup():
    z = zclip.ZClip(warmup=2, max_norm=1.0)
    model = _grad_model([torch.full((4,), 1.0)])  # ||g|| = 2
    z(0, {"model": model, "grad_norm": 2.0})
    assert math.isclose(_total_norm(model), 1.0, rel_tol=1e-5)  # warmup: plain max-norm clip


def test_zclip_shadow_priming_matches_live_and_never_touches_grads():
    norms = [1.0 + 0.1 * (i % 7) for i in range(40)] + [9.0, 1.2, 1.1]
    live, shadow = zclip.ZClip(max_norm=None), zclip.ZClip(max_norm=None)
    prime = shadow.observer()
    clips, raw_ema = 0, zclip.ZClip(max_norm=None, z_thresh=float("inf"))  # never clips: EMA of raw norms
    for n in norms:
        model = _grad_model([torch.full((1,), n)])
        ctx = {"model": model, "grad_norm": n}
        live(0, ctx)
        clips += ctx["zclip"]["clipped"]
        raw_ema.observe(n)
        g = torch.full((1,), n)
        model = _grad_model([g])
        prime(0, {"grad_norm": n, "model": model})
        assert torch.equal(model.p0.grad, g)  # shadow: grads untouched
    assert clips >= 1  # the 9.0 spike was clipped, so the match below is not vacuous
    assert live.state_dict() == shadow.state_dict()
    assert shadow.mean < raw_ema.mean  # priming fed the CLIPPED size, not the raw spike, into the EMA
    # One primed state seeds independent copies.
    a, b = zclip.as_pre_opt_step(live.state_dict()), zclip.as_pre_opt_step(live.state_dict())
    a.observe(100.0)
    assert b.state_dict() == live.state_dict() != a.state_dict()


# --------------------------------------------------------------------------------------
# AdaGC
# --------------------------------------------------------------------------------------


def test_adagc_warmup_global_clip_and_min_seed():
    a = adagc.AdaGC(start_clip_steps=2, max_norm=1.0)
    m = _grad_model([torch.full((4,), 1.0), torch.full((4,), 0.5)])  # tensor norms 2, 1; total sqrt(5)
    a(0, {"model": m, "grad_norm": _total_norm(m)})
    c = 1.0 / (math.sqrt(5) + 1e-6)
    assert math.isclose(_total_norm(m), 1.0, rel_tol=1e-5)  # warm-up = global clip
    assert math.isclose(float(a.gamma["p0"]), 2 * c, rel_tol=1e-5)
    m = _grad_model([torch.full((4,), 0.1), torch.full((4,), 0.1)])  # under max_norm: not clipped
    a(1, {"model": m, "grad_norm": _total_norm(m)})
    assert math.isclose(float(a.gamma["p0"]), 0.2, rel_tol=1e-5)  # the MIN post-clip norm wins
    assert math.isclose(float(a.gamma["p1"]), 0.2, rel_tol=1e-5)


def test_adagc_clips_only_the_spiking_tensor():
    a = adagc.AdaGC(start_clip_steps=0)
    a.gamma = {"p0": torch.tensor(1.0), "p1": torch.tensor(1.0)}
    calm = torch.full((4,), 0.5)  # norm 1.0 <= 1.04
    m = _grad_model([torch.full((4,), 5.0), calm])  # p0 norm 10 -> clipped to 1.04
    ctx = {"model": m, "grad_norm": _total_norm(m)}
    a(0, ctx)
    assert ctx["adagc"]["clipped_tensors"] == 1
    assert math.isclose(float(m.p0.grad.norm()), 1.04 * 10 / (10 + 1e-6), rel_tol=1e-6)
    assert torch.equal(m.p1.grad, calm)  # tensor-wise locality: the calm tensor is bitwise untouched
    assert math.isclose(float(a.gamma["p0"]), 0.99 * 1.0 + 0.01 * 1.04, rel_tol=1e-6)  # EMA of CLIPPED norm
    assert math.isclose(float(a.gamma["p1"]), 0.99 * 1.0 + 0.01 * 1.0, rel_tol=1e-6)


def test_adagc_shadow_priming_undoes_trunk_clip_and_matches_live():
    # Live AdaGC sees raw grads. The trunk's on_step sees the SAME grads after its own global clip;
    # priming must land on the identical gamma anyway, in both the warm-up and adaptive phases.
    trunk_clip = 0.5
    live = adagc.AdaGC(start_clip_steps=3, max_norm=0.8)
    shadow = adagc.AdaGC(start_clip_steps=3, max_norm=0.8)
    prime = shadow.observer(trunk_clip)
    for k in range(10):
        scale = 4.0 if k == 7 else 1.0 + 0.05 * k  # a spike at k=7, in the adaptive phase
        raw = [torch.full((4,), 0.3 * scale), torch.full((2,), 0.2 * scale)]
        m_live = _grad_model(raw)
        live(k, {"model": m_live, "grad_norm": _total_norm(m_live)})

        m_trunk = _grad_model(raw)
        raw_norm = _total_norm(m_trunk)
        torch.nn.utils.clip_grad_norm_(m_trunk.parameters(), trunk_clip)  # what the trunk applied
        seen = [p.grad.clone() for p in m_trunk.parameters()]
        prime(k, {"model": m_trunk, "grad_norm": raw_norm})
        assert all(torch.equal(p.grad, g) for p, g in zip(m_trunk.parameters(), seen))  # shadow: untouched
    assert live.steps == shadow.steps == 10
    for name in live.gamma:
        assert math.isclose(float(live.gamma[name]), float(shadow.gamma[name]), rel_tol=1e-5), name

    # state_dict round-trips by name into a fresh instance (fresh params, as in a fork).
    fresh = adagc.as_pre_opt_step(shadow.state_dict())
    assert fresh.steps == 10 and set(fresh.gamma) == {"p0", "p1"}


# --------------------------------------------------------------------------------------
# Skip-step and v-only reset
# --------------------------------------------------------------------------------------


def test_v_only_reset_keeps_m():
    p, q, opt = _adamw_with_state(v_val=1.0)
    m_before = {x: opt.state[x]["exp_avg"].clone() for x in (p, q)}
    step_before = opt.state[p]["step"].clone()
    assert reset.zero_moments(opt, reset.V_ONLY) == 2  # one v per param
    for x in (p, q):
        assert float(opt.state[x]["exp_avg_sq"].abs().sum()) == 0.0
        assert torch.equal(opt.state[x]["exp_avg"], m_before[x]) and float(m_before[x].abs().sum()) > 0
    assert torch.equal(opt.state[p]["step"], step_before)


def _opt_state(opt):
    return [(st["step"].clone(), st["exp_avg"].clone(), st["exp_avg_sq"].clone()) for st in opt.state.values()]


def test_skip_step_freezes_w_m_v_and_step_count_but_keeps_data_aligned():
    with _tmpdir() as d:
        data = _data_dir(d)
        trace = []

        def record(step, info):
            trace.append((_weights(info["model"]), _opt_state(info["opt"])))

        model, opt = _fresh()
        skip = skip_step.as_pre_opt_step(inject_step=1, width=2)  # skip the updates of steps 1 and 2
        losses = train_forward(model, opt, _CFG, data, 0, 5, "cpu", on_step=record, pre_opt_step=skip)
        assert len(losses) == 5 and all(math.isfinite(x) for x in losses)  # batches still consumed + logged
        w0, s0 = trace[0]
        for w, s in trace[1:3]:  # inside the window: nothing moved
            assert torch.equal(w, w0)
            assert all(torch.equal(a, b) for sa, sb in zip(s, s0) for a, b in zip(sa, sb))
        assert not torch.equal(trace[3][0], w0)  # updates resume after the window
        assert all(float(st["step"]) == 3 for st in opt.state.values())  # 5 steps - 2 skipped

        # Data alignment: step 3 trains on step 3's batch, not step 1's. Replaying steps 0 and 3..4 by hand
        # (no hook, start_step jumps the skipped slots) must give the same losses and weights.
        model2, opt2 = _fresh()
        ref = train_forward(model2, opt2, _CFG, data, 0, 1, "cpu")
        ref += train_forward(model2, opt2, _CFG, data, 3, 2, "cpu")
        assert [losses[0], losses[3], losses[4]] == ref
        assert torch.equal(_weights(model), _weights(model2))


# --------------------------------------------------------------------------------------
# Hook wiring in train_forward / run_fork
# --------------------------------------------------------------------------------------


def test_noop_pre_opt_step_is_bitwise_identical():
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        base = train_forward(model, opt, _CFG, data, 0, 4, "cpu")
        w_base = _weights(model)

        calls = []
        model, opt = _fresh()
        hooked = train_forward(model, opt, _CFG, data, 0, 4, "cpu", pre_opt_step=lambda s, c: calls.append(s))
        assert calls == [0, 1, 2, 3]
        assert hooked == base
        assert torch.equal(_weights(model), w_base)


def test_pre_opt_step_sees_final_grad_and_undo_runs_after_step():
    # A tiny clip norm, so the global clip fires every step: a hook on the FINAL gradient must see
    # ||g|| == clip, not the raw per-microbatch or pre-clip norm.
    clip = 1e-3
    cfg = {**_CFG, "optim": {**_CFG["optim"], "grad_clip": clip}, "train": {**_CFG["train"], "grad_accum": 2}}
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh(cfg)
        seen = []

        def freeze_step_2(step, ctx):
            g_norm = float(torch.cat([p.grad.reshape(-1) for p in ctx["model"].parameters()]).norm())
            seen.append((step, ctx["grad_norm"], g_norm))
            if step != 2:
                return None
            ctx["w_before"] = _weights(ctx["model"])
            saved = [(g["lr"], g["weight_decay"]) for g in ctx["opt"].param_groups]
            for g in ctx["opt"].param_groups:
                g["lr"], g["weight_decay"] = 0.0, 0.0

            def undo():
                assert torch.equal(_weights(ctx["model"]), ctx["w_before"])  # step ran with lr=0 ...
                for g, (lr, wd) in zip(ctx["opt"].param_groups, saved):
                    g["lr"], g["weight_decay"] = lr, wd

            return undo

        train_forward(model, opt, cfg, data, 0, 4, "cpu", pre_opt_step=freeze_step_2)
        assert [s for s, _, _ in seen] == [0, 1, 2, 3]  # once per optimizer step, not per microbatch
        assert all(raw > clip and math.isclose(g, clip, rel_tol=1e-3) for _, raw, g in seen), seen
        assert all(g["lr"] == cfg["optim"]["lr"] for g in opt.param_groups)  # ... and undo restored it


def test_run_fork_and_branch_pass_pre_opt_step_through():
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        train_forward(model, opt, _CFG, data, 0, 3, "cpu")
        snap = capture(model, opt, step=3, meta={"scale": "test"})

        calls = []
        run_fork(_CFG, data, snap, steps=2, device="cpu", seed=False, pre_opt_step=lambda s, c: calls.append(s))
        assert calls == [3, 4]  # global steps: the fork resumes at the snapshot's step

        recipe = SimpleNamespace(pre_step=lambda s, c: None, inject_step=4, width=1)
        br = make_baseline_branches(_CFG, recipe)
        assert set(br) == {"Bskipstep", "Bvreset", "Bspam", "Bzclip", "Badagc"}
        assert all(isinstance(b, Branch) for b in br.values())
        hooked = [b for n, b in br.items() if n != "Bvreset"]  # Bvreset acts via pre_step, not the grad hook
        assert all(b.pre_step is recipe.pre_step and b.pre_opt_step is not None for b in hooked)
        assert br["Bvreset"].pre_opt_step is None
        # ZClip/AdaGC replace the global clip and inherit its value as their cap; SPAM keeps the trunk cfg.
        assert br["Bspam"].cfg is _CFG
        assert br["Bzclip"].cfg["optim"]["grad_clip"] == br["Badagc"].cfg["optim"]["grad_clip"] == 0.0
        assert br["Bzclip"].pre_opt_step.max_norm == br["Badagc"].pre_opt_step.max_norm == 1.0
        for b in br.values():
            res = run_branch(data, snap, b, steps=2, device="cpu", seed=False)
            assert res["survival"] == 1 and len(res["losses"]) == 2, b.name


def test_prime_trunk_matches_plain_trunk_and_primed_state_is_json_safe():
    import json

    fork_step = 4
    with _tmpdir() as d:
        data = _data_dir(d)
        snap = prime_trunk(_CFG, data, fork_step, device="cpu")
        assert snap["step"] == fork_step

        # The observers only watch: the snapshot is bitwise the plain trunk's at the same step.
        plain = {}

        def grab(step, info):
            if step == fork_step - 1:
                plain["snap"] = capture(info["model"], info["opt"], step=fork_step)

        run_trunk(_CFG, data, steps=fork_step, on_step=grab, deterministic=True, device="cpu")
        for key in ("w", "m", "v"):
            assert snap[key].keys() == plain["snap"][key].keys()
            assert all(torch.equal(snap[key][n], plain["snap"][key][n]) for n in snap[key]), key

        primed = snap["meta"]["baseline_priming"]
        assert len(primed["zclip_state"]["buffer"]) == fork_step  # still inside ZClip's 25-step warmup
        assert primed["adagc_state"]["steps"] == fork_step
        assert set(primed["adagc_state"]["gamma"]) == {n for n, _ in _fresh()[0].named_parameters()}

        # `snapshot.save` stores meta as JSON in the safetensors header (safetensors itself isn't in the
        # CI image), so a JSON round-trip is the property that matters: exact, floats included.
        assert json.loads(json.dumps(snap["meta"])) == snap["meta"]

        br = make_baseline_branches(_CFG, SimpleNamespace(pre_step=None, inject_step=5, width=1), **primed)
        assert br["Bzclip"].pre_opt_step.state_dict() == primed["zclip_state"]
        assert br["Badagc"].pre_opt_step.state_dict() == primed["adagc_state"]
        for b in br.values():
            res = run_branch(data, snap, b, steps=2, device="cpu", seed=False)
            assert res["survival"] == 1, b.name


def test_never_firing_zclip_is_bitwise_identical_to_no_clip():
    # A ZClip that can never clip (no cap, huge threshold, already initialized) must leave the
    # trajectory bitwise equal to running with no clip at all: the hook has no side effects.
    cfg = zclip.apply_to_cfg(_CFG)
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh(cfg)
        base = train_forward(model, opt, cfg, data, 0, 4, "cpu")
        w_base = _weights(model)

        z = zclip.ZClip(max_norm=None, z_thresh=1e9)
        z.load_state_dict({"buffer": [], "initialized": True, "mean": 1.0, "var": 1.0})
        model, opt = _fresh(cfg)
        assert train_forward(model, opt, cfg, data, 0, 4, "cpu", pre_opt_step=z) == base
        assert torch.equal(_weights(model), w_base)


def _main() -> None:
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
    print(f"baselines selfcheck OK: {len(fns)} tests passed (baselines, trunk priming, hook wiring)")


if __name__ == "__main__":
    _main()
