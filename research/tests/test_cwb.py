"""CPU checks for research/autopsy/cwb.py (EXPERIMENT_CATALOG A5): exact on gradients that don't depend on w (A5 test
a), a closed-loop error that is first order in eta (A5 test b), and lag 0 equal to Bskipstep bitwise on the real
training loop."""

import pytest
import torch

from research.autopsy import cwb, undo
from research.baselines import skip_step
from research.harness.schedule import lr_at

B1, B2 = 0.9, 0.95
SHAPES = [(3, 4), (3,)]  # nn.Linear(4, 3): weight, bias
NAMES = ["weight", "bias"]


def _cfg(lr, schedule=None):
    o = {"lr": lr, "betas": [B1, B2], "eps": 1e-8, "weight_decay": 0.1}
    return {"optim": o | ({"schedule": schedule} if schedule else {}), "train": {"max_steps": 20}}


def _run(cfg, grad_fn, steps, pre_step=None, pre_opt_step=None):
    """What trunk.train_forward does around its hooks, on a float64 Linear whose gradients come from
    grad_fn(step, model) -> [g per param]. The weight decays and the bias doesn't, as in configure_optimizers."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, dtype=torch.float64)
    o = cfg["optim"]
    groups = [{"params": [model.weight], "weight_decay": o["weight_decay"]}, {"params": [model.bias]}]
    opt = torch.optim.AdamW(groups, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"], weight_decay=0.0)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        ctx = {"model": model, "opt": opt, "cfg": cfg}
        if pre_step:
            pre_step(step, ctx)
        for p, g in zip(model.parameters(), grad_fn(step, model)):
            p.grad = g
        if pre_opt_step:
            pre_opt_step(step, ctx)
        if not ctx.get("skip_update"):
            opt.step()
    return model, opt


def _state(model, opt):
    st = [opt.state[p] for p in model.parameters()]
    return [p.detach() for p in model.parameters()] + [s[k] for s in st for k in ("exp_avg", "exp_avg_sq", "step")]


@pytest.mark.parametrize("mode", cwb.MODES)
def test_exact_when_gradients_do_not_depend_on_w(mode):
    # A5 test (a): with an exogenous gradient sequence the open loop is exact, so CWB lands on its counterfactual:
    # Bskipstep for "skip", the clean run for "clean". A width-2 window, lag 3, LR still warming up.
    gen = torch.Generator().manual_seed(1)
    live = {k: [torch.randn(s, generator=gen, dtype=torch.float64) for s in SHAPES] for k in range(12)}
    clean = {k: [torch.randn(s, generator=gen, dtype=torch.float64) for s in SHAPES] for k in (4, 5)}
    for k in clean:
        live[k] = [30 * g for g in live[k]]  # the spike window
    cfg = _cfg(1e-2, {"kind": "cosine", "warmup": 8})

    def grads(spiked):
        return lambda step, model: [g.clone() for g in (clean[step] if not spiked and step in clean else live[step])]

    out = {}
    clean_fn = (lambda s, ctx: dict(zip(NAMES, clean[s]))) if mode == "clean" else None
    fix, track = cwb.as_hooks([4, 5], undo.undo_step(5, 3), mode, clean_fn=clean_fn, out=out)
    got = _run(cfg, grads(True), 12, fix, track)
    if mode == "skip":
        want = _run(cfg, grads(True), 12, pre_opt_step=skip_step.as_pre_opt_step(4, 2))
    else:
        want = _run(cfg, grads(False), 12)
    for a, b in zip(_state(*got), _state(*want)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)
    assert out["a_norm"] > 1e-3 * out["w_norm"]  # the commit really moved w (else the match above is vacuous)


def _quadratic_errors(lr, lag):
    """(e, d) on f(w) = sum h (w - c)^2 / 2 with a spike added to step 6's gradient: e = ||CWB - Bskipstep||,
    d = ||live - Bskipstep||, both on w at the end of the decision step. Here the gradient depends on w."""
    gen = torch.Generator().manual_seed(2)
    h = [0.5 + 1.5 * torch.rand(s, generator=gen, dtype=torch.float64) for s in SHAPES]
    c = [torch.randn(s, generator=gen, dtype=torch.float64) for s in SHAPES]
    spike = [50 * torch.randn(s, generator=gen, dtype=torch.float64) for s in SHAPES]

    def grad_fn(step, model):
        g = [hi * (p.detach() - ci) for hi, p, ci in zip(h, model.parameters(), c)]
        return [gi + si for gi, si in zip(g, spike)] if step == 6 else g

    at, cfg = undo.undo_step(6, lag), _cfg(lr)
    w = [
        torch.cat([p.detach().flatten() for p in _run(cfg, grad_fn, at + 1, *hooks)[0].parameters()])
        for hooks in ((None, skip_step.as_pre_opt_step(6, 1)), cwb.as_hooks([6], at), ())
    ]
    return float((w[1] - w[0]).norm()), float((w[2] - w[0]).norm())


def test_closed_loop_error_is_first_order_in_lr():
    # A5 test (b): on a real loss the shadow sees gradients taken at the live w, so CWB is approximate. Its error
    # relative to the damage it removes, e / d, is O(eta ||H|| tau): exactly 0 at lag 0 (no open-loop step yet),
    # small at lag 5, and about 4x smaller when eta is cut 4x.
    e0, d0 = _quadratic_errors(1e-3, 0)
    assert d0 > 0 and e0 == 0.0
    e, d = _quadratic_errors(1e-3, 5)
    e4, d4 = _quadratic_errors(2.5e-4, 5)
    assert 0 < e < 0.05 * d  # nonzero: the loop really is closed
    assert 0 < e4 / d4 < 0.6 * (e / d)


def test_lag_zero_is_bskipstep_bitwise_on_the_real_loop():
    # The tests above mimic train_forward by hand; this runs the hooks through run_fork on a 1-layer GPT. At lag 0
    # the shadow has skipped the window and replayed nothing, so the commit restores the pre-window state exactly.
    from research.harness.fork import run_fork
    from research.harness.snapshot import capture
    from research.harness.trunk import train_forward
    from research.tests.test_baselines import _CFG, _data_dir, _fresh, _tmpdir

    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        train_forward(model, opt, _CFG, data, 0, 3, "cpu")
        snap = capture(model, opt, step=3)

        def fork(pre=None, pre_opt=None):
            return run_fork(_CFG, data, snap, steps=4, device="cpu", seed=False, pre_step=pre, pre_opt_step=pre_opt)

        live, out = fork(), {}
        skip = fork(pre_opt=skip_step.as_pre_opt_step(3, 1))
        assert fork(*cwb.as_hooks([3], undo.undo_step(3, 0), out=out)) == skip != live
        assert out["a_norm"] > 0
        # No spike in this fork, so the clean slot is the live slot: the clean shadow replays the live run, up to
        # the fp32 rounding of a step taken in fp64.
        clean = fork(*cwb.as_hooks([3], 5, "clean", data_dir=data))
        assert max(abs(a - b) for a, b in zip(clean, live)) < 1e-6


def test_as_hooks_rejects_bad_arguments():
    with pytest.raises(ValueError):
        cwb.as_hooks([4], 4)  # commit inside the window
    with pytest.raises(ValueError):
        cwb.as_hooks([4], 6, "clean")  # clean mode with no clean gradients
    with pytest.raises(ValueError):
        cwb.as_hooks([4], 6, "rollback")
    fix, _ = cwb.as_hooks([4], 6)
    with pytest.raises(RuntimeError):
        fix(6, {})  # the window never ran: nothing to commit
