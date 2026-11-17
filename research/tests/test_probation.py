"""CPU checks for research/autopsy/probation.py (RML-2): null-repair identity, the two exact corners, the verdict."""

import math

import torch

from research.autopsy import probation

B1, B2 = 0.9, 0.95


def _model_opt():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, dtype=torch.float64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(B1, B2), weight_decay=0.1)
    return model, opt


def _grads(model, scale=1.0):
    return [scale * torch.randn_like(p) for p in model.parameters()]


def _step(model, opt, step, grads, hook=None, update=True):
    """What trunk.train_forward does around `pre_opt_step`."""
    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone()
    fix = hook(step, {"model": model, "opt": opt}) if hook else None
    if update:
        opt.step()
    if fix:
        fix()


def _moments(opt, model):
    return [(opt.state[p]["exp_avg"].clone(), opt.state[p]["exp_avg_sq"].clone()) for p in model.parameters()]


def test_never_armed_is_a_bitwise_noop():
    torch.manual_seed(1)
    seq = [_grads(torch.nn.Linear(4, 3, dtype=torch.float64)) for _ in range(6)]
    (ma, oa), (mb, ob) = _model_opt(), _model_opt()
    hook = probation.as_pre_opt_step(x_arm=math.inf)
    for s, g in enumerate(seq):
        _step(ma, oa, s, g, hook)
        _step(mb, ob, s, g)
    assert all(torch.equal(a, b) for a, b in zip(ma.parameters(), mb.parameters()))
    for (am, av), (bm, bv) in zip(_moments(oa, ma), _moments(ob, mb)):
        assert torch.equal(am, bm) and torch.equal(av, bv)


def test_refute_leaves_the_moments_of_a_run_that_skipped_the_step():
    model, opt = _model_opt()
    out = {}
    hook = probation.as_pre_opt_step(x_arm=1e-12, K=3, arm_steps={2}, out=out)
    for s in (0, 1):
        _step(model, opt, s, _grads(model), hook)
    pre = _moments(opt, model)
    _step(model, opt, 2, _grads(model, 50.0), hook)  # the spike, armed on every coordinate
    later = [m.clone() for m, _ in pre]  # fresh gradients with no component along the spike: y = 0
    for s in (3, 4):
        _step(model, opt, s, later, hook)
    _step(model, opt, 5, later, hook, update=False)  # k = K: closes before step 5's update
    assert out["verdicts"][0][:4] == (2, 5, "refute", 0.0)
    assert out["verdicts"][0][4] == sum(p.numel() for p in model.parameters())
    for p, (m0, v0) in zip(model.parameters(), pre):
        v_ref = v0.clone()
        for _ in (3, 4):  # the skip run: step 2 left m, v alone, steps 3-4 saw `later`
            v_ref = B2 * v_ref + (1 - B2) * m0.square()
        assert torch.allclose(opt.state[p]["exp_avg"], m0, rtol=1e-12, atol=1e-15)
        # the refute subtracts a v inflation of up to ~1e3 to leave ~1e-1 (method.md §2.1: cancellation costs bits),
        # so the error is a few ulps of 1e3, not of the result; 1e-10 is still 1e-13 of the inflation
        assert torch.allclose(opt.state[p]["exp_avg_sq"], v_ref, rtol=1e-12, atol=1e-10)


def test_phi_one_restores_the_moments_of_the_run_that_kept_the_step():
    torch.manual_seed(2)
    seq = [_grads(torch.nn.Linear(4, 3, dtype=torch.float64), 50.0 if s == 2 else 1.0) for s in range(5)]
    (ma, oa), (mb, ob) = _model_opt(), _model_opt()
    out = {}
    hook = probation.as_pre_opt_step(x_arm=1e-12, K=10, level=math.inf, arm_steps={2}, out=out)
    for s, g in enumerate(seq):
        _step(ma, oa, s, g, hook)
        _step(mb, ob, s, g)
    probation.close(ma, oa, out["open"][0], phi=1.0, keep_scale=False, at_step=5)
    for (am, av), (bm, bv) in zip(_moments(oa, ma), _moments(ob, mb)):
        assert torch.allclose(am, bm, rtol=1e-12, atol=1e-15)
        assert torch.allclose(av, bv, rtol=1e-12, atol=1e-15)


def test_verdict_confirms_refutes_and_flags_opposition():
    R = {"Q": 16.0}
    assert probation.verdict({**R, "ys": [16.0]}, K=3, level=3.0) == ("confirm", 0.25)  # t = 4, phi = 1 - 3/4
    assert probation.verdict({**R, "ys": [-16.0]}, K=3, level=3.0) == ("unstable", 0.0)
    assert probation.verdict({**R, "ys": [0.0]}, K=3, level=3.0) is None
    assert probation.verdict({**R, "ys": [0.0, 0.0, 0.0]}, K=3, level=3.0) == ("refute", 0.0)
