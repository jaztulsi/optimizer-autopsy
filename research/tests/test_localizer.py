"""CPU tests for the localizer's Week 1-3 code (toy HVP, curvature.py, poison.py's v_bar).

Everything here is small and runs in seconds on the CI runner. The real-scale checks (the proxy
model on a GPU) are research/gpu/curvature_run.py, not pytest.
"""

import importlib

import pytest
import torch

from research.localizer import curvature, poison
from research.localizer.toy_hvp import hvp as toy_hvp
from research.localizer.toy_hvp import model as toy_model
from research.localizer.toy_hvp import verify as toy_verify
from research.model.nanogpt import GPT, GPTConfig

# ---- Week 1: the toy HVP -------------------------------------------------------------------------


def test_toy_hvp_matches_finite_differences():
    worst = toy_verify.check(seed=0)
    assert worst < toy_verify.TOLERANCE


def test_toy_check_fails_when_hvp_is_wrong(monkeypatch):
    # The gate must be able to fail: a deliberately wrong HVP (twice the right answer) must raise.
    real = toy_verify.hvp
    monkeypatch.setattr(toy_verify, "hvp", lambda *a: 2.0 * real(*a))
    with pytest.raises(AssertionError):
        toy_verify.check(seed=0)


def test_toy_modules_do_not_seed_at_import():
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    for mod in (toy_model, toy_hvp, toy_verify):
        importlib.reload(mod)
    assert torch.equal(before, torch.get_rng_state()), "importing toy_hvp reset the global random stream"


# ---- Week 2: name-keyed HVP on the real GPT loss -------------------------------------------------


def _tiny_gpt():
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=8, vocab_size=32, n_layer=1, n_head=2, n_embd=8, dropout=0.0)
    model = GPT(cfg).double()
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), generator=gen)
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), generator=gen)
    return model, (x, y)


def test_gpt_index_counts_tied_weight_once():
    model, _ = _tiny_gpt()
    curvature.assert_tied_once(model)
    idx = curvature.param_index(model)
    unique = {id(p): p.numel() for p in model.parameters()}
    assert curvature.flat_dim(idx) == sum(unique.values())


def test_gpt_hvp_matches_finite_differences():
    # float64 so the check is decisive; the fp32 sweep at real scale is the GPU run.
    model, batch = _tiny_gpt()
    report = curvature.fd_check(model, batch, steps=[1e-3, 3e-4, 1e-4], seed=0)
    best = report["best"]
    assert best["rel_l2"] < 1e-5, f"GPT HVP vs finite differences: rel_l2 {best['rel_l2']:.3e}"
    # fd_check computes cos in float32, whose resolution near 1 is ~6e-8, so 1e-6 is the tightest honest bound.
    assert best["cos"] > 1 - 1e-6, f"GPT HVP vs finite differences: cos {best['cos']:.12f}"


def test_gpt_hvp_is_symmetric():
    # u . (H v) must equal v . (H u): a Hessian is symmetric.
    model, batch = _tiny_gpt()
    idx = curvature.param_index(model)
    dim = curvature.flat_dim(idx)
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(dim, generator=gen, dtype=torch.float64)
    v = torch.randn(dim, generator=gen, dtype=torch.float64)
    a = torch.dot(u, curvature.hvp(model, batch, v, index=idx)).item()
    b = torch.dot(v, curvature.hvp(model, batch, u, index=idx)).item()
    assert abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)


# ---- Week 3: eigensolve and v_bar (the CPU self-checks, which assert internally) -----------------


def test_eigsh_recovers_known_quadratic():
    assert curvature._selfcheck_quadratic()["passed"]


def test_preconditioner_sandwich():
    assert curvature._selfcheck_precond()["passed"]


def test_asymmetry_guard_fires():
    assert curvature._selfcheck_asymmetry_is_caught()["passed"]


def test_vbar_debias_lag_freeze_roundtrip():
    assert poison._selfcheck_vbar()["passed"]
