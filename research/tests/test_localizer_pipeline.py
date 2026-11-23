"""End-to-end CPU regression tests for the localizer (Week 8).

A tiny GPT is trained for a few AdamW steps on random tokens while a v_bar EMA is kept, exactly the
shape of what research/spikes/capture.py saves: normal snapshots with their v_bar, then a t0
snapshot. A spike is PLANTED by inflating v in one known tensor of the t0 snapshot (PLAN_V8 §6.2's
v_inflate, which has exact ground truth). The full pipeline -- restore, eigensolve of H~, directional
SNR, null thresholds from the normal steps, `localize` -- must point at that tensor. Also: the
eigensolver's stability harness on a model with a known spectrum, against a dense reference.
"""

import copy

import numpy as np
import pytest
import torch

from research.harness.snapshot import capture
from research.localizer import curvature, pipeline, poison, stability
from research.model.nanogpt import GPT, GPTConfig

PLANT = "transformer.h.0.mlp.c_fc.weight"


def _batch(gen, cfg, rows=8):
    x = torch.randint(0, cfg.vocab_size, (rows, cfg.block_size), generator=gen)
    y = torch.randint(0, cfg.vocab_size, (rows, cfg.block_size), generator=gen)
    return x, y


def _run(n_steps=10, rho=0.9):
    """Train a tiny GPT; return (model, opt, cfg, normals [(snap, vbar)], t0 snap, t0 vbar, data fn)."""
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=8, vocab_size=32, n_layer=1, n_head=2, n_embd=8, dropout=0.0)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    batches = [_batch(gen, cfg) for _ in range(n_steps + 1)]
    vbar, normals = None, []
    for s in range(n_steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(*batches[s])
        loss.backward()
        opt.step()
        v = {n: opt.state[p]["exp_avg_sq"].detach().clone() for n, p in model.named_parameters()}
        vbar = v if vbar is None else {n: rho * vbar[n] + (1 - rho) * v[n] for n in v}
        if 3 <= s < n_steps - 1:
            normals.append((capture(model, opt, s + 1), copy.deepcopy(vbar)))
    t0 = capture(model, opt, n_steps)
    t0_vbar = copy.deepcopy(vbar)

    def micro(snapshot):  # the batch of the step that produced this snapshot, split into 4
        x, y = batches[int(snapshot["step"]) - 1]
        return list(zip(x.chunk(4), y.chunk(4)))

    return model, opt, cfg, normals, t0, t0_vbar, micro, batches[n_steps]


def _plant(snapshot, name=PLANT, kappa=1e4):
    s = copy.deepcopy(snapshot)
    s["v"][name] = s["v"][name] * kappa
    return s


def test_pipeline_finds_planted_v_inflation_curvature_path():
    model, opt, cfg, normals, t0, t0_vbar, micro, probe = _run()
    loc = pipeline.localize_spike(
        model, opt, _plant(t0), t0_vbar, normals, probe_batch=probe, microbatches=micro, k=3, n_head=cfg.n_head
    )
    poison.validate_localization(loc)
    assert loc["selected"] and loc["selected"][0] == PLANT
    assert loc["groups"][PLANT]["sign"] == 1  # v went up: over-preconditioned
    ev = loc["provenance"]["eigen"]
    assert len(ev["eigvals"]) == 3 and max(ev["residuals"]) <= 1e-3
    assert 0.0 <= loc["provenance"]["psi_k"] <= 1.0
    assert "transformer.wte.weight" in loc["groups"] and "lm_head.weight" not in loc["groups"]  # tied once


def test_pipeline_no_hvp_path_and_head_groups():
    model, opt, cfg, normals, t0, t0_vbar, micro, probe = _run()
    loc = pipeline.localize_spike(
        model, opt, _plant(t0), t0_vbar, normals, use_eig=False, groups="head", n_head=cfg.n_head
    )
    assert loc["selected"][0] == PLANT and loc["method"] == "drift (no HVP)"
    assert "transformer.h.0.attn.c_attn.weight[q0]" in loc["groups"]
    # c_attn.weight is (3C, C) = (24, 8); the v block is rows [16, 24), head 1 of 2 owns its upper half
    assert loc["groups"]["transformer.h.0.attn.c_attn.weight[v1]"]["rows"] == [20, 24]


def test_pipeline_refuses_without_normals():
    model, opt, cfg, normals, t0, t0_vbar, micro, probe = _run()
    with pytest.raises(ValueError, match="normal"):
        pipeline.localize_spike(model, opt, t0, t0_vbar, [], use_eig=False)


def test_pipeline_leaves_model_at_t0():
    model, opt, cfg, normals, t0, t0_vbar, micro, probe = _run()
    pipeline.localize_spike(model, opt, t0, t0_vbar, normals, probe_batch=probe, microbatches=micro, k=2)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), t0["w"][n]), n


def test_summary_stats_and_report():
    from research.localizer import validate

    model, opt, cfg, normals, t0, t0_vbar, micro, probe = _run()
    spike = pipeline.summary_stats(pipeline.localize_spike(model, opt, _plant(t0), t0_vbar, normals, use_eig=False))
    calm = pipeline.summary_stats(pipeline.localize_spike(model, opt, t0, t0_vbar, normals, use_eig=False))
    assert spike["max_drift_ratio"] > calm["max_drift_ratio"]
    rows = [{**spike, "label": "spike", "seed": 0, "kind": "v_inflate"}, {**calm, "label": "normal", "seed": 0}]
    rep = validate.report(rows)
    assert rep["separation"]["max_drift_ratio"]["auc"] == 1.0


# ---- Week 4: solver stability on a model with a known spectrum ------------------------------------


class _FakeOpt:
    def __init__(self, p, dim):
        self.state = {p: {"exp_avg_sq": torch.ones(dim)}}


def test_eigen_stability_repeatable_and_matches_dense():
    torch.manual_seed(0)
    dim = 60
    B = torch.randn(dim, dim, dtype=torch.float64)
    A = ((B + B.T) / 2).float()
    m = curvature._quad_model(A, dim, seed=0)
    idx = curvature.param_index(m)
    opt = _FakeOpt(idx[0][1], dim)
    rep = stability.eigen_stability(m, [(None, None), (None, None)], seeds=(0, 1, 2), k=3, optimizer=opt, tol=1e-7)
    assert rep["n_ok"] == 6, rep["failures"]
    assert rep["same_batch"]["min_overlap"] > 0.999
    assert rep["cross_batch"]["min_overlap"] > 0.999  # the quadratic does not depend on the batch

    op = curvature.HTilde(m, (None, None), optimizer=opt)
    ref = stability.dense_reference(op, 3)
    got = curvature.top_k_htilde(m, (None, None), k=3, op=op, tol=1e-7)
    assert np.allclose(got["eigvals"], ref["eigvals"], rtol=1e-4)
    assert stability.subspace_overlap(got["eigvecs"], ref["eigvecs"]) > 0.999
    assert ref["asymmetry"] < 1e-5


def test_eigen_stability_records_failures_instead_of_raising():
    torch.manual_seed(0)
    dim = 30
    A = torch.eye(dim)
    m = curvature._quad_model(A, dim, seed=0)
    opt = _FakeOpt(curvature.param_index(m)[0][1], dim)
    rep = stability.eigen_stability(m, [(None, None)], seeds=(0,), k=3, optimizer=opt, mem_budget_gb=1e-9)
    assert rep["n_ok"] == 0 and "MemoryError" in rep["failures"][0]["reason"]
