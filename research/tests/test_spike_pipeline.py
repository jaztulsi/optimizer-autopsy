"""CPU-only tests for the spike-engineer pipeline (PLAN V7 §6): no training, tiny arrays/tensors only.

The GPU paths (capture.run_one, natural --search) are exercised by `capture --selfcheck` on the AMD box;
everything they are built from is checked here, and each test fails if its logic breaks.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.spikes import k2
from research.spikes.capture import SIGNALS, _battery_specs, fork_from_ring, pick_normals, tensor_signals
from research.spikes.induce import RECIPES, _repeated_rows, induce
from research.spikes.natural import STRESS, alignment_test, find_spikes
from research.spikes.tune_detector import DetectorParams, OnlineDetector, trigger_mask


def _reference_trigger_mask(loss_hist, gradnorm_hist, p):
    """The pre-refactor trigger_mask, verbatim: OnlineDetector must reproduce it exactly."""
    loss = np.asarray(loss_hist, dtype=float)
    mask, ema = [False] * len(loss), None
    for t in range(len(loss)):
        gt = float(gradnorm_hist[t]) if gradnorm_hist is not None else None
        fired = False
        if t >= p.warmup:
            if not np.isfinite(loss[t]):
                fired = True
            else:
                past = loss[max(0, t - p.win) : t]
                if len(past) >= 2:
                    mu, sd = past.mean(), past.std()
                    if sd > 0 and (loss[t] - mu) / sd > p.z:
                        fired = True
                if p.use_gradnorm and ema is not None and gt is not None and np.isfinite(gt):
                    if gt > p.ema_mult * ema:
                        fired = True
        mask[t] = fired
        if gt is not None and np.isfinite(gt):
            ema = gt if ema is None else p.ema_beta * ema + (1 - p.ema_beta) * gt
    return mask


def test_online_detector_matches_reference_exactly():
    rng = np.random.default_rng(0)
    grid = [
        DetectorParams(z=z, ema_mult=m, win=w, warmup=wu)
        for z in (2.0, 4.0)
        for m in (2.0, 4.0)
        for w in (5, 20)
        for wu in (0, 10)
    ]
    for trial in range(40):
        n = 120
        loss = 3 + rng.normal(0, 0.1, n)
        grad = 1 + np.abs(rng.normal(0, 0.2, n))
        for t in rng.integers(0, n, 4):
            loss[t] += rng.uniform(0, 3)
            grad[t] *= rng.uniform(1, 8)
        if trial % 5 == 0:
            loss[rng.integers(0, n)] = np.nan
            grad[rng.integers(0, n)] = np.inf
        for p in grid:
            assert trigger_mask(loss.tolist(), grad.tolist(), p) == _reference_trigger_mask(loss, grad, p)
            assert trigger_mask(loss.tolist(), None, p) == _reference_trigger_mask(loss, None, p)


def test_online_detector_reports_trigger_stat():
    p = DetectorParams(z=4.0, win=5, warmup=5)
    det = OnlineDetector(p)
    losses = [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 5.0]
    fired = [det.update(x, 1.0) for x in losses]
    assert fired == [False] * 6 + [True]
    past = np.asarray(losses[1:6])
    assert det.stat["loss_z"] == pytest.approx((5.0 - past.mean()) / past.std())
    assert det.stat["gradnorm_over_ema"] == pytest.approx(1.0) and det.stat["step"] == 6


def test_repeated_rows_tiles_one_ngram_per_row_and_keeps_the_rest():
    x = torch.arange(4 * 12).reshape(4, 12)
    y = x + 1
    gen = torch.Generator().manual_seed(7)
    xr, yr = _repeated_rows(x, y, n=3, frac=0.5, vocab=50, gen=gen)
    assert torch.equal(xr[2:], x[2:]) and torch.equal(yr[2:], y[2:])  # untouched real rows
    for i in range(2):
        assert torch.equal(xr[i, 3:], xr[i, :-3])  # period-3 tiling
        assert torch.equal(yr[i, :-1], xr[i, 1:])  # targets are the next token
        assert int(xr[i].max()) < 50
    gen2 = torch.Generator().manual_seed(7)
    assert torch.equal(_repeated_rows(x, y, n=3, frac=0.5, vocab=50, gen=gen2)[0], xr)  # deterministic


def test_repeat_ngram_recipe_uses_the_real_batch(tmp_path):
    np.arange(5000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    cfg = {"train": {"batch_size": 4, "grad_accum": 1}}
    r = induce("repeat_ngram", cfg, inject_step=3, width=1, n=4, frac=0.5, data_dir=str(tmp_path))
    mcfg = SimpleNamespace(block_size=8, vocab_size=50304)
    model = SimpleNamespace(cfg=mcfg, parameters=lambda: iter([torch.zeros(1)]))
    ctx = {"model": model, "cfg": cfg, "batch": None}
    r.pre_step(2, ctx)
    assert ctx["batch"] is None  # outside the window
    r.pre_step(3, ctx)
    x, y = ctx["batch"]
    from research.data.prepare import get_batch

    rx, ry = get_batch("train", 3, 4, 8, str(tmp_path))
    assert torch.equal(x[2:], rx[2:]) and torch.equal(y[2:], ry[2:])
    assert not torch.equal(x[:2], rx[:2])
    with pytest.raises(ValueError):
        induce("repeat_ngram", cfg, inject_step=3, frac=0.5)  # frac<1 without real data


def test_v_inflate_scales_only_the_targets_inside_the_window():
    a, b = torch.ones(3), torch.ones(2)
    model = SimpleNamespace(named_parameters=lambda: [("A", a), ("B", b)])
    opt = SimpleNamespace(state={a: {"exp_avg_sq": torch.full((3,), 2.0)}, b: {"exp_avg_sq": torch.full((2,), 2.0)}})
    r = induce("v_inflate", {}, inject_step=5, width=1, factor=10.0, targets=["A"])
    assert r.params["targets"] == ["A"]  # the ground truth travels with the recipe
    r.pre_step(4, {"model": model, "opt": opt})
    assert torch.equal(opt.state[a]["exp_avg_sq"], torch.full((3,), 2.0))
    r.pre_step(5, {"model": model, "opt": opt})
    assert torch.equal(opt.state[a]["exp_avg_sq"], torch.full((3,), 20.0))
    assert torch.equal(opt.state[b]["exp_avg_sq"], torch.full((2,), 2.0))
    with pytest.raises(KeyError):
        induce("v_inflate", {}, inject_step=5, targets=["nope"]).pre_step(5, {"model": model, "opt": opt})


def _trace(n=600, spikes=(), nan_at=None, seed=0):
    rng = np.random.default_rng(seed)
    loss = 3 + 4 * np.exp(-np.arange(n) / 30) + rng.normal(0, 0.02, n)
    for start, height, length in spikes:
        loss[start : start + length] += height
    if nan_at is not None:
        loss[nan_at:] = np.nan
    return loss.tolist()


def test_find_spikes_flags_real_spikes_only():
    assert find_spikes(_trace()) == []  # the initial descent is not a spike
    ev = find_spikes(_trace(spikes=[(300, 3.0, 4), (450, 3.0, 150)]))
    assert [e["start"] for e in ev] == [300, 450]
    assert ev[0]["peak_step"] in range(300, 304) and ev[0]["ratio"] > 1.9 and not ev[0]["malignant"]
    assert ev[1]["malignant"]  # never came back below 1.1 x baseline
    div = find_spikes(_trace(nan_at=400))
    assert div[0]["start"] == 400 and div[0]["malignant"] and div[0]["ratio"] == float("inf")


def test_alignment_test_separates_data_from_dynamics():
    aligned = [[200, 900, 1500], [202, 899, 1503], [198, 901, 1499]]
    scattered = [[200, 900, 1500], [450, 1180, 2700], [610, 1990, 2400]]
    assert alignment_test(aligned, 3000, n_perm=500)["p_value"] < 0.01
    assert alignment_test(scattered, 3000, n_perm=500)["p_value"] > 0.05


def test_stress_presets_are_the_preregistered_grid():
    assert len(STRESS) == 8
    assert {tuple(v["optim.betas"]) for v in STRESS.values()} == {(0.9, 0.95), (0.9, 0.999)}


def _run(**kw):
    run = {
        "config_path": "research/experiments/proxy/config.yaml",
        "overrides": {"train.batch_size": 16, "seed": 1},
        "data_preset": "proxy",
        "data_revision": "f54c09fd",
        "seed": 1,
        "stress": None,
        "detector_params": {"z": 4.0},
        "normal_snapshots": ["normal_120.safetensors"],
        "hardware": {"gpu": "MI300X", "torch": "x", "backend": "rocm", "deterministic": True},
        "triggers": [],
    }
    run.update(kw)
    return run


def _trigger(t0, fork_step):
    return {
        "t0": t0,
        "stat": {"loss_z": 9.0},
        "t0_snapshot": f"t0_{t0}.st",
        "fork_snapshot": f"fork_{fork_step}.st",
        "fork_step": fork_step,
        "vbar": f"vbar_{fork_step}.st",
    }


def test_k2_entries_from_induced_and_natural_runs(tmp_path):
    losses = [3.0] * 200
    for k_, h in enumerate([1.0, 2.0, 4.0, 6.0, 7.0, 5.0, 2.0]):  # injected at 160, loss peaks at 164
        losses[160 + k_] += h
    rec = {
        "kind": "lr_bump",
        "inject_step": 160,
        "width": 3,
        "params": {"factor": 50.0},
        "min_lead": 2,
        "allow_at_peak": False,
        "peak_offset": 0,
    }
    induced = _run(recipe=rec, losses=losses, triggers=[_trigger(161, 160), _trigger(165, 160)])
    entries = k2.entries_from_run(induced, evidence="run.json")
    assert [e.t0 for e in entries] == [161]  # the at-peak trigger fails the predictive policy
    e = entries[0]
    assert (e.peak_step, e.lead, e.detection_mode, e.fork_step) == (164, 3, "predictive", 160)
    k2.validate(e)

    nat_losses = _trace(spikes=[(300, 3.0, 4)])
    natural = _run(
        recipe=None, stress="lr10x_b20.999_clip0", losses=nat_losses, triggers=[_trigger(250, 230), _trigger(299, 280)]
    )  # 250 is a false alarm
    nat = k2.entries_from_run(natural, evidence="run.json")
    assert [x.t0 for x in nat] == [299] and nat[0].source == "natural" and nat[0].recipe is None
    k2.validate(nat[0])

    path = tmp_path / "k2.json"
    k2.save_k2(k2.K2Manifest(entries=entries + nat), str(path))
    assert k2.load_k2(str(path)).entries == entries + nat
    json.loads(path.read_text())  # plain JSON on disk


def test_k2_validate_rejects_broken_entries():
    base = k2.entries_from_run(
        _run(
            recipe={
                "kind": "lr_bump",
                "inject_step": 10,
                "width": 1,
                "params": {},
                "min_lead": 2,
                "allow_at_peak": False,
                "peak_offset": 0,
            },
            losses=[1.0] * 10 + [1.5, 3.0, 5.0] + [1.0] * 20,
            triggers=[_trigger(10, 10)],
        ),
        evidence="run.json",
    )[0]
    from dataclasses import replace

    for bad in (
        replace(base, fork_step=11),  # fork snapshot already contains the injection
        replace(base, detection_mode="predictive", lead=1, peak_step=base.t0 + 1),
        replace(base, label="confirmed", seed=k2.TUNING_SEED),  # tuning seed can't confirm
        replace(base, normal_snapshot_uris=()),
        replace(base, hardware={"backend": "cpu"}),
        replace(base, source="natural"),
    ):
        with pytest.raises(ValueError):
            k2.validate(bad)


def test_tensor_signals_match_the_formulas():
    g = torch.tensor([1.0, -2.0])
    m = torch.tensor([0.5, -0.5])
    v = torch.tensor([4.0, 1.0])
    vbar = torch.tensor([2.0, 1.0])
    out = tensor_signals(g, m, v, vbar, step=1, betas=(0.9, 0.95), eps=1e-8)
    vh, mh = v / 0.05, m / 0.1
    r = g**2 / vh
    want = [r.mean().sqrt(), r.max(), (mh / (vh.sqrt() + 1e-8)).square().mean().sqrt(), (v - vbar).norm() / vbar.norm()]
    assert out.shape == (len(SIGNALS),) and torch.allclose(out, torch.stack(want))


def test_fork_and_normal_selection():
    assert fork_from_ring([10, 20, 30, 40], t0=45, lookback=20) == 1  # newest step <= 26 is 20
    assert fork_from_ring([40, 50], t0=45, lookback=20) is None
    assert pick_normals([40, 80, 120, 160], fork_step=160, n_normal=3, gap=20) == [40, 80, 120]
    assert pick_normals([40, 80], fork_step=50, n_normal=3, gap=20) == []


def test_battery_is_the_preregistered_design_and_spike_run_policies_exist():
    from research.kaggle.spike_run import SPEC

    specs = _battery_specs("out")
    assert len(specs) == 2 * 2 * 5
    assert {(s["recipe"]["kind"], s["recipe"]["inject_step"]) for s in specs} == {
        (k_, i) for k_ in ("lr_bump", "corrupt_batch") for i in (160, 480)
    }
    assert all(s["steps"] == s["recipe"]["inject_step"] + 60 for s in specs)
    assert set(SPEC) <= set(RECIPES) and "v_inflate" not in SPEC
    for kind, sp in SPEC.items():
        assert {"inject", "width", "params", "min_lead", "allow_at_peak", "peak_offset"} <= set(sp), kind
