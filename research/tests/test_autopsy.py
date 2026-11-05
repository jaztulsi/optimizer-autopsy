"""CPU checks for research/autopsy: the chaos-floor placebo, RML-sel / cooldown edits, branch bookkeeping, report."""

import json

import pytest
import torch

from research.autopsy import report, undo
from research.autopsy.mve import expected_branches, parser
from research.autopsy.probes import data_order, nudge_ulp, ulp_placebo


def _model_opt():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    model(torch.randn(4, 8)).square().mean().backward()
    opt.step()
    return model, opt


def test_nudge_ulp_moves_every_element_one_ulp_deterministically():
    t = torch.randn(1000)
    a, b, c = t.clone(), t.clone(), t.clone()
    nudge_ulp([a], seed=1), nudge_ulp([b], seed=1), nudge_ulp([c], seed=2)
    assert torch.equal(a, b) and not torch.equal(a, c)
    up = torch.nextafter(t, torch.full_like(t, float("inf")))
    down = torch.nextafter(t, torch.full_like(t, -float("inf")))
    assert bool(((a == up) | (a == down)).all()) and bool((a != t).all())


def _log(model, scale):
    return {
        5: {
            n: (scale * torch.randn_like(p).double(), scale * torch.rand_like(p).double())
            for n, p in model.named_parameters()
        }
    }


def test_undo_v_top_edits_only_the_top_fraction():
    model, opt = _model_opt()
    v0 = {n: opt.state[p]["exp_avg_sq"].clone() for n, p in model.named_parameters()}
    for p in model.parameters():  # inflate v by the logged contribution so the undo has something to remove
        opt.state[p]["exp_avg_sq"] += 1.0
    log = _log(model, 1.0)
    frac = undo.undo_v_top(model, opt, log, at_step=6, frac=0.1)
    n_all = sum(p.numel() for p in model.parameters())
    changed = sum(int((opt.state[p]["exp_avg_sq"] != v0[n] + 1.0).sum()) for n, p in model.named_parameters())
    assert changed == round(frac * n_all) and 0 < frac <= 0.1 + 1 / n_all


def test_cooldown_is_identity_without_inflation_and_shrinks_with_it():
    # cfg lr 2e-3 != the optimizer's 1e-3: the cooldown must scale the scheduled LR, not the group's current one
    for scale, shrink in ((0.0, False), (1.0, True)):
        model, opt = _model_opt()
        pre = undo.as_cooldown_pre_step(_log(model, scale), 6, "v", {})
        pre(6, {"model": model, "opt": opt, "cfg": {"optim": {"lr": 2e-3}}})
        lr = opt.param_groups[0]["lr"]
        assert 0 < lr < 2e-3 if shrink else lr == 2e-3


def test_expected_branches_count_placebo_dose_and_cooldown_arms():
    base = ["--config", "x", "--inject-step", "160", "--taus", "0", "--c2-taus", "0", "--dts", "8"]
    extra = ["--placebo", "8", "--plant-coefs", "1,10", "--dose-taus", "0"]
    cb = expected_branches(parser().parse_args(base + ["--kind", "corrupt_batch"] + extra))
    lr = expected_branches(parser().parse_args(base + ["--kind", "lr_bump"] + extra))
    assert len(cb) == 2 + 1 + 2 + 1 + 16 + (2 + 6) + 2 * 5 and "Px10|cool@0" in cb and "Pl_0" in cb
    assert len(lr) == 2 + 1 + 2 + 1 + 16  # no G for lr_bump -> no planted/dose arms
    n = {
        k: len(expected_branches(parser().parse_args(base + ["--kind", k, "--fault-bit", "29"] + extra)))
        for k in ("sdc_grad", "v_bitflip", "m_bitflip")
    }
    # a moment flip leaves g alone: no G, so no planted arms, and no U_mv@0 / U_m@0 / U_v@0 (each is bitwise B0)
    assert n["v_bitflip"] == n["m_bitflip"] == len(lr) - 3
    assert n["sdc_grad"] == len(cb)  # a gradient flip is a G, planted like corrupt_batch's


def test_parity_primed_at_fork_start_flags_a_v_flip_on_the_first_step():
    # The fork's first step is the fault step: unprimed, parity_residual has no previous m, v and the flip reads clean.
    from research.autopsy.faults import parity_residual, v_bitflip
    from research.autopsy.mve import _parity_summary, _prime_parity
    from research.harness.fork import compose

    for flip, primed in ((True, True), (True, False), (False, True)):
        model, opt = _model_opt()
        for p in model.parameters():
            p.grad = None  # a freshly restored fork
        pr = {}
        par = parity_residual(pr)
        flip_hook = v_bitflip({}, 1, 1, target="weight")[0] if flip else None
        pre = compose(_prime_parity(par, 1) if primed else None, flip_hook)
        if pre:
            pre(1, {"model": model, "opt": opt})
        assert all(p.grad is None for p in model.parameters())  # priming leaves no grads behind
        model(torch.randn(4, 8)).square().mean().backward()
        opt.step()
        par(1, {"model": model, "opt": opt})
        if not primed:
            assert pr == {}  # the gap the priming closes
            continue
        assert pr[1]["m"] < 1e-4 and (pr[1]["v"] > 1.0) == flip  # honest steps sit at fp32 rounding
    s = _parity_summary({0: {"m": 1, "v": 2}, 5: {"m": 3, "v": 4}, 20: {"m": 5, "v": 6}}, inject=5)
    assert s == {"window": {"5": {"m": 3, "v": 4}}, "max_outside": {"m": 5, "v": 6}}


def test_opt_placebo_moves_every_moment_one_ulp_and_keeps_v_nonnegative():
    model, opt = _model_opt()
    params = list(model.parameters())
    opt.state[params[0]]["exp_avg_sq"][0, 0] = 0.0  # nudged down it would go negative
    m0 = [opt.state[p]["exp_avg"].clone() for p in params]
    v0 = [opt.state[p]["exp_avg_sq"].clone() for p in params]
    w0 = [p.detach().clone() for p in params]
    hook = ulp_placebo(3, at=7, channel="opt")
    hook(6, {"model": model, "opt": opt})
    assert all(torch.equal(opt.state[p]["exp_avg"], m) for p, m in zip(params, m0))  # other steps: no-op
    hook(7, {"model": model, "opt": opt})
    for p, m, v, w in zip(params, m0, v0, w0):
        m1, v1 = opt.state[p]["exp_avg"], opt.state[p]["exp_avg_sq"]
        up = torch.nextafter(m, torch.full_like(m, float("inf")))
        down = torch.nextafter(m, torch.full_like(m, -float("inf")))
        assert bool(((m1 == up) | (m1 == down)).all()) and torch.equal(p.detach(), w)
        assert bool((v1 >= 0).all()) and bool((v1[v > 0] != v[v > 0]).all())
    with pytest.raises(ValueError):
        ulp_placebo(0, 7, "m")  # a typo must not silently fall through to the gradient channel


def test_data_order_replays_the_same_slots_in_a_new_order(monkeypatch):
    from research.data import prepare

    monkeypatch.setattr(prepare, "get_batch", lambda split, step, batch, block, data_dir, device: step)
    model, _ = _model_opt()
    cfg = {"train": {"batch_size": 2, "max_steps": 20}, "model": {"block_size": 4}}
    orders = []
    for seed in (0, 1):
        hook, seen = data_order(seed, 10, cfg, "unused"), []
        for step in range(10, 20):
            ctx = {"model": model, "batch": None}
            hook(step, ctx)
            seen.append(ctx["batch"])
        orders.append(seen)
    assert sorted(orders[0]) == list(range(10, 20)) and orders[0] != orders[1]
    with pytest.raises(NotImplementedError):
        data_order(0, 10, {**cfg, "train": {**cfg["train"], "grad_accum": 2}}, "unused")


def test_chaos_battery_runs_only_bstar_zero_pert_placebos_and_data_forks():
    argv = ["--config", "x", "--inject-step", "160", "--battery", "chaos", "--placebo", "4"]
    names = expected_branches(parser().parse_args(argv + ["--placebo-channels", "w,g,opt", "--data-forks", "2"]))
    assert names[:2] == ["B*", "Pl_0"] and len(names) == 2 + 3 * 4 + 2
    assert "B0" not in names and "Pl_opt#3" in names and "Dord#1" in names


def test_report_zero_pert_check_borrowed_floor_and_release_rows(tmp_path):
    cfg = {"seed": 0, "optim": {"betas": [0.9, 0.95]}}
    chaos = ["mve", "--config", "x", "--inject-step", "10", "--battery", "chaos", "--placebo", "3"]
    spike = ["mve", "--config", "x", "--kind", "corrupt_batch", "--inject-step", "10", "--c2-taus", "0"]

    def run(argv, vals):
        branches = {n: {"val": {"99": x}, "survival": 1} for n, x in vals.items()}
        return {"stamp": {"argv": argv, "cfg": cfg}, "horizons": [99], "branches": branches}

    pl = {"B*": 2.0, "Pl_0": 2.0, "Pl_w#0": 2.1, "Pl_w#1": 1.9, "Pl_w#2": 2.0}
    json.dump(run(chaos + ["--placebo-channels", "w"], pl), open(tmp_path / "chaos.json", "w"))
    json.dump(run(spike, {"B*": 2.0, "B0": 2.5}), open(tmp_path / "spike.json", "w"))
    report.main([str(tmp_path)], str(tmp_path / "out"))
    inst = json.load(open(tmp_path / "out" / "report.json"))
    sealed = json.load(open(tmp_path / "out" / "sealed.json"))
    assert inst["chaos"]["zero_pert_exact"] and inst["chaos"]["sigma_unit"]["source"] == "own"
    assert inst["spike"]["sigma_unit"]["source"] == "borrowed"  # same trunk + inject step -> chaos run's floor
    assert abs(sealed["spike"]["_d_end"]["B0"] - 5.0) < 1e-6  # D = 0.5 over sigma = 0.1
    rows = [json.loads(line) for line in open(tmp_path / "out" / "chaos_forks.jsonl")]
    assert len(rows) == 4 and {r["arm"] for r in rows} == {"zero", "w"}


def test_report_floor_and_dose_units():
    argv = [
        "mve",
        "--config",
        "x",
        "--kind",
        "corrupt_batch",
        "--inject-step",
        "10",
        "--taus",
        "0",
        "--c2-taus",
        "0",
        "--dts",
        "8",
        "--placebo",
        "3",
        "--placebo-channels",
        "w",
        "--plant-coefs",
        "10",
    ]
    v = {"B*": 2.0, "Pl_w#0": 2.1, "Pl_w#1": 1.9, "Pl_w#2": 2.0, "Px10": 2.5, "Px10|U_mv@0": 2.1}
    res = {
        "stamp": {"argv": argv, "cfg": {"seed": 0, "optim": {"betas": [0.9, 0.95]}}},
        "horizons": [99],
        "branches": {
            n: {"val": {"99": x}, "survival": 1, "metrics": {"xbar": 0.3} if n == "Px10" else {}} for n, x in v.items()
        },
    }
    inst, sealed = report._instance(res)
    assert abs(inst["sigma_chaos"]["w"]["99"] - 0.1) < 1e-9
    dose = sealed["_dose"]["10"]
    assert abs(dose["d"]["none"] - 5.0) < 1e-6 and abs(dose["indirect_share"] - 0.8) < 1e-6 and dose["xbar"] == 0.3


def test_repair_battery_times_detector_arms_at_the_rml_lag(monkeypatch):
    # Without --detector-t0 the detector-timed baselines (skip-step, rollback) silently dropped out of C-4.
    from types import SimpleNamespace

    from research.autopsy import mve
    from research.harness import fork

    seen = {}
    monkeypatch.setattr(fork, "make_branches", lambda cfg, recipe, **kw: {})
    monkeypatch.setattr(fork, "make_baseline_branches", lambda cfg, recipe, **kw: seen.update(kw) or {})
    recipe = SimpleNamespace(inject_step=160, width=1, pre_step=None)
    cfg = {"optim": {"grad_clip": 1.0}}
    mve.repair_battery(cfg, recipe, "unused", {"meta": {}}, lag=3)
    assert seen["detector_t0"] == 163 == undo.undo_step(160, 3) - 1  # RML acts at the top of 164
    mve.repair_battery(cfg, recipe, "unused", {"meta": {}}, lag=3, detector_t0=170)
    assert seen["detector_t0"] == 170
