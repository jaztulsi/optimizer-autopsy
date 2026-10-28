"""CPU checks for research/autopsy: the chaos-floor placebo, RML-sel / cooldown edits, branch bookkeeping, report."""

import torch

from research.autopsy import report, undo
from research.autopsy.mve import expected_branches, parser
from research.autopsy.probes import nudge_ulp


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
    for scale, shrink in ((0.0, False), (1.0, True)):
        model, opt = _model_opt()
        out = {}
        pre = undo.as_cooldown_pre_step(_log(model, scale), 6, "v", out)
        pre(6, {"model": model, "opt": opt})
        assert (opt.param_groups[0]["lr"] < 1e-3) == shrink and out["lr0"] == 1e-3


def test_expected_branches_count_placebo_dose_and_cooldown_arms():
    base = ["--config", "x", "--inject-step", "160", "--taus", "0", "--c2-taus", "0", "--dts", "8"]
    extra = ["--placebo", "8", "--plant-coefs", "1,10", "--dose-taus", "0"]
    cb = expected_branches(parser().parse_args(base + ["--kind", "corrupt_batch"] + extra))
    lr = expected_branches(parser().parse_args(base + ["--kind", "lr_bump"] + extra))
    assert len(cb) == 2 + 1 + 2 + 16 + (2 + 6) + 2 * 5 and "Px10|cool@0" in cb
    assert len(lr) == 2 + 1 + 2 + 16  # no G for lr_bump -> no planted/dose arms


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
